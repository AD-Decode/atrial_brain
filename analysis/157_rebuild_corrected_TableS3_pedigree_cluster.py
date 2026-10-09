#!/usr/bin/env python3

from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path("/data/qiallab/Framingham")
RAW = ROOT / "downloads" / "clinical"
RES = ROOT / "results"

MAIN = RES / "neurocardiac_metadata_analysis_ready_metabolic.csv"

LAVI_RESULT = (
    RES
    / "kinship_sensitivity_corrected"
    / "LAVI_middle_occipital_corrected_pedigree_sensitivity.csv"
)

OUTDIR = RES / "kinship_sensitivity_corrected"
OUTDIR.mkdir(parents=True, exist_ok=True)

OUTCSV = OUTDIR / "TableS3_corrected_primary_pedigree_cluster.csv"
OUTXLSX = OUTDIR / "TableS3_corrected_primary_pedigree_cluster.xlsx"
DIAGCSV = OUTDIR / "TableS3_corrected_primary_pedigree_diagnostics.csv"

print("=" * 115)
print("157: REBUILD TABLE S3 — EXACT PRIMARY MODELS + PEDIGREE-FAMILY CLUSTERED INFERENCE")
print("=" * 115)


# ============================================================================
# FIND PEDIGREE FILE
# ============================================================================

ped_files = sorted(
    RAW.glob("*pht000183*.txt.gz")
)

if not ped_files:
    raise RuntimeError(
        "Could not find pht000183 pedigree file."
    )

PED = ped_files[0]

print("\nMain data:")
print(MAIN)

print("\nPedigree:")
print(PED)


# ============================================================================
# LOAD
# ============================================================================

df = pd.read_csv(MAIN)

ped = pd.read_csv(
    PED,
    sep="\t",
    compression="gzip",
    comment="#",
    low_memory=False,
)

print("\nMain shape:", df.shape)
print("Pedigree shape:", ped.shape)


# ============================================================================
# ID NORMALIZATION + PEDIGREE MERGE
# ============================================================================

def norm_id(s):

    s = (
        s.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )

    return s.replace({
        "nan": np.nan,
        "None": np.nan,
        "": np.nan,
    })


if "shareid" not in df.columns:
    raise RuntimeError("shareid missing from main dataset.")

if "shareid" not in ped.columns:
    raise RuntimeError("shareid missing from pedigree dataset.")

df["join_id"] = norm_id(
    df["shareid"]
)

ped["join_id"] = norm_id(
    ped["shareid"]
)

ped = (
    ped[
        [
            "join_id",
            "pedno",
            "fshare",
            "mshare",
            "twinid",
            "idtype",
        ]
    ]
    .drop_duplicates("join_id")
)

df = df.merge(
    ped,
    on="join_id",
    how="left",
)

print("\nAfter pedigree merge:", df.shape)
print(
    "Participants with pedno:",
    df["pedno"].notna().sum()
)
print(
    "Unique pedigrees:",
    df["pedno"].nunique()
)


# ============================================================================
# REPRODUCE DERIVED VARIABLES USED IN ORIGINAL TIER-2 PIPELINE
# ============================================================================

# Exact total cortical surface area construction from script 32
for c in [
    "lh_WhiteSurfArea_DesKil_area",
    "rh_WhiteSurfArea_DesKil_area",
]:
    if c not in df.columns:
        raise RuntimeError(
            f"Required cortical surface-area column missing: {c}"
        )

df["TotalCorticalSurfaceArea"] = (
    pd.to_numeric(
        df["lh_WhiteSurfArea_DesKil_area"],
        errors="coerce",
    )
    +
    pd.to_numeric(
        df["rh_WhiteSurfArea_DesKil_area"],
        errors="coerce",
    )
)

print(
    "Constructed TotalCorticalSurfaceArea:",
    int(df["TotalCorticalSurfaceArea"].notna().sum()),
    "nonmissing"
)


# Sex
if "_sex" not in df.columns:

    if "sex_clinical" not in df.columns:
        raise RuntimeError(
            "Neither _sex nor sex_clinical is available."
        )

    df["_sex"] = pd.to_numeric(
        df["sex_clinical"],
        errors="coerce",
    )


# Diabetes
if "diabetes_history_any" not in df.columns:

    if "diabetes_history_nearest_exam" not in df.columns:
        raise RuntimeError(
            "Cannot derive diabetes_history_any."
        )

    x = pd.to_numeric(
        df["diabetes_history_nearest_exam"],
        errors="coerce",
    )

    d = pd.Series(
        np.nan,
        index=df.index,
        dtype=float,
    )

    d.loc[x == 0] = 0
    d.loc[x.isin([1, 2])] = 1

    df["diabetes_history_any"] = d


# BSA / indexed LV measures if needed
need_indexed = [
    x for x in ["LVESVi", "LV_MASSi"]
    if x not in df.columns
]

if need_indexed:

    if (
        "height_in" not in df.columns
        or "weight_lb" not in df.columns
    ):
        raise RuntimeError(
            "Need height_in and weight_lb to derive indexed LV variables."
        )

    height_cm = (
        pd.to_numeric(
            df["height_in"],
            errors="coerce",
        )
        * 2.54
    )

    weight_kg = (
        pd.to_numeric(
            df["weight_lb"],
            errors="coerce",
        )
        * 0.45359237
    )

    BSA = np.sqrt(
        height_cm * weight_kg / 3600.0
    )

    if "LVESVi" not in df.columns:

        if "LVESV" not in df.columns:
            raise RuntimeError(
                "Cannot derive LVESVi: LVESV missing."
            )

        df["LVESVi"] = (
            pd.to_numeric(
                df["LVESV"],
                errors="coerce",
            )
            / BSA
        )

        print("Derived LVESVi from LVESV/BSA.")

    if "LV_MASSi" not in df.columns:

        if "LV_MASS" not in df.columns:
            raise RuntimeError(
                "Cannot derive LV_MASSi: LV_MASS missing."
            )

        df["LV_MASSi"] = (
            pd.to_numeric(
                df["LV_MASS"],
                errors="coerce",
            )
            / BSA
        )

        print("Derived LV_MASSi from LV_MASS/BSA.")


# ============================================================================
# EXACT ORIGINAL STANDARDIZATION
#
# Script 32 uses pandas .std() with default ddof=1.
# ============================================================================

def z(x):

    x = pd.to_numeric(
        x,
        errors="coerce",
    )

    sd = x.std()

    if not np.isfinite(sd) or sd == 0:
        return x * np.nan

    return (
        (x - x.mean())
        / sd
    )


# ============================================================================
# PRIMARY TARGETS
# ============================================================================

TARGETS = [
    {
        "analysis":
            "LVEF × APOE4 → left middle temporal volume",
        "region":
            "lh_middletemporal_vol",
        "heart":
            "LVEF",
        "metric":
            "volume",
        "expected_beta":
            0.2333,
        "expected_p":
            0.00176,
    },

    {
        "analysis":
            "LVESVi × APOE4 → right superior frontal area",
        "region":
            "rh_superiorfrontal_area",
        "heart":
            "LVESVi",
        "metric":
            "surface_area",
        "expected_beta":
            -0.1703,
        "expected_p":
            1.97e-4,
    },

    {
        "analysis":
            "LV mass index × APOE4 → right lingual volume",
        "region":
            "rh_lingual_vol",
        "heart":
            "LV_MASSi",
        "metric":
            "volume",
        "expected_beta":
            0.3053,
        "expected_p":
            2.16e-4,
    },

    {
        "analysis":
            "LV mass index × APOE4 → right lingual area",
        "region":
            "rh_lingual_area",
        "heart":
            "LV_MASSi",
        "metric":
            "surface_area",
        "expected_beta":
            0.2259,
        "expected_p":
            7.09e-4,
    },
]


# ============================================================================
# EXACT M3 MODEL DATA CONSTRUCTION
# ============================================================================

def make_model_data(
    data,
    region,
    heart,
    metric,
    require_pedigree=False,
):

    # Exact phenotype definition from script 32.
    y = pd.to_numeric(
        data[region],
        errors="coerce",
    )

    if metric == "volume":

        icv = pd.to_numeric(
            data["IntraCranialVol"],
            errors="coerce",
        )

        y = y / icv

    dat = pd.DataFrame({
        "y":
            y,

        "heart":
            pd.to_numeric(
                data[heart],
                errors="coerce",
            ),

        "age":
            pd.to_numeric(
                data["age_at_mri"],
                errors="coerce",
            ),

        "sex":
            pd.to_numeric(
                data["_sex"],
                errors="coerce",
            ),

        "interval":
            pd.to_numeric(
                data["abs_delta_years"],
                errors="coerce",
            ),

        "BMI":
            pd.to_numeric(
                data["BMI_nearest_exam"],
                errors="coerce",
            ),

        "SBP":
            pd.to_numeric(
                data["SBP_nearest_exam"],
                errors="coerce",
            ),

        "smoking":
            pd.to_numeric(
                data["current_smoker_nearest_exam"],
                errors="coerce",
            ),

        "diabetes":
            pd.to_numeric(
                data["diabetes_history_any"],
                errors="coerce",
            ),

        "APOE4":
            pd.to_numeric(
                data["APOE4_carrier"],
                errors="coerce",
            ),

        "pedno":
            data["pedno"],
    })

    if metric == "surface_area":

        if "TotalCorticalSurfaceArea" not in data.columns:
            raise RuntimeError(
                "TotalCorticalSurfaceArea is required for surface-area models "
                "but is not present in this dataset."
            )

        dat["total_area"] = pd.to_numeric(
            data["TotalCorticalSurfaceArea"],
            errors="coerce",
        )

    need = [
        "y",
        "heart",
        "age",
        "sex",
        "interval",
        "BMI",
        "SBP",
        "smoking",
        "diabetes",
        "APOE4",
    ]

    if metric == "surface_area":
        need.append(
            "total_area"
        )

    if require_pedigree:
        need.append(
            "pedno"
        )

    dat = (
        dat
        .dropna(subset=need)
        .copy()
    )

    # Exact original standardization,
    # performed within the actual analysis sample.
    dat["y_z"] = z(
        dat["y"]
    )

    dat["heart_z"] = z(
        dat["heart"]
    )

    dat["age_z"] = z(
        dat["age"]
    )

    dat["interval_z"] = z(
        dat["interval"]
    )

    dat["BMI_z"] = z(
        dat["BMI"]
    )

    dat["SBP_z"] = z(
        dat["SBP"]
    )

    dat["heart_x_APOE4"] = (
        dat["heart_z"]
        * dat["APOE4"]
    )

    if metric == "surface_area":

        dat["total_area_z"] = z(
            dat["total_area"]
        )

    return dat


# ============================================================================
# FIT EXACT M3
# ============================================================================

def fit_m3(
    dat,
    metric,
    covariance,
):

    xcols = [
        "heart_z",
        "age_z",
        "sex",
        "interval_z",
    ]

    if metric == "surface_area":

        xcols.append(
            "total_area_z"
        )

    xcols += [
        "BMI_z",
        "SBP_z",
        "smoking",
        "diabetes",
        "APOE4",
        "heart_x_APOE4",
    ]

    X = sm.add_constant(
        dat[xcols].astype(float),
        has_constant="add",
    )

    model = sm.OLS(
        dat["y_z"].astype(float),
        X,
    )

    if covariance == "HC3":

        fit = model.fit(
            cov_type="HC3"
        )

    elif covariance == "PEDIGREE_CLUSTER":

        if dat["pedno"].isna().any():
            raise RuntimeError(
                "Missing pedno in clustered model."
            )

        fit = model.fit(
            cov_type="cluster",
            cov_kwds={
                "groups":
                    dat["pedno"],
            },
        )

    else:
        raise ValueError(
            covariance
        )

    term = "heart_x_APOE4"

    ci = fit.conf_int().loc[
        term
    ]

    return {
        "N":
            int(fit.nobs),

        "N_APOE4_noncarrier":
            int(
                (dat["APOE4"] == 0).sum()
            ),

        "N_APOE4_carrier":
            int(
                (dat["APOE4"] == 1).sum()
            ),

        "N_pedigrees":
            (
                int(dat["pedno"].nunique())
                if dat["pedno"].notna().any()
                else np.nan
            ),

        "beta_interaction":
            float(
                fit.params[term]
            ),

        "SE":
            float(
                fit.bse[term]
            ),

        "CI_low":
            float(
                ci.iloc[0]
            ),

        "CI_high":
            float(
                ci.iloc[1]
            ),

        "P":
            float(
                fit.pvalues[term]
            ),

        "R2":
            float(
                fit.rsquared
            ),
    }


# ============================================================================
# RUN FOUR PRIMARY VENTRICULAR FINDINGS
# ============================================================================

diagnostic_rows = []
table_rows = []

print("\n" + "=" * 115)
print("FOUR PRIMARY VENTRICULAR FINDINGS")
print("=" * 115)

for target in TARGETS:

    region = target["region"]
    heart = target["heart"]
    metric = target["metric"]

    print(
        "\n",
        target["analysis"]
    )

    # ------------------------------------------------------------
    # Full reference sample, HC3
    # ------------------------------------------------------------

    d_full = make_model_data(
        df,
        region,
        heart,
        metric,
        require_pedigree=False,
    )

    ref = fit_m3(
        d_full,
        metric,
        "HC3",
    )

    diagnostic_rows.append({
        "analysis":
            target["analysis"],

        "sample":
            "FULL_REFERENCE_HC3",

        **ref,
    })

    print(
        "  FULL HC3:",
        f"N={ref['N']}",
        f"beta={ref['beta_interaction']:.6f}",
        f"P={ref['P']:.6g}",
    )

    # ------------------------------------------------------------
    # Reproduction check
    # ------------------------------------------------------------

    if ref["N"] != 765:

        raise RuntimeError(
            f"\nREPRODUCTION FAILURE for {target['analysis']}\n"
            f"Expected N=765 but obtained N={ref['N']}.\n"
            "Do not interpret pedigree results until resolved."
        )

    if (
        abs(
            ref["beta_interaction"]
            - target["expected_beta"]
        )
        > 0.002
    ):

        raise RuntimeError(
            f"\nREPRODUCTION FAILURE for {target['analysis']}\n"
            f"Expected beta about {target['expected_beta']}, "
            f"obtained {ref['beta_interaction']}.\n"
            "Do not interpret pedigree results until resolved."
        )

    # ------------------------------------------------------------
    # Pedigree-complete sample
    # ------------------------------------------------------------

    d_ped = make_model_data(
        df,
        region,
        heart,
        metric,
        require_pedigree=True,
    )

    ped_hc3 = fit_m3(
        d_ped,
        metric,
        "HC3",
    )

    ped_cluster = fit_m3(
        d_ped,
        metric,
        "PEDIGREE_CLUSTER",
    )

    # Coefficients MUST be identical because only covariance changes.
    if not np.isclose(
        ped_hc3["beta_interaction"],
        ped_cluster["beta_interaction"],
        rtol=0,
        atol=1e-12,
    ):

        raise RuntimeError(
            "HC3 and cluster coefficients differ on identical sample."
        )

    diagnostic_rows.append({
        "analysis":
            target["analysis"],

        "sample":
            "PEDIGREE_COMPLETE_HC3",

        **ped_hc3,
    })

    diagnostic_rows.append({
        "analysis":
            target["analysis"],

        "sample":
            "PEDIGREE_COMPLETE_FAMILY_CLUSTER",

        **ped_cluster,
    })

    fam_sizes = (
        d_ped
        .groupby("pedno")
        .size()
    )

    print(
        "  PEDIGREE HC3:",
        f"N={ped_hc3['N']}",
        f"beta={ped_hc3['beta_interaction']:.6f}",
        f"P={ped_hc3['P']:.6g}",
    )

    print(
        "  PEDIGREE CLUSTER:",
        f"N={ped_cluster['N']}",
        f"families={ped_cluster['N_pedigrees']}",
        f"beta={ped_cluster['beta_interaction']:.6f}",
        f"SE={ped_cluster['SE']:.6f}",
        f"P={ped_cluster['P']:.6g}",
    )

    print(
        "  Families >1 participant:",
        int(
            (fam_sizes > 1).sum()
        )
    )

    table_rows.append({
        "Analysis":
            target["analysis"],

        "Outcome":
            region,

        "Cardiac/metabolic predictor":
            heart,

        "N":
            ped_cluster["N"],

        "N pedigrees":
            ped_cluster["N_pedigrees"],

        "Interaction beta":
            ped_cluster["beta_interaction"],

        "Cluster-robust SE":
            ped_cluster["SE"],

        "95% CI low":
            ped_cluster["CI_low"],

        "95% CI high":
            ped_cluster["CI_high"],

        "Cluster-robust P":
            ped_cluster["P"],

        "R²":
            ped_cluster["R2"],

        "Pedigree-cluster robust":
            (
                "Yes"
                if ped_cluster["P"] < 0.05
                else "No"
            ),
    })


# ============================================================================
# ADD VALIDATED PRIMARY EXAM-8 LAVI RESULT FROM SCRIPT 156
# ============================================================================

if not LAVI_RESULT.exists():

    raise RuntimeError(
        f"Corrected LAVI result not found: {LAVI_RESULT}"
    )

lavi = pd.read_csv(
    LAVI_RESULT
)

wanted = lavi[
    lavi["analysis"]
    ==
    "PRIMARY_EXAM8_PEDIGREE_FAMILY_CLUSTER"
].copy()

if len(wanted) != 1:

    raise RuntimeError(
        "Could not uniquely identify corrected LAVI clustered row."
    )

r = wanted.iloc[0]

print("\n" + "=" * 115)
print("ADDING VALIDATED LAVI RESULT")
print("=" * 115)

print(
    f"N={int(r['N'])}",
    f"families={int(r['N_families'])}",
    f"beta={r['beta_interaction']:.6f}",
    f"SE={r['SE']:.6f}",
    f"P={r['P']:.6g}",
)

table_rows.append({
    "Analysis":
        "LAVI × APOE4 → left middle-occipital cortical thickness variability",

    "Outcome":
        "lh_G_occipital_middle_tksd",

    "Cardiac/metabolic predictor":
        "LAVI_max",

    "N":
        int(r["N"]),

    "N pedigrees":
        int(r["N_families"]),

    "Interaction beta":
        float(r["beta_interaction"]),

    "Cluster-robust SE":
        float(r["SE"]),

    "95% CI low":
        float(r["CI_low"]),

    "95% CI high":
        float(r["CI_high"]),

    "Cluster-robust P":
        float(r["P"]),

    "R²":
        float(r["R2"]),

    "Pedigree-cluster robust":
        (
            "Yes"
            if float(r["P"]) < 0.05
            else "No"
        ),
})


# ============================================================================
# BUILD TABLE
#
# IMPORTANT:
# The exploratory LVEDVi × APOE4 × HbA1c row from the old Table S3 is
# intentionally NOT copied here. Its original pedigree sensitivity also used
# the simplified age/sex script and should be audited separately rather than
# silently retained in a corrected table.
# ============================================================================

table = pd.DataFrame(
    table_rows
)

diagnostics = pd.DataFrame(
    diagnostic_rows
)

table.to_csv(
    OUTCSV,
    index=False,
)

diagnostics.to_csv(
    DIAGCSV,
    index=False,
)


# ============================================================================
# NOTES
# ============================================================================

notes = pd.DataFrame({
    "Notes": [
        (
            "Sensitivity analyses accounting for familial clustering "
            "in the Framingham cohort."
        ),

        (
            "For the four ventricular cardiac–brain associations, "
            "the original fully adjusted M3 APOE4 models were refit "
            "on pedigree-complete subsets with standard errors clustered "
            "by pedigree family number (pedno)."
        ),

        (
            "The ventricular M3 models included standardized cardiac phenotype, "
            "APOE ε4 carrier status, cardiac phenotype × APOE ε4 interaction, "
            "age at brain MRI, sex, absolute MRI–CMR interval, BMI, systolic "
            "blood pressure, current smoking, and diabetes history."
        ),

        (
            "Regional cortical volumes were normalized to intracranial volume "
            "before standardization. Surface-area models additionally adjusted "
            "for total cortical surface area."
        ),

        (
            "For the atrial association, the primary Exam 8 model was restricted "
            "to participants with pedigree information and retained the original "
            "atrial base-model covariates: age at brain MRI, sex, examination-to-MRI "
            "interval, BMI, systolic blood pressure, current smoking, diabetes history, "
            "and APOE ε4 carrier status."
        ),

        (
            "Continuous outcomes and continuous predictors were standardized "
            "within each analysis sample, consistent with the corresponding "
            "primary model implementation."
        ),

        (
            "Pedigree-family clustered inference accounts for within-family "
            "dependence but is not equivalent to fitting a genetic kinship matrix "
            "or a pedigree random-effects model."
        ),

        (
            "The prior LAVI row based on left lateral occipital thickness variability "
            "was erroneous. The corrected analysis evaluates the prespecified "
            "left middle-occipital cortical-thickness variability phenotype."
        ),

        (
            "The exploratory LVEDVi × APOE4 × HbA1c row from the previous Table S3 "
            "is omitted from this corrected primary table pending a separate audit "
            "against its exact original covariate specification."
        ),
    ]
})


# ============================================================================
# WRITE XLSX
# ============================================================================

with pd.ExcelWriter(
    OUTXLSX,
    engine="openpyxl",
) as writer:

    table.to_excel(
        writer,
        sheet_name="Pedigree sensitivity",
        index=False,
    )

    notes.to_excel(
        writer,
        sheet_name="Notes",
        index=False,
    )

    diagnostics.to_excel(
        writer,
        sheet_name="Reproduction diagnostics",
        index=False,
    )


# ============================================================================
# DISPLAY
# ============================================================================

print("\n" + "=" * 115)
print("CORRECTED PRIMARY TABLE S3")
print("=" * 115)

print(
    table.to_string(
        index=False,
        float_format=lambda x: f"{x:.6g}",
    )
)

print("\n" + "=" * 115)
print("REPRODUCTION DIAGNOSTICS")
print("=" * 115)

print(
    diagnostics.to_string(
        index=False,
        float_format=lambda x: f"{x:.6g}",
    )
)

print("\nSaved:")
print(OUTCSV)
print(OUTXLSX)
print(DIAGCSV)

print("\nIMPORTANT:")
print(
    "No file in FINAL_MANUSCRIPT_REPO_20260928 was overwritten."
)
print(
    "Review reproduction checks and pedigree results before promoting this table."
)
