#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path("/data/qiallab/Framingham")
RAW = ROOT / "downloads" / "clinical"
RES = ROOT / "results"

ATRIA = RES / "neurocardiac_metadata_with_atria.csv"

OUTDIR = RES / "kinship_sensitivity_corrected"
OUTDIR.mkdir(parents=True, exist_ok=True)

OUTCOME = "lh_G_occipital_middle_tksd"
ATRIAL = "LAVI_max"

print("=" * 110)
print("156: CORRECTED PEDIGREE-FAMILY CLUSTER SENSITIVITY")
print("LAVI x APOE4 -> LEFT MIDDLE-OCCIPITAL CORTICAL THICKNESS VARIABILITY")
print("=" * 110)


# ============================================================================
# FIND PEDIGREE FILE
# ============================================================================

ped_files = sorted(RAW.glob("*pht000183*.txt.gz"))

if not ped_files:
    raise RuntimeError("Could not find pht000183 pedigree data file.")

PED = ped_files[0]

print("\nAtrial data:")
print(ATRIA)

print("\nPedigree file:")
print(PED)


# ============================================================================
# LOAD
# ============================================================================

df = pd.read_csv(ATRIA)

ped = pd.read_csv(
    PED,
    sep="\t",
    compression="gzip",
    comment="#",
    low_memory=False,
)

print("\nRaw atrial dataframe:", df.shape)
print("Raw pedigree dataframe:", ped.shape)


# ============================================================================
# ID NORMALIZATION + PEDIGREE MERGE
# ============================================================================

def norm_id(s):
    s = s.astype(str).str.strip()
    s = s.str.replace(r"\.0$", "", regex=True)

    return s.replace({
        "nan": np.nan,
        "None": np.nan,
        "": np.nan,
    })


df["join_id"] = norm_id(df["shareid"])
ped["join_id"] = norm_id(ped["shareid"])

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
print("Subjects with pedigree ID:", df["pedno"].notna().sum())
print("Unique pedigrees:", df["pedno"].nunique())


# ============================================================================
# REPRODUCE VARIABLE DEFINITIONS FROM SCRIPT 42
# ============================================================================

df["_sex"] = pd.to_numeric(
    df["sex_clinical"],
    errors="coerce",
)


def diabetes_binary(x):
    x = pd.to_numeric(x, errors="coerce")

    out = pd.Series(
        np.nan,
        index=x.index,
        dtype=float,
    )

    out.loc[x == 0] = 0
    out.loc[x.isin([1, 2])] = 1

    return out


df["diabetes_history_any"] = diabetes_binary(
    df["diabetes_history_nearest_exam"]
)


# Recompute LV mass index if it is not already present.
if "LV_MASSi_recomputed" not in df.columns:

    needed_bsa = [
        "LV_MASS",
        "height_in",
        "weight_lb",
    ]

    missing_bsa = [
        x for x in needed_bsa
        if x not in df.columns
    ]

    if missing_bsa:
        raise RuntimeError(
            "Cannot construct LV_MASSi_recomputed. Missing: "
            + ", ".join(missing_bsa)
        )

    height_cm = (
        pd.to_numeric(
            df["height_in"],
            errors="coerce"
        )
        * 2.54
    )

    weight_kg = (
        pd.to_numeric(
            df["weight_lb"],
            errors="coerce"
        )
        * 0.45359237
    )

    BSA = np.sqrt(
        height_cm * weight_kg / 3600.0
    )

    df["LV_MASSi_recomputed"] = (
        pd.to_numeric(
            df["LV_MASS"],
            errors="coerce"
        )
        / BSA
    )


if "LVEF" not in df.columns:
    raise RuntimeError("LVEF not found.")


# ============================================================================
# REQUIRED VARIABLES
#
# Script 42 fits M0_ATRIAL_BASE on the common complete-case sample that
# also has LV mass index and LVEF available. Preserve that sample here.
# ============================================================================

BASE_VARS = [
    OUTCOME,
    ATRIAL,
    "age_at_mri",
    "_sex",
    "abs_years_exam_mri",
    "BMI_nearest_exam",
    "SBP_nearest_exam",
    "current_smoker_nearest_exam",
    "diabetes_history_any",
    "APOE4_carrier",
]

COMMON_VARS = (
    BASE_VARS
    + [
        "LV_MASSi_recomputed",
        "LVEF",
    ]
)


missing = [
    x for x in COMMON_VARS
    if x not in df.columns
]

if missing:
    raise RuntimeError(
        "Missing required variables: "
        + ", ".join(missing)
    )


for c in COMMON_VARS:
    df[c] = pd.to_numeric(
        df[c],
        errors="coerce",
    )


# ============================================================================
# STANDARDIZATION
# ============================================================================

def zscore(x):
    x = pd.to_numeric(
        x,
        errors="coerce",
    )

    return (
        (x - x.mean())
        / x.std(ddof=0)
    )


# ============================================================================
# MODEL DATA
# ============================================================================

def prepare_model_data(data):

    d = (
        data[
            COMMON_VARS
            + ["pedno"]
        ]
        .copy()
        .dropna(
            subset=COMMON_VARS
        )
    )

    d["y_z"] = zscore(
        d[OUTCOME]
    )

    d["atrial_z"] = zscore(
        d[ATRIAL]
    )

    d["age_z"] = zscore(
        d["age_at_mri"]
    )

    d["interval_z"] = zscore(
        d["abs_years_exam_mri"]
    )

    d["BMI_z"] = zscore(
        d["BMI_nearest_exam"]
    )

    d["SBP_z"] = zscore(
        d["SBP_nearest_exam"]
    )

    d["atrial_x_APOE4"] = (
        d["atrial_z"]
        * d["APOE4_carrier"]
    )

    return d


Xvars = [
    "atrial_z",
    "age_z",
    "_sex",
    "interval_z",
    "BMI_z",
    "SBP_z",
    "current_smoker_nearest_exam",
    "diabetes_history_any",
    "APOE4_carrier",
    "atrial_x_APOE4",
]


# ============================================================================
# FITTER
# ============================================================================

def fit_one(d, label, covariance):

    X = sm.add_constant(
        d[Xvars].astype(float),
        has_constant="add",
    )

    model = sm.OLS(
        d["y_z"].astype(float),
        X,
    )

    if covariance == "HC3":

        fit = model.fit(
            cov_type="HC3"
        )

    elif covariance == "PEDIGREE_CLUSTER":

        if d["pedno"].isna().any():
            raise RuntimeError(
                "Missing pedno in pedigree-cluster model."
            )

        fit = model.fit(
            cov_type="cluster",
            cov_kwds={
                "groups": d["pedno"],
            },
        )

    else:
        raise ValueError(
            f"Unknown covariance: {covariance}"
        )

    term = "atrial_x_APOE4"

    ci = fit.conf_int().loc[term]

    return {
        "analysis": label,
        "outcome": OUTCOME,
        "predictor": ATRIAL,

        "N": int(fit.nobs),

        "N_APOE4_noncarrier": int(
            (d["APOE4_carrier"] == 0).sum()
        ),

        "N_APOE4_carrier": int(
            (d["APOE4_carrier"] == 1).sum()
        ),

        "N_families": (
            int(d["pedno"].nunique())
            if d["pedno"].notna().any()
            else np.nan
        ),

        "beta_interaction":
            fit.params[term],

        "SE":
            fit.bse[term],

        "CI_low":
            ci.iloc[0],

        "CI_high":
            ci.iloc[1],

        "P":
            fit.pvalues[term],

        "R2":
            fit.rsquared,

        "covariance":
            covariance,
    }


# ============================================================================
# DEFINE PRIMARY EXAM 8 AND ALL-ATRIAL SAMPLES
# ============================================================================

if "nearest_exam" not in df.columns:
    raise RuntimeError("nearest_exam not found.")

primary = df[
    df["nearest_exam"] == 8
].copy()

all_atrial = df[
    df[ATRIAL].notna()
].copy()


# ============================================================================
# 1. PRIMARY EXAM 8 FULL REFERENCE
#
# This should reproduce script 42 PRIMARY_EXAM8 / M0_ATRIAL_BASE:
# expected approximately:
# N = 451
# beta interaction = -0.54497
# P = 1.56e-5
# ============================================================================

d_primary_full = prepare_model_data(
    primary
)

print("\n" + "=" * 110)
print("PRIMARY EXAM 8 FULL REFERENCE")
print("=" * 110)

print("N:", len(d_primary_full))
print(
    "APOE4 noncarriers:",
    int(
        (d_primary_full["APOE4_carrier"] == 0).sum()
    )
)
print(
    "APOE4 carriers:",
    int(
        (d_primary_full["APOE4_carrier"] == 1).sum()
    )
)

rows = []

rows.append(
    fit_one(
        d_primary_full,
        "PRIMARY_EXAM8_FULL_HC3",
        "HC3",
    )
)


# ============================================================================
# 2. PRIMARY EXAM 8 PEDIGREE-COMPLETE SAMPLE, HC3
#
# This isolates the effect of restricting to participants with pedigree data.
# ============================================================================

primary_ped_raw = (
    primary[
        COMMON_VARS
        + ["pedno"]
    ]
    .dropna(
        subset=
        COMMON_VARS
        + ["pedno"]
    )
    .copy()
)

d_primary_ped = prepare_model_data(
    primary_ped_raw
)

d_primary_ped = (
    d_primary_ped
    .dropna(subset=["pedno"])
    .copy()
)

print("\n" + "=" * 110)
print("PRIMARY EXAM 8 PEDIGREE-COMPLETE SAMPLE")
print("=" * 110)

print("N:", len(d_primary_ped))
print(
    "APOE4 noncarriers:",
    int(
        (d_primary_ped["APOE4_carrier"] == 0).sum()
    )
)
print(
    "APOE4 carriers:",
    int(
        (d_primary_ped["APOE4_carrier"] == 1).sum()
    )
)

print(
    "Unique pedigrees:",
    d_primary_ped["pedno"].nunique()
)

fam_sizes = (
    d_primary_ped
    .groupby("pedno")
    .size()
)

print(
    "Families with >1 participant:",
    int(
        (fam_sizes > 1).sum()
    )
)

print(
    "Participants in multi-member families:",
    int(
        fam_sizes[
            fam_sizes > 1
        ].sum()
    )
)

rows.append(
    fit_one(
        d_primary_ped,
        "PRIMARY_EXAM8_PEDIGREE_HC3",
        "HC3",
    )
)


# ============================================================================
# 3. PRIMARY EXAM 8 PEDIGREE-COMPLETE SAMPLE,
#    FAMILY-CLUSTER ROBUST SE
#
# Same observations and coefficients as row 2.
# Only the covariance estimator changes.
# ============================================================================

rows.append(
    fit_one(
        d_primary_ped,
        "PRIMARY_EXAM8_PEDIGREE_FAMILY_CLUSTER",
        "PEDIGREE_CLUSTER",
    )
)


# ============================================================================
# 4. ALL-ATRIAL SENSITIVITY REFERENCE
#
# This should reproduce the previously observed broader sample:
# N = 476
# beta ~ -0.531711
# P ~ 8.94e-6
# ============================================================================

d_all_full = prepare_model_data(
    all_atrial
)

rows.append(
    fit_one(
        d_all_full,
        "SENSITIVITY_ALL_FULL_HC3",
        "HC3",
    )
)


# ============================================================================
# SAVE RESULTS
# ============================================================================

res = pd.DataFrame(rows)

outfile = (
    OUTDIR
    / "LAVI_middle_occipital_corrected_pedigree_sensitivity.csv"
)

res.to_csv(
    outfile,
    index=False,
)


print("\n" + "=" * 110)
print("CORRECTED LAVI PEDIGREE SENSITIVITY RESULTS")
print("=" * 110)

print(
    res.to_string(
        index=False,
        float_format=lambda x: f"{x:.6g}",
    )
)

print("\nSaved:")
print(outfile)
