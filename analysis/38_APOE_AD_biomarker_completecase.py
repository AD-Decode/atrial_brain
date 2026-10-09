#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

ROOT = Path("/data/qiallab/Framingham")
RES = ROOT / "results"
CLIN = ROOT / "downloads/clinical"

DATAFILE = RES / "neurocardiac_metadata_analysis_ready_metabolic.csv"
OUTDIR = RES / "tier3_AD_biomarkers_completecase"
OUTDIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATAFILE)
df["shareid"] = pd.to_numeric(df["shareid"], errors="coerce")

# ------------------------------------------------------------
# Load biomarkers
# ------------------------------------------------------------

def load_dbgap(pattern, keep):
    files = sorted(CLIN.glob(pattern))
    if not files:
        raise RuntimeError(f"No file found: {pattern}")

    x = pd.read_csv(
        files[0],
        sep="\t",
        compression="gzip",
        comment="#",
        low_memory=False
    )

    x["shareid"] = pd.to_numeric(
        x["shareid"],
        errors="coerce"
    )

    return x[keep].copy()


amy = load_dbgap(
    "*pht003309*.HMB-IRB-MDS.txt.gz",
    ["shareid", "amyloid40", "amyloid42"]
)

ptau = load_dbgap(
    "*pht015135*.HMB-IRB-MDS.txt.gz",
    ["shareid", "pTau_181"]
)

glial = load_dbgap(
    "*pht012879*.HMB-IRB-MDS.txt.gz",
    ["shareid", "gfap", "nf_l"]
)

for c in ["amyloid40", "amyloid42"]:
    amy[c] = pd.to_numeric(amy[c], errors="coerce")

ptau["pTau_181"] = pd.to_numeric(
    ptau["pTau_181"],
    errors="coerce"
)

for c in ["gfap", "nf_l"]:
    glial[c] = pd.to_numeric(
        glial[c],
        errors="coerce"
    )

df = (
    df.merge(amy, on="shareid", how="left")
      .merge(ptau, on="shareid", how="left")
      .merge(glial, on="shareid", how="left")
)

df["Abeta42_40"] = (
    df["amyloid42"]
    / df["amyloid40"]
)

df["log_pTau181"] = np.log1p(
    df["pTau_181"]
)

df["log_GFAP"] = np.log1p(
    df["gfap"]
)

df["log_NFL"] = np.log1p(
    df["nf_l"]
)

# ------------------------------------------------------------
# Cardiac indexing
# ------------------------------------------------------------

height_cm = pd.to_numeric(
    df["height_in"],
    errors="coerce"
) * 2.54

weight_kg = pd.to_numeric(
    df["weight_lb"],
    errors="coerce"
) * 0.45359237

df["BSA_m2"] = np.sqrt(
    height_cm * weight_kg / 3600.0
)

df["LVESVi"] = (
    pd.to_numeric(
        df["LVESV"],
        errors="coerce"
    ) / df["BSA_m2"]
)

df["LV_MASSi"] = (
    pd.to_numeric(
        df["LV_MASS"],
        errors="coerce"
    ) / df["BSA_m2"]
)

# ------------------------------------------------------------
# Brain scaling
# ------------------------------------------------------------

df["TotalCorticalSurfaceArea"] = (
    pd.to_numeric(
        df["lh_WhiteSurfArea_DesKil_area"],
        errors="coerce"
    )
    +
    pd.to_numeric(
        df["rh_WhiteSurfArea_DesKil_area"],
        errors="coerce"
    )
)

# ------------------------------------------------------------
# Diabetes
# ------------------------------------------------------------

dm = pd.to_numeric(
    df["diabetes_history_nearest_exam"],
    errors="coerce"
)

df["diabetes_any"] = np.where(
    dm.isna(),
    np.nan,
    dm.isin([1, 2]).astype(float)
)

# ------------------------------------------------------------
# Sex
# ------------------------------------------------------------

sex = df["sex_clinical"]

if pd.api.types.is_numeric_dtype(sex):
    df["_sex"] = pd.to_numeric(
        sex,
        errors="coerce"
    )
else:
    vals = list(
        sex.dropna()
        .astype(str)
        .unique()
    )

    if len(vals) != 2:
        raise RuntimeError(
            f"Unexpected sex categories: {vals}"
        )

    mapping = {
        vals[0]: 0,
        vals[1]: 1
    }

    print("Sex mapping:", mapping)

    df["_sex"] = (
        sex.astype(str)
        .map(mapping)
    )

# ------------------------------------------------------------
# Frozen pairs
# ------------------------------------------------------------

pairs = [
    {
        "region": "rh_superiorfrontal_area",
        "metric": "surface_area",
        "cardiac": "LVESVi",
    },
    {
        "region": "rh_lingual_vol",
        "metric": "volume",
        "cardiac": "LV_MASSi",
    },
    {
        "region": "rh_lingual_area",
        "metric": "surface_area",
        "cardiac": "LV_MASSi",
    },
    {
        "region": "lh_middletemporal_vol",
        "metric": "volume",
        "cardiac": "LVEF",
    },
]

specs = {
    "M1_AMYLOID": [
        "Abeta42_40"
    ],
    "M2_PTAU": [
        "log_pTau181"
    ],
    "M3_GFAP": [
        "log_GFAP"
    ],
    "M4_NFL": [
        "log_NFL"
    ],
    "M5_AMYLOID_PTAU": [
        "Abeta42_40",
        "log_pTau181"
    ],
    "M6_GFAP_NFL": [
        "log_GFAP",
        "log_NFL"
    ],
    "M7_ALL4": [
        "Abeta42_40",
        "log_pTau181",
        "log_GFAP",
        "log_NFL"
    ],
}

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def z(x):
    x = pd.to_numeric(
        x,
        errors="coerce"
    )

    sd = x.std()

    if not np.isfinite(sd) or sd == 0:
        return x * np.nan

    return (
        x - x.mean()
    ) / sd


def get_brain(region, metric):

    y = pd.to_numeric(
        df[region],
        errors="coerce"
    )

    if metric == "volume":
        icv = pd.to_numeric(
            df["IntraCranialVol"],
            errors="coerce"
        )

        y = y / icv

    return y


def fit_model(dat, biomarkers):

    d = dat.copy()

    d["brain_z"] = z(d["brain"])
    d["heart_z"] = z(d["heart"])
    d["age_z"] = z(d["age"])
    d["interval_z"] = z(d["interval"])
    d["BMI_z"] = z(d["BMI"])
    d["SBP_z"] = z(d["SBP"])

    xcols = [
        "heart_z",
        "APOE4",
        "age_z",
        "sex",
        "interval_z",
        "BMI_z",
        "SBP_z",
        "smoking",
        "diabetes",
    ]

    if "total_area" in d.columns:
        d["total_area_z"] = z(
            d["total_area"]
        )
        xcols.append(
            "total_area_z"
        )

    for b in biomarkers:
        bz = b + "_z"
        d[bz] = z(d[b])
        xcols.append(bz)

    d["heart_x_APOE4"] = (
        d["heart_z"]
        * d["APOE4"]
    )

    xcols.append(
        "heart_x_APOE4"
    )

    X = sm.add_constant(
        d[xcols]
    )

    fit = sm.OLS(
        d["brain_z"],
        X
    ).fit(cov_type="HC3")

    return fit


# ------------------------------------------------------------
# Run paired complete-case comparisons
# ------------------------------------------------------------

rows = []

for pair in pairs:

    base_dat = pd.DataFrame({
        "brain": get_brain(
            pair["region"],
            pair["metric"]
        ),

        "heart": pd.to_numeric(
            df[pair["cardiac"]],
            errors="coerce"
        ),

        "APOE4": pd.to_numeric(
            df["APOE4_carrier"],
            errors="coerce"
        ),

        "age": pd.to_numeric(
            df["age_at_mri"],
            errors="coerce"
        ),

        "sex": pd.to_numeric(
            df["_sex"],
            errors="coerce"
        ),

        "interval": pd.to_numeric(
            df["abs_delta_years"],
            errors="coerce"
        ),

        "BMI": pd.to_numeric(
            df["BMI_nearest_exam"],
            errors="coerce"
        ),

        "SBP": pd.to_numeric(
            df["SBP_nearest_exam"],
            errors="coerce"
        ),

        "smoking": pd.to_numeric(
            df["current_smoker_nearest_exam"],
            errors="coerce"
        ),

        "diabetes": pd.to_numeric(
            df["diabetes_any"],
            errors="coerce"
        ),

        "Abeta42_40": pd.to_numeric(
            df["Abeta42_40"],
            errors="coerce"
        ),

        "log_pTau181": pd.to_numeric(
            df["log_pTau181"],
            errors="coerce"
        ),

        "log_GFAP": pd.to_numeric(
            df["log_GFAP"],
            errors="coerce"
        ),

        "log_NFL": pd.to_numeric(
            df["log_NFL"],
            errors="coerce"
        ),
    })

    if pair["metric"] == "surface_area":
        base_dat["total_area"] = pd.to_numeric(
            df["TotalCorticalSurfaceArea"],
            errors="coerce"
        )

    base_need = [
        "brain",
        "heart",
        "APOE4",
        "age",
        "sex",
        "interval",
        "BMI",
        "SBP",
        "smoking",
        "diabetes",
    ]

    if pair["metric"] == "surface_area":
        base_need.append(
            "total_area"
        )

    for spec_name, biomarkers in specs.items():

        # Same exact sample for baseline and adjusted model
        need = base_need + biomarkers

        dat = (
            base_dat[need]
            .dropna()
            .copy()
        )

        fit0 = fit_model(
            dat,
            biomarkers=[]
        )

        fit1 = fit_model(
            dat,
            biomarkers=biomarkers
        )

        b0 = fit0.params[
            "heart_x_APOE4"
        ]

        b1 = fit1.params[
            "heart_x_APOE4"
        ]

        attenuation = (
            100.0
            * (
                abs(b0)
                - abs(b1)
            )
            / abs(b0)
        )

        rows.append({
            "spec": spec_name,
            "region": pair["region"],
            "cardiac": pair["cardiac"],
            "metric": pair["metric"],
            "N": len(dat),

            "beta_interaction_matched_base":
                b0,

            "p_interaction_matched_base":
                fit0.pvalues[
                    "heart_x_APOE4"
                ],

            "beta_interaction_adjusted":
                b1,

            "p_interaction_adjusted":
                fit1.pvalues[
                    "heart_x_APOE4"
                ],

            "attenuation_pct":
                attenuation,

            "R2_base":
                fit0.rsquared,

            "R2_adjusted":
                fit1.rsquared,

            "delta_R2":
                fit1.rsquared
                - fit0.rsquared,
        })

res = pd.DataFrame(rows)

# ------------------------------------------------------------
# FDR across 4 frozen pairs within each specification
# ------------------------------------------------------------

res["q_base"] = np.nan
res["q_adjusted"] = np.nan

for spec, idx in res.groupby(
    "spec"
).groups.items():

    res.loc[
        idx,
        "q_base"
    ] = multipletests(
        res.loc[
            idx,
            "p_interaction_matched_base"
        ],
        method="fdr_bh"
    )[1]

    res.loc[
        idx,
        "q_adjusted"
    ] = multipletests(
        res.loc[
            idx,
            "p_interaction_adjusted"
        ],
        method="fdr_bh"
    )[1]

# ------------------------------------------------------------
# Stability
# ------------------------------------------------------------

res["same_direction"] = (
    np.sign(
        res["beta_interaction_matched_base"]
    )
    ==
    np.sign(
        res["beta_interaction_adjusted"]
    )
)

res["adjusted_q05"] = (
    res["q_adjusted"] < 0.05
)

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

res.to_csv(
    OUTDIR /
    "APOE_AD_biomarker_completecase_all.csv",
    index=False
)

summary = (
    res.groupby(
        [
            "region",
            "cardiac"
        ]
    )
    .agg(
        n_specs=(
            "spec",
            "size"
        ),
        n_same_direction=(
            "same_direction",
            "sum"
        ),
        n_adjusted_q05=(
            "adjusted_q05",
            "sum"
        ),
        min_attenuation_pct=(
            "attenuation_pct",
            "min"
        ),
        max_attenuation_pct=(
            "attenuation_pct",
            "max"
        ),
    )
    .reset_index()
)

summary.to_csv(
    OUTDIR /
    "APOE_AD_biomarker_completecase_summary.csv",
    index=False
)

print("\nMATCHED COMPLETE-CASE BIOMARKER ANALYSIS")
print("========================================")

print(
    res[
        [
            "spec",
            "region",
            "cardiac",
            "N",
            "beta_interaction_matched_base",
            "beta_interaction_adjusted",
            "attenuation_pct",
            "p_interaction_adjusted",
            "q_adjusted",
            "delta_R2",
            "same_direction",
        ]
    ]
    .sort_values(
        [
            "region",
            "spec"
        ]
    )
    .to_string(
        index=False
    )
)

print("\nSUMMARY")
print("-------")

print(
    summary.to_string(
        index=False
    )
)

print("\nSaved to:")
print(OUTDIR)
