#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

ROOT = Path("/data/qiallab/Framingham")
RES = ROOT / "results"

DATAFILE = RES / "neurocardiac_metadata_analysis_ready_metabolic.csv"
INVFILE  = RES / "tier2_regional" / "tier2_region_family_inventory.csv"

OUTDIR = RES / "tier2_corrected"
OUTDIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATAFILE)
inv = pd.read_csv(INVFILE)

# ============================================================
# 1. Body surface area and indexed cardiac phenotypes
# ============================================================

height_cm = pd.to_numeric(df["height_in"], errors="coerce") * 2.54
weight_kg = pd.to_numeric(df["weight_lb"], errors="coerce") * 0.45359237

df["BSA_m2"] = np.sqrt(
    height_cm * weight_kg / 3600.0
)

df["LVEDVi"] = (
    pd.to_numeric(df["LVEDV"], errors="coerce")
    / df["BSA_m2"]
)

df["LVESVi"] = (
    pd.to_numeric(df["LVESV"], errors="coerce")
    / df["BSA_m2"]
)

df["LV_MASSi"] = (
    pd.to_numeric(df["LV_MASS"], errors="coerce")
    / df["BSA_m2"]
)

cardiac_vars = [
    "LVEDVi",
    "LVESVi",
    "LV_MASSi",
    "LVEF",
    "LV_3D_LONG",
]

# ============================================================
# 2. Total cortical surface area
# ============================================================

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

# ============================================================
# 3. Sex
# ============================================================

sex = df["sex_clinical"]

if pd.api.types.is_numeric_dtype(sex):
    df["_sex"] = pd.to_numeric(sex, errors="coerce")
else:
    vals = list(sex.dropna().astype(str).unique())

    if len(vals) != 2:
        raise RuntimeError(
            f"Unexpected sex categories: {vals}"
        )

    mapping = {
        vals[0]: 0,
        vals[1]: 1
    }

    print("Sex mapping:", mapping)

    df["_sex"] = sex.astype(str).map(mapping)

# ============================================================
# 4. Recode diabetes-history variable
# ============================================================

dm_raw = pd.to_numeric(
    df["diabetes_history_nearest_exam"],
    errors="coerce"
)

# Existing coding:
# 0 = no
# 1 = yes/current
# 2 = yes/not current
#
# For adjustment here, collapse 1/2 to any positive history.
df["diabetes_history_any"] = np.where(
    dm_raw.isna(),
    np.nan,
    dm_raw.isin([1, 2]).astype(float)
)

# ============================================================
# 5. Safety checks for binary variables
# ============================================================

binary_vars = [
    "current_smoker_nearest_exam",
    "diabetes_history_any",
    "APOE4_carrier",
]

print("\nBINARY VARIABLE CHECK")
print("=====================")

for c in binary_vars:

    x = pd.to_numeric(
        df[c],
        errors="coerce"
    )

    vals = sorted(
        pd.Series(x.dropna().unique()).tolist()
    )

    print(c, vals)

    if not set(vals).issubset({0, 1, 0.0, 1.0}):
        raise RuntimeError(
            f"{c} is not binary 0/1: {vals}"
        )

# ============================================================
# 6. Helpers
# ============================================================

def z(x):
    x = pd.to_numeric(
        x,
        errors="coerce"
    )

    sd = x.std()

    if not np.isfinite(sd) or sd == 0:
        return x * np.nan

    return (x - x.mean()) / sd


def outcome(region, metric_type):

    y = pd.to_numeric(
        df[region],
        errors="coerce"
    )

    # Regional volume: normalize to ICV
    if metric_type == "volume":

        icv = pd.to_numeric(
            df["IntraCranialVol"],
            errors="coerce"
        )

        y = y / icv

    # Thickness: raw
    # Surface area: raw, with total cortical surface area covariate
    return y


def fit_model(
    region,
    family,
    metric_type,
    heart,
    model
):

    dat = pd.DataFrame({
        "y": outcome(region, metric_type),

        "heart": pd.to_numeric(
            df[heart],
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
            df["diabetes_history_any"],
            errors="coerce"
        ),

        "APOE4": pd.to_numeric(
            df["APOE4_carrier"],
            errors="coerce"
        ),

        "total_area": pd.to_numeric(
            df["TotalCorticalSurfaceArea"],
            errors="coerce"
        ),
    })

    need = [
        "y",
        "heart",
        "age",
        "sex",
        "interval"
    ]

    if metric_type == "surface_area":
        need.append("total_area")

    if model in [
        "M2_vascular",
        "M3_APOE4"
    ]:
        need += [
            "BMI",
            "SBP",
            "smoking",
            "diabetes"
        ]

    if model == "M3_APOE4":
        need.append("APOE4")

    dat = dat[need].dropna()

    if len(dat) < 100:
        return None

    # Standardized outcome and continuous predictors
    dat["y_z"] = z(dat["y"])
    dat["heart_z"] = z(dat["heart"])
    dat["age_z"] = z(dat["age"])
    dat["interval_z"] = z(dat["interval"])

    xcols = [
        "heart_z",
        "age_z",
        "sex",
        "interval_z"
    ]

    # Surface-area-specific adjustment
    if metric_type == "surface_area":

        dat["total_area_z"] = z(
            dat["total_area"]
        )

        xcols.append(
            "total_area_z"
        )

    if model in [
        "M2_vascular",
        "M3_APOE4"
    ]:

        dat["BMI_z"] = z(dat["BMI"])
        dat["SBP_z"] = z(dat["SBP"])

        xcols += [
            "BMI_z",
            "SBP_z",
            "smoking",
            "diabetes"
        ]

    if model == "M3_APOE4":

        dat["heart_x_APOE4"] = (
            dat["heart_z"]
            * dat["APOE4"]
        )

        xcols += [
            "APOE4",
            "heart_x_APOE4"
        ]

    X = sm.add_constant(
        dat[xcols]
    )

    fit = sm.OLS(
        dat["y_z"],
        X
    ).fit(cov_type="HC3")

    ci = fit.conf_int().loc["heart_z"]

    out = {
        "metric_type": metric_type,
        "family": family,
        "region": region,
        "cardiac": heart,
        "model": model,
        "N": len(dat),

        "beta_cardiac":
            fit.params["heart_z"],

        "SE_cardiac":
            fit.bse["heart_z"],

        "p_cardiac":
            fit.pvalues["heart_z"],

        "CI_low":
            ci.iloc[0],

        "CI_high":
            ci.iloc[1],

        "R2":
            fit.rsquared,
    }

    if model == "M3_APOE4":

        ci2 = fit.conf_int().loc[
            "heart_x_APOE4"
        ]

        out.update({
            "beta_interaction":
                fit.params["heart_x_APOE4"],

            "SE_interaction":
                fit.bse["heart_x_APOE4"],

            "p_interaction":
                fit.pvalues["heart_x_APOE4"],

            "CI_interaction_low":
                ci2.iloc[0],

            "CI_interaction_high":
                ci2.iloc[1],
        })

    return out


# ============================================================
# 7. Run all Tier 2 models
# ============================================================

rows = []

models = [
    "M1_basic",
    "M2_vascular",
    "M3_APOE4"
]

for _, r in inv.iterrows():

    region = r["variable"]
    family = r["family"]
    metric_type = r["metric_type"]

    if region not in df.columns:
        continue

    for heart in cardiac_vars:

        for model in models:

            result = fit_model(
                region,
                family,
                metric_type,
                heart,
                model
            )

            if result is not None:
                rows.append(result)

res = pd.DataFrame(rows)

# ============================================================
# 8. FDR within metric × family × cardiac × model
# ============================================================

res["q_cardiac"] = np.nan
res["q_interaction"] = np.nan

for _, idx in res.groupby(
    [
        "metric_type",
        "family",
        "cardiac",
        "model"
    ]
).groups.items():

    p = res.loc[
        idx,
        "p_cardiac"
    ].fillna(1)

    res.loc[
        idx,
        "q_cardiac"
    ] = multipletests(
        p,
        method="fdr_bh"
    )[1]

# APOE4 interaction FDR
m3mask = res["model"] == "M3_APOE4"

for _, idx in res[m3mask].groupby(
    [
        "metric_type",
        "family",
        "cardiac"
    ]
).groups.items():

    p = res.loc[
        idx,
        "p_interaction"
    ].fillna(1)

    res.loc[
        idx,
        "q_interaction"
    ] = multipletests(
        p,
        method="fdr_bh"
    )[1]

# ============================================================
# 9. Save full and primary results
# ============================================================

res.to_csv(
    OUTDIR / "tier2_corrected_all.csv",
    index=False
)

m2 = res[
    res["model"] == "M2_vascular"
].copy()

sig_main = m2[
    m2["q_cardiac"] < 0.05
].copy()

sig_int = res[
    (res["model"] == "M3_APOE4")
    &
    (res["q_interaction"] < 0.05)
].copy()

sig_main.to_csv(
    OUTDIR / "tier2_corrected_M2_FDR_main.csv",
    index=False
)

sig_int.to_csv(
    OUTDIR / "tier2_corrected_FDR_APOE4.csv",
    index=False
)

# ============================================================
# 10. Summary
# ============================================================

print("\nCORRECTED TIER 2 ANALYSIS")
print("=========================")

print("\nCardiac phenotypes:")
print(cardiac_vars)

print("\nBSA summary:")
print(df["BSA_m2"].describe().to_string())

print(
    "\nM2 FDR-significant main effects:",
    len(sig_main)
)

print(
    "FDR-significant APOE4 interactions:",
    len(sig_int)
)

print("\nM2 MAIN EFFECTS")
print("---------------")

if len(sig_main):

    print(
        sig_main[
            [
                "metric_type",
                "family",
                "region",
                "cardiac",
                "N",
                "beta_cardiac",
                "p_cardiac",
                "q_cardiac"
            ]
        ]
        .sort_values(
            ["q_cardiac", "p_cardiac"]
        )
        .head(150)
        .to_string(index=False)
    )

else:
    print("None.")

print("\nAPOE4 INTERACTIONS")
print("------------------")

if len(sig_int):

    print(
        sig_int[
            [
                "metric_type",
                "family",
                "region",
                "cardiac",
                "N",
                "beta_cardiac",
                "beta_interaction",
                "p_interaction",
                "q_interaction"
            ]
        ]
        .sort_values(
            ["q_interaction", "p_interaction"]
        )
        .head(150)
        .to_string(index=False)
    )

else:
    print("None.")

print("\nSIGNIFICANT MAIN-EFFECT COUNTS BY METRIC/CARDIAC")
print("-------------------------------------------------")

if len(sig_main):

    print(
        sig_main.groupby(
            ["metric_type", "cardiac"]
        )
        .size()
        .sort_values(ascending=False)
        .to_string()
    )

else:
    print("None.")

print("\nSIGNIFICANT APOE4 COUNTS BY METRIC/CARDIAC")
print("-------------------------------------------")

if len(sig_int):

    print(
        sig_int.groupby(
            ["metric_type", "cardiac"]
        )
        .size()
        .sort_values(ascending=False)
        .to_string()
    )

else:
    print("None.")

print("\nSaved to:")
print(OUTDIR)
