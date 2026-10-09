#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path("/data/qiallab/Framingham")

INFILE = (
    ROOT / "results" /
    "neurocardiac_metadata_with_atria.csv"
)

OUTDIR = (
    ROOT / "results" /
    "atrial_APOE4" /
    "independence_LV"
)

OUTDIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INFILE)

OUTCOME = "lh_G_occipital_middle_tksd"
ATRIAL = "LAVI_max"

# ============================================================
# Construct LV mass index exactly as in prior pipeline
# ============================================================


# ============================================================
# Reconstruct BSA from stored anthropometrics
# ============================================================

if "height_cm" in df.columns and "weight_kg" in df.columns:

    height_cm = pd.to_numeric(
        df["height_cm"],
        errors="coerce"
    )

    weight_kg = pd.to_numeric(
        df["weight_kg"],
        errors="coerce"
    )

elif "height_in" in df.columns and "weight_lb" in df.columns:

    height_cm = (
        pd.to_numeric(
            df["height_in"],
            errors="coerce"
        ) * 2.54
    )

    weight_kg = (
        pd.to_numeric(
            df["weight_lb"],
            errors="coerce"
        ) * 0.45359237
    )

else:
    raise RuntimeError(
        "Could not construct BSA. "
        "Need either height_cm/weight_kg "
        "or height_in/weight_lb."
    )

df["BSA_m2_exact"] = np.sqrt(
    height_cm * weight_kg / 3600.0
)

if "LV_MASS" not in df.columns:
    raise RuntimeError("LV_MASS not found.")

df["LV_MASSi_recomputed"] = (
    pd.to_numeric(
        df["LV_MASS"],
        errors="coerce"
    )
    / df["BSA_m2_exact"]
)

if "LVEF" not in df.columns:
    raise RuntimeError("LVEF not found.")

# ============================================================
# Other covariates
# ============================================================

df["_sex"] = pd.to_numeric(
    df["sex_clinical"],
    errors="coerce"
)

def diabetes_binary(x):
    x = pd.to_numeric(x, errors="coerce")

    out = pd.Series(
        np.nan,
        index=x.index,
        dtype=float
    )

    out.loc[x == 0] = 0
    out.loc[x.isin([1, 2])] = 1

    return out

df["diabetes_history_any"] = diabetes_binary(
    df["diabetes_history_nearest_exam"]
)

def zscore(x):
    x = pd.to_numeric(x, errors="coerce")
    return (x - x.mean()) / x.std(ddof=0)

# ============================================================
# Nested models
# ============================================================

MODELS = {
    "M0_ATRIAL_BASE": [],

    "M1_PLUS_LV_MASSi": [
        "LV_MASSi_recomputed"
    ],

    "M2_PLUS_LVEF": [
        "LVEF"
    ],

    "M3_PLUS_LV_MASSi_LVEF": [
        "LV_MASSi_recomputed",
        "LVEF"
    ],
}

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

# Common complete-case sample across ALL nested models
COMMON_VARS = (
    BASE_VARS
    + [
        "LV_MASSi_recomputed",
        "LVEF",
    ]
)

def fit_analysis(data, analysis_label):

    d = data[COMMON_VARS].copy().dropna()

    print("\n" + "=" * 100)
    print(analysis_label)
    print("=" * 100)

    print("Common complete-case N:", len(d))
    print(
        "APOE4 noncarriers:",
        int((d["APOE4_carrier"] == 0).sum())
    )
    print(
        "APOE4 carriers:",
        int((d["APOE4_carrier"] == 1).sum())
    )

    if len(d) < 100:
        raise RuntimeError(
            f"Too few complete cases for {analysis_label}: N={len(d)}"
        )

    # Standardize ONCE on common sample
    d["y_z"] = zscore(d[OUTCOME])
    d["atrial_z"] = zscore(d[ATRIAL])

    d["age_z"] = zscore(d["age_at_mri"])
    d["interval_z"] = zscore(d["abs_years_exam_mri"])

    d["BMI_z"] = zscore(d["BMI_nearest_exam"])
    d["SBP_z"] = zscore(d["SBP_nearest_exam"])

    d["LV_MASSi_z"] = zscore(
        d["LV_MASSi_recomputed"]
    )

    d["LVEF_z"] = zscore(
        d["LVEF"]
    )

    d["atrial_x_APOE4"] = (
        d["atrial_z"]
        * pd.to_numeric(
            d["APOE4_carrier"],
            errors="coerce"
        )
    )

    rows = []

    for model_name, extra in MODELS.items():

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

        if "LV_MASSi_recomputed" in extra:
            Xvars.append("LV_MASSi_z")

        if "LVEF" in extra:
            Xvars.append("LVEF_z")

        X = sm.add_constant(
            d[Xvars].astype(float),
            has_constant="add"
        )

        fit = sm.OLS(
            d["y_z"].astype(float),
            X
        ).fit(cov_type="HC3")

        b0 = fit.params["atrial_z"]
        bi = fit.params["atrial_x_APOE4"]

        ci_int = fit.conf_int().loc[
            "atrial_x_APOE4"
        ]

        # Correct carrier slope variance
        cov = fit.cov_params()

        var_carrier = (
            cov.loc["atrial_z", "atrial_z"]
            +
            cov.loc[
                "atrial_x_APOE4",
                "atrial_x_APOE4"
            ]
            +
            2
            * cov.loc[
                "atrial_z",
                "atrial_x_APOE4"
            ]
        )

        se_carrier = np.sqrt(var_carrier)

        beta_carrier = b0 + bi

        row = {
            "analysis": analysis_label,
            "model": model_name,

            "outcome": OUTCOME,
            "atrial_metric": ATRIAL,

            "N": len(d),

            "N_APOE4_noncarrier":
                int((d["APOE4_carrier"] == 0).sum()),

            "N_APOE4_carrier":
                int((d["APOE4_carrier"] == 1).sum()),

            "beta_noncarrier": b0,
            "SE_noncarrier":
                fit.bse["atrial_z"],

            "beta_interaction": bi,
            "SE_interaction":
                fit.bse["atrial_x_APOE4"],

            "CI_interaction_low":
                ci_int.iloc[0],

            "CI_interaction_high":
                ci_int.iloc[1],

            "p_interaction":
                fit.pvalues["atrial_x_APOE4"],

            "beta_carrier":
                beta_carrier,

            "SE_carrier":
                se_carrier,

            "CI_carrier_low":
                beta_carrier
                - 1.96 * se_carrier,

            "CI_carrier_high":
                beta_carrier
                + 1.96 * se_carrier,

            "beta_LV_MASSi":
                fit.params.get(
                    "LV_MASSi_z",
                    np.nan
                ),

            "p_LV_MASSi":
                fit.pvalues.get(
                    "LV_MASSi_z",
                    np.nan
                ),

            "beta_LVEF":
                fit.params.get(
                    "LVEF_z",
                    np.nan
                ),

            "p_LVEF":
                fit.pvalues.get(
                    "LVEF_z",
                    np.nan
                ),

            "R2": fit.rsquared,
        }

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# Primary: Exam 8 matched
# Sensitivity: all atrial participants
# ============================================================

primary = df[
    df["nearest_exam"] == 8
].copy()

all_atrial = df[
    df[ATRIAL].notna()
].copy()

r_primary = fit_analysis(
    primary,
    "PRIMARY_EXAM8"
)

r_all = fit_analysis(
    all_atrial,
    "SENSITIVITY_ALL"
)

res = pd.concat(
    [r_primary, r_all],
    ignore_index=True
)

# ============================================================
# Interaction attenuation relative to M0
# ============================================================

res["interaction_abs_change_pct"] = np.nan

for analysis in res["analysis"].unique():

    idx = res["analysis"] == analysis

    base = res.loc[
        idx
        & (
            res["model"]
            == "M0_ATRIAL_BASE"
        ),
        "beta_interaction"
    ].iloc[0]

    res.loc[
        idx,
        "interaction_abs_change_pct"
    ] = (
        (
            np.abs(
                res.loc[
                    idx,
                    "beta_interaction"
                ]
            )
            - np.abs(base)
        )
        / np.abs(base)
        * 100
    )

# ============================================================
# Save
# ============================================================

OUTCSV = (
    OUTDIR /
    "LAVI_APOE4_independence_LV_models.csv"
)

res.to_csv(
    OUTCSV,
    index=False
)

OUTTXT = (
    OUTDIR /
    "LAVI_APOE4_independence_LV_summary.txt"
)

with open(OUTTXT, "w") as f:

    f.write(
        "LAVI × APOE4 INDEPENDENCE FROM LV PHYSIOLOGY\n"
        "===========================================\n\n"
    )

    f.write(
        "Outcome: "
        "lh_G_occipital_middle_tksd\n"
    )

    f.write(
        "Exposure: LAVI_max\n\n"
    )

    f.write(
        "Nested models use a common complete-case "
        "sample within each analysis.\n\n"
    )

    f.write(
        "M0: atrial model\n"
        "M1: + LV mass index\n"
        "M2: + LVEF\n"
        "M3: + LV mass index + LVEF\n\n"
    )

    f.write(
        res.to_string(index=False)
    )

    f.write("\n")

print("\n")
print("=" * 100)
print("ATRIAL INDEPENDENCE RESULTS")
print("=" * 100)

print(
    res[
        [
            "analysis",
            "model",
            "N",
            "N_APOE4_carrier",
            "beta_noncarrier",
            "beta_interaction",
            "CI_interaction_low",
            "CI_interaction_high",
            "p_interaction",
            "beta_carrier",
            "interaction_abs_change_pct",
            "R2",
        ]
    ].to_string(index=False)
)

print("\nSaved:")
print(OUTCSV)
print(OUTTXT)
