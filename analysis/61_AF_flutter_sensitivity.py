#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

ROOT = Path("/data/qiallab/Framingham")

DATA = ROOT / "results" / "neurocardiac_metadata_analysis_ready_metabolic.csv"

OUTDIR = ROOT / "results" / "AF_flutter_sensitivity"
OUTDIR.mkdir(parents=True, exist_ok=True)

OUTFILE = OUTDIR / "AF_flutter_primary4_sensitivity.csv"

APOE = "APOE4_carrier"

TESTS = [
    ("lh_middletemporal_vol", "LVEF", "volume"),
    ("rh_superiorfrontal_area", "LVESVi", "area"),
    ("rh_lingual_vol", "LV_MASSi", "volume"),
    ("rh_lingual_area", "LV_MASSi", "area"),
]


def zscore(s):
    sd = s.std(ddof=0)
    if pd.isna(sd) or sd == 0:
        return pd.Series(np.nan, index=s.index)
    return (s - s.mean()) / sd


def clean_binary(series):
    if pd.api.types.is_bool_dtype(series):
        return series.astype(float)

    s = pd.to_numeric(series, errors="coerce")

    # Only allow explicit 0/1 values
    out = pd.Series(np.nan, index=series.index, dtype=float)
    out.loc[s == 0] = 0.0
    out.loc[s == 1] = 1.0

    return out


print("=" * 90)
print("AF / FLUTTER SENSITIVITY ANALYSIS")
print("=" * 90)

df = pd.read_csv(DATA, low_memory=False)

print("\nInput:")
print(DATA)
print("N =", len(df))


# ============================================================
# 1. RECREATE VENTRICULAR INDICES EXACTLY AS EXISTING PIPELINE
# ============================================================

height_in = pd.to_numeric(
    df["height_in"],
    errors="coerce"
)

weight_lb = pd.to_numeric(
    df["weight_lb"],
    errors="coerce"
)

height_cm = height_in * 2.54
weight_kg = weight_lb * 0.45359237

BSA = np.sqrt(
    height_cm * weight_kg / 3600.0
)

df = df.copy()

df["BSA_m2"] = BSA

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


# ============================================================
# 2. TOTAL CORTICAL SURFACE AREA
# ============================================================

df["total_area"] = (
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
# 3. DIABETES VARIABLE
# ============================================================

dm = pd.to_numeric(
    df["diabetes_history_nearest_exam"],
    errors="coerce"
)

df["diabetes_any"] = np.where(
    dm.isna(),
    np.nan,
    dm.isin([1, 2]).astype(float)
)


# ============================================================
# 4. AF HISTORY VARIABLE
# ============================================================

af = pd.to_numeric(
    df["af_history_nearest_exam"],
    errors="coerce"
)

df["AF_history_binary"] = np.select(
    [
        af == 0,
        af == 1,
    ],
    [
        0.0,
        1.0,
    ],
    default=np.nan,
)

print("\nAF history raw:")
print(
    df["af_history_nearest_exam"]
    .value_counts(dropna=False)
)

print("\nAF analysis variable:")
print(
    df["AF_history_binary"]
    .value_counts(dropna=False)
)

if "ecg_rhythm_nearest_exam" in df.columns:

    ecg = pd.to_numeric(
        df["ecg_rhythm_nearest_exam"],
        errors="coerce"
    )

    print("\nECG AF/flutter code 6:")
    print("N ECG code 6 =", int((ecg == 6).sum()))

    extra = (
        (ecg == 6)
        & (df["AF_history_binary"] != 1)
    )

    print(
        "Additional AF/flutter cases from ECG =",
        int(extra.sum())
    )


# ============================================================
# 5. QC DERIVED VARIABLES
# ============================================================

print("\nDerived cardiac measures:")

for c in [
    "BSA_m2",
    "LVESVi",
    "LV_MASSi",
    "total_area",
    "diabetes_any",
]:
    x = pd.to_numeric(df[c], errors="coerce")
    print(
        f"{c:16s} "
        f"N={x.notna().sum():3d} "
        f"mean={x.mean():.4f} "
        f"SD={x.std():.4f}"
    )


# ============================================================
# 6. MODEL FUNCTION
# ============================================================

def fit_one(
    data,
    region,
    cardiac,
    brain_type,
    af_adjust=False,
):

    required = [
        region,
        cardiac,
        APOE,
        "age_at_mri",
        "sex_clinical",
        "abs_delta_years",
        "BMI_nearest_exam",
        "SBP_nearest_exam",
        "current_smoker_nearest_exam",
        "diabetes_any",
    ]

    if brain_type == "area":
        required.append("total_area")

    if brain_type == "volume":
        required.append("IntraCranialVol")

    if af_adjust:
        required.append("AF_history_binary")

    missing = [
        c for c in required
        if c not in data.columns
    ]

    if missing:
        raise KeyError(
            "Missing:\n" + "\n".join(missing)
        )

    d = data[required].copy()

    continuous = [
        region,
        cardiac,
        "age_at_mri",
        "abs_delta_years",
        "BMI_nearest_exam",
        "SBP_nearest_exam",
    ]

    if brain_type == "area":
        continuous.append("total_area")

    if brain_type == "volume":
        continuous.append("IntraCranialVol")

    for c in continuous:
        d[c] = pd.to_numeric(
            d[c],
            errors="coerce"
        )

    d[APOE] = clean_binary(d[APOE])

    d["current_smoker_nearest_exam"] = clean_binary(
        d["current_smoker_nearest_exam"]
    )

    d["diabetes_any"] = clean_binary(
        d["diabetes_any"]
    )

    if af_adjust:
        d["AF_history_binary"] = clean_binary(
            d["AF_history_binary"]
        )

    d = d.dropna(
        subset=required
    ).copy()

    if brain_type == "volume":
        d["brain_y"] = (
            d[region]
            / d["IntraCranialVol"]
        )
    else:
        d["brain_y"] = d[region]

    d["brain_z"] = zscore(
        d["brain_y"]
    )

    d["cardiac_z"] = zscore(
        d[cardiac]
    )

    d["age_z"] = zscore(
        d["age_at_mri"]
    )

    d["interval_z"] = zscore(
        d["abs_delta_years"]
    )

    d["BMI_z"] = zscore(
        d["BMI_nearest_exam"]
    )

    d["SBP_z"] = zscore(
        d["SBP_nearest_exam"]
    )

    if brain_type == "area":
        d["total_area_z"] = zscore(
            d["total_area"]
        )

    formula = (
        f"brain_z ~ cardiac_z * {APOE}"
        " + age_z"
        " + C(sex_clinical)"
        " + interval_z"
        " + BMI_z"
        " + SBP_z"
        " + current_smoker_nearest_exam"
        " + diabetes_any"
    )

    if brain_type == "area":
        formula += " + total_area_z"

    if af_adjust:
        formula += " + AF_history_binary"

    model = smf.ols(
        formula,
        data=d
    ).fit(
        cov_type="HC3"
    )

    interaction = f"cardiac_z:{APOE}"

    if interaction not in model.params:
        interaction = f"{APOE}:cardiac_z"

    ci = model.conf_int().loc[
        interaction
    ]

    beta_noncarrier = model.params[
        "cardiac_z"
    ]

    beta_interaction = model.params[
        interaction
    ]

    beta_carrier = (
        beta_noncarrier
        + beta_interaction
    )

    return {
        "N": len(d),
        "N_APOE4_noncarrier":
            int((d[APOE] == 0).sum()),
        "N_APOE4_carrier":
            int((d[APOE] == 1).sum()),
        "N_AF_positive":
            int(
                (
                    d["AF_history_binary"] == 1
                ).sum()
            )
            if "AF_history_binary" in d.columns
            else np.nan,
        "beta_noncarrier":
            beta_noncarrier,
        "beta_interaction":
            beta_interaction,
        "beta_carrier":
            beta_carrier,
        "CI_low":
            ci.iloc[0],
        "CI_high":
            ci.iloc[1],
        "p_interaction":
            model.pvalues[
                interaction
            ],
        "R2":
            model.rsquared,
    }


# ============================================================
# 7. RUN MODELS
# ============================================================

results = []

for region, cardiac, brain_type in TESTS:

    print(
        "\nRunning:",
        cardiac,
        "x APOE4 ->",
        region
    )

    # Original M3
    r = fit_one(
        df,
        region,
        cardiac,
        brain_type,
        af_adjust=False,
    )

    r.update({
        "region": region,
        "cardiac": cardiac,
        "analysis": "Original M3",
    })

    results.append(r)

    # M3 + AF covariate
    r = fit_one(
        df,
        region,
        cardiac,
        brain_type,
        af_adjust=True,
    )

    r.update({
        "region": region,
        "cardiac": cardiac,
        "analysis": "M3 + AF history",
    })

    results.append(r)

    # Exclude AF-positive
    # Keep known AF-negative only
    no_af = df.loc[
        df["AF_history_binary"] == 0
    ].copy()

    r = fit_one(
        no_af,
        region,
        cardiac,
        brain_type,
        af_adjust=False,
    )

    r.update({
        "region": region,
        "cardiac": cardiac,
        "analysis": "Exclude AF positive",
    })

    results.append(r)


# ============================================================
# 8. FDR
# ============================================================

res = pd.DataFrame(results)

res["q_interaction"] = np.nan

for analysis in res["analysis"].unique():

    mask = (
        res["analysis"] == analysis
    )

    _, qvals, _, _ = multipletests(
        res.loc[
            mask,
            "p_interaction"
        ],
        method="fdr_bh",
    )

    res.loc[
        mask,
        "q_interaction"
    ] = qvals


# ============================================================
# 9. COMPARE WITH ORIGINAL
# ============================================================

orig = (
    res.loc[
        res["analysis"] == "Original M3"
    ]
    .set_index(
        ["region", "cardiac"]
    )["beta_interaction"]
)

res["beta_original"] = res.apply(
    lambda r:
        orig.loc[
            (
                r["region"],
                r["cardiac"]
            )
        ],
    axis=1,
)

res["delta_beta"] = (
    res["beta_interaction"]
    - res["beta_original"]
)

res["percent_change_abs"] = (
    100
    * (
        res["beta_interaction"].abs()
        - res["beta_original"].abs()
    )
    / res["beta_original"].abs()
)


# ============================================================
# 10. SAVE / PRINT
# ============================================================

res.to_csv(
    OUTFILE,
    index=False
)

show = [
    "region",
    "cardiac",
    "analysis",
    "N",
    "N_APOE4_noncarrier",
    "N_APOE4_carrier",
    "N_AF_positive",
    "beta_noncarrier",
    "beta_interaction",
    "beta_carrier",
    "CI_low",
    "CI_high",
    "p_interaction",
    "q_interaction",
    "percent_change_abs",
]

print("\n")
print("=" * 130)
print("AF / FLUTTER SENSITIVITY RESULTS")
print("=" * 130)

print(
    res[show].to_string(
        index=False,
        float_format=lambda x: f"{x:.6g}"
    )
)

print("\nSaved:")
print(OUTFILE)


# ============================================================
# 11. REPRODUCTION CHECK
# ============================================================

expected = {
    (
        "lh_middletemporal_vol",
        "LVEF",
    ): 0.2333,

    (
        "rh_superiorfrontal_area",
        "LVESVi",
    ): -0.1703,

    (
        "rh_lingual_vol",
        "LV_MASSi",
    ): 0.3053,

    (
        "rh_lingual_area",
        "LV_MASSi",
    ): 0.2259,
}

print("\n")
print("=" * 90)
print("ORIGINAL M3 REPRODUCTION CHECK")
print("=" * 90)

for _, row in res.loc[
    res["analysis"] == "Original M3"
].iterrows():

    key = (
        row["region"],
        row["cardiac"]
    )

    target = expected[key]

    print(
        f"{row['cardiac']:9s} "
        f"{row['region']:28s} "
        f"observed={row['beta_interaction']:+.4f}  "
        f"target={target:+.4f}  "
        f"diff={row['beta_interaction']-target:+.4f}"
    )

