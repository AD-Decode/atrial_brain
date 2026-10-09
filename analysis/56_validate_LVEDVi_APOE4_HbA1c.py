#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

# ============================================================
# PATHS
# ============================================================

ROOT = Path("/data/qiallab/Framingham")
RES = ROOT / "results"

INPUT = RES / "neurocardiac_metadata_analysis_ready_metabolic.csv"

OUTDIR = RES / "LVEDVi_APOE4_HbA1c_validation"
OUTDIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(INPUT)

print("Loaded:", INPUT)
print("Shape:", df.shape)

# ============================================================
# VARIABLES
# ============================================================

APOE = "APOE4_carrier"
HBA1C = "hba1c_exam7"

AGE = "age_at_cmr"
SEX = "sex_clinical"

BMI = "BMI_nearest_exam"
SBP = "SBP_nearest_exam"
HTNMED = "htn_med_nearest_exam"

HBA1C_CMR_INTERVAL = "hba1c_exam7_to_cmr_years"

# ============================================================
# DERIVE LVEDVi
# ============================================================

for c in [
    "LVEDV",
    "height_in",
    "weight_lb",
    APOE,
    HBA1C,
    AGE,
    BMI,
    SBP,
    HTNMED,
    HBA1C_CMR_INTERVAL
]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

df["height_cm"] = df["height_in"] * 2.54
df["weight_kg"] = df["weight_lb"] * 0.45359237

df["BSA"] = np.sqrt(
    df["height_cm"] * df["weight_kg"] / 3600.0
)

df["LVEDVi"] = df["LVEDV"] / df["BSA"]

# ============================================================
# STANDARDIZE CONTINUOUS PREDICTORS
# ============================================================

def zscore(s):
    s = pd.to_numeric(s, errors="coerce")
    return (s - s.mean()) / s.std(ddof=0)

df["HbA1c_z"] = zscore(df[HBA1C])

if BMI in df.columns:
    df["BMI_z"] = zscore(df[BMI])

if SBP in df.columns:
    df["SBP_z"] = zscore(df[SBP])

if HBA1C_CMR_INTERVAL in df.columns:
    df["HbA1c_CMR_interval_z"] = zscore(
        df[HBA1C_CMR_INTERVAL]
    )

# ============================================================
# QC
# ============================================================

print("\nAPOE4:")
print(df[APOE].value_counts(dropna=False))

print("\nHbA1c:")
print(df[HBA1C].describe())

print("\nLVEDVi:")
print(df["LVEDVi"].describe())

print("\nHbA1c-to-CMR interval:")
if HBA1C_CMR_INTERVAL in df.columns:
    print(df[HBA1C_CMR_INTERVAL].describe())

# ============================================================
# MODEL DEFINITIONS
# ============================================================

models = {
    "M0_BASE":
        f"LVEDVi ~ {APOE} * HbA1c_z + {AGE} + C({SEX})",

    "M1_INTERVAL":
        f"LVEDVi ~ {APOE} * HbA1c_z + {AGE} + C({SEX})"
        f" + HbA1c_CMR_interval_z",

    "M2_BMI":
        f"LVEDVi ~ {APOE} * HbA1c_z + {AGE} + C({SEX})"
        f" + HbA1c_CMR_interval_z + BMI_z",

    "M3_VASCULAR":
        f"LVEDVi ~ {APOE} * HbA1c_z + {AGE} + C({SEX})"
        f" + HbA1c_CMR_interval_z + BMI_z + SBP_z + {HTNMED}",
}

# ============================================================
# FIT MODELS
# ============================================================

rows = []
fits = {}

for name, formula in models.items():

    vars_needed = [
        "LVEDVi",
        APOE,
        "HbA1c_z",
        AGE,
        SEX
    ]

    if name in ["M1_INTERVAL", "M2_BMI", "M3_VASCULAR"]:
        vars_needed.append("HbA1c_CMR_interval_z")

    if name in ["M2_BMI", "M3_VASCULAR"]:
        vars_needed.append("BMI_z")

    if name == "M3_VASCULAR":
        vars_needed += ["SBP_z", HTNMED]

    d = df[vars_needed].dropna().copy()

    fit = smf.ols(
        formula,
        data=d
    ).fit(cov_type="HC3")

    fits[name] = fit

    int_term = f"{APOE}:HbA1c_z"

    if int_term not in fit.params.index:
        int_term = f"HbA1c_z:{APOE}"

    beta_int = fit.params[int_term]
    se_int = fit.bse[int_term]
    p_int = fit.pvalues[int_term]
    ci_int = fit.conf_int().loc[int_term]

    # --------------------------------------------------------
    # SIMPLE SLOPE APOE4 = 0
    # --------------------------------------------------------

    beta0 = fit.params["HbA1c_z"]
    se0 = fit.bse["HbA1c_z"]
    p0 = fit.pvalues["HbA1c_z"]
    ci0 = fit.conf_int().loc["HbA1c_z"]

    # --------------------------------------------------------
    # SIMPLE SLOPE APOE4 = 1
    #
    # slope = HbA1c + interaction
    # variance = var(b1) + var(b3) + 2cov(b1,b3)
    # --------------------------------------------------------

    cov = fit.cov_params()

    beta1 = beta0 + beta_int

    var1 = (
        cov.loc["HbA1c_z", "HbA1c_z"]
        + cov.loc[int_term, int_term]
        + 2 * cov.loc["HbA1c_z", int_term]
    )

    se1 = np.sqrt(var1)

    z1 = beta1 / se1

    from scipy.stats import norm

    p1 = 2 * norm.sf(abs(z1))

    ci1_low = beta1 - 1.96 * se1
    ci1_high = beta1 + 1.96 * se1

    rows.append({
        "model": name,
        "N": int(fit.nobs),

        "beta_interaction": beta_int,
        "SE_interaction": se_int,
        "CI_interaction_low": ci_int.iloc[0],
        "CI_interaction_high": ci_int.iloc[1],
        "p_interaction": p_int,

        "beta_HbA1c_APOE0": beta0,
        "SE_HbA1c_APOE0": se0,
        "CI_APOE0_low": ci0.iloc[0],
        "CI_APOE0_high": ci0.iloc[1],
        "p_APOE0": p0,

        "beta_HbA1c_APOE1": beta1,
        "SE_HbA1c_APOE1": se1,
        "CI_APOE1_low": ci1_low,
        "CI_APOE1_high": ci1_high,
        "p_APOE1": p1,

        "R2": fit.rsquared,
        "R2_adj": fit.rsquared_adj,
    })

results = pd.DataFrame(rows)

# ============================================================
# SAVE MODEL RESULTS
# ============================================================

results_file = (
    OUTDIR /
    "LVEDVi_APOE4_HbA1c_validation_models.csv"
)

results.to_csv(
    results_file,
    index=False
)

print("\n" + "="*110)
print("LVEDVi × APOE4 × HbA1c VALIDATION")
print("="*110)

print(
    results.to_string(
        index=False,
        float_format=lambda x: f"{x:.6g}"
    )
)

# ============================================================
# FIGURE
#
# Use M0_BASE for transparent visualization.
# Show raw LVEDVi against HbA1c, with fitted lines adjusted
# at reference covariate values.
# ============================================================

plot_vars = [
    "LVEDVi",
    APOE,
    HBA1C,
    "HbA1c_z",
    AGE,
    SEX
]

plotdat = df[plot_vars].dropna().copy()

fit = fits["M0_BASE"]

# Reference age = sample mean
age_ref = plotdat[AGE].mean()

# Reference sex = most common category
sex_ref = plotdat[SEX].mode().iloc[0]

hba_min = plotdat[HBA1C].quantile(0.02)
hba_max = plotdat[HBA1C].quantile(0.98)

hba_grid = np.linspace(
    hba_min,
    hba_max,
    100
)

hba_mean = df[HBA1C].mean()
hba_sd = df[HBA1C].std(ddof=0)

grid_z = (
    hba_grid - hba_mean
) / hba_sd

fig, ax = plt.subplots(
    figsize=(8, 6)
)

# Raw data
for apoe_value, label in [
    (0, "APOE4 noncarrier"),
    (1, "APOE4 carrier")
]:

    subset = plotdat[
        plotdat[APOE] == apoe_value
    ]

    ax.scatter(
        subset[HBA1C],
        subset["LVEDVi"],
        alpha=0.35,
        s=22,
        label=f"{label} observations"
    )

    pred = pd.DataFrame({
        APOE: apoe_value,
        "HbA1c_z": grid_z,
        AGE: age_ref,
        SEX: sex_ref
    })

    sf = fit.get_prediction(
        pred
    ).summary_frame(
        alpha=0.05
    )

    ax.plot(
        hba_grid,
        sf["mean"],
        linewidth=2.5,
        label=f"{label} fitted"
    )

    ax.fill_between(
        hba_grid,
        sf["mean_ci_lower"],
        sf["mean_ci_upper"],
        alpha=0.15
    )

ax.set_xlabel("HbA1c (%)")
ax.set_ylabel("LV end-diastolic volume index (mL/m²)")

ax.set_title(
    "HbA1c × APOE4 interaction for LVEDVi"
)

ax.legend(
    frameon=False,
    fontsize=9
)

fig.tight_layout()

png = (
    OUTDIR /
    "LVEDVi_HbA1c_APOE4_interaction.png"
)

pdf = (
    OUTDIR /
    "LVEDVi_HbA1c_APOE4_interaction.pdf"
)

fig.savefig(
    png,
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    pdf,
    bbox_inches="tight"
)

plt.close(fig)

# ============================================================
# SIMPLE-SLOPE FOREST PLOT
# ============================================================

r = results[
    results["model"] == "M0_BASE"
].iloc[0]

labels = [
    "APOE4 noncarriers",
    "APOE4 carriers"
]

betas = [
    r["beta_HbA1c_APOE0"],
    r["beta_HbA1c_APOE1"]
]

lower = [
    r["CI_APOE0_low"],
    r["CI_APOE1_low"]
]

upper = [
    r["CI_APOE0_high"],
    r["CI_APOE1_high"]
]

y = np.arange(2)

fig, ax = plt.subplots(
    figsize=(7, 3.5)
)

ax.errorbar(
    betas,
    y,
    xerr=[
        np.array(betas) - np.array(lower),
        np.array(upper) - np.array(betas)
    ],
    fmt="o",
    capsize=4
)

ax.axvline(
    0,
    linewidth=1
)

ax.set_yticks(y)
ax.set_yticklabels(labels)

ax.set_xlabel(
    "Change in LVEDVi per 1-SD higher HbA1c (mL/m²)"
)

ax.set_title(
    "HbA1c simple slopes by APOE4 status"
)

fig.tight_layout()

forest_png = (
    OUTDIR /
    "LVEDVi_HbA1c_APOE4_simple_slopes.png"
)

fig.savefig(
    forest_png,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print("\nSaved:")
print(results_file)
print(png)
print(pdf)
print(forest_png)

