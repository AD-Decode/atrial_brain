#!/usr/bin/env python3

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import patsy
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

# ============================================================
# PATHS
# ============================================================

ROOT = Path("/data/qiallab/Framingham")

VALID = (
    ROOT / "results"
    / "longitudinal_heart_brain"
    / "temporal_prediction"
    / "gee_validation"
)

DATAFILE = (
    ROOT / "data"
    / "longitudinal_derived"
    / "E4_E6_repeated_MRI_temporal_analysis.tsv"
)

FIGDIR = (
    ROOT / "results"
    / "longitudinal_heart_brain"
    / "temporal_prediction"
    / "figures"
)
FIGDIR.mkdir(parents=True, exist_ok=True)

EFFECT_FILE = VALID / "FOCUS_LA_lateral_ventricle_effects.tsv"
SLOPE_FILE = VALID / "FOCUS_LA_lateral_ventricle_simple_slopes.tsv"

# ============================================================
# LOAD 108 RESULTS
# ============================================================

effects = pd.read_csv(EFFECT_FILE, sep="\t")
slopes = pd.read_csv(SLOPE_FILE, sep="\t")

METHOD_LABEL = {
    "MixedLM_random_intercept": "MixedLM",
    "GEE_exchangeable_robust": "GEE",
}

SUBSET_LABEL = {
    "prospective": "Prospective",
    "gap_ge2y": "≥2-year gap",
}

effects["method_label"] = effects["method"].map(METHOD_LABEL)
effects["subset_label"] = effects["subset"].map(SUBSET_LABEL)

slopes["method_label"] = slopes["method"].map(METHOD_LABEL)
slopes["subset_label"] = slopes["subset"].map(SUBSET_LABEL)

# ============================================================
# LOAD ANALYSIS DATA AND RECREATE PROSPECTIVE LA MODEL
# ============================================================

long = pd.read_csv(
    DATAFILE,
    sep="\t",
    dtype={"shareid": str},
    low_memory=False
)

CARDIAC_BASE = "la_dim"
NCOL = f"{CARDIAC_BASE}_n"
SCOL = f"{CARDIAC_BASE}_slope"
OUTCOME = "Lateralvent"

d = long[
    (pd.to_numeric(long[NCOL], errors="coerce") >= 3)
    & long[SCOL].notna()
    & long["APOE4_carrier"].notna()
    & long["age6_c"].notna()
    & long["sex"].notna()
    & (
        pd.to_numeric(
            long["last_echo_to_first_mri_years"],
            errors="coerce"
        ) >= 0
    )
    & long[OUTCOME].notna()
    & long["mri_time_years"].notna()
].copy()

# Require >=2 lateral-ventricle measurements
counts = d.groupby("shareid")[OUTCOME].count()
good_ids = counts[counts >= 2].index
d = d[d["shareid"].isin(good_ids)].copy()

# Exactly reproduce cardiac scaling from 104/108
subject_slopes = (
    d[["shareid", SCOL]]
    .drop_duplicates("shareid")
)

cardiac_mean = subject_slopes[SCOL].mean()
cardiac_sd = subject_slopes[SCOL].std()

d["cardiac_z"] = (
    d[SCOL] - cardiac_mean
) / cardiac_sd

# Exactly reproduce outcome scaling
brain_mean = d[OUTCOME].mean()
brain_sd = d[OUTCOME].std()

d["brain_z"] = (
    d[OUTCOME] - brain_mean
) / brain_sd

FORMULA = (
    "brain_z ~ "
    "mri_time_years * cardiac_z * APOE4_carrier "
    "+ age6_c + C(sex)"
)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    model = smf.mixedlm(
        FORMULA,
        data=d,
        groups=d["shareid"],
        re_formula="1"
    )

    res = model.fit(
        reml=False,
        method="lbfgs",
        maxiter=2000,
        disp=False
    )

    if not getattr(res, "converged", False):
        res = model.fit(
            reml=False,
            method="powell",
            maxiter=5000,
            disp=False
        )

print("Prospective LA model converged:", res.converged)
print("N subjects:", d["shareid"].nunique())
print("N observations:", len(d))

# ============================================================
# PREDICTION GRID
#
# -1 SD and +1 SD LA trajectory
# APOE4 noncarrier and carrier
# age6 centered at 0 = cohort mean
# sex set to most common category
# ============================================================

sex_ref = d["sex"].mode().iloc[0]

# Use central observed follow-up range rather than extreme tail
max_time = float(
    d.groupby("shareid")["mri_time_years"]
    .max()
    .quantile(0.95)
)

max_time = min(max_time, 15.0)

time_grid = np.linspace(0, max_time, 100)

prediction_rows = []

for apoe in [0, 1]:

    for cardiac_z in [-1.0, 1.0]:

        new = pd.DataFrame({
            "mri_time_years": time_grid,
            "cardiac_z": cardiac_z,
            "APOE4_carrier": apoe,
            "age6_c": 0.0,
            "sex": sex_ref,
        })

        # ------------------------------------------------------------
        # Fixed-effect predictions + 95% CI
        #
        # Reconstruct the fixed-effect design matrix from the same
        # formula used in model fitting. MixedLM does not retain
        # Patsy's design_info in this fitted object.

        rhs_formula = (
            "mri_time_years * cardiac_z * APOE4_carrier "
            "+ age6_c + C(sex)"
        )

        design_vars = [
            "mri_time_years",
            "cardiac_z",
            "APOE4_carrier",
            "age6_c",
            "sex",
        ]

        # Include original model rows when Patsy determines categorical
        # levels so that sex coding matches the fitted model.
        design_base = d[design_vars].copy()

        combined = pd.concat(
            [design_base, new[design_vars]],
            axis=0,
            ignore_index=True
        )

        X_all = patsy.dmatrix(
            rhs_formula,
            combined,
            return_type="dataframe"
        )

        # Prediction-grid rows are at the end of combined.
        X = X_all.iloc[-len(new):].copy()
        X.index = new.index

        fe_names = list(res.fe_params.index)

        X = X.reindex(
            columns=fe_names,
            fill_value=0.0
        )

        beta = res.fe_params.loc[fe_names].to_numpy()

        cov_beta = (
            res.cov_params()
            .loc[fe_names, fe_names]
            .to_numpy()
        )

        pred_values = X.to_numpy() @ beta

        pred_var = np.einsum(
            "ij,jk,ik->i",
            X.to_numpy(),
            cov_beta,
            X.to_numpy()
        )

        pred_se = np.sqrt(
            np.clip(pred_var, 0, None)
        )

        ci_low = pred_values - 1.96 * pred_se
        ci_high = pred_values + 1.96 * pred_se

        for t, y, se, lo, hi in zip(
            time_grid,
            pred_values,
            pred_se,
            ci_low,
            ci_high
        ):

            prediction_rows.append({
                "mri_time_years": t,
                "predicted_brain_z": y,
                "predicted_SE": se,
                "CI_low": lo,
                "CI_high": hi,
                "APOE4_carrier": apoe,
                "APOE_group":
                    "APOE ε4 carrier"
                    if apoe == 1
                    else "APOE ε4 noncarrier",
                "LA_trajectory":
                    "+1 SD LA remodeling"
                    if cardiac_z == 1
                    else "−1 SD LA remodeling",
                "cardiac_z": cardiac_z,
            })

pred = pd.DataFrame(prediction_rows)

# Save source data
pred.to_csv(
    FIGDIR / "Figure109_predicted_trajectory_source_data.tsv",
    sep="\t",
    index=False
)

# ============================================================
# FIGURE
# ============================================================

fig = plt.figure(
    figsize=(16, 6.5),
    constrained_layout=True
)

gs = fig.add_gridspec(
    1,
    3,
    width_ratios=[1.0, 1.15, 1.35]
)

axA = fig.add_subplot(gs[0, 0])
axB = fig.add_subplot(gs[0, 1])
axC = fig.add_subplot(gs[0, 2])

# ============================================================
# PANEL A
# LA × MRI TIME × APOE4:
# MixedLM vs GEE
# ============================================================

A = effects[
    effects["effect_type"]
    ==
    "MRI_time_x_cardiac_x_APOE4"
].copy()

order_A = [
    ("prospective", "MixedLM_random_intercept"),
    ("prospective", "GEE_exchangeable_robust"),
    ("gap_ge2y", "MixedLM_random_intercept"),
    ("gap_ge2y", "GEE_exchangeable_robust"),
]

rows_A = []

for subset, method in order_A:

    tmp = A[
        (A["subset"] == subset)
        & (A["method"] == method)
    ]

    if len(tmp):
        r = tmp.iloc[0].copy()
        r["display"] = (
            f'{SUBSET_LABEL[subset]} — '
            f'{METHOD_LABEL[method]}'
        )
        rows_A.append(r)

A = pd.DataFrame(rows_A)

yA = np.arange(len(A))[::-1]

for i, (_, r) in enumerate(A.iterrows()):

    marker = "D" if r["method_label"] == "MixedLM" else "o"

    axA.errorbar(
        r["beta"],
        yA[i],
        xerr=[
            [r["beta"] - r["CI_low"]],
            [r["CI_high"] - r["beta"]]
        ],
        fmt=marker,
        capsize=4,
        markersize=7,
        linewidth=1.5,
    )

axA.axvline(
    0,
    linestyle="--",
    linewidth=1
)

axA.set_yticks(yA)
axA.set_yticklabels(A["display"])
axA.set_xlabel("Interaction β (95% CI)")
axA.set_title(
    "A. LA remodeling × MRI time × APOE4",
    loc="left",
    weight="bold"
)
axA.grid(
    axis="x",
    linestyle=":",
    linewidth=0.6
)

# Add P/q text under labels using annotation
for i, (_, r) in enumerate(A.iterrows()):

    if r["method_label"] == "MixedLM":
        txt = f'P={r["P"]:.3g}; q={r["FDR_q"]:.3g}'
    else:
        txt = f'P={r["P"]:.3g}; q={r["FDR_q"]:.3g}'

    axA.annotate(
        txt,
        xy=(r["beta"], yA[i]),
        xytext=(6, -12),
        textcoords="offset points",
        fontsize=8
    )

# ============================================================
# PANEL B
# APOE-STRATIFIED SIMPLE SLOPES
# ============================================================

B = slopes.copy()

order_B = [
    ("prospective", "MixedLM_random_intercept", 0),
    ("prospective", "MixedLM_random_intercept", 1),
    ("prospective", "GEE_exchangeable_robust", 0),
    ("prospective", "GEE_exchangeable_robust", 1),
    ("gap_ge2y", "MixedLM_random_intercept", 0),
    ("gap_ge2y", "MixedLM_random_intercept", 1),
    ("gap_ge2y", "GEE_exchangeable_robust", 0),
    ("gap_ge2y", "GEE_exchangeable_robust", 1),
]

rows_B = []

for subset, method, apoe in order_B:

    tmp = B[
        (B["subset"] == subset)
        & (B["method"] == method)
        & (B["APOE4_carrier"] == apoe)
    ]

    if len(tmp):

        r = tmp.iloc[0].copy()

        group = (
            "APOE4+"
            if apoe == 1
            else "APOE4−"
        )

        r["display"] = (
            f'{SUBSET_LABEL[subset]} — '
            f'{METHOD_LABEL[method]} — {group}'
        )

        rows_B.append(r)

B = pd.DataFrame(rows_B)

yB = np.arange(len(B))[::-1]

for i, (_, r) in enumerate(B.iterrows()):

    marker = "D" if r["method_label"] == "MixedLM" else "o"

    axB.errorbar(
        r["beta"],
        yB[i],
        xerr=[
            [r["beta"] - r["CI_low"]],
            [r["CI_high"] - r["beta"]]
        ],
        fmt=marker,
        capsize=4,
        markersize=7,
        linewidth=1.4,
    )

axB.axvline(
    0,
    linestyle="--",
    linewidth=1
)

axB.set_yticks(yB)
axB.set_yticklabels(B["display"], fontsize=8.5)

axB.set_xlabel(
    "LA trajectory effect on ventricular slope\n"
    "(standardized β, 95% CI)"
)

axB.set_title(
    "B. APOE4-stratified simple slopes",
    loc="left",
    weight="bold"
)

axB.grid(
    axis="x",
    linestyle=":",
    linewidth=0.6
)

# ============================================================
# PANEL C
# MODEL-BASED PREDICTED VENTRICULAR TRAJECTORIES
# ============================================================

line_styles = {
    "−1 SD LA remodeling": "--",
    "+1 SD LA remodeling": "-",
}

markers = {
    "APOE ε4 noncarrier": None,
    "APOE ε4 carrier": None,
}

for apoe_group in [
    "APOE ε4 noncarrier",
    "APOE ε4 carrier"
]:

    for la_group in [
        "−1 SD LA remodeling",
        "+1 SD LA remodeling"
    ]:

        z = pred[
            (pred["APOE_group"] == apoe_group)
            & (pred["LA_trajectory"] == la_group)
        ]

        label = f"{apoe_group}, {la_group}"

        axC.plot(
            z["mri_time_years"],
            z["predicted_brain_z"],
            linestyle=line_styles[la_group],
            linewidth=2,
            label=label
        )

axC.axhline(
    0,
    linewidth=0.8,
    linestyle=":"
)

axC.set_xlabel(
    "Years since first brain MRI"
)

axC.set_ylabel(
    "Predicted lateral ventricular volume (SD)"
)

axC.set_title(
    "C. Model-predicted ventricular trajectories",
    loc="left",
    weight="bold"
)

axC.grid(
    linestyle=":",
    linewidth=0.6
)

axC.legend(
    frameon=False,
    fontsize=8,
    loc="best"
)

# ============================================================
# OVERALL TITLE / NOTE
# ============================================================

fig.suptitle(
    "Early left atrial remodeling and subsequent lateral ventricular change",
    fontsize=15,
    weight="bold"
)

fig.text(
    0.5,
    -0.01,
    (
        "Early cardiac trajectory derived from Offspring Exams 4–6. "
        "Panel C shows fixed-effect predictions from the prospective "
        "random-intercept MixedLM at mean Exam-6 age and reference sex. "
        "Diamonds indicate MixedLM; circles indicate GEE."
    ),
    ha="center",
    fontsize=9
)

# ============================================================
# SAVE
# ============================================================

PNG = FIGDIR / "Figure109_LA_APOE4_ventricular_validation.png"
PDF = FIGDIR / "Figure109_LA_APOE4_ventricular_validation.pdf"
SVG = FIGDIR / "Figure109_LA_APOE4_ventricular_validation.svg"

fig.savefig(
    PNG,
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    PDF,
    bbox_inches="tight"
)

fig.savefig(
    SVG,
    bbox_inches="tight"
)

plt.close(fig)

# ============================================================
# FIGURE SOURCE TABLES
# ============================================================

A.to_csv(
    FIGDIR / "Figure109_panelA_interaction_estimates.tsv",
    sep="\t",
    index=False
)

B.to_csv(
    FIGDIR / "Figure109_panelB_simple_slopes.tsv",
    sep="\t",
    index=False
)

print("\nSaved:")
print(PNG)
print(PDF)
print(SVG)
print(FIGDIR / "Figure109_panelA_interaction_estimates.tsv")
print(FIGDIR / "Figure109_panelB_simple_slopes.tsv")
print(FIGDIR / "Figure109_predicted_trajectory_source_data.tsv")

print("\n109 COMPLETE")
