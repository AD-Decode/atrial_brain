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

# ============================================================
# BASELINE LA DIMENSION
# From the original cardiac subject-trajectory file.
# la_dim_baseline is the first observed LA dimension after
# sorting cardiac observations by actual examination time.
# ============================================================

BASELINE_FILE = Path(
    "/data/qiallab/Framingham/downloads/longitudinal_20260924/"
    "longitudinal_analysis/cardiac_subject_trajectories.tsv"
)

base = pd.read_csv(
    BASELINE_FILE,
    sep="\t",
    dtype={"shareid": str},
    low_memory=False
)

base["shareid"] = base["shareid"].astype(str).str.strip()

assert "la_dim_baseline" in base.columns, (
    "STOP: la_dim_baseline not found in trajectory file"
)

base["la_dim_baseline"] = pd.to_numeric(
    base["la_dim_baseline"],
    errors="coerce"
)

base = (
    base[["shareid", "la_dim_baseline"]]
    .drop_duplicates("shareid")
)

long["shareid"] = long["shareid"].astype(str).str.strip()

long = long.merge(
    base,
    on="shareid",
    how="left",
    validate="many_to_one"
)

print(
    "Baseline LA available:",
    long.loc[
        long["la_dim_baseline"].notna(),
        "shareid"
    ].nunique(),
    "subjects"
)


CARDIAC_BASE = "la_dim"
NCOL = f"{CARDIAC_BASE}_n"
SCOL = f"{CARDIAC_BASE}_slope"
OUTCOME = "Lateralvent"

d = long[
    (pd.to_numeric(long[NCOL], errors="coerce") >= 3)
    & long[SCOL].notna()
    & long["la_dim_baseline"].notna()
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

# Standardize baseline LA using one value per subject,
# not repeated MRI observations.
subject_baseline = (
    d[["shareid", "la_dim_baseline"]]
    .drop_duplicates("shareid")
)

baseline_mean = subject_baseline["la_dim_baseline"].mean()
baseline_sd = subject_baseline["la_dim_baseline"].std()

d["baseline_LA_z"] = (
    d["la_dim_baseline"] - baseline_mean
) / baseline_sd

print(
    f"Baseline LA mean = {baseline_mean:.6f}; "
    f"SD = {baseline_sd:.6f}"
)


# Exactly reproduce outcome scaling
brain_mean = d[OUTCOME].mean()
brain_sd = d[OUTCOME].std()

d["brain_z"] = (
    d[OUTCOME] - brain_mean
) / brain_sd

FORMULA = (
    "brain_z ~ "
    "mri_time_years * cardiac_z * APOE4_carrier "
    "+ baseline_LA_z "
    "+ baseline_LA_z:mri_time_years "
    "+ age6_c + C(sex) "
    "+ age6_c:mri_time_years "
    "+ C(sex):mri_time_years"
)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    model = smf.mixedlm(
        FORMULA,
        data=d,
        groups=d["shareid"],
        re_formula="~mri_time_years"
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
# KEY SENSITIVITY RESULT
# ============================================================

TERM = "mri_time_years:cardiac_z:APOE4_carrier"
ci = res.conf_int()

print("\\n" + "=" * 80)
print("BASELINE LA + RANDOM MRI-TIME SLOPE SENSITIVITY")
print("=" * 80)
print("Formula:", FORMULA)
print("Three-way term:", TERM)
print(f"beta      = {res.params[TERM]:.8f}")
print(f"SE        = {res.bse[TERM]:.8f}")
print(f"95% CI    = [{ci.loc[TERM, 0]:.8f}, {ci.loc[TERM, 1]:.8f}]")
print(f"P         = {res.pvalues[TERM]:.8g}")
print(f"N subjects = {d['shareid'].nunique()}")
print(f"N obs      = {len(d)}")
print(f"Converged  = {res.converged}")
print("=" * 80)


print("\n142 COMPLETE — random MRI-time slope sensitivity.")
# raise SystemExit(0)  # disabled for 146

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


# =============================================================================
# 146: ICV / HEAD-SIZE SENSITIVITY
# =============================================================================
#
# Purpose:
#   Address head-size handling for the prospective LA-remodeling ->
#   lateral-ventricular trajectory association.
#
# Strategy:
#   Restrict to the SAME ICV-complete observations and compare:
#
#   MODEL A: robust longitudinal model without ICV
#   MODEL B: identical model + ICV_z + ICV_z:mri_time_years
#
#   Both models include:
#       cardiac_z × MRI time × APOE4
#       baseline LA + baseline LA × MRI time
#       age + age × MRI time
#       sex + sex × MRI time
#       subject-specific random intercept + random MRI-time slope
#
# =============================================================================

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

print("\n\n")
print("=" * 100)
print("146: LONGITUDINAL ICV / HEAD-SIZE SENSITIVITY")
print("=" * 100)

ICV_FILE = Path(
    "/data/qiallab/Framingham/downloads/longitudinal_20260924/"
    "phs000007.v35.pht004364.v3.p16.c1."
    "t_mrbrfs_2010_1_0900s.HMB-IRB-MDS.txt.gz"
)

# -------------------------------------------------------------------------
# The validated 143 script leaves its prospective analytic dataframe as d.
# Verify required variables before proceeding.
# -------------------------------------------------------------------------

required = [
    "shareid",
    "brain_z",
    "mri_time_years",
    "cardiac_z",
    "APOE4_carrier",
    "baseline_LA_z",
    "age6_c",
    "sex",
]

missing = [x for x in required if x not in d.columns]

if missing:
    raise RuntimeError(
        "146 cannot proceed because variables expected from 143 are missing: "
        + ", ".join(missing)
    )

# -------------------------------------------------------------------------
# Read subject-level ICV
# -------------------------------------------------------------------------

raw_icv = pd.read_csv(
    ICV_FILE,
    sep="\t",
    dtype=str,
    low_memory=False,
    comment="#"
)

raw_icv.columns = [
    str(c).strip()
    for c in raw_icv.columns
]

assert "shareid" in raw_icv.columns
assert "IntraCranialVol" in raw_icv.columns

icv = raw_icv[
    ["shareid", "IntraCranialVol"]
].copy()

icv["shareid"] = (
    icv["shareid"]
    .astype(str)
    .str.strip()
)

icv["IntraCranialVol"] = pd.to_numeric(
    icv["IntraCranialVol"],
    errors="coerce"
)

icv = icv.dropna(
    subset=["IntraCranialVol"]
).copy()

if icv["shareid"].duplicated().any():
    raise RuntimeError(
        "ICV file unexpectedly contains duplicate shareid values."
    )

# -------------------------------------------------------------------------
# Harmonize IDs and merge
# -------------------------------------------------------------------------

dd = d.copy()

dd["shareid"] = (
    dd["shareid"]
    .astype(str)
    .str.strip()
)

# Prevent accidental duplicate column if script is rerun interactively.
if "IntraCranialVol" in dd.columns:
    dd = dd.drop(
        columns=["IntraCranialVol"]
    )

dd = dd.merge(
    icv,
    on="shareid",
    how="inner",
    validate="many_to_one"
)

# Complete cases for ALL variables used in either model.
model_vars = [
    "shareid",
    "brain_z",
    "mri_time_years",
    "cardiac_z",
    "APOE4_carrier",
    "baseline_LA_z",
    "age6_c",
    "sex",
    "IntraCranialVol",
]

dd = dd.dropna(
    subset=model_vars
).copy()

# Require >=2 observations AFTER ICV restriction.
nobs = (
    dd.groupby("shareid")
    ["brain_z"]
    .count()
)

keep_ids = nobs[
    nobs >= 2
].index

dd = dd[
    dd["shareid"].isin(keep_ids)
].copy()

# -------------------------------------------------------------------------
# Standardize ICV across SUBJECTS, not observations.
#
# Since each subject contributes multiple MRI observations, calculating the
# mean/SD directly across rows would weight subjects by their number of scans.
# -------------------------------------------------------------------------

subject_icv = (
    dd[
        ["shareid", "IntraCranialVol"]
    ]
    .drop_duplicates("shareid")
    .copy()
)

icv_mean = subject_icv[
    "IntraCranialVol"
].mean()

icv_sd = subject_icv[
    "IntraCranialVol"
].std(ddof=0)

if not np.isfinite(icv_sd) or icv_sd <= 0:
    raise RuntimeError(
        "Invalid ICV SD."
    )

dd["ICV_z"] = (
    dd["IntraCranialVol"]
    - icv_mean
) / icv_sd

# -------------------------------------------------------------------------
# QC
# -------------------------------------------------------------------------

n_subjects = dd["shareid"].nunique()
n_obs = len(dd)

carrier_counts = (
    dd[
        ["shareid", "APOE4_carrier"]
    ]
    .drop_duplicates("shareid")
    ["APOE4_carrier"]
    .value_counts()
    .sort_index()
)

print("\nMATCHED ICV-COMPLETE ANALYTIC SAMPLE")
print("-" * 100)

print("Subjects :", n_subjects)
print("MRI obs  :", n_obs)

print("\nAPOE4 counts:")
print(carrier_counts.to_string())

print("\nICV:")
print(f"Mean = {icv_mean:.6f}")
print(f"SD   = {icv_sd:.6f}")

assert n_subjects > 0
assert n_obs > n_subjects

# -------------------------------------------------------------------------
# MODEL A
# Matched robust model WITHOUT ICV.
# -------------------------------------------------------------------------

FORMULA_A = (
    "brain_z ~ "
    "mri_time_years * cardiac_z * APOE4_carrier "
    "+ baseline_LA_z "
    "+ baseline_LA_z:mri_time_years "
    "+ age6_c "
    "+ age6_c:mri_time_years "
    "+ C(sex) "
    "+ C(sex):mri_time_years"
)

print("\n" + "=" * 100)
print("MODEL A: MATCHED SAMPLE, NO ICV")
print("=" * 100)
print(FORMULA_A)

model_A = smf.mixedlm(
    FORMULA_A,
    data=dd,
    groups=dd["shareid"],
    re_formula="~mri_time_years",
)

fit_A = model_A.fit(
    reml=False,
    method="lbfgs",
    maxiter=2000,
    disp=False,
)

# -------------------------------------------------------------------------
# MODEL B
# SAME subjects/observations + ICV and ICV × MRI time.
# -------------------------------------------------------------------------

FORMULA_B = (
    "brain_z ~ "
    "mri_time_years * cardiac_z * APOE4_carrier "
    "+ baseline_LA_z "
    "+ baseline_LA_z:mri_time_years "
    "+ age6_c "
    "+ age6_c:mri_time_years "
    "+ C(sex) "
    "+ C(sex):mri_time_years "
    "+ ICV_z "
    "+ ICV_z:mri_time_years"
)

print("\n" + "=" * 100)
print("MODEL B: SAME SAMPLE + ICV + ICV x MRI TIME")
print("=" * 100)
print(FORMULA_B)

model_B = smf.mixedlm(
    FORMULA_B,
    data=dd,
    groups=dd["shareid"],
    re_formula="~mri_time_years",
)

fit_B = model_B.fit(
    reml=False,
    method="lbfgs",
    maxiter=2000,
    disp=False,
)

# -------------------------------------------------------------------------
# Extract key three-way interaction.
# -------------------------------------------------------------------------

TERM = (
    "mri_time_years:"
    "cardiac_z:"
    "APOE4_carrier"
)

def extract(fit, label):
    if TERM not in fit.params.index:
        raise RuntimeError(
            f"Three-way term not found in {label}"
        )

    beta = fit.params[TERM]
    se = fit.bse[TERM]
    p = fit.pvalues[TERM]

    ci = fit.conf_int().loc[TERM]

    return {
        "model": label,
        "beta": beta,
        "SE": se,
        "CI_low": ci.iloc[0],
        "CI_high": ci.iloc[1],
        "P": p,
        "N_subjects": n_subjects,
        "N_obs": n_obs,
        "converged": fit.converged,
    }

rA = extract(
    fit_A,
    "Matched sample, no ICV"
)

rB = extract(
    fit_B,
    "Matched sample + ICV + ICV*time"
)

res = pd.DataFrame(
    [rA, rB]
)

# -------------------------------------------------------------------------
# Quantify coefficient change due specifically to ICV adjustment.
# -------------------------------------------------------------------------

beta_A = rA["beta"]
beta_B = rB["beta"]

pct_change_signed = (
    (beta_B - beta_A)
    / abs(beta_A)
    * 100
)

pct_change_magnitude = (
    (abs(beta_B) - abs(beta_A))
    / abs(beta_A)
    * 100
)

print("\n" + "=" * 100)
print("PRIMARY ICV SENSITIVITY RESULT")
print("=" * 100)

print(
    res.to_string(
        index=False,
        float_format=lambda x: f"{x:.8g}"
    )
)

print("\nEffect of ICV adjustment on three-way coefficient:")
print(
    f"Matched no-ICV beta : {beta_A:+.8f}"
)
print(
    f"ICV-adjusted beta    : {beta_B:+.8f}"
)
print(
    f"Signed change        : {pct_change_signed:+.2f}%"
)
print(
    f"Change in |beta|     : {pct_change_magnitude:+.2f}%"
)

# -------------------------------------------------------------------------
# Save results
# -------------------------------------------------------------------------

OUTDIR = Path(
    "/data/qiallab/Framingham/results/"
    "longitudinal_heart_brain/"
    "ICV_sensitivity"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True
)

res.to_csv(
    OUTDIR /
    "146_longitudinal_ICV_matched_models.csv",
    index=False
)

with open(
    OUTDIR /
    "146_longitudinal_ICV_model_summary.txt",
    "w"
) as f:

    f.write(
        "146 LONGITUDINAL ICV SENSITIVITY\n"
    )
    f.write(
        "=" * 80 + "\n\n"
    )

    f.write(
        f"N subjects: {n_subjects}\n"
    )
    f.write(
        f"N observations: {n_obs}\n\n"
    )

    f.write(
        "MODEL A — MATCHED SAMPLE, NO ICV\n"
    )
    f.write(
        fit_A.summary().as_text()
    )

    f.write(
        "\n\nMODEL B — SAME SAMPLE + ICV + ICV x MRI TIME\n"
    )
    f.write(
        fit_B.summary().as_text()
    )

    f.write(
        "\n\nKEY COMPARISON\n"
    )
    f.write(
        res.to_string(index=False)
    )

    f.write(
        f"\n\nMatched no-ICV beta: {beta_A:+.8f}\n"
    )
    f.write(
        f"ICV-adjusted beta: {beta_B:+.8f}\n"
    )
    f.write(
        f"Signed change: {pct_change_signed:+.2f}%\n"
    )
    f.write(
        f"Change in |beta|: {pct_change_magnitude:+.2f}%\n"
    )

print("\nSaved:")
print(
    OUTDIR /
    "146_longitudinal_ICV_matched_models.csv"
)
print(
    OUTDIR /
    "146_longitudinal_ICV_model_summary.txt"
)

print("\n" + "=" * 100)
print("146 COMPLETE")
print("=" * 100)
