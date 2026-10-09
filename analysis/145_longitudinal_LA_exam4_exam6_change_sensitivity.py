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
# EXPLICIT EXAM 4 -> EXAM 6 LA CHANGE
# ============================================================

ECHO_FILE = Path(
    "/data/qiallab/Framingham/downloads/longitudinal_20260924/"
    "harmonized/echo_harmonized_long.tsv"
)

echo = pd.read_csv(
    ECHO_FILE,
    sep="\t",
    dtype={"shareid": str},
    low_memory=False
)

echo["shareid"] = echo["shareid"].astype(str).str.strip()
echo["exam"] = pd.to_numeric(echo["exam"], errors="coerce")
echo["la_dim"] = pd.to_numeric(echo["la_dim"], errors="coerce")

assert "la_dim" in echo.columns, "STOP: la_dim not found"
assert "exam" in echo.columns, "STOP: exam not found"

# Keep only literal Exam 4 and Exam 6 measurements.
e46 = echo.loc[
    echo["exam"].isin([4, 6]),
    ["shareid", "exam", "la_dim"]
].copy()

# Check that there is at most one LA measurement per subject/exam.
dup = e46.duplicated(["shareid", "exam"], keep=False)

if dup.any():
    print("\nWARNING: duplicate subject/exam LA records detected:")
    print(
        e46.loc[dup]
        .sort_values(["shareid", "exam"])
        .head(30)
        .to_string(index=False)
    )
    raise RuntimeError(
        "STOP: duplicate Exam 4/6 LA measurements; inspect before proceeding."
    )

wide = e46.pivot(
    index="shareid",
    columns="exam",
    values="la_dim"
).reset_index()

assert 4 in wide.columns, "STOP: no Exam 4 LA column"
assert 6 in wide.columns, "STOP: no Exam 6 LA column"

wide = wide.rename(
    columns={
        4: "la_dim_exam4",
        6: "la_dim_exam6",
    }
)

# ============================================================
# Obtain actual Exam 4 and Exam 6 dates from the same date
# backbone used by script 97.
# ============================================================

DATE_ROOT = Path(
    "/data/qiallab/Framingham/downloads/longitudinal_20260924"
)

date_hits = sorted(
    DATE_ROOT.glob("*pht003099*.HMB-IRB-MDS.txt.gz")
)

assert date_hits, "STOP: pht003099 date file not found"

dates = pd.read_csv(
    date_hits[0],
    sep="\t",
    dtype=str,
    low_memory=False,
    comment="#"
)

dates.columns = [c.strip() for c in dates.columns]
dates["shareid"] = dates["shareid"].astype(str).str.strip()

assert "date4" in dates.columns, "STOP: date4 not found"
assert "date6" in dates.columns, "STOP: date6 not found"

dates["date4"] = pd.to_numeric(
    dates["date4"],
    errors="coerce"
)
dates["date6"] = pd.to_numeric(
    dates["date6"],
    errors="coerce"
)

dates = (
    dates[["shareid", "date4", "date6"]]
    .drop_duplicates("shareid")
)

wide = wide.merge(
    dates,
    on="shareid",
    how="left",
    validate="one_to_one"
)

# Same date units/conversion used by script 97:
# elapsed years = date difference / 365.25
wide["exam4_exam6_years"] = (
    wide["date6"] - wide["date4"]
) / 365.25

wide["la_dim_exam4_exam6_delta"] = (
    wide["la_dim_exam6"] -
    wide["la_dim_exam4"]
)

wide["la_dim_exam4_exam6_change"] = (
    wide["la_dim_exam4_exam6_delta"] /
    wide["exam4_exam6_years"]
)

# Require actual positive elapsed time and both LA measurements.
bad_time = (
    wide["exam4_exam6_years"].isna()
    | (wide["exam4_exam6_years"] <= 0)
)

wide.loc[
    bad_time,
    "la_dim_exam4_exam6_change"
] = float("nan")

# Baseline LA for this sensitivity is explicitly Exam 4 LA.
wide["la_dim_baseline"] = wide["la_dim_exam4"]

print("\nEXPLICIT EXAM 4 -> EXAM 6 LA DATA")
print("=" * 80)
print(
    "Subjects with Exam 4 LA:",
    wide["la_dim_exam4"].notna().sum()
)
print(
    "Subjects with Exam 6 LA:",
    wide["la_dim_exam6"].notna().sum()
)
print(
    "Subjects with both + valid dates:",
    wide["la_dim_exam4_exam6_change"].notna().sum()
)

q = wide["exam4_exam6_years"].dropna()
print(
    "Exam 4 -> 6 interval, years:",
    f"mean={q.mean():.3f}, "
    f"SD={q.std():.3f}, "
    f"min={q.min():.3f}, "
    f"max={q.max():.3f}"
)

long["shareid"] = long["shareid"].astype(str).str.strip()

long = long.merge(
    wide[
        [
            "shareid",
            "la_dim_baseline",
            "la_dim_exam4",
            "la_dim_exam6",
            "exam4_exam6_years",
            "la_dim_exam4_exam6_delta",
            "la_dim_exam4_exam6_change",
        ]
    ],
    on="shareid",
    how="left",
    validate="many_to_one"
)

print(
    "Explicit Exam 4 -> 6 LA change available in longitudinal data:",
    long.loc[
        long["la_dim_exam4_exam6_change"].notna(),
        "shareid"
    ].nunique(),
    "subjects"
)



CARDIAC_BASE = "la_dim"
NCOL = f"{CARDIAC_BASE}_n"
SCOL = "la_dim_exam4_exam6_change"
OUTCOME = "Lateralvent"

d = long[
    long[SCOL].notna()
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
print("EXPLICIT EXAM 4 -> EXAM 6 LA CHANGE SENSITIVITY")
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
raise SystemExit(0)

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
