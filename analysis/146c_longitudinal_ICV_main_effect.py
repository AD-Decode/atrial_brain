from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

print("=" * 100)
print("146: LONGITUDINAL ICV SENSITIVITY — MATCHED COMPLETE-CASE ANALYSIS")
print("=" * 100)

# =============================================================================
# FILES
# =============================================================================

DATAFILE = Path(
    "/data/qiallab/Framingham/data/longitudinal_derived/"
    "E4_E6_repeated_MRI_temporal_analysis.tsv"
)

BASELINE_FILE = Path(
    "/data/qiallab/Framingham/downloads/longitudinal_20260924/"
    "longitudinal_analysis/cardiac_subject_trajectories.tsv"
)

ICV_FILE = Path(
    "/data/qiallab/Framingham/downloads/longitudinal_20260924/"
    "phs000007.v35.pht004364.v3.p16.c1."
    "t_mrbrfs_2010_1_0900s.HMB-IRB-MDS.txt.gz"
)

OUTDIR = Path(
    "/data/qiallab/Framingham/results/"
    "longitudinal_heart_brain/ICV_sensitivity"
)
OUTDIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# LOAD LONGITUDINAL DATA
# =============================================================================

long = pd.read_csv(
    DATAFILE,
    sep="\t",
    dtype={"shareid": str},
    low_memory=False
)

long["shareid"] = long["shareid"].astype(str).str.strip()

# =============================================================================
# BASELINE LA — EXACTLY AS VALIDATED 143
# =============================================================================

base = pd.read_csv(
    BASELINE_FILE,
    sep="\t",
    dtype={"shareid": str},
    low_memory=False
)

base["shareid"] = base["shareid"].astype(str).str.strip()
base["la_dim_baseline"] = pd.to_numeric(
    base["la_dim_baseline"],
    errors="coerce"
)

base = (
    base[["shareid", "la_dim_baseline"]]
    .drop_duplicates("shareid")
)

long = long.merge(
    base,
    on="shareid",
    how="left",
    validate="many_to_one"
)

# =============================================================================
# RECREATE EXACT 143 PRIMARY SAMPLE
# =============================================================================

CARDIAC_BASE = "la_dim"
NCOL = "la_dim_n"
SCOL = "la_dim_slope"
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

counts = d.groupby("shareid")[OUTCOME].count()
good_ids = counts[counts >= 2].index
d = d[d["shareid"].isin(good_ids)].copy()

print("\nPrimary 143 sample:")
print("Subjects:", d["shareid"].nunique())
print("Observations:", len(d))

assert d["shareid"].nunique() == 1305, (
    "STOP: did not reproduce 143 N=1305."
)
assert len(d) == 3787, (
    "STOP: did not reproduce 143 Nobs=3787."
)

# =============================================================================
# ORIGINAL 143 SCALING — BEFORE ICV RESTRICTION
# =============================================================================

subject_slopes = (
    d[["shareid", SCOL]]
    .drop_duplicates("shareid")
)

cardiac_mean = subject_slopes[SCOL].mean()
cardiac_sd = subject_slopes[SCOL].std()

d["cardiac_z"] = (
    d[SCOL] - cardiac_mean
) / cardiac_sd

subject_baseline = (
    d[["shareid", "la_dim_baseline"]]
    .drop_duplicates("shareid")
)

baseline_mean = subject_baseline["la_dim_baseline"].mean()
baseline_sd = subject_baseline["la_dim_baseline"].std()

d["baseline_LA_z"] = (
    d["la_dim_baseline"] - baseline_mean
) / baseline_sd

brain_mean = d[OUTCOME].mean()
brain_sd = d[OUTCOME].std()

d["brain_z"] = (
    d[OUTCOME] - brain_mean
) / brain_sd

print("\nFrozen 143 scaling:")
print(f"Cardiac mean = {cardiac_mean:.8f}")
print(f"Cardiac SD   = {cardiac_sd:.8f}")
print(f"Baseline LA mean = {baseline_mean:.8f}")
print(f"Baseline LA SD   = {baseline_sd:.8f}")
print(f"Brain mean = {brain_mean:.8f}")
print(f"Brain SD   = {brain_sd:.8f}")

# =============================================================================
# LOAD SUBJECT-LEVEL ICV
# =============================================================================

raw_icv = pd.read_csv(
    ICV_FILE,
    sep="\t",
    dtype=str,
    low_memory=False,
    comment="#"
)

raw_icv.columns = [str(c).strip() for c in raw_icv.columns]

assert "shareid" in raw_icv.columns
assert "IntraCranialVol" in raw_icv.columns

icv = raw_icv[
    ["shareid", "IntraCranialVol"]
].copy()

icv["shareid"] = icv["shareid"].astype(str).str.strip()

icv["IntraCranialVol"] = pd.to_numeric(
    icv["IntraCranialVol"],
    errors="coerce"
)

icv = icv.dropna(subset=["IntraCranialVol"]).copy()

assert not icv["shareid"].duplicated().any()

# =============================================================================
# MERGE ICV — IDENTICAL SAMPLE FOR BOTH MODELS
# =============================================================================

dd = d.merge(
    icv,
    on="shareid",
    how="inner",
    validate="many_to_one"
)

# Retain subjects with >=2 observations after merge.
counts = dd.groupby("shareid")[OUTCOME].count()
good_ids = counts[counts >= 2].index
dd = dd[dd["shareid"].isin(good_ids)].copy()

print("\nICV-complete matched sample:")
print("Subjects:", dd["shareid"].nunique())
print("Observations:", len(dd))

print("\nAPOE4 subject counts:")
print(
    dd[
        ["shareid", "APOE4_carrier"]
    ]
    .drop_duplicates("shareid")
    ["APOE4_carrier"]
    .value_counts()
    .sort_index()
)

# We expect the overlap already audited.
assert dd["shareid"].nunique() == 795, (
    "STOP: expected 795 ICV-complete subjects."
)
assert len(dd) == 2405, (
    "STOP: expected 2405 observations."
)

# =============================================================================
# STANDARDIZE ICV AT SUBJECT LEVEL
# =============================================================================

subject_icv = (
    dd[
        ["shareid", "IntraCranialVol"]
    ]
    .drop_duplicates("shareid")
)

icv_mean = subject_icv["IntraCranialVol"].mean()
icv_sd = subject_icv["IntraCranialVol"].std()

dd["ICV_z"] = (
    dd["IntraCranialVol"] - icv_mean
) / icv_sd

print("\nICV scaling:")
print(f"Mean = {icv_mean:.4f}")
print(f"SD   = {icv_sd:.4f}")

# =============================================================================
# MODEL A — MATCHED SAMPLE, NO ICV
# =============================================================================

FORMULA_A = (
    "brain_z ~ "
    "mri_time_years * cardiac_z * APOE4_carrier "
    "+ baseline_LA_z "
    "+ baseline_LA_z:mri_time_years "
    "+ age6_c + C(sex) "
    "+ age6_c:mri_time_years "
    "+ C(sex):mri_time_years"
)

# =============================================================================
# MODEL B — IDENTICAL SAMPLE + ICV + ICV × TIME
# =============================================================================

FORMULA_B = (
    "brain_z ~ "
    "mri_time_years * cardiac_z * APOE4_carrier "
    "+ baseline_LA_z "
    "+ baseline_LA_z:mri_time_years "
    "+ age6_c + C(sex) "
    "+ age6_c:mri_time_years "
    "+ C(sex):mri_time_years "
    "+ ICV_z"
)

def fit_model(formula):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        model = smf.mixedlm(
            formula,
            data=dd,
            groups=dd["shareid"],
            re_formula="1"
        )

        return model.fit(
            reml=False,
            method="lbfgs",
            maxiter=2000,
            disp=False
        )

print("\nFitting Model A...")
fit_A = fit_model(FORMULA_A)

print("Fitting Model B...")
fit_B = fit_model(FORMULA_B)

# =============================================================================
# EXTRACT PRIMARY THREE-WAY
# =============================================================================

TERM = "mri_time_years:cardiac_z:APOE4_carrier"

def extract(fit, label):
    beta = fit.params[TERM]
    se = fit.bse[TERM]
    lo, hi = fit.conf_int().loc[TERM]
    p = fit.pvalues[TERM]

    return {
        "model": label,
        "beta": beta,
        "SE": se,
        "CI_low": lo,
        "CI_high": hi,
        "P": p,
        "N_subjects": dd["shareid"].nunique(),
        "N_obs": len(dd),
        "converged": fit.converged
    }

A = extract(
    fit_A,
    "Matched ICV sample — no ICV covariate"
)

B = extract(
    fit_B,
    "Matched ICV sample — ICV + ICV*time"
)

res = pd.DataFrame([A, B])

beta_A = A["beta"]
beta_B = B["beta"]

pct_magnitude_change = (
    (abs(beta_B) - abs(beta_A))
    / abs(beta_A)
    * 100
)

pct_relative_difference = (
    abs(beta_B - beta_A)
    / abs(beta_A)
    * 100
)

# =============================================================================
# RESULTS
# =============================================================================

print("\n" + "=" * 100)
print("PRIMARY ICV SENSITIVITY RESULT")
print("=" * 100)

print(
    res.to_string(
        index=False,
        float_format=lambda x: f"{x:.8g}"
    )
)

print("\nMatched comparison:")
print(f"No ICV beta       = {beta_A:+.8f}")
print(f"ICV-adjusted beta = {beta_B:+.8f}")
print(
    f"Absolute relative beta difference = "
    f"{pct_relative_difference:.2f}%"
)
print(
    f"Change in |beta| = "
    f"{pct_magnitude_change:+.2f}%"
)

print("\nICV terms:")
for term in [
    "ICV_z",
    "ICV_z:mri_time_years"
]:
    if term in fit_B.params.index:
        print(
            f"{term}: "
            f"beta={fit_B.params[term]:+.8f}, "
            f"P={fit_B.pvalues[term]:.8g}"
        )

# =============================================================================
# SAVE
# =============================================================================

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
        + "=" * 80
        + "\n\n"
    )

    f.write(
        f"Original 143 sample: "
        f"1305 subjects / 3787 observations\n"
    )

    f.write(
        f"ICV-complete sample: "
        f"{dd['shareid'].nunique()} subjects / "
        f"{len(dd)} observations\n\n"
    )

    f.write(
        "MODEL A — MATCHED SAMPLE WITHOUT ICV\n"
    )
    f.write(fit_A.summary().as_text())

    f.write(
        "\n\nMODEL B — SAME SAMPLE + ICV + ICV×TIME\n"
    )
    f.write(fit_B.summary().as_text())

    f.write(
        "\n\nPRIMARY COMPARISON\n"
    )
    f.write(res.to_string(index=False))

    f.write(
        f"\n\nNo ICV beta = {beta_A:+.8f}\n"
    )
    f.write(
        f"ICV-adjusted beta = {beta_B:+.8f}\n"
    )
    f.write(
        f"Absolute relative beta difference = "
        f"{pct_relative_difference:.2f}%\n"
    )
    f.write(
        f"Change in |beta| = "
        f"{pct_magnitude_change:+.2f}%\n"
    )

print("\nSaved to:")
print(OUTDIR)

print("\n" + "=" * 100)
print("146 COMPLETE")
print("=" * 100)
