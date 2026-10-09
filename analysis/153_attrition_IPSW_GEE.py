#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

ROOT = Path("/data/qiallab/Framingham")

ATRISK_FILE = (
    ROOT / "data/longitudinal_derived"
    / "attrition_atrisk_source.tsv"
)

MRI_FILE = (
    ROOT / "data/longitudinal_derived"
    / "E4_E6_repeated_MRI_temporal_analysis.tsv"
)

OUT = (
    ROOT / "results/longitudinal_heart_brain"
    / "temporal_prediction/attrition_IPSW"
)
OUT.mkdir(parents=True, exist_ok=True)

print("=" * 100)
print("ATTRITION / IPSW SENSITIVITY")
print("=" * 100)


def numeric(s):
    return pd.to_numeric(s, errors="coerce")


# =============================================================================
# 1. LOAD AT-RISK COHORT
# =============================================================================

a = pd.read_csv(
    ATRISK_FILE,
    sep="\t",
    dtype={"shareid": str},
    low_memory=False
)

print("At-risk N:", len(a))
print("Included N:", int(a["included"].sum()))

if len(a) != 2522:
    raise RuntimeError(f"Expected at-risk N=2522, observed {len(a)}")

if int(a["included"].sum()) != 1305:
    raise RuntimeError(
        f"Expected included N=1305, observed {int(a['included'].sum())}"
    )


# =============================================================================
# 2. STANDARDIZE LA SLOPE IN FULL AT-RISK COHORT
# =============================================================================

a["la_dim_slope"] = numeric(a["la_dim_slope"])

mu = a["la_dim_slope"].mean()
sd = a["la_dim_slope"].std(ddof=1)

a["la_slope_std"] = (
    a["la_dim_slope"] - mu
) / sd

print("LA slope mean:", mu)
print("LA slope SD  :", sd)


# =============================================================================
# 3. COMPLETE-CASE SELECTION MODEL
# =============================================================================

selection_vars = [
    "included",
    "la_slope_std",
    "APOE4_carrier",
    "age6",
    "sex",
    "BMI4",
    "SBP4",
    "diabetes4",
    "smoking4",
    "baseline_LA",
]

for c in selection_vars:
    if c != "included":
        a[c] = numeric(a[c])

sel = a.dropna(
    subset=selection_vars
).copy()

print("\nSelection-model complete-case N:", len(sel))
print(" Included:", int(sel["included"].sum()))
print(" Excluded:", int((sel["included"] == 0).sum()))


# =============================================================================
# 4. FULL SELECTION MODEL
# =============================================================================

full_formula = (
    "included ~ "
    "la_slope_std * APOE4_carrier "
    "+ age6 + C(sex) "
    "+ BMI4 + SBP4 + diabetes4 + smoking4 "
    "+ baseline_LA"
)

full = smf.logit(
    full_formula,
    data=sel
).fit(disp=False)

print("\nFULL SELECTION MODEL")
print(full.summary())

term = "la_slope_std:APOE4_carrier"

b = full.params[term]
se = full.bse[term]
p = full.pvalues[term]

OR = np.exp(b)
lo = np.exp(b - 1.96 * se)
hi = np.exp(b + 1.96 * se)

print("\nLA slope x APOE4 selection interaction:")
print(f"log-odds beta = {b:.6f}")
print(f"OR = {OR:.4f}")
print(f"95% CI = {lo:.4f} to {hi:.4f}")
print(f"P = {p:.6g}")


# =============================================================================
# 5. STABILIZED IPS WEIGHTS
# numerator = P(included | APOE4)
# denominator = P(included | full covariates)
# =============================================================================

num = smf.logit(
    "included ~ APOE4_carrier",
    data=sel
).fit(disp=False)

sel["p_num"] = num.predict(sel)
sel["p_den"] = full.predict(sel)

# For included subjects, probability of observed inclusion status
sel["sw_raw"] = np.where(
    sel["included"] == 1,
    sel["p_num"] / sel["p_den"],
    (1 - sel["p_num"]) / (1 - sel["p_den"])
)

q01 = sel["sw_raw"].quantile(0.01)
q99 = sel["sw_raw"].quantile(0.99)

sel["sw_trunc"] = sel["sw_raw"].clip(
    lower=q01,
    upper=q99
)

print("\nWEIGHT DISTRIBUTION")
print("Raw:")
print(sel["sw_raw"].describe(
    percentiles=[.01, .05, .50, .95, .99]
))
print("\nTruncated at:")
print("1% =", q01)
print("99% =", q99)
print("\nTruncated:")
print(sel["sw_trunc"].describe(
    percentiles=[.01, .05, .50, .95, .99]
))


# =============================================================================
# 6. KEEP INCLUDED SUBJECTS WITH SELECTION-MODEL COVARIATES
# =============================================================================

w = sel.loc[
    sel["included"] == 1,
    ["shareid", "sw_raw", "sw_trunc"]
].copy()

print("\nIncluded subjects with usable IPS weights:", len(w))


# =============================================================================
# 7. LOAD REPEATED MRI OUTCOME DATA
# =============================================================================

long = pd.read_csv(
    MRI_FILE,
    sep="\t",
    dtype={"shareid": str},
    low_memory=False
)

for c in [
    "Lateralvent",
    "mri_time_years",
    "la_dim_slope",
    "la_dim_n",
    "APOE4_carrier",
    "age6_c",
]:
    long[c] = numeric(long[c])

# exact primary analysis rows
long = long[
    (long["la_dim_n"] >= 3)
    & long["Lateralvent"].notna()
    & long["mri_time_years"].notna()
    & long["la_dim_slope"].notna()
    & long["APOE4_carrier"].notna()
    & long["age6_c"].notna()
    & long["sex"].notna()
].copy()

counts = long.groupby("shareid")["Lateralvent"].count()
good = counts[counts >= 2].index

long = long[
    long["shareid"].isin(good)
].copy()

print("\nPrimary repeated-MRI subjects:", long["shareid"].nunique())
print("Primary repeated-MRI observations:", len(long))

if long["shareid"].nunique() != 1305:
    raise RuntimeError("Primary MRI cohort does not reproduce N=1305")

if len(long) != 3787:
    raise RuntimeError(
        f"Primary MRI observations should be 3787; observed {len(long)}"
    )


# =============================================================================
# 8. STANDARDIZE EXACTLY WITHIN PRIMARY ANALYSIS DATA
# =============================================================================

long["cardiac_z"] = (
    long["la_dim_slope"]
    - long["la_dim_slope"].mean()
) / long["la_dim_slope"].std(ddof=1)

long["brain_z"] = (
    long["Lateralvent"]
    - long["Lateralvent"].mean()
) / long["Lateralvent"].std(ddof=1)


# =============================================================================
# 9. MERGE IPS WEIGHTS
# =============================================================================

longw = long.merge(
    w,
    on="shareid",
    how="inner",
    validate="many_to_one"
)

print("\nIPSW-complete MRI subjects:", longw["shareid"].nunique())
print("IPSW-complete MRI observations:", len(longw))


# =============================================================================
# 10. LIKE-FOR-LIKE UNWEIGHTED GEE
# =============================================================================

formula = (
    "brain_z ~ "
    "mri_time_years * cardiac_z * APOE4_carrier "
    "+ age6_c + C(sex)"
)

gee_unw = smf.gee(
    formula=formula,
    groups="shareid",
    data=longw,
    family=sm.families.Gaussian(),
    cov_struct=sm.cov_struct.Exchangeable()
).fit()

threeway = "mri_time_years:cardiac_z:APOE4_carrier"

print("\nUNWEIGHTED GEE — IPSW-COMPLETE SAMPLE")
print("beta =", gee_unw.params[threeway])
print("SE   =", gee_unw.bse[threeway])
print("P    =", gee_unw.pvalues[threeway])


# =============================================================================
# 11. WEIGHTED GEE
# =============================================================================

gee_w = smf.gee(
    formula=formula,
    groups="shareid",
    data=longw,
    family=sm.families.Gaussian(),
    cov_struct=sm.cov_struct.Exchangeable(),
    weights=longw["sw_trunc"]
).fit()

print("\nWEIGHTED GEE — TRUNCATED IPSW")
print("beta =", gee_w.params[threeway])
print("SE   =", gee_w.bse[threeway])
print("P    =", gee_w.pvalues[threeway])

ci = gee_w.conf_int().loc[threeway]

print("95% CI =", ci.iloc[0], "to", ci.iloc[1])


# =============================================================================
# 12. SAVE AGGREGATE RESULTS
# =============================================================================

results = pd.DataFrame([
    {
        "model": "Primary mixed model",
        "N_subjects": 1305,
        "N_observations": 3787,
        "beta_3way": -0.00858,
        "SE": np.nan,
        "CI_low": -0.01300,
        "CI_high": -0.00415,
        "P": 1.45e-4,
    },
    {
        "model": "Unweighted GEE, IPSW-complete sample",
        "N_subjects": longw["shareid"].nunique(),
        "N_observations": len(longw),
        "beta_3way": gee_unw.params[threeway],
        "SE": gee_unw.bse[threeway],
        "CI_low": gee_unw.conf_int().loc[threeway].iloc[0],
        "CI_high": gee_unw.conf_int().loc[threeway].iloc[1],
        "P": gee_unw.pvalues[threeway],
    },
    {
        "model": "IPSW-weighted GEE, weights truncated 1/99%",
        "N_subjects": longw["shareid"].nunique(),
        "N_observations": len(longw),
        "beta_3way": gee_w.params[threeway],
        "SE": gee_w.bse[threeway],
        "CI_low": gee_w.conf_int().loc[threeway].iloc[0],
        "CI_high": gee_w.conf_int().loc[threeway].iloc[1],
        "P": gee_w.pvalues[threeway],
    },
])

results.to_csv(
    OUT / "IPSW_GEE_effect_comparison.tsv",
    sep="\t",
    index=False
)

selection_result = pd.DataFrame([{
    "N_at_risk": len(a),
    "N_selection_complete_case": len(sel),
    "N_included_complete_case": int(sel["included"].sum()),
    "LAxAPOE_logOR": b,
    "LAxAPOE_OR": OR,
    "CI_low": lo,
    "CI_high": hi,
    "P": p,
    "weight_p01": q01,
    "weight_p99": q99,
}])

selection_result.to_csv(
    OUT / "selection_model_summary.tsv",
    sep="\t",
    index=False
)

print("\n" + "=" * 100)
print("FINAL COMPARISON")
print("=" * 100)
print(results.to_string(index=False))

print("\nSaved:")
print(OUT / "IPSW_GEE_effect_comparison.tsv")
print(OUT / "selection_model_summary.tsv")
