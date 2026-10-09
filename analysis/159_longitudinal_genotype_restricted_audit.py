#!/usr/bin/env python3

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import norm

ROOT = Path("/data/qiallab/Framingham")

DATA = (
    ROOT
    / "data"
    / "longitudinal_derived"
    / "E4_E6_repeated_MRI_temporal_analysis.tsv"
)

OUTDIR = (
    ROOT
    / "results"
    / "longitudinal_heart_brain"
    / "temporal_prediction"
    / "genotype_restricted_audit"
)

OUTDIR.mkdir(parents=True, exist_ok=True)

OUTCSV = OUTDIR / "restricted_E3E3_E3E4_E4E4_slopes.csv"
OUTMODEL = OUTDIR / "restricted_E3E3_E3E4_E4E4_model_terms.csv"

print("=" * 115)
print("159: LONGITUDINAL GENOTYPE AUDIT")
print("Exact prospective LA-remodeling -> lateral ventricular model")
print("Restricted to E3/E3, E3/E4, E4/E4")
print("=" * 115)

long = pd.read_csv(
    DATA,
    sep="\t",
    low_memory=False,
)

print("Loaded:", DATA)
print("Shape :", long.shape)

# ============================================================
# EXACT PRIMARY VARIABLES
# ============================================================

SUBJ = "shareid"
OUTCOME = "Lateralvent"

CARDIAC_BASE = "la_dim"
NCOL = "la_dim_n"
SCOL = "la_dim_slope"

TIME = "mri_time_years"
AGE = "age6"
AGEC = "age6_c"
SEX = "sex"

APOE4 = "APOE4_carrier"
GENO = "APOE_genotype"

GAP = "last_echo_to_first_mri_years"

required = [
    SUBJ,
    OUTCOME,
    NCOL,
    SCOL,
    TIME,
    AGE,
    AGEC,
    SEX,
    APOE4,
    GENO,
    GAP,
]

for c in required:
    if c not in long.columns:
        raise RuntimeError(f"Missing required variable: {c}")

# ============================================================
# EXACT PRIMARY PROSPECTIVE SUBSET FROM SCRIPT 104
# ============================================================

d = long.copy()

d = d[
    (pd.to_numeric(d[NCOL], errors="coerce") >= 3)
    & d[SCOL].notna()
    & d[APOE4].notna()
    & d[AGE].notna()
    & d[SEX].notna()
    & (
        pd.to_numeric(
            d[GAP],
            errors="coerce"
        ) >= 0.0
    )
    & d[OUTCOME].notna()
    & d[TIME].notna()
].copy()

# Need repeated MRI outcome
counts = (
    d.groupby(SUBJ)[OUTCOME]
    .count()
)

good_ids = counts[
    counts >= 2
].index

d = d[
    d[SUBJ].isin(good_ids)
].copy()

nsub = d[SUBJ].nunique()
nobs = len(d)

print("\nPRIMARY PROSPECTIVE SAMPLE")
print("Subjects    :", nsub)
print("Observations:", nobs)

# ============================================================
# EXACT STANDARDIZATION FROM SCRIPT 104
# ============================================================

# subject-level cardiac slope mean/SD
cs = (
    d[
        [SUBJ, SCOL]
    ]
    .drop_duplicates(SUBJ)
)

cmean = cs[SCOL].mean()
csd = cs[SCOL].std()

d["cardiac_z"] = (
    d[SCOL] - cmean
) / csd

# outcome standardized across observations
om = d[OUTCOME].mean()
osd = d[OUTCOME].std()

d["brain_z"] = (
    d[OUTCOME] - om
) / osd

# ============================================================
# REPRODUCE PRIMARY APOE4-CARRIER MODEL FIRST
# ============================================================

formula_primary = (
    "brain_z ~ "
    "mri_time_years * cardiac_z * APOE4_carrier "
    "+ age6_c + C(sex)"
)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    model = smf.mixedlm(
        formula_primary,
        data=d,
        groups=d[SUBJ],
        re_formula="1",
    )

    fit = model.fit(
        reml=False,
        method="lbfgs",
        maxiter=2000,
        disp=False,
    )

term = (
    "mri_time_years:"
    "cardiac_z:"
    "APOE4_carrier"
)

if term not in fit.params.index:
    matches = [
        x for x in fit.params.index
        if (
            "mri_time_years" in x
            and "cardiac_z" in x
            and "APOE4_carrier" in x
        )
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"Cannot identify primary interaction term: {matches}"
        )

    term = matches[0]

beta_primary = float(
    fit.params[term]
)

se_primary = float(
    fit.bse[term]
)

p_primary = float(
    fit.pvalues[term]
)

print("\nPRIMARY CARRIER REPRODUCTION")
print(
    f"beta = {beta_primary:.9f}"
)
print(
    f"SE   = {se_primary:.9f}"
)
print(
    f"P    = {p_primary:.9g}"
)

# benchmark
if nsub != 1305:
    raise RuntimeError(
        f"Benchmark failed: expected 1305 subjects, got {nsub}"
    )

if nobs != 3787:
    raise RuntimeError(
        f"Benchmark failed: expected 3787 observations, got {nobs}"
    )

if abs(
    beta_primary - (-0.00858)
) > 0.001:
    raise RuntimeError(
        f"Benchmark failed: beta={beta_primary:.6f}"
    )

print("\nPRIMARY BENCHMARK PASSED.")

# ============================================================
# NORMALIZE GENOTYPE
# ============================================================

def clean_genotype(x):

    if pd.isna(x):
        return np.nan

    s = (
        str(x)
        .upper()
        .strip()
        .replace(" ", "")
        .replace("*", "")
        .replace("\\", "/")
        .replace("-", "/")
        .replace("_", "/")
    )

    mapping = {
        "E3/E3": "E3/E3",
        "3/3": "E3/E3",
        "33": "E3/E3",

        "E3/E4": "E3/E4",
        "E4/E3": "E3/E4",
        "3/4": "E3/E4",
        "4/3": "E3/E4",
        "34": "E3/E4",
        "43": "E3/E4",

        "E4/E4": "E4/E4",
        "4/4": "E4/E4",
        "44": "E4/E4",
    }

    return mapping.get(s, np.nan)


d["geno3"] = (
    d[GENO]
    .apply(clean_genotype)
)

allowed = [
    "E3/E3",
    "E3/E4",
    "E4/E4",
]

g = d[
    d["geno3"].isin(
        allowed
    )
].copy()

g["geno3"] = pd.Categorical(
    g["geno3"],
    categories=allowed,
)

# ============================================================
# RE-STANDARDIZE IN RESTRICTED THREE-GENOTYPE SAMPLE
# ============================================================

cs = (
    g[
        [SUBJ, SCOL]
    ]
    .drop_duplicates(SUBJ)
)

cmean = cs[SCOL].mean()
csd = cs[SCOL].std()

g["cardiac_z"] = (
    g[SCOL] - cmean
) / csd

om = g[OUTCOME].mean()
osd = g[OUTCOME].std()

g["brain_z"] = (
    g[OUTCOME] - om
) / osd

# ============================================================
# COUNTS
# ============================================================

subject_geno = (
    g[
        [SUBJ, "geno3"]
    ]
    .drop_duplicates(SUBJ)
)

counts = (
    subject_geno["geno3"]
    .value_counts()
    .reindex(allowed)
)

print("\n" + "=" * 115)
print("RESTRICTED THREE-GENOTYPE SAMPLE")
print("=" * 115)

print(
    "Subjects    :",
    g[SUBJ].nunique(),
)

print(
    "Observations:",
    len(g),
)

print("\nSubjects by genotype:")
print(counts)

# ============================================================
# CATEGORICAL GENOTYPE MODEL
# ============================================================

formula = (
    "brain_z ~ "
    "mri_time_years * cardiac_z * "
    "C(geno3, Treatment(reference='E3/E3')) "
    "+ age6_c + C(sex)"
)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    model_g = smf.mixedlm(
        formula,
        data=g,
        groups=g[SUBJ],
        re_formula="1",
    )

    try:

        fit_g = model_g.fit(
            reml=False,
            method="lbfgs",
            maxiter=2000,
            disp=False,
        )

    except Exception:

        fit_g = model_g.fit(
            reml=False,
            method="powell",
            maxiter=4000,
            disp=False,
        )

print(
    "\nConverged:",
    getattr(
        fit_g,
        "converged",
        np.nan,
    )
)

# ============================================================
# IDENTIFY TERMS
# ============================================================

base = "mri_time_years:cardiac_z"

if base not in fit_g.params.index:
    raise RuntimeError(
        "Could not identify base time × cardiac term."
    )


def find_threeway(level):

    matches = [
        x for x in fit_g.params.index
        if (
            "mri_time_years" in x
            and "cardiac_z" in x
            and f"[T.{level}]" in x
        )
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"Cannot identify three-way term for {level}: {matches}"
        )

    return matches[0]


t34 = find_threeway(
    "E3/E4"
)

t44 = find_threeway(
    "E4/E4"
)

cov = fit_g.cov_params()

# ============================================================
# LINEAR COMBINATIONS = GENOTYPE-SPECIFIC CARDIAC × TIME SLOPES
# ============================================================

def lincomb(terms, weights):

    beta = sum(
        w * fit_g.params[t]
        for t, w in zip(
            terms,
            weights,
        )
    )

    var = 0.0

    for ti, wi in zip(
        terms,
        weights,
    ):
        for tj, wj in zip(
            terms,
            weights,
        ):
            var += (
                wi
                * wj
                * cov.loc[
                    ti,
                    tj,
                ]
            )

    se = np.sqrt(var)

    z = beta / se

    p = 2 * norm.sf(
        abs(z)
    )

    return {
        "beta": float(beta),
        "SE": float(se),
        "CI_low": float(
            beta - 1.96 * se
        ),
        "CI_high": float(
            beta + 1.96 * se
        ),
        "P": float(p),
    }


rows = []

r = lincomb(
    [base],
    [1],
)
r["genotype"] = "E3/E3"
r["N_subjects"] = int(
    counts["E3/E3"]
)
rows.append(r)

r = lincomb(
    [base, t34],
    [1, 1],
)
r["genotype"] = "E3/E4"
r["N_subjects"] = int(
    counts["E3/E4"]
)
rows.append(r)

r = lincomb(
    [base, t44],
    [1, 1],
)
r["genotype"] = "E4/E4"
r["N_subjects"] = int(
    counts["E4/E4"]
)
rows.append(r)

slopes = pd.DataFrame(
    rows
)[
    [
        "genotype",
        "N_subjects",
        "beta",
        "SE",
        "CI_low",
        "CI_high",
        "P",
    ]
]

# ============================================================
# 2-df JOINT TEST OF GENOTYPE MODIFICATION
# ============================================================

names = list(
    fit_g.params.index
)

R = np.zeros(
    (
        2,
        len(names),
    )
)

R[
    0,
    names.index(t34)
] = 1

R[
    1,
    names.index(t44)
] = 1

wald = fit_g.wald_test(
    R,
    scalar=True,
)

joint_chi2 = float(
    wald.statistic
)

joint_p = float(
    wald.pvalue
)

# ============================================================
# DESCRIPTIVE MONOTONICITY
# ============================================================

b33 = slopes.loc[
    slopes.genotype == "E3/E3",
    "beta",
].iloc[0]

b34 = slopes.loc[
    slopes.genotype == "E3/E4",
    "beta",
].iloc[0]

b44 = slopes.loc[
    slopes.genotype == "E4/E4",
    "beta",
].iloc[0]

monotonic = bool(
    (b33 > b34 > b44)
    or
    (b33 < b34 < b44)
)

print("\n" + "=" * 115)
print("GENOTYPE-SPECIFIC LA × MRI-TIME SLOPES")
print("=" * 115)

print(
    slopes.to_string(
        index=False,
        float_format=lambda x: f"{x:.8g}",
    )
)

print(
    "\nJoint genotype modification:"
)

print(
    f"Wald chi-square = {joint_chi2:.8g}"
)

print(
    f"P = {joint_p:.8g}"
)

print(
    "\nMonotonic E3/E3 -> E3/E4 -> E4/E4:",
    monotonic,
)

# ============================================================
# SAVE
# ============================================================

slopes[
    "joint_Wald_chi2"
] = joint_chi2

slopes[
    "joint_Wald_P"
] = joint_p

slopes[
    "monotonic"
] = monotonic

slopes[
    "N_total_subjects"
] = g[SUBJ].nunique()

slopes[
    "N_total_observations"
] = len(g)

slopes.to_csv(
    OUTCSV,
    index=False,
)

model_terms = pd.DataFrame({
    "term":
        fit_g.params.index,

    "beta":
        fit_g.params.values,

    "SE":
        fit_g.bse.values,

    "P":
        fit_g.pvalues.values,
})

model_terms.to_csv(
    OUTMODEL,
    index=False,
)

print("\nSaved:")
print(OUTCSV)
print(OUTMODEL)
