#!/usr/bin/env python3

from pathlib import Path
import gzip
import re

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests


ROOT = Path("/data/qiallab/Framingham")

LONG_FILE = (
    ROOT
    / "data/longitudinal_derived/"
      "E4_E6_repeated_MRI_temporal_analysis.tsv"
)

CLINICAL = ROOT / "downloads/clinical"

OUTDIR = (
    ROOT
    / "results/longitudinal_heart_brain/"
      "plasma_biomarker_prediction"
)

OUTDIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def read_dbgap(path):
    return pd.read_csv(
        path,
        sep="\t",
        compression="gzip",
        low_memory=False,
        comment="#"
    )


def find_one(pattern):
    hits = sorted(CLINICAL.glob(pattern))
    if not hits:
        raise FileNotFoundError(
            f"No file matching {pattern} under {CLINICAL}"
        )
    print(f"{pattern} -> {hits[0]}")
    return hits[0]


def resolve_col(df, candidates):
    lookup = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lookup:
            return lookup[cand.lower()]
    return None


def zscore(x):
    x = pd.to_numeric(x, errors="coerce")
    return (x - x.mean()) / x.std(ddof=0)


# ============================================================
# LOAD LONGITUDINAL CARDIAC / APOE DATA
# ============================================================

d = pd.read_csv(
    LONG_FILE,
    sep="\t",
    low_memory=False
)

# One row per subject for predictor/covariate merge
subject = (
    d.sort_values("mri_date")
     .drop_duplicates("shareid")
     .copy()
)

required_long = [
    "shareid",
    "la_dim_n",
    "la_dim_slope",
    "APOE4_carrier",
    "sex",
    "age6",
    "last_echo_date",
]

missing = [
    c for c in required_long
    if c not in subject.columns
]

if missing:
    raise RuntimeError(
        f"Missing longitudinal columns: {missing}"
    )

# Require complete 3-exam LA trajectory
subject = subject[
    (subject["la_dim_n"] == 3)
    & subject["la_dim_slope"].notna()
    & subject["APOE4_carrier"].notna()
].copy()

subject["LA_slope_z"] = zscore(
    subject["la_dim_slope"]
)

subject["APOE4_carrier"] = (
    pd.to_numeric(
        subject["APOE4_carrier"],
        errors="coerce"
    )
)

subject["last_echo_date"] = pd.to_datetime(
    subject["last_echo_date"],
    errors="coerce"
)

print("\nLongitudinal cardiac subjects:")
print(subject["shareid"].nunique())


# ============================================================
# BIOMARKER FILES
# ============================================================

amyloid_file = find_one("*pht003309*.HMB-IRB-MDS.txt.gz")
ptau_file = find_one("*pht015135*.HMB-IRB-MDS.txt.gz")
glial_file = find_one("*pht012879*.HMB-IRB-MDS.txt.gz")

amy = read_dbgap(amyloid_file)
ptau = read_dbgap(ptau_file)
glial = read_dbgap(glial_file)

print("\nAmyloid columns:")
print(amy.columns.tolist())

print("\npTau columns:")
print(ptau.columns.tolist())

print("\nGFAP/NfL columns:")
print(glial.columns.tolist())


# ============================================================
# RESOLVE IDs AND BIOMARKERS
# ============================================================

for name, df in [
    ("amyloid", amy),
    ("ptau", ptau),
    ("glial", glial),
]:
    sid = resolve_col(
        df,
        ["shareid", "SHAREID"]
    )
    if sid is None:
        raise RuntimeError(
            f"Could not find shareid in {name}"
        )
    if sid != "shareid":
        df.rename(columns={sid: "shareid"}, inplace=True)

a40 = resolve_col(
    amy,
    ["amyloid40", "ABETA40", "abeta40"]
)

a42 = resolve_col(
    amy,
    ["amyloid42", "ABETA42", "abeta42"]
)

ptau_col = resolve_col(
    ptau,
    ["pTau_181", "ptau_181", "PTAU181"]
)

gfap_col = resolve_col(
    glial,
    ["gfap", "GFAP"]
)

nfl_col = resolve_col(
    glial,
    ["nf_l", "NfL", "NFL"]
)

for label, col in [
    ("amyloid40", a40),
    ("amyloid42", a42),
    ("pTau181", ptau_col),
    ("GFAP", gfap_col),
    ("NfL", nfl_col),
]:
    if col is None:
        raise RuntimeError(
            f"Could not resolve {label} column"
        )

amy["Abeta42_40"] = (
    pd.to_numeric(amy[a42], errors="coerce")
    /
    pd.to_numeric(amy[a40], errors="coerce")
)

ptau["pTau181"] = pd.to_numeric(
    ptau[ptau_col],
    errors="coerce"
)

glial["GFAP"] = pd.to_numeric(
    glial[gfap_col],
    errors="coerce"
)

glial["NfL"] = pd.to_numeric(
    glial[nfl_col],
    errors="coerce"
)


# ============================================================
# TRY TO RECOVER BIOMARKER DATES
# ============================================================

def candidate_date_cols(df):
    return [
        c for c in df.columns
        if (
            "date" in c.lower()
            or "exam" in c.lower()
            or "visit" in c.lower()
        )
    ]

print("\nCandidate amyloid timing columns:")
print(candidate_date_cols(amy))

print("\nCandidate pTau timing columns:")
print(candidate_date_cols(ptau))

print("\nCandidate GFAP/NfL timing columns:")
print(candidate_date_cols(glial))


# ============================================================
# REDUCE TO SUBJECT LEVEL
#
# For now, if duplicated subjects exist, retain first nonmissing
# biomarker row. We will tighten temporal ordering once actual
# biomarker-date columns are identified.
# ============================================================

amy_s = (
    amy.loc[
        amy["Abeta42_40"].notna(),
        ["shareid", "Abeta42_40"]
    ]
    .drop_duplicates("shareid")
)

ptau_s = (
    ptau.loc[
        ptau["pTau181"].notna(),
        ["shareid", "pTau181"]
    ]
    .drop_duplicates("shareid")
)

glial_s = (
    glial.loc[
        glial["GFAP"].notna()
        | glial["NfL"].notna(),
        ["shareid", "GFAP", "NfL"]
    ]
    .drop_duplicates("shareid")
)

m = (
    subject
    .merge(amy_s, on="shareid", how="left")
    .merge(ptau_s, on="shareid", how="left")
    .merge(glial_s, on="shareid", how="left")
)


# ============================================================
# COVARIATES
#
# Reuse early Exam 4 vascular covariates if present in the
# longitudinal file. Otherwise fit minimal models only.
# ============================================================

covariate_candidates = {
    "BMI4": ["BMI4", "bmi4"],
    "SBP4": ["SBP4", "sbp4"],
    "smoking4": ["smoking4", "smoker4"],
    "diabetes4": ["diabetes4", "CURR_DIAB4"],
}

resolved_covars = {}

for outname, candidates in covariate_candidates.items():
    for c in candidates:
        if c in m.columns:
            resolved_covars[outname] = c
            break

print("\nResolved early covariates:")
print(resolved_covars)


# ============================================================
# TRANSFORM BIOMARKERS
# ============================================================

m["Abeta42_40_z"] = zscore(m["Abeta42_40"])

for raw, out in [
    ("pTau181", "log_pTau181"),
    ("GFAP", "log_GFAP"),
    ("NfL", "log_NFL"),
]:
    x = pd.to_numeric(
        m[raw],
        errors="coerce"
    )

    m[out] = np.where(
        x > 0,
        np.log(x),
        np.nan
    )

    m[out + "_z"] = zscore(m[out])


# ============================================================
# MODELS
# ============================================================

outcomes = {
    "Aβ42/40": "Abeta42_40_z",
    "p-tau181": "log_pTau181_z",
    "GFAP": "log_GFAP_z",
    "NfL": "log_NFL_z",
}

rows = []

for biomarker_name, outcome in outcomes.items():

    base_cols = [
        outcome,
        "LA_slope_z",
        "APOE4_carrier",
        "age6",
        "sex",
    ]

    dx = m[
        base_cols
    ].dropna().copy()

    if len(dx) < 50:
        print(
            f"Skipping {biomarker_name}: "
            f"N={len(dx)}"
        )
        continue

    formula = (
        f"{outcome} ~ "
        "LA_slope_z * APOE4_carrier "
        "+ age6 + C(sex)"
    )

    res = smf.ols(
        formula,
        data=dx
    ).fit(
        cov_type="HC3"
    )

    term = "LA_slope_z:APOE4_carrier"

    rows.append({
        "biomarker": biomarker_name,
        "model": "minimal",
        "N": len(dx),
        "N_APOE4_noncarrier":
            int((dx["APOE4_carrier"] == 0).sum()),
        "N_APOE4_carrier":
            int((dx["APOE4_carrier"] == 1).sum()),
        "interaction_beta":
            float(res.params[term]),
        "interaction_SE":
            float(res.bse[term]),
        "interaction_CI_low":
            float(res.conf_int().loc[term, 0]),
        "interaction_CI_high":
            float(res.conf_int().loc[term, 1]),
        "interaction_P":
            float(res.pvalues[term]),
    })


results = pd.DataFrame(rows)

if not results.empty:

    _, q, _, _ = multipletests(
        results["interaction_P"],
        method="fdr_bh"
    )

    results["interaction_q_FDR4"] = q

results.to_csv(
    OUTDIR / "LA_E4E6_APOE4_plasma_biomarkers.tsv",
    sep="\t",
    index=False
)

m.to_csv(
    OUTDIR / "LA_E4E6_biomarker_merged_analysis_data.tsv",
    sep="\t",
    index=False
)

print("\n" + "=" * 90)
print("LONGITUDINAL CARDIAC → AD BIOMARKER RESULTS")
print("=" * 90)

if results.empty:
    print("No models completed.")
else:
    print(
        results.to_string(
            index=False
        )
    )

print("\nSaved:")
print(
    OUTDIR
    / "LA_E4E6_APOE4_plasma_biomarkers.tsv"
)
print(
    OUTDIR
    / "LA_E4E6_biomarker_merged_analysis_data.tsv"
)
