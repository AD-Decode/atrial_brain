#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/data/qiallab/Framingham")
RAW  = ROOT / "downloads" / "clinical"
RES  = ROOT / "results"

OVERLAP = RES / "brain_cardiac_overlap.csv"
NPLONG  = RES / "neuropsych_sessions_long.csv"
COGOUT  = RES / "neurocardiac_cognitive_outcomes.csv"

OUTDOM  = RES / "cognitive_domains_nearest_mri.csv"
OUTQC   = RES / "cognitive_domains_QC.txt"


def read_dbgap(path):
    return pd.read_csv(
        path,
        sep="\t",
        compression="gzip",
        comment="#",
        low_memory=False
    )


def norm_id(s):
    s = s.astype(str).str.strip()
    s = s.str.replace(r"\.0$", "", regex=True)
    s = s.replace({
        "nan": np.nan,
        "": np.nan,
        "None": np.nan
    })
    return s


def zscore(s):
    s = pd.to_numeric(s, errors="coerce")
    sd = s.std(ddof=0)
    if pd.isna(sd) or sd == 0:
        return pd.Series(np.nan, index=s.index)
    return (s - s.mean()) / sd


# ============================================================
# 1. Load datasets
# ============================================================

overlap = pd.read_csv(OVERLAP)
np_scores = pd.read_csv(NPLONG)
cogout = pd.read_csv(COGOUT)

datefile = sorted(
    RAW.glob("*pht003910*.txt.gz")
)[0]

npdates = read_dbgap(datefile)

for df in [overlap, np_scores, cogout, npdates]:
    if "shareid" in df.columns:
        df["join_id"] = norm_id(df["shareid"])
    elif "join_id" in df.columns:
        df["join_id"] = norm_id(df["join_id"])


overlap["mri_date"] = pd.to_numeric(
    overlap["mri_date"],
    errors="coerce"
)

npdates["npdate"] = pd.to_numeric(
    npdates["npdate"],
    errors="coerce"
)


# ============================================================
# 2. Restrict NP dates to subjects with actual NP scores
# ============================================================

score_ids = set(np_scores["join_id"].dropna())

dates = npdates[
    npdates["join_id"].isin(score_ids)
].copy()

mri = overlap[
    ["join_id", "mri_date"]
].drop_duplicates("join_id")

dates = dates.merge(
    mri,
    on="join_id",
    how="left"
)

dates["np_minus_mri_days"] = (
    dates["npdate"] - dates["mri_date"]
)

dates["abs_np_mri_days"] = (
    dates["np_minus_mri_days"].abs()
)

nearest_date = (
    dates
    .sort_values(
        ["join_id", "abs_np_mri_days", "npdate"]
    )
    .drop_duplicates("join_id", keep="first")
    [
        [
            "join_id",
            "npdate",
            "mri_date",
            "np_minus_mri_days",
            "abs_np_mri_days"
        ]
    ]
)

nearest_date = nearest_date.rename(
    columns={
        "npdate": "np_date_days_from_exam1"
    }
)

nearest_date["years_np_to_mri"] = (
    nearest_date["np_minus_mri_days"] / 365.25
)


# ============================================================
# 3. Resolve multiple pht004374 score rows
#
# Only 9 subjects have >1 score row.
# For now preserve one score row per subject by choosing the
# row with the greatest number of nonmissing primary tests.
# ============================================================

PRIMARY_TESTS = [
    "LMi", "LMd", "LMr",
    "PASi", "PASd", "PASr",
    "VRi", "VRd", "VRr",
    "DSF", "DSB",
    "trailsA", "trailsB",
    "FAS_animal",
    "BD"
]

available_tests = [
    x for x in PRIMARY_TESTS
    if x in np_scores.columns
]

np_scores["n_primary_tests_available"] = (
    np_scores[available_tests]
    .notna()
    .sum(axis=1)
)

# Prefer most complete cognitive battery if multiple rows
np_one = (
    np_scores
    .sort_values(
        ["join_id", "n_primary_tests_available"],
        ascending=[True, False]
    )
    .drop_duplicates("join_id", keep="first")
    .copy()
)


# ============================================================
# 4. Attach nearest NP date
# ============================================================

dat = np_one.merge(
    nearest_date,
    on="join_id",
    how="left",
    validate="one_to_one"
)


# ============================================================
# 5. Clean numeric tests
# ============================================================

TESTS = [
    "LMi", "LMd", "LMr",
    "PASi", "PASd", "PASr",
    "VRi", "VRd", "VRr",
    "DSF", "DSB",
    "trailsA", "trailsB",
    "FAS_animal",
    "BD",
    "Incidental_free",
    "Math_correct"
]

for x in TESTS:
    if x in dat.columns:
        dat[x] = pd.to_numeric(
            dat[x],
            errors="coerce"
        )


# ============================================================
# 6. Standardize individual tests
#
# Trails are completion time:
# lower raw score = better performance,
# so reverse after z-scoring.
# ============================================================

HIGHER_BETTER = [
    "LMi", "LMd", "LMr",
    "PASi", "PASd", "PASr",
    "VRi", "VRd", "VRr",
    "DSF", "DSB",
    "FAS_animal",
    "BD"
]

LOWER_BETTER = [
    "trailsA",
    "trailsB"
]

for x in HIGHER_BETTER:
    if x in dat:
        dat[f"{x}_z"] = zscore(dat[x])

for x in LOWER_BETTER:
    if x in dat:
        dat[f"{x}_z"] = -zscore(dat[x])


# ============================================================
# 7. Domain composites
#
# Require at least half the domain's component measures.
# ============================================================

DOMAINS = {
    "verbal_memory_z": [
        "LMi_z", "LMd_z", "LMr_z",
        "PASi_z", "PASd_z", "PASr_z"
    ],

    "visual_memory_z": [
        "VRi_z", "VRd_z", "VRr_z"
    ],

    "attention_working_memory_z": [
        "DSF_z", "DSB_z"
    ],

    "executive_speed_z": [
        "trailsA_z", "trailsB_z"
    ],

    "language_z": [
        "FAS_animal_z"
    ],

    "visuospatial_z": [
        "BD_z"
    ]
}


for domain, vars_ in DOMAINS.items():

    vars_ = [
        x for x in vars_
        if x in dat.columns
    ]

    if not vars_:
        dat[domain] = np.nan
        continue

    n_req = int(np.ceil(len(vars_) / 2))

    n_avail = dat[vars_].notna().sum(axis=1)

    dat[domain] = (
        dat[vars_]
        .mean(axis=1)
        .where(n_avail >= n_req)
    )


# ============================================================
# 8. Overall memory and global cognition
# ============================================================

memory_domains = [
    "verbal_memory_z",
    "visual_memory_z"
]

dat["memory_global_z"] = (
    dat[memory_domains]
    .mean(axis=1)
)

global_domains = [
    "verbal_memory_z",
    "visual_memory_z",
    "attention_working_memory_z",
    "executive_speed_z",
    "language_z",
    "visuospatial_z"
]

n_domain_available = (
    dat[global_domains]
    .notna()
    .sum(axis=1)
)

# Require at least 3 cognitive domains
dat["global_cognition_z"] = (
    dat[global_domains]
    .mean(axis=1)
    .where(n_domain_available >= 3)
)

dat["n_cognitive_domains"] = (
    n_domain_available
)


# ============================================================
# 9. Save cognitive-domain table
# ============================================================

KEEP = [
    "join_id",
    "shareid",
    "sex",
    "age",
    "battery",
    "test_language",

    "np_date_days_from_exam1",
    "mri_date",
    "np_minus_mri_days",
    "abs_np_mri_days",
    "years_np_to_mri",

    "LMi", "LMd", "LMr",
    "PASi", "PASd", "PASr",
    "VRi", "VRd", "VRr",
    "DSF", "DSB",
    "trailsA", "trailsB",
    "FAS_animal",
    "BD",

    "verbal_memory_z",
    "visual_memory_z",
    "memory_global_z",
    "attention_working_memory_z",
    "executive_speed_z",
    "language_z",
    "visuospatial_z",
    "global_cognition_z",
    "n_cognitive_domains",
    "n_primary_tests_available"
]

KEEP = [
    x for x in KEEP
    if x in dat.columns
]

domains = dat[KEEP].copy()

domains.to_csv(
    OUTDOM,
    index=False
)


# ============================================================
# 10. Merge summary cognition columns into 785-row outcome file
# ============================================================

summary_cols = [
    "join_id",
    "np_date_days_from_exam1",
    "np_minus_mri_days",
    "abs_np_mri_days",
    "years_np_to_mri",
    "verbal_memory_z",
    "visual_memory_z",
    "memory_global_z",
    "attention_working_memory_z",
    "executive_speed_z",
    "language_z",
    "visuospatial_z",
    "global_cognition_z",
    "n_cognitive_domains"
]

summary_cols = [
    x for x in summary_cols
    if x in domains.columns
]

# remove old versions if rerunning
for col in summary_cols:
    if col != "join_id" and col in cogout.columns:
        cogout = cogout.drop(columns=col)

cogout = cogout.merge(
    domains[summary_cols],
    on="join_id",
    how="left",
    validate="one_to_one"
)

cogout.to_csv(
    COGOUT,
    index=False
)


# ============================================================
# 11. QC report
# ============================================================

with OUTQC.open("w") as f:

    def p(*args):
        print(*args, file=f)

    p("COGNITIVE DOMAIN QC")
    p("===================")
    p()

    p(f"Subjects with cognitive scores: {len(domains)}")
    p()

    p("NP-MRI timing:")
    p(
        domains["abs_np_mri_days"]
        .describe()
        .to_string()
    )

    p()

    for yr in [0.25, 0.5, 1, 2, 3]:
        n = (
            domains["abs_np_mri_days"]
            <= yr * 365.25
        ).sum()

        p(
            f"within_{yr}y: "
            f"{n}/{len(domains)}"
        )

    p()
    p("Domain availability:")

    for x in [
        "verbal_memory_z",
        "visual_memory_z",
        "memory_global_z",
        "attention_working_memory_z",
        "executive_speed_z",
        "language_z",
        "visuospatial_z",
        "global_cognition_z"
    ]:
        if x in domains.columns:
            p(
                f"{x}: "
                f"{domains[x].notna().sum()}"
            )

    p()
    p("Number of cognitive domains per subject:")
    p(
        domains["n_cognitive_domains"]
        .value_counts()
        .sort_index()
        .to_string()
    )


print("Created:")
print(OUTDOM)
print(COGOUT)
print(OUTQC)

print("\nDomain coverage:")
for x in [
    "verbal_memory_z",
    "visual_memory_z",
    "memory_global_z",
    "attention_working_memory_z",
    "executive_speed_z",
    "language_z",
    "visuospatial_z",
    "global_cognition_z"
]:
    if x in domains.columns:
        print(
            f"{x}: "
            f"{domains[x].notna().sum()}"
        )
