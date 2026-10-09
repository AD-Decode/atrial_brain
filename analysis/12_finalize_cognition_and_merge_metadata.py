#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/data/qiallab/Framingham")
RES  = ROOT / "results"

DOM = RES / "cognitive_domains_nearest_mri.csv"
COG = RES / "neurocardiac_cognitive_outcomes.csv"

# We may or may not already have the final master metadata file.
MASTER_CANDIDATES = [
    RES / "neurocardiac_metadata.csv",
    RES / "brain_cardiac_overlap.csv"
]

OUT_MASTER = RES / "neurocardiac_metadata_with_cognition.csv"
OUT_QC     = RES / "cognition_diagnosis_overlap_QC.txt"

domains = pd.read_csv(DOM)
cog = pd.read_csv(COG)

def _nid(s):   # same normalization as norm_id() below: IDs as text, no trailing ".0"
    s = s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    return s.replace({"nan": np.nan, "None": np.nan, "": np.nan})
domains["join_id"] = _nid(domains["join_id"])
cog["join_id"] = _nid(cog["join_id"])

# ------------------------------------------------------------
# 1. Rebuild primary global cognition using the 5 domains
#    with good coverage; do NOT require visuospatial.
# ------------------------------------------------------------

primary_domains = [
    "verbal_memory_z",
    "visual_memory_z",
    "attention_working_memory_z",
    "executive_speed_z",
    "language_z"
]

n_primary = domains[primary_domains].notna().sum(axis=1)

domains["global_cognition5_z"] = (
    domains[primary_domains]
    .mean(axis=1)
    .where(n_primary >= 3)
)

domains["n_primary_cognitive_domains"] = n_primary

# keep visuospatial separately
domains.to_csv(DOM, index=False)

# ------------------------------------------------------------
# 2. Update cognitive outcomes table
# ------------------------------------------------------------

add_cols = [
    "join_id",
    "global_cognition5_z",
    "n_primary_cognitive_domains"
]

for c in add_cols:
    if c != "join_id" and c in cog.columns:
        cog = cog.drop(columns=c)

cog = cog.merge(
    domains[add_cols],
    on="join_id",
    how="left",
    validate="one_to_one"
)

cog.to_csv(COG, index=False)

# ------------------------------------------------------------
# 3. Choose current master source
# ------------------------------------------------------------

master_file = next(
    (p for p in MASTER_CANDIDATES if p.exists()),
    None
)

if master_file is None:
    raise RuntimeError("No master metadata/overlap file found.")

master = pd.read_csv(master_file)

def norm_id(s):
    s = s.astype(str).str.strip()
    s = s.str.replace(r"\.0$", "", regex=True)
    return s.replace({"nan": np.nan, "None": np.nan, "": np.nan})

if "shareid" in master.columns:
    master["join_id"] = norm_id(master["shareid"])
elif "join_id" in master.columns:
    master["join_id"] = norm_id(master["join_id"])
else:
    raise RuntimeError("No shareid/join_id in master.")

# ------------------------------------------------------------
# 4. Compact cognitive variables for main metadata
# ------------------------------------------------------------

compact = [
    "join_id",
    "np_date_days_from_exam1",
    "np_minus_mri_days",
    "verbal_memory_z",
    "visual_memory_z",
    "memory_global_z",
    "attention_working_memory_z",
    "executive_speed_z",
    "language_z",
    "visuospatial_z",
    "global_cognition5_z",
    "n_primary_cognitive_domains"
]

compact = [c for c in compact if c in domains.columns]

# diagnosis/event variables already built
dxcols = [
    "join_id",
    "dx_ever_adjudicated",
    "DEMRV103",
    "DEMRV115",
    "DEMRV116",
    "dementia_subtype_label",
    "mci_ever",
    "dementia_ever",
    "ad_dementia_ever",
    "vascular_dementia_ever",
    "normal_date_days_from_exam1",
    "impairment_date_days_from_exam1",
    "mild_dementia_date_days_from_exam1",
    "earliest_dementia_date_days_from_exam1"
]

dxcols = [c for c in dxcols if c in cog.columns]

summary = domains[compact].merge(
    cog[dxcols],
    on="join_id",
    how="outer",
    validate="one_to_one"
)

# remove old versions on rerun
for c in summary.columns:
    if c != "join_id" and c in master.columns:
        master = master.drop(columns=c)

master = master.merge(
    summary,
    on="join_id",
    how="left",
    validate="one_to_one"
)

master.to_csv(OUT_MASTER, index=False)

# ------------------------------------------------------------
# 5. QC: cognition by eventual diagnosis
# ------------------------------------------------------------

with OUT_QC.open("w") as f:

    def p(*x):
        print(*x, file=f)

    p("COGNITION / DIAGNOSIS OVERLAP")
    p("============================")
    p()

    p(f"Master N: {len(master)}")
    p(f"With global cognition: {master['global_cognition5_z'].notna().sum()}")
    p(f"With adjudicated endpoint diagnosis: {master['dx_ever_adjudicated'].notna().sum()}")
    p(
        "With BOTH cognition and diagnosis: "
        f"{master[['global_cognition5_z','dx_ever_adjudicated']].notna().all(axis=1).sum()}"
    )
    p()

    p("Global cognition by eventual diagnosis:")
    tab = (
        master.dropna(
            subset=["global_cognition5_z","dx_ever_adjudicated"]
        )
        .groupby("dx_ever_adjudicated")["global_cognition5_z"]
        .agg(["count","mean","std","median"])
    )
    p(tab.to_string())
    p()

    p("Memory by eventual diagnosis:")
    tab = (
        master.dropna(
            subset=["memory_global_z","dx_ever_adjudicated"]
        )
        .groupby("dx_ever_adjudicated")["memory_global_z"]
        .agg(["count","mean","std","median"])
    )
    p(tab.to_string())
    p()

    p("Endpoint diagnosis among subjects with cognition:")
    p(
        master.loc[
            master["global_cognition5_z"].notna(),
            "dx_ever_adjudicated"
        ]
        .value_counts(dropna=False)
        .to_string()
    )

print("Created:")
print(OUT_MASTER)
print(OUT_QC)
