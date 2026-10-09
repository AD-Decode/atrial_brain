#!/usr/bin/env python3

import pandas as pd
from pathlib import Path
from io import StringIO
import gzip

ROOT = Path("/data/qiallab/Framingham")

SAMPLE_INFO = ROOT / "results/genetics_pathways/sample_metadata/TOPMed_c1_sample_info.txt"
WANTED = ROOT / "results/genetics_pathways/WGS_541_NWD_samples.txt"
PED = ROOT / "downloads/clinical/phs000007.v35.pht000183.v14.p16.Framingham_Pedigree.MULTI.txt.gz"

OUT = ROOT / "results/genetics_pathways/sample_metadata/TOPMed_541_metadata.tsv"

# ------------------------------------------------------------
# 1. TOPMed sample-info
# ------------------------------------------------------------

rows = []

with open(SAMPLE_INFO) as f:
    for line in f:
        p = line.strip().split()

        if len(p) < 10:
            continue

        rows.append({
            "topmed_family_id": p[0],
            "shareid": p[1],
            "topmed_mother_id": p[2],
            "topmed_father_id": p[3],
            "sample_id": p[4],
            "topmed_sex": p[5],
            "consent_code": p[6],
            "consent_abbrev": p[7],
            "data_type": p[-1],
            "consent_text": " ".join(p[8:-1]),
        })

topmed = pd.DataFrame(rows)

for c in [
    "topmed_family_id",
    "shareid",
    "topmed_mother_id",
    "topmed_father_id",
    "topmed_sex",
]:
    topmed[c] = pd.to_numeric(topmed[c], errors="coerce").astype("Int64")

wanted = set(WANTED.read_text().split())

topmed = topmed[topmed["sample_id"].isin(wanted)].copy()

print("TOPMed c1 rows parsed:", len(rows))
print("Requested WGS samples:", len(wanted))
print("Matched TOPMed samples:", len(topmed))
print("Unique samples:", topmed.sample_id.nunique())
print("Unique shareids:", topmed.shareid.nunique())

# ------------------------------------------------------------
# 2. Official FHS pedigree
# IMPORTANT: preserve empty TAB-delimited fields
# ------------------------------------------------------------

with gzip.open(PED, "rt") as f:
    lines = [
        line for line in f
        if not line.startswith("#") and line.strip()
    ]

ped = pd.read_csv(
    StringIO("".join(lines)),
    sep="\t",
    dtype=str,
    keep_default_na=True
)

# remove accidental empty columns if present
ped = ped.loc[:, ~ped.columns.str.match(r"^Unnamed")]

print("\nPedigree columns:")
print(ped.columns.tolist())

for c in [
    "dbGaP_Subject_ID",
    "pedno",
    "shareid",
    "fshare",
    "mshare",
    "sex",
    "twinid",
    "idtype",
]:
    if c in ped.columns:
        ped[c] = pd.to_numeric(
            ped[c].replace("", pd.NA),
            errors="coerce"
        ).astype("Int64")

print("Pedigree rows:", len(ped))
print("Unique pedigree shareids:", ped.shareid.nunique())

# ------------------------------------------------------------
# 3. Merge
# ------------------------------------------------------------

m = topmed.merge(
    ped,
    on="shareid",
    how="left",
    validate="one_to_one"
)

m["sex_label"] = m["sex"].map({1: "M", 2: "F"})
m["topmed_sex_label"] = m["topmed_sex"].map({1: "M", 2: "F"})

# ------------------------------------------------------------
# 4. Diagnostics
# ------------------------------------------------------------

print("\n===== FINAL 541 METADATA =====")
print("Rows:", len(m))
print("Unique NWD samples:", m.sample_id.nunique())
print("Unique shareids:", m.shareid.nunique())

print("\nOfficial pedigree sex:")
print(m["sex_label"].value_counts(dropna=False))

print("\nTOPMed sex:")
print(m["topmed_sex_label"].value_counts(dropna=False))

print("\nPedigree coverage:")
print("pedno present:", m["pedno"].notna().sum())
print("father present:", m["fshare"].notna().sum())
print("mother present:", m["mshare"].notna().sum())
print("sex present:", m["sex"].notna().sum())

print("\nUnique families:", m["pedno"].nunique(dropna=True))

family_sizes = (
    m.dropna(subset=["pedno"])
     .groupby("pedno")
     .size()
     .sort_values(ascending=False)
)

print("\nLargest family sizes:")
print(family_sizes.head(20).to_string())

fam_n = m["pedno"].map(family_sizes)

print(
    "\nParticipants in families with >1 WGS member:",
    (fam_n > 1).sum()
)

print(
    "Singleton pedigree families:",
    (fam_n == 1).sum()
)

sex_disagree = (
    m["topmed_sex"].notna()
    & m["sex"].notna()
    & (m["topmed_sex"] != m["sex"])
)

print("\nTOPMed vs pedigree sex disagreements:")
print(sex_disagree.sum())

print("\nMissing official pedigree row:")
print(m["dbGaP_Subject_ID"].isna().sum())

print("Missing pedno:")
print(m["pedno"].isna().sum())

print("Missing requested NWD samples:")
print(len(wanted - set(m.sample_id)))

# ------------------------------------------------------------
# 5. Save
# ------------------------------------------------------------

cols = [
    "sample_id",
    "shareid",
    "dbGaP_Subject_ID",
    "pedno",
    "fshare",
    "mshare",
    "sex",
    "sex_label",
    "twinid",
    "idtype",
    "topmed_sex",
    "topmed_sex_label",
    "topmed_mother_id",
    "topmed_father_id",
]

cols = [c for c in cols if c in m.columns]

m[cols].sort_values("sample_id").to_csv(
    OUT,
    sep="\t",
    index=False
)

print("\nSaved:")
print(OUT)
