#!/usr/bin/env python3

from pathlib import Path
from io import BytesIO
import tarfile
import numpy as np
import pandas as pd

ROOT = Path("/data/qiallab/Framingham")
RAW  = ROOT / "downloads" / "clinical"
RES  = ROOT / "results"

MASTER = RES / "neurocardiac_metadata_with_cognition.csv"
OUT    = RES / "neurocardiac_metadata_with_cognition_APOE.csv"
QC     = RES / "APOE_merge_QC.txt"

APOE_TAR = sorted(
    RAW.glob("*FHS_SHARe_APOE.original_submission_matrixfmt*.tar")
)[0]


def norm_id(s):
    s = s.astype(str).str.strip()
    s = s.str.replace(r"\.0$", "", regex=True)
    return s.replace({
        "nan": np.nan,
        "None": np.nan,
        "": np.nan
    })


# ============================================================
# 1. Load master metadata
# ============================================================

master = pd.read_csv(MASTER)

if "shareid" not in master.columns:
    raise RuntimeError("shareid not found in master metadata.")

master["shareid_norm"] = norm_id(master["shareid"])

print("Master N:", len(master))
print("Master unique shareid:", master["shareid_norm"].nunique())


# ============================================================
# 2. Read APOE matrix directly from tar
# ============================================================

with tarfile.open(APOE_TAR, "r:*") as tar:

    members = [
        m for m in tar.getmembers()
        if m.isfile() and m.name.lower().endswith(".csv")
    ]

    if len(members) != 1:
        raise RuntimeError(
            f"Expected one APOE CSV in TAR; found {len(members)}"
        )

    raw = tar.extractfile(members[0]).read()

apoe = pd.read_csv(BytesIO(raw), low_memory=False)

needed = {"shareid", "idtype", "APOE", "source"}

if not needed.issubset(apoe.columns):
    raise RuntimeError(
        f"Missing expected APOE columns. Found: {apoe.columns.tolist()}"
    )

apoe["shareid_norm"] = norm_id(apoe["shareid"])

apoe["APOE"] = pd.to_numeric(
    apoe["APOE"],
    errors="coerce"
).astype("Int64")


# ============================================================
# 3. QC genotype values and duplicates
# ============================================================

valid_genotypes = {22, 23, 24, 33, 34, 44}

observed = set(
    apoe["APOE"]
    .dropna()
    .astype(int)
    .unique()
)

unexpected = observed - valid_genotypes

print("\nObserved APOE genotypes:", sorted(observed))
print("Unexpected APOE values:", sorted(unexpected))

dup = apoe[
    apoe["shareid_norm"].duplicated(keep=False)
].sort_values("shareid_norm")

print("\nDuplicate shareid rows in APOE matrix:", len(dup))
print(
    "Subjects duplicated:",
    dup["shareid_norm"].nunique()
)

if len(dup):
    print(dup.head(30).to_string(index=False))

# one row per shareid expected
if apoe["shareid_norm"].duplicated().any():
    raise RuntimeError(
        "APOE matrix has duplicate shareid values. "
        "Do not merge until resolved."
    )


# ============================================================
# 4. Derive genotype variables
# ============================================================

def genotype_string(x):
    if pd.isna(x):
        return np.nan
    s = str(int(x))
    if len(s) != 2:
        return np.nan
    return f"E{s[0]}/E{s[1]}"


def allele_dose(x, allele):
    if pd.isna(x):
        return np.nan
    s = str(int(x))
    return s.count(str(allele))


apoe["APOE_genotype"] = apoe["APOE"].apply(
    genotype_string
)

apoe["APOE4_dose"] = apoe["APOE"].apply(
    lambda x: allele_dose(x, 4)
)

apoe["APOE2_dose"] = apoe["APOE"].apply(
    lambda x: allele_dose(x, 2)
)

apoe["APOE4_carrier"] = np.where(
    apoe["APOE4_dose"].notna(),
    (apoe["APOE4_dose"] > 0).astype(int),
    np.nan
)

apoe["APOE2_carrier"] = np.where(
    apoe["APOE2_dose"].notna(),
    (apoe["APOE2_dose"] > 0).astype(int),
    np.nan
)

# Useful mutually exclusive simplified grouping
def risk_group(x):
    if pd.isna(x):
        return np.nan

    x = int(x)

    if x == 33:
        return "E3/E3"
    elif x in [34, 44]:
        return "E4_carrier_no_E2"
    elif x in [22, 23]:
        return "E2_carrier_no_E4"
    elif x == 24:
        return "E2/E4"
    else:
        return np.nan


apoe["APOE_risk_group"] = apoe["APOE"].apply(
    risk_group
)


# ============================================================
# 5. Merge into 785-person metadata
# ============================================================

keep = [
    "shareid_norm",
    "idtype",
    "APOE",
    "APOE_genotype",
    "APOE4_carrier",
    "APOE4_dose",
    "APOE2_carrier",
    "APOE2_dose",
    "APOE_risk_group",
    "source"
]

apoe_small = apoe[keep].rename(
    columns={
        "idtype": "APOE_idtype",
        "APOE": "APOE_code",
        "source": "APOE_source"
    }
)

# remove old APOE columns if rerunning
for c in apoe_small.columns:
    if c != "shareid_norm" and c in master.columns:
        master = master.drop(columns=c)

merged = master.merge(
    apoe_small,
    on="shareid_norm",
    how="left",
    validate="one_to_one"
)

if len(merged) != 785:
    raise RuntimeError(
        f"Expected 785 rows after merge, got {len(merged)}"
    )

merged.to_csv(OUT, index=False)


# ============================================================
# 6. QC
# ============================================================

with QC.open("w") as f:

    def p(*x):
        print(*x, file=f)

    p("APOE MERGE QC")
    p("=============")
    p()

    p(f"Master N: {len(merged)}")
    p(
        "APOE available: "
        f"{merged['APOE_code'].notna().sum()}"
    )
    p(
        "APOE missing: "
        f"{merged['APOE_code'].isna().sum()}"
    )
    p()

    p("Genotype counts:")
    p(
        merged["APOE_genotype"]
        .value_counts(dropna=False)
        .to_string()
    )
    p()

    p("APOE4 carrier:")
    p(
        merged["APOE4_carrier"]
        .value_counts(dropna=False)
        .to_string()
    )
    p()

    p("APOE4 dose:")
    p(
        merged["APOE4_dose"]
        .value_counts(dropna=False)
        .sort_index()
        .to_string()
    )
    p()

    p("APOE2 carrier:")
    p(
        merged["APOE2_carrier"]
        .value_counts(dropna=False)
        .to_string()
    )
    p()

    p("APOE risk groups:")
    p(
        merged["APOE_risk_group"]
        .value_counts(dropna=False)
        .to_string()
    )
    p()

    p("APOE source:")
    p(
        merged["APOE_source"]
        .value_counts(dropna=False)
        .to_string()
    )

print("\nCreated:")
print(OUT)
print(QC)

print("\n" + QC.read_text())
