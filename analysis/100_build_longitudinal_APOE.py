#!/usr/bin/env python3

from pathlib import Path
from io import BytesIO
import tarfile
import pandas as pd
import numpy as np

ROOT = Path("/data/qiallab/Framingham")
DL = ROOT / "downloads/longitudinal_20260924"
LONG = DL / "longitudinal_analysis"
OUT = LONG
OUT.mkdir(exist_ok=True)

# ============================================================
# 1. Locate full original APOE submission TAR
# ============================================================

hits = sorted(
    ROOT.rglob("*FHS_SHARe_APOE.original_submission_matrixfmt*.tar*")
)

if not hits:
    raise FileNotFoundError(
        "Could not find FHS_SHARe_APOE.original_submission_matrixfmt TAR"
    )

print("\nAPOE TAR candidates:")
for h in hits:
    print(" ", h)

APOE_TAR = hits[0]
print("\nUsing:", APOE_TAR)

# ============================================================
# 2. Read APOE CSV directly from TAR
# ============================================================

with tarfile.open(APOE_TAR, "r:*") as tar:
    members = [
        m for m in tar.getmembers()
        if m.isfile() and m.name.lower().endswith(".csv")
    ]

    if len(members) != 1:
        print("\nCSV members in TAR:")
        for m in members:
            print(" ", m.name)

    if not members:
        raise RuntimeError("No CSV found in APOE TAR")

    # Prefer a matrix-like CSV if multiple exist
    member = members[0]

    raw = tar.extractfile(member).read()

apoe = pd.read_csv(
    BytesIO(raw),
    low_memory=False
)

print("\n" + "="*90)
print("FULL APOE MATRIX")
print("="*90)
print("Shape:", apoe.shape)
print("Columns:", apoe.columns.tolist())

required = {"shareid", "APOE"}

if not required.issubset(apoe.columns):
    raise RuntimeError(
        f"Expected at least {required}; found {apoe.columns.tolist()}"
    )

# ============================================================
# 3. Normalize identifiers and genotype code
# ============================================================

def norm_id(x):
    x = x.astype(str).str.strip()
    x = x.str.replace(r"\.0$", "", regex=True)
    x = x.replace(
        ["", "nan", "NaN", "NA", "N/A", "."],
        np.nan
    )
    return x

apoe["shareid"] = norm_id(apoe["shareid"])

apoe["APOE"] = pd.to_numeric(
    apoe["APOE"],
    errors="coerce"
)

print("\nObserved APOE codes:")
print(
    apoe["APOE"]
    .value_counts(dropna=False)
    .sort_index()
    .to_string()
)

# Standard FHS coding used in our prior pipeline
GENOTYPE_MAP = {
    22: "E2/E2",
    23: "E2/E3",
    24: "E2/E4",
    33: "E3/E3",
    34: "E3/E4",
    44: "E4/E4",
}

observed = set(
    apoe["APOE"].dropna().astype(int).unique()
)

unexpected = observed - set(GENOTYPE_MAP)

print("\nUnexpected nonmissing APOE codes:", sorted(unexpected))

if unexpected:
    raise RuntimeError(
        f"Unexpected APOE codes found: {sorted(unexpected)}"
    )

# ============================================================
# 4. Check duplicate SHAREIDs
# ============================================================

dup = apoe[
    apoe["shareid"].notna()
    & apoe["shareid"].duplicated(keep=False)
].copy()

print("\nDuplicate SHAREID rows:", len(dup))
print(
    "Unique duplicated subjects:",
    dup["shareid"].nunique()
)

if len(dup):
    # Determine whether duplicates disagree in APOE genotype
    conflict = (
        dup.groupby("shareid")["APOE"]
        .nunique(dropna=True)
    )

    conflicts = conflict[conflict > 1]

    print(
        "Subjects with conflicting APOE codes:",
        len(conflicts)
    )

    if len(conflicts):
        print(conflicts.head(20).to_string())
        raise RuntimeError(
            "Conflicting APOE calls found for duplicate SHAREIDs."
        )

# Collapse exact/consistent duplicates
keep_cols = [
    c for c in ["shareid", "idtype", "APOE", "source"]
    if c in apoe.columns
]

apoe = (
    apoe[keep_cols]
    .dropna(subset=["shareid"])
    .sort_values("shareid")
    .drop_duplicates("shareid", keep="first")
    .copy()
)

# ============================================================
# 5. Derive APOE variables
# ============================================================

apoe["APOE_genotype"] = (
    apoe["APOE"]
    .map(GENOTYPE_MAP)
)

def e4_dose(code):
    if pd.isna(code):
        return np.nan
    code = int(code)

    return {
        22: 0,
        23: 0,
        24: 1,
        33: 0,
        34: 1,
        44: 2,
    }[code]

def e2_dose(code):
    if pd.isna(code):
        return np.nan
    code = int(code)

    return {
        22: 2,
        23: 1,
        24: 1,
        33: 0,
        34: 0,
        44: 0,
    }[code]

apoe["APOE4_dose"] = apoe["APOE"].apply(e4_dose)
apoe["APOE2_dose"] = apoe["APOE"].apply(e2_dose)

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

def risk_group(genotype):
    if pd.isna(genotype):
        return np.nan

    if genotype == "E3/E3":
        return "E3/E3"

    if genotype in ["E3/E4", "E4/E4"]:
        return "E4_carrier_no_E2"

    if genotype in ["E2/E2", "E2/E3"]:
        return "E2_carrier_no_E4"

    if genotype == "E2/E4":
        return "E2/E4"

    return np.nan

apoe["APOE_risk_group"] = (
    apoe["APOE_genotype"]
    .apply(risk_group)
)

# ============================================================
# 6. Full APOE coverage
# ============================================================

print("\n" + "="*90)
print("FULL APOE SUBJECT COVERAGE")
print("="*90)

print("Unique subjects:", apoe["shareid"].nunique())
print("APOE genotype available:", apoe["APOE_genotype"].notna().sum())

print("\nGenotypes:")
print(
    apoe["APOE_genotype"]
    .value_counts(dropna=False)
    .to_string()
)

print("\nAPOE4 carrier:")
print(
    apoe["APOE4_carrier"]
    .value_counts(dropna=False)
    .to_string()
)

# ============================================================
# 7. Load cardiac windows + MRI from script 98
# ============================================================

cw = pd.read_csv(
    LONG / "cardiac_window_trajectories_with_mri.tsv",
    sep="\t",
    dtype={"shareid": str}
)

cw["shareid"] = norm_id(cw["shareid"])

# Primary prospective E4-E6 cohort
e46 = cw[
    cw["window"] == "E4_E6"
].copy()

# Merge broad APOE
e46 = e46.merge(
    apoe,
    on="shareid",
    how="left",
    validate="many_to_one"
)

# ============================================================
# 8. APOE availability in primary prospective analyses
# ============================================================

TRAITS = [
    "lv_mass_derived_g",
    "la_dim",
    "lvdd",
    "fs_derived_pct",
]

print("\n" + "="*90)
print("APOE COVERAGE IN PROSPECTIVE E4-E6 -> MRI COHORT")
print("="*90)

rows = []

for trait in TRAITS:

    ncol = f"{trait}_n"
    scol = f"{trait}_slope"

    q = e46[
        (e46[ncol] >= 2)
        & e46[scol].notna()
        & (e46["n_hippo"] >= 2)
        & (e46["last_echo_to_first_mri_years"] >= 0)
    ].copy()

    q2 = q[
        q["last_echo_to_first_mri_years"] >= 2
    ].copy()

    for label, z in [
        ("prospective", q),
        ("prospective_gap_ge2y", q2),
    ]:

        n_apoe = int(
            z["APOE4_carrier"].notna().sum()
        )

        n_carrier = int(
            (z["APOE4_carrier"] == 1).sum()
        )

        n_noncarrier = int(
            (z["APOE4_carrier"] == 0).sum()
        )

        rows.append({
            "trait": trait,
            "subset": label,
            "N_total": len(z),
            "N_APOE_available": n_apoe,
            "N_APOE4_carrier": n_carrier,
            "N_APOE4_noncarrier": n_noncarrier,
            "APOE_coverage_pct":
                100 * n_apoe / len(z)
                if len(z) else np.nan
        })

        print(
            f"{trait:20s} {label:24s} "
            f"N={len(z):4d}  "
            f"APOE={n_apoe:4d} "
            f"({100*n_apoe/len(z):5.1f}%)  "
            f"E4+={n_carrier:4d}  "
            f"E4-={n_noncarrier:4d}"
        )

coverage = pd.DataFrame(rows)

# ============================================================
# 9. Sensitivity: all three E4/E5/E6 measurements
# ============================================================

print("\n" + "="*90)
print("APOE COVERAGE WITH ALL 3 ECHO OBSERVATIONS")
print("="*90)

for trait in TRAITS:

    ncol = f"{trait}_n"
    scol = f"{trait}_slope"

    q = e46[
        (e46[ncol] >= 3)
        & e46[scol].notna()
        & (e46["n_hippo"] >= 2)
        & (e46["last_echo_to_first_mri_years"] >= 0)
        & e46["APOE4_carrier"].notna()
    ]

    print(
        f"{trait:20s} "
        f"N={len(q):4d}  "
        f"E4+={(q['APOE4_carrier']==1).sum():4d}  "
        f"E4-={(q['APOE4_carrier']==0).sum():4d}"
    )

# ============================================================
# 10. Save
# ============================================================

apoe.to_csv(
    OUT / "APOE_full_offspring_subjects.tsv",
    sep="\t",
    index=False
)

e46.to_csv(
    OUT / "E4_E6_cardiac_MRI_with_APOE.tsv",
    sep="\t",
    index=False
)

coverage.to_csv(
    OUT / "E4_E6_APOE_coverage.tsv",
    sep="\t",
    index=False
)

print("\nSaved:")
print(OUT / "APOE_full_offspring_subjects.tsv")
print(OUT / "E4_E6_cardiac_MRI_with_APOE.tsv")
print(OUT / "E4_E6_APOE_coverage.tsv")
