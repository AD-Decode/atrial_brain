#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/data/qiallab/Framingham")

OUTDIR = (
    ROOT / "data" / "longitudinal_derived"
)
OUTDIR.mkdir(parents=True, exist_ok=True)

OUTFILE = OUTDIR / "attrition_atrisk_source.tsv"


def norm_id(s):
    s = s.astype(str).str.strip()
    s = s.str.replace(r"\.0$", "", regex=True)
    return s.replace(
        ["", "nan", "NaN", "NA", "N/A", "."],
        np.nan
    )


def numeric(s):
    return pd.to_numeric(s, errors="coerce")


def find_pht(pht):
    roots = [
        ROOT / "data/longitudinal_20260924",
        ROOT / "downloads",
    ]

    hits = []

    for r in roots:
        if r.exists():
            hits.extend(
                sorted(
                    r.glob(
                        f"**/*{pht}*.HMB-IRB-MDS.txt.gz"
                    )
                )
            )

    hits = [
        x for x in hits
        if "data_dict" not in x.name.lower()
        and "var_report" not in x.name.lower()
    ]

    return hits[0] if hits else None


def read_pht(pht):
    f = find_pht(pht)

    if f is None:
        raise FileNotFoundError(
            f"Could not find phenotype table {pht}"
        )

    print("READ", pht, ":", f)

    d = pd.read_csv(
        f,
        sep="\t",
        comment="#",
        dtype=str,
        low_memory=False,
    )

    d.columns = [x.strip() for x in d.columns]

    if "shareid" in d.columns:
        d["shareid"] = norm_id(d["shareid"])

    return d


print("=" * 100)
print("BUILD ATTRITION AT-RISK SOURCE TABLE")
print("=" * 100)


# =============================================================================
# 1. CARDIAC TRAJECTORIES
# =============================================================================

traj_file = (
    ROOT
    / "downloads/longitudinal_20260924/longitudinal_analysis"
    / "cardiac_subject_trajectories.tsv"
)

traj = pd.read_csv(
    traj_file,
    sep="\t",
    dtype={"shareid": str},
    low_memory=False,
)

traj["shareid"] = norm_id(traj["shareid"])

needed = [
    "shareid",
    "la_dim_n",
    "la_dim_slope_per_year",
    "la_dim_baseline",
    "first_echo_date",
    "last_echo_date",
]

missing = [x for x in needed if x not in traj.columns]

if missing:
    raise RuntimeError(
        f"Trajectory file missing: {missing}"
    )

traj = traj[needed].copy()

traj = traj[
    numeric(traj["la_dim_n"]) >= 3
].copy()

traj = traj[
    traj["la_dim_slope_per_year"].notna()
].copy()

traj = traj.rename(
    columns={
        "la_dim_slope_per_year": "la_dim_slope",
        "la_dim_baseline": "baseline_LA",
    }
)

print("Valid three-exam LA trajectories:", traj["shareid"].nunique())


# =============================================================================
# 2. APOE
# =============================================================================

apoe_file = (
    ROOT
    / "downloads/longitudinal_20260924/longitudinal_analysis"
    / "APOE_full_offspring_subjects.tsv"
)

apoe = pd.read_csv(
    apoe_file,
    sep="\t",
    dtype={"shareid": str},
    low_memory=False,
)

apoe["shareid"] = norm_id(apoe["shareid"])

apoe = (
    apoe[
        [
            "shareid",
            "APOE4_carrier",
            "APOE4_dose",
            "APOE_genotype",
        ]
    ]
    .drop_duplicates("shareid")
)

apoe["APOE4_carrier"] = numeric(
    apoe["APOE4_carrier"]
)

x = traj.merge(
    apoe,
    on="shareid",
    how="left",
    validate="one_to_one",
)

x = x[
    x["APOE4_carrier"].notna()
].copy()

print("LA trajectory + APOE classified:", x["shareid"].nunique())

if x["shareid"].nunique() != 2522:
    raise RuntimeError(
        "STOP: expected 2522 subjects with valid 3-exam LA trajectory + APOE "
        "before conditioning on MRI participation; "
        f"observed {x['shareid'].nunique()}."
    )


# =============================================================================
# 3. AGE, SEX, EXAM-6 DATE
# =============================================================================

dates = read_pht("pht003099")

for c in ["sex", "age6", "date6"]:
    if c not in dates.columns:
        raise RuntimeError(
            f"Required date-backbone variable missing: {c}"
        )

dates = dates[
    ["shareid", "sex", "age6", "date6"]
].copy()

dates["sex"] = numeric(dates["sex"])
dates["age6"] = numeric(dates["age6"])
dates["date6"] = numeric(dates["date6"])

dates = dates.drop_duplicates("shareid")

x = x.merge(
    dates,
    on="shareid",
    how="left",
    validate="one_to_one",
)


# =============================================================================
# 4. EXAM-4 BMI / SBP / SMOKING
# =============================================================================

e4 = read_pht("pht000033")

MAP = {
    "BMI4": "D443",
    "SBP4_1": "D192",
    "SBP4_2": "D288",
    "smoking4": "D093",
}

for label, col in MAP.items():
    if col not in e4.columns:
        raise RuntimeError(
            f"Exam-4 variable not found: {label} = {col}"
        )

cov = e4[
    [
        "shareid",
        MAP["BMI4"],
        MAP["SBP4_1"],
        MAP["SBP4_2"],
        MAP["smoking4"],
    ]
].copy()

cov["BMI4"] = numeric(
    cov[MAP["BMI4"]]
)

sbp1 = numeric(
    cov[MAP["SBP4_1"]]
)

sbp2 = numeric(
    cov[MAP["SBP4_2"]]
)

cov["SBP4"] = pd.concat(
    [sbp1, sbp2],
    axis=1
).mean(
    axis=1,
    skipna=True,
)

cov["smoking4"] = numeric(
    cov[MAP["smoking4"]]
)

cov = (
    cov[
        [
            "shareid",
            "BMI4",
            "SBP4",
            "smoking4",
        ]
    ]
    .drop_duplicates("shareid")
)


# =============================================================================
# 5. EXAM-4 DIABETES
# =============================================================================

db = read_pht("pht000041")

if "CURR_DIAB4" not in db.columns:
    raise RuntimeError(
        "CURR_DIAB4 not found in pht000041."
    )

db["diabetes4"] = numeric(
    db["CURR_DIAB4"]
)

db = (
    db[
        ["shareid", "diabetes4"]
    ]
    .drop_duplicates("shareid")
)

cov = cov.merge(
    db,
    on="shareid",
    how="left",
    validate="one_to_one",
)

x = x.merge(
    cov,
    on="shareid",
    how="left",
    validate="one_to_one",
)


# =============================================================================
# 6. MRI PARTICIPATION / EXACT PRIMARY OUTCOME ELIGIBILITY
# =============================================================================

mri_file = (
    ROOT
    / "data/longitudinal_derived"
    / "E4_E6_repeated_MRI_temporal_analysis.tsv"
)

mri = pd.read_csv(
    mri_file,
    sep="\t",
    dtype={"shareid": str},
    low_memory=False,
)

mri["shareid"] = norm_id(
    mri["shareid"]
)

for c in [
    "APOE4_carrier",
    "age6_c",
    "la_dim_slope",
    "mri_time_years",
    "Lateralvent",
    "mri_date",
]:
    if c not in mri.columns:
        raise RuntimeError(
            f"MRI analysis variable missing: {c}"
        )


# Exact primary-model eligible observations
eligible = mri.dropna(
    subset=[
        "shareid",
        "APOE4_carrier",
        "sex",
        "age6_c",
        "la_dim_slope",
        "mri_time_years",
        "Lateralvent",
    ]
).copy()

eligible = eligible[
    pd.to_numeric(
        eligible["la_dim_n"],
        errors="coerce"
    ) >= 3
].copy()

eligible_count = (
    eligible
    .groupby("shareid")
    .size()
    .rename("n_eligible_vent_mri")
)

# Any MRI date
mri["mri_date"] = numeric(
    mri["mri_date"]
)

first_mri = (
    mri.dropna(
        subset=["mri_date"]
    )
    .groupby("shareid")["mri_date"]
    .min()
    .rename("first_mri_date")
)

x = x.merge(
    eligible_count,
    on="shareid",
    how="left",
)

x = x.merge(
    first_mri,
    on="shareid",
    how="left",
)

x["n_eligible_vent_mri"] = (
    x["n_eligible_vent_mri"]
    .fillna(0)
    .astype(int)
)

x["included"] = (
    x["n_eligible_vent_mri"] >= 2
).astype(int)

print(
    "Exact primary included:",
    int(x["included"].sum())
)

if int(x["included"].sum()) != 1305:
    raise RuntimeError(
        "STOP: exact primary eligibility should give N=1305; "
        f"observed {int(x['included'].sum())}."
    )


# =============================================================================
# 7. SUMMARY ONLY — NO SUBJECT IDS PRINTED
# =============================================================================

print("\nAT-RISK N:", len(x))
print("Included:", int(x["included"].sum()))
print(
    "Exactly one eligible ventricle MRI:",
    int((x["n_eligible_vent_mri"] == 1).sum())
)
print(
    "Zero eligible ventricle MRI:",
    int((x["n_eligible_vent_mri"] == 0).sum())
)

print("\nCOVARIATE AVAILABILITY")
for c in [
    "age6",
    "sex",
    "BMI4",
    "SBP4",
    "diabetes4",
    "smoking4",
    "baseline_LA",
]:
    print(
        f"{c:15s}",
        int(x[c].notna().sum())
    )


# =============================================================================
# 8. WRITE LOCAL CONTROLLED SUBJECT-LEVEL TABLE
# =============================================================================

x.to_csv(
    OUTFILE,
    sep="\t",
    index=False,
)

print("\nWROTE:")
print(OUTFILE)
print("This file contains controlled participant-level data; do not share it.")

