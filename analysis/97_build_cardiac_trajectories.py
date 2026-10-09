#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("/data/qiallab/Framingham")
DL = ROOT / "downloads/longitudinal_20260924"
HARM = DL / "harmonized"
OUT = DL / "longitudinal_analysis"
OUT.mkdir(exist_ok=True)

ECHO_LONG = HARM / "echo_harmonized_long.tsv"


# ============================================================
# Helpers
# ============================================================

def getfile(pht):
    hits = sorted(DL.glob(f"*{pht}*.HMB-IRB-MDS.txt.gz"))
    if not hits:
        raise FileNotFoundError(f"Could not find {pht}")
    return hits[0]


def read_pht(pht):
    d = pd.read_csv(
        getfile(pht),
        sep="\t",
        dtype=str,
        low_memory=False,
        comment="#"
    )
    d.columns = [c.strip() for c in d.columns]
    if "shareid" in d.columns:
        d["shareid"] = d["shareid"].astype(str).str.strip()
    return d


def slope_stats(g, variable):
    """
    OLS slope per year using actual examination dates.
    Requires >=2 nonmissing observations.
    """
    q = g[["time_years", variable]].dropna().copy()

    if len(q) < 2:
        return pd.Series({
            f"{variable}_n": len(q),
            f"{variable}_slope_per_year": np.nan,
            f"{variable}_baseline": np.nan,
            f"{variable}_last": np.nan,
            f"{variable}_followup_years": np.nan,
            f"{variable}_delta": np.nan,
        })

    q = q.sort_values("time_years")

    x = q["time_years"].to_numpy(float)
    y = q[variable].to_numpy(float)

    # Need distinct dates
    if np.ptp(x) <= 0:
        slope = np.nan
    else:
        slope = np.polyfit(x, y, 1)[0]

    return pd.Series({
        f"{variable}_n": len(q),
        f"{variable}_slope_per_year": slope,
        f"{variable}_baseline": y[0],
        f"{variable}_last": y[-1],
        f"{variable}_followup_years": x[-1] - x[0],
        f"{variable}_delta": y[-1] - y[0],
    })


# ============================================================
# 1. Read harmonized echo file
# ============================================================

echo = pd.read_csv(
    ECHO_LONG,
    sep="\t",
    dtype={"shareid": str}
)

echo["shareid"] = echo["shareid"].astype(str).str.strip()
echo["exam"] = pd.to_numeric(echo["exam"], errors="coerce").astype("Int64")


# ============================================================
# 2. Read Framingham exam-date backbone
# ============================================================

dates = read_pht("pht003099")

keep = [
    "shareid",
    "idtype",
    "sex",
    "date4",
    "date5",
    "date6",
    "date8",
    "age4",
    "age5",
    "age6",
    "age8",
]

keep = [c for c in keep if c in dates.columns]
dates = dates[keep].copy()

for c in ["date4", "date5", "date6", "date8",
          "age4", "age5", "age6", "age8"]:
    if c in dates.columns:
        dates[c] = pd.to_numeric(dates[c], errors="coerce")


# ============================================================
# 3. Convert wide dates to long format
# ============================================================

date_rows = []

for examno in [4, 5, 6, 8]:
    dc = f"date{examno}"
    ac = f"age{examno}"

    z = pd.DataFrame({
        "shareid": dates["shareid"],
        "exam": examno,
        "exam_date": dates[dc] if dc in dates.columns else np.nan,
        "exam_age": dates[ac] if ac in dates.columns else np.nan,
    })

    date_rows.append(z)

date_long = pd.concat(date_rows, ignore_index=True)

date_long = date_long.dropna(
    subset=["shareid", "exam_date"]
).copy()


# ============================================================
# 4. Merge exact exam dates into echo measurements
# ============================================================

echo = echo.merge(
    date_long,
    on=["shareid", "exam"],
    how="left",
    validate="many_to_one"
)

print("\n" + "=" * 90)
print("EXAM DATE MATCHING")
print("=" * 90)

for examno in [4, 5, 6, 8]:
    d = echo[echo["exam"] == examno]

    print(
        f"Exam {examno}: rows={len(d):4d}  "
        f"date matched={d['exam_date'].notna().sum():4d}  "
        f"missing date={d['exam_date'].isna().sum():4d}"
    )


# ============================================================
# 5. Time relative to each subject's first available echo
# ============================================================

echo["first_echo_date"] = (
    echo.groupby("shareid")["exam_date"]
        .transform("min")
)

echo["time_years"] = (
    echo["exam_date"] - echo["first_echo_date"]
) / 365.25


# ============================================================
# 6. Derive cardiac traits identically across all exams
# ============================================================

# ASE-style cube formula, matching the documented Exam 8 equation:
#
# LV mass = 0.8 * 1.04 *
#           [ (LVDD + IVSd + PWTd)^3 - LVDD^3 ] + 0.6
#
# Dimensions are in cm -> mass in grams.

echo["lv_mass_derived_g"] = (
    0.8 * 1.04 * (
        (echo["lvdd"] + echo["ivs_d"] + echo["pwt_d"]) ** 3
        - echo["lvdd"] ** 3
    ) + 0.6
)

# Relative wall thickness
echo["rwt"] = (
    2 * echo["pwt_d"] / echo["lvdd"]
)

# We already derived FS consistently in script 96/earlier harmonization.
# Retain it as fs_derived_pct.


# ============================================================
# 7. Basic QC of derived measures
# ============================================================

print("\n" + "=" * 90)
print("DERIVED CARDIAC PHENOTYPE DISTRIBUTIONS BY EXAM")
print("=" * 90)

derived = [
    "lv_mass_derived_g",
    "rwt",
    "fs_derived_pct",
    "la_dim",
    "lvdd",
]

qc_rows = []

for examno in [4, 5, 6, 8]:
    d = echo[echo["exam"] == examno]

    for v in derived:
        x = pd.to_numeric(d[v], errors="coerce").dropna()

        if len(x) == 0:
            continue

        row = {
            "exam": examno,
            "variable": v,
            "N": len(x),
            "mean": x.mean(),
            "sd": x.std(),
            "p01": x.quantile(.01),
            "p05": x.quantile(.05),
            "median": x.median(),
            "p95": x.quantile(.95),
            "p99": x.quantile(.99),
            "min": x.min(),
            "max": x.max(),
        }

        qc_rows.append(row)

qc = pd.DataFrame(qc_rows)

print(
    qc.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}"
    )
)


# ============================================================
# 8. Examination timing
# ============================================================

print("\n" + "=" * 90)
print("ACTUAL ECHO TIMING")
print("=" * 90)

for examno in [4, 5, 6, 8]:
    d = echo[echo["exam"] == examno]

    x = d["exam_date"].dropna()

    if len(x):
        print(
            f"Exam {examno}: N={len(x)}, "
            f"median relative day={x.median():.1f}, "
            f"range={x.min():.1f} to {x.max():.1f}"
        )

# Subject-level total cardiac follow-up
time_summary = (
    echo.dropna(subset=["exam_date"])
        .groupby("shareid")
        .agg(
            n_echo_dates=("exam_date", "nunique"),
            first_echo_date=("exam_date", "min"),
            last_echo_date=("exam_date", "max")
        )
        .reset_index()
)

time_summary["echo_followup_years"] = (
    time_summary["last_echo_date"]
    - time_summary["first_echo_date"]
) / 365.25

print("\nCARDIAC FOLLOW-UP AMONG SUBJECTS WITH >=2 DATES")

z = time_summary[
    time_summary["n_echo_dates"] >= 2
]

print(
    z["echo_followup_years"]
    .describe(
        percentiles=[.05, .10, .25, .50, .75, .90, .95]
    )
    .to_string()
)


# ============================================================
# 9. Calculate subject-specific trajectories
# ============================================================

TRAITS = [
    "lv_mass_derived_g",
    "la_dim",
    "lvdd",
    "fs_derived_pct",
    "rwt",

    # Secondary traits
    "lvds",
    "ivs_d",
    "pwt_d",
    "ao_root",
]

trajectory_parts = []

for trait in TRAITS:

    t = (
        echo.groupby("shareid", group_keys=False)
            .apply(
                lambda g: slope_stats(g, trait),
                include_groups=False
            )
            .reset_index()
    )

    trajectory_parts.append(t)

# Merge all trait-specific results
traj = trajectory_parts[0]

for t in trajectory_parts[1:]:
    traj = traj.merge(
        t,
        on="shareid",
        how="outer",
        validate="one_to_one"
    )

traj = traj.merge(
    time_summary,
    on="shareid",
    how="left",
    validate="one_to_one"
)


# ============================================================
# 10. Trajectory coverage
# ============================================================

print("\n" + "=" * 90)
print("CARDIAC TRAJECTORY COVERAGE")
print("=" * 90)

coverage_rows = []

for trait in TRAITS:
    nc = f"{trait}_n"
    sc = f"{trait}_slope_per_year"

    row = {
        "trait": trait,
        "N_any": int((traj[nc] >= 1).sum()),
        "N_ge2": int((traj[nc] >= 2).sum()),
        "N_ge3": int((traj[nc] >= 3).sum()),
        "N_ge4": int((traj[nc] >= 4).sum()),
        "N_slope": int(traj[sc].notna().sum()),
    }

    coverage_rows.append(row)
    print(row)

coverage = pd.DataFrame(coverage_rows)


# ============================================================
# 11. Distribution of slopes
# ============================================================

print("\n" + "=" * 90)
print("CARDIAC SLOPE DISTRIBUTIONS")
print("=" * 90)

slope_rows = []

for trait in TRAITS:
    sc = f"{trait}_slope_per_year"

    x = traj[sc].dropna()

    if len(x) == 0:
        continue

    row = {
        "trait": trait,
        "N": len(x),
        "mean": x.mean(),
        "sd": x.std(),
        "p01": x.quantile(.01),
        "p05": x.quantile(.05),
        "median": x.median(),
        "p95": x.quantile(.95),
        "p99": x.quantile(.99),
    }

    slope_rows.append(row)

slope_qc = pd.DataFrame(slope_rows)

print(
    slope_qc.to_string(
        index=False,
        float_format=lambda x: f"{x:.5f}"
    )
)


# ============================================================
# 12. Merge with FRAM4 longitudinal MRI availability
# ============================================================

fram4 = read_pht("pht015152")

fram4["mri_date"] = pd.to_numeric(
    fram4["mri_date"],
    errors="coerce"
)

# Count usable repeated structural MRI observations.
# Hippo/Total_brain coverage is essentially identical, but use both.
for v in ["Hippo", "Total_brain", "Lateralvent"]:
    if v in fram4.columns:
        fram4[v] = pd.to_numeric(
            fram4[v],
            errors="coerce"
        )

brain_subject = (
    fram4.groupby("shareid")
         .agg(
             n_mri_dates=("mri_date", "nunique"),
             first_mri_date=("mri_date", "min"),
             last_mri_date=("mri_date", "max"),
             n_hippo=("Hippo", lambda x: x.notna().sum()),
             n_totalbrain=("Total_brain", lambda x: x.notna().sum()),
             n_lateralvent=("Lateralvent", lambda x: x.notna().sum()),
         )
         .reset_index()
)

brain_subject["mri_followup_years"] = (
    brain_subject["last_mri_date"]
    - brain_subject["first_mri_date"]
) / 365.25

combined = traj.merge(
    brain_subject,
    on="shareid",
    how="inner",
    validate="one_to_one"
)


# ============================================================
# 13. Heart -> brain temporal relationship
# ============================================================

# Difference from LAST echo used in the cardiac trajectory
# to FIRST MRI.
combined["last_echo_to_first_mri_years"] = (
    combined["first_mri_date"]
    - combined["last_echo_date"]
) / 365.25

# Difference from first echo to first MRI
combined["first_echo_to_first_mri_years"] = (
    combined["first_mri_date"]
    - combined["first_echo_date"]
) / 365.25


# ============================================================
# 14. Exact analytic N for primary traits
# ============================================================

print("\n" + "=" * 90)
print("LONGITUDINAL CARDIAC -> BRAIN ANALYTIC OVERLAP")
print("=" * 90)

primary_traits = [
    "lv_mass_derived_g",
    "la_dim",
    "lvdd",
    "fs_derived_pct",
]

overlap_rows = []

for trait in primary_traits:
    nc = f"{trait}_n"
    sc = f"{trait}_slope_per_year"

    for n_echo in [2, 3, 4]:
        q = combined[
            (combined[nc] >= n_echo)
            & combined[sc].notna()
            & (combined["n_hippo"] >= 2)
        ]

        row = {
            "trait": trait,
            "min_echo_observations": n_echo,
            "brain_outcome": "Hippo",
            "min_mri_observations": 2,
            "N": len(q),
        }

        overlap_rows.append(row)

        print(
            f"{trait:20s} >= {n_echo} echo obs "
            f"+ >=2 longitudinal hippocampal MRI: {len(q)}"
        )

overlap = pd.DataFrame(overlap_rows)


# ============================================================
# 15. Restrict to cardiac history before MRI if desired
# ============================================================

print("\n" + "=" * 90)
print("TEMPORALLY ORDERED CARDIAC -> BRAIN SUBSETS")
print("=" * 90)

for trait in primary_traits:
    nc = f"{trait}_n"
    sc = f"{trait}_slope_per_year"

    q = combined[
        (combined[nc] >= 3)
        & combined[sc].notna()
        & (combined["n_hippo"] >= 2)
    ].copy()

    print(f"\n{trait}")

    print(
        "  >=3 echo + >=2 MRI:",
        len(q)
    )

    print(
        "  last echo <= first MRI:",
        int(
            (
                q["last_echo_to_first_mri_years"] >= 0
            ).sum()
        )
    )

    print(
        "  last echo >=2 years before first MRI:",
        int(
            (
                q["last_echo_to_first_mri_years"] >= 2
            ).sum()
        )
    )


# ============================================================
# 16. Save outputs
# ============================================================

echo.to_csv(
    OUT / "echo_harmonized_with_dates.tsv",
    sep="\t",
    index=False
)

traj.to_csv(
    OUT / "cardiac_subject_trajectories.tsv",
    sep="\t",
    index=False
)

coverage.to_csv(
    OUT / "cardiac_trajectory_coverage.tsv",
    sep="\t",
    index=False
)

qc.to_csv(
    OUT / "cardiac_derived_trait_qc.tsv",
    sep="\t",
    index=False
)

slope_qc.to_csv(
    OUT / "cardiac_slope_qc.tsv",
    sep="\t",
    index=False
)

brain_subject.to_csv(
    OUT / "fram4_subject_longitudinal_summary.tsv",
    sep="\t",
    index=False
)

combined.to_csv(
    OUT / "cardiac_brain_longitudinal_subjects.tsv",
    sep="\t",
    index=False
)

overlap.to_csv(
    OUT / "cardiac_brain_analytic_overlap.tsv",
    sep="\t",
    index=False
)

print("\n" + "=" * 90)
print("SAVED OUTPUTS")
print("=" * 90)

for f in sorted(OUT.glob("*.tsv")):
    print(f)
