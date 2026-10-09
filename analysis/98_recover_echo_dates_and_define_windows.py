#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("/data/qiallab/Framingham")
DL = ROOT / "downloads/longitudinal_20260924"
HARM = DL / "harmonized"
LONG = DL / "longitudinal_analysis"
OUT = LONG
OUT.mkdir(exist_ok=True)

def getfile(pht):
    return next(DL.glob(f"*{pht}*.HMB-IRB-MDS.txt.gz"))

def read_pht(pht):
    d = pd.read_csv(
        getfile(pht),
        sep="\t",
        dtype=str,
        low_memory=False,
        comment="#"
    )
    d.columns = [c.strip() for c in d.columns]
    d["shareid"] = d["shareid"].astype(str).str.strip()
    return d

# ============================================================
# 1. Load harmonized echo file with dates from script 97
# ============================================================

echo = pd.read_csv(
    LONG / "echo_harmonized_with_dates.tsv",
    sep="\t",
    dtype={"shareid": str}
)

echo["shareid"] = echo["shareid"].astype(str).str.strip()
echo["exam"] = pd.to_numeric(echo["exam"], errors="coerce")

# ============================================================
# 2. Read raw Exam 6 and Exam 8 echo date fields
# ============================================================

e6 = read_pht("pht000150")
e8 = read_pht("pht002572")

e6_dates = e6[["shareid", "SDATE"]].copy()
e8_dates = e8[["shareid", "sdate"]].copy()

e6_dates["echo_raw_date"] = pd.to_numeric(
    e6_dates["SDATE"], errors="coerce"
)
e8_dates["echo_raw_date"] = pd.to_numeric(
    e8_dates["sdate"], errors="coerce"
)

e6_dates = e6_dates[["shareid", "echo_raw_date"]]
e8_dates = e8_dates[["shareid", "echo_raw_date"]]

# ============================================================
# 3. Compare visit-date table vs raw echo dates
# ============================================================

print("\n" + "="*90)
print("RAW ECHO DATE RECOVERY")
print("="*90)

for examno, raw in [(6, e6_dates), (8, e8_dates)]:

    q = echo[echo["exam"] == examno][
        ["shareid", "exam_date"]
    ].merge(
        raw,
        on="shareid",
        how="left"
    )

    both = q.dropna(
        subset=["exam_date", "echo_raw_date"]
    ).copy()

    both["difference"] = (
        both["echo_raw_date"] - both["exam_date"]
    )

    print(f"\nExam {examno}")
    print("rows:", len(q))
    print("visit-table dates:", q["exam_date"].notna().sum())
    print("raw echo dates:", q["echo_raw_date"].notna().sum())
    print("both available:", len(both))

    if len(both):
        print("\nDifference raw - visit table:")
        print(
            both["difference"]
            .describe(
                percentiles=[.01,.05,.25,.50,.75,.95,.99]
            )
            .to_string()
        )

        print("\nMost common differences:")
        print(
            both["difference"]
            .value_counts()
            .head(10)
            .to_string()
        )

# ============================================================
# 4. Fill missing Exam 6 / Exam 8 dates IF raw dates align
#    with visit-date time scale.
#    We infer alignment based on median difference.
# ============================================================

echo["exam_date_recovered"] = echo["exam_date"]
echo["date_source"] = np.where(
    echo["exam_date"].notna(),
    "visit_table",
    "missing"
)

for examno, raw in [(6, e6_dates), (8, e8_dates)]:

    q = echo[echo["exam"] == examno][
        ["shareid", "exam_date"]
    ].merge(
        raw,
        on="shareid",
        how="left"
    )

    both = q.dropna(
        subset=["exam_date", "echo_raw_date"]
    ).copy()

    if len(both) == 0:
        print(f"\nCannot validate raw dates for Exam {examno}")
        continue

    median_diff = np.median(
        both["echo_raw_date"] - both["exam_date"]
    )

    # check whether dates are on same scale / nearly identical
    close = (
        np.abs(
            (both["echo_raw_date"] - both["exam_date"])
            - median_diff
        ) <= 7
    ).mean()

    print(
        f"\nExam {examno}: median offset={median_diff:.1f} days, "
        f"fraction within ±7 days of offset={close:.3f}"
    )

    if close < 0.95:
        print(
            f"WARNING: Exam {examno} raw dates do not align "
            "cleanly enough; not filling missing values."
        )
        continue

    raw2 = raw.copy()
    raw2["converted_raw_date"] = (
        raw2["echo_raw_date"] - median_diff
    )

    mask_exam = echo["exam"] == examno

    tmp = echo.loc[
        mask_exam,
        ["shareid", "exam_date_recovered"]
    ].merge(
        raw2[["shareid", "converted_raw_date"]],
        on="shareid",
        how="left"
    )

    idx = echo.index[mask_exam]

    fillmask = (
        echo.loc[idx, "exam_date_recovered"].isna().to_numpy()
        & tmp["converted_raw_date"].notna().to_numpy()
    )

    echo.loc[
        idx[fillmask],
        "exam_date_recovered"
    ] = tmp.loc[
        fillmask,
        "converted_raw_date"
    ].to_numpy()

    echo.loc[
        idx[fillmask],
        "date_source"
    ] = "raw_echo_recovered"

    print(
        f"Recovered {fillmask.sum()} missing dates "
        f"for Exam {examno}"
    )

# ============================================================
# 5. Date completeness after recovery
# ============================================================

print("\n" + "="*90)
print("DATE COMPLETENESS AFTER RECOVERY")
print("="*90)

for examno in [4,5,6,8]:
    q = echo[echo["exam"] == examno]

    print(
        f"Exam {examno}: "
        f"N={len(q)}, "
        f"dated={q['exam_date_recovered'].notna().sum()}, "
        f"missing={q['exam_date_recovered'].isna().sum()}"
    )

# ============================================================
# 6. Load FRAM4 MRI data
# ============================================================

fram4 = read_pht("pht015152")

fram4["mri_date"] = pd.to_numeric(
    fram4["mri_date"],
    errors="coerce"
)

for v in [
    "Hippo",
    "Total_brain",
    "Lateralvent",
    "Total_gray",
    "Total_white",
]:
    if v in fram4.columns:
        fram4[v] = pd.to_numeric(
            fram4[v],
            errors="coerce"
        )

brain = (
    fram4.groupby("shareid")
    .agg(
        n_mri=("mri_date", "nunique"),
        first_mri=("mri_date", "min"),
        last_mri=("mri_date", "max"),
        n_hippo=("Hippo", lambda x: x.notna().sum()),
    )
    .reset_index()
)

brain["mri_followup_years"] = (
    brain["last_mri"] - brain["first_mri"]
) / 365.25

# ============================================================
# 7. Helper to calculate slopes within a chosen exam window
# ============================================================

TRAITS = [
    "lv_mass_derived_g",
    "la_dim",
    "lvdd",
    "fs_derived_pct",
]

# Recreate derived LV mass if necessary
if "lv_mass_derived_g" not in echo.columns:
    echo["lv_mass_derived_g"] = (
        0.8 * 1.04 * (
            (echo["lvdd"] + echo["ivs_d"] + echo["pwt_d"])**3
            - echo["lvdd"]**3
        ) + 0.6
    )

def make_window(window_name, exams):

    d = echo[
        echo["exam"].isin(exams)
    ].copy()

    rows = []

    for sid, g in d.groupby("shareid"):

        base = {
            "shareid": sid,
            "window": window_name,
            "exam_set": ",".join(map(str, exams)),
        }

        dated = g.dropna(
            subset=["exam_date_recovered"]
        )

        if len(dated):
            base["first_echo_date"] = (
                dated["exam_date_recovered"].min()
            )
            base["last_echo_date"] = (
                dated["exam_date_recovered"].max()
            )
        else:
            base["first_echo_date"] = np.nan
            base["last_echo_date"] = np.nan

        for trait in TRAITS:

            q = g[
                ["exam_date_recovered", trait]
            ].dropna().copy()

            q = q.sort_values(
                "exam_date_recovered"
            )

            base[f"{trait}_n"] = len(q)

            if (
                len(q) >= 2
                and q["exam_date_recovered"].nunique() >= 2
            ):
                x = (
                    q["exam_date_recovered"].to_numpy(float)
                    - q["exam_date_recovered"].iloc[0]
                ) / 365.25

                y = q[trait].to_numpy(float)

                base[f"{trait}_slope"] = (
                    np.polyfit(x, y, 1)[0]
                )
            else:
                base[f"{trait}_slope"] = np.nan

        rows.append(base)

    return pd.DataFrame(rows)

# Important windows
windows = {
    "E4_E5": [4,5],
    "E4_E6": [4,5,6],
    "E4_E8": [4,5,6,8],
    "E5_E6": [5,6],
    "E5_E8": [5,6,8],
    "E6_E8": [6,8],
}

all_windows = []

for name, exams in windows.items():
    print(f"\nBuilding {name} ...")
    all_windows.append(
        make_window(name, exams)
    )

cw = pd.concat(
    all_windows,
    ignore_index=True
)

# ============================================================
# 8. Merge each cardiac window with MRI
# ============================================================

cw = cw.merge(
    brain,
    on="shareid",
    how="left"
)

cw["last_echo_to_first_mri_years"] = (
    cw["first_mri"] - cw["last_echo_date"]
) / 365.25

# ============================================================
# 9. Prospective overlap counts
# ============================================================

print("\n" + "="*90)
print("PROSPECTIVE CARDIAC-WINDOW -> LONGITUDINAL MRI OVERLAP")
print("="*90)

results = []

for window_name in windows:

    w = cw[cw["window"] == window_name]

    print(f"\n### {window_name}")

    for trait in TRAITS:

        ncol = f"{trait}_n"
        scol = f"{trait}_slope"

        q = w[
            (w[ncol] >= 2)
            & w[scol].notna()
            & (w["n_hippo"] >= 2)
        ].copy()

        q0 = q[
            q["last_echo_to_first_mri_years"] >= 0
        ]

        q2 = q[
            q["last_echo_to_first_mri_years"] >= 2
        ]

        q5 = q[
            q["last_echo_to_first_mri_years"] >= 5
        ]

        row = {
            "window": window_name,
            "trait": trait,
            "N_echo_slope_plus_mri2": len(q),
            "N_last_echo_before_first_mri": len(q0),
            "N_gap_ge2y": len(q2),
            "N_gap_ge5y": len(q5),
        }

        results.append(row)

        print(
            f"{trait:20s} "
            f"all={len(q):4d}  "
            f"prospective={len(q0):4d}  "
            f"gap>=2y={len(q2):4d}  "
            f"gap>=5y={len(q5):4d}"
        )

results = pd.DataFrame(results)

# ============================================================
# 10. Follow-up duration in prospective subsets
# ============================================================

print("\n" + "="*90)
print("MRI FOLLOW-UP IN PROSPECTIVE E4-E6 SUBSET")
print("="*90)

for trait in TRAITS:

    ncol = f"{trait}_n"
    scol = f"{trait}_slope"

    q = cw[
        (cw["window"] == "E4_E6")
        & (cw[ncol] >= 2)
        & cw[scol].notna()
        & (cw["n_hippo"] >= 2)
        & (cw["last_echo_to_first_mri_years"] >= 0)
    ]

    print(f"\n{trait}: N={len(q)}")

    if len(q):
        print(
            q["mri_followup_years"]
            .describe(
                percentiles=[.10,.25,.50,.75,.90]
            )
            .to_string()
        )

# ============================================================
# 11. Save
# ============================================================

echo.to_csv(
    OUT / "echo_harmonized_with_recovered_dates.tsv",
    sep="\t",
    index=False
)

cw.to_csv(
    OUT / "cardiac_window_trajectories_with_mri.tsv",
    sep="\t",
    index=False
)

results.to_csv(
    OUT / "prospective_window_overlap.tsv",
    sep="\t",
    index=False
)

print("\nSaved:")
print(OUT / "echo_harmonized_with_recovered_dates.tsv")
print(OUT / "cardiac_window_trajectories_with_mri.tsv")
print(OUT / "prospective_window_overlap.tsv")
