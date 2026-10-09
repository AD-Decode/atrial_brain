#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/data/qiallab/Framingham")
RAW  = ROOT / "downloads" / "clinical"
RES  = ROOT / "results"

MASTER = RES / "neurocardiac_metadata_with_cognition_APOE.csv"
CLIN_LONG = RES / "clinical_exam8_10_long.csv"

OUT_CLIN = RES / "clinical_nearest_exam.csv"
OUT_MASTER = RES / "neurocardiac_metadata_analysis_ready.csv"
OUT_QC = RES / "nearest_exam_clinical_QC.txt"


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
    return s.replace({
        "nan": np.nan,
        "None": np.nan,
        "": np.nan
    })


# ============================================================
# 1. Load data
# ============================================================

master = pd.read_csv(MASTER)
clin = pd.read_csv(CLIN_LONG)

datefile = sorted(
    RAW.glob("*pht003099*.txt.gz")
)[0]

dates = read_dbgap(datefile)

for df in [master, clin, dates]:
    if "shareid" in df.columns:
        df["join_id"] = norm_id(df["shareid"])
    else:
        df["join_id"] = norm_id(df["join_id"])

master["mri_date"] = pd.to_numeric(
    master["mri_date"],
    errors="coerce"
)


# ============================================================
# 2. Extract sex, age, exam dates
# ============================================================

keep = [
    "join_id",
    "sex",
    "age8",
    "age9",
    "age10",
    "date8",
    "date9",
    "date10"
]

dates = dates[keep].copy()

for c in [
    "age8", "age9", "age10",
    "date8", "date9", "date10"
]:
    dates[c] = pd.to_numeric(
        dates[c],
        errors="coerce"
    )

dates = dates[
    dates["join_id"].isin(set(master["join_id"]))
].copy()

dates = dates.drop_duplicates("join_id")

print("Date table subjects:", dates["join_id"].nunique())


# ============================================================
# 3. Reshape dates/ages into long format
# ============================================================

rows = []

for exam in [8, 9, 10]:

    tmp = dates[
        [
            "join_id",
            "sex",
            f"age{exam}",
            f"date{exam}"
        ]
    ].copy()

    tmp = tmp.rename(
        columns={
            f"age{exam}": "age_at_exam",
            f"date{exam}": "exam_date_days_from_exam1"
        }
    )

    tmp["exam"] = exam

    rows.append(tmp)

exam_dates = pd.concat(
    rows,
    ignore_index=True
)


# ============================================================
# 4. Merge exam dates into clinical exam table
# ============================================================

clin["exam"] = pd.to_numeric(
    clin["exam"],
    errors="coerce"
).astype("Int64")

exam_dates["exam"] = exam_dates["exam"].astype("Int64")

clin = clin.merge(
    exam_dates,
    on=["join_id", "exam"],
    how="left",
    validate="many_to_one"
)


# ============================================================
# 5. Attach MRI date and calculate timing
# ============================================================

mri = master[
    ["join_id", "mri_date"]
].drop_duplicates("join_id")

clin = clin.merge(
    mri,
    on="join_id",
    how="left",
    validate="many_to_one"
)

clin["exam_minus_mri_days"] = (
    clin["exam_date_days_from_exam1"]
    - clin["mri_date"]
)

clin["abs_exam_mri_days"] = (
    clin["exam_minus_mri_days"].abs()
)

clin["years_exam_to_mri"] = (
    clin["exam_minus_mri_days"] / 365.25
)

clin["abs_years_exam_mri"] = (
    clin["abs_exam_mri_days"] / 365.25
)


# ============================================================
# 6. Choose nearest Exam 8/9/10 to MRI
# ============================================================

nearest = (
    clin[
        clin["exam_date_days_from_exam1"].notna()
    ]
    .sort_values(
        ["join_id", "abs_exam_mri_days", "exam"]
    )
    .drop_duplicates(
        "join_id",
        keep="first"
    )
    .copy()
)

nearest = nearest.rename(
    columns={
        "exam": "nearest_exam",
        "sex": "sex_clinical",
        "age_at_exam": "age_nearest_exam"
    }
)


# ============================================================
# 7. Rename clinical variables clearly
# ============================================================

rename = {
    "BMI": "BMI_nearest_exam",
    "SBP_mean": "SBP_nearest_exam",
    "DBP_mean": "DBP_nearest_exam",
    "waist_umbilicus_in": "waist_nearest_exam_in",

    "htn_med": "htn_med_nearest_exam",
    "lipid_med": "lipid_med_nearest_exam",
    "diabetes_history": "diabetes_history_nearest_exam",
    "diabetes_med": "diabetes_med_nearest_exam",
    "cv_med": "cv_med_nearest_exam",

    "current_smoker": "current_smoker_nearest_exam",
    "cigarettes_per_day": "cigarettes_per_day_nearest_exam",

    "heart_failure_history": "heart_failure_history_nearest_exam",
    "mi_history": "mi_history_nearest_exam",
    "af_history": "af_history_nearest_exam",

    "memory_dementia_history": "memory_history_nearest_exam",
    "subjective_memory_impairment": "subjective_memory_nearest_exam",

    "ecg_rhythm": "ecg_rhythm_nearest_exam",
    "atrial_enlargement": "atrial_enlargement_nearest_exam",
    "lvh": "lvh_nearest_exam",
    "renal_disease_clinical": "renal_disease_nearest_exam"
}

nearest = nearest.rename(columns=rename)


# ============================================================
# 8. Keep compact analysis-ready clinical columns
# ============================================================

keep_cols = [
    "join_id",

    "nearest_exam",
    "exam_date_days_from_exam1",
    "mri_date",
    "exam_minus_mri_days",
    "abs_exam_mri_days",
    "years_exam_to_mri",
    "abs_years_exam_mri",

    "sex_clinical",
    "age_nearest_exam",

    "BMI_nearest_exam",
    "weight_lb",
    "height_in",
    "waist_nearest_exam_in",

    "SBP_nearest_exam",
    "DBP_nearest_exam",
    "n_sbp_readings",
    "n_dbp_readings",

    "htn_med_nearest_exam",
    "lipid_med_nearest_exam",
    "diabetes_history_nearest_exam",
    "diabetes_med_nearest_exam",
    "cv_med_nearest_exam",

    "current_smoker_nearest_exam",
    "cigarettes_per_day_nearest_exam",

    "heart_failure_history_nearest_exam",
    "mi_history_nearest_exam",
    "af_history_nearest_exam",

    "memory_history_nearest_exam",
    "subjective_memory_nearest_exam",

    "ecg_rhythm_nearest_exam",
    "atrial_enlargement_nearest_exam",
    "lvh_nearest_exam",
    "renal_disease_nearest_exam"
]

keep_cols = [
    x for x in keep_cols
    if x in nearest.columns
]

nearest = nearest[keep_cols].copy()

nearest.to_csv(
    OUT_CLIN,
    index=False
)


# ============================================================
# 9. Merge into master
# ============================================================

for c in nearest.columns:
    if c != "join_id" and c in master.columns:
        master = master.drop(columns=c)

master = master.merge(
    nearest,
    on="join_id",
    how="left",
    validate="one_to_one"
)

master.to_csv(
    OUT_MASTER,
    index=False
)


# ============================================================
# 10. QC
# ============================================================

with OUT_QC.open("w") as f:

    def p(*x):
        print(*x, file=f)

    p("NEAREST EXAM CLINICAL QC")
    p("========================")
    p()

    p(f"Master N: {len(master)}")
    p(
        "Subjects with nearest Exam 8-10: "
        f"{master['nearest_exam'].notna().sum()}"
    )
    p()

    p("Nearest exam distribution:")
    p(
        master["nearest_exam"]
        .value_counts(dropna=False)
        .sort_index()
        .to_string()
    )
    p()

    p("Timing relative to MRI:")
    p(
        master["abs_exam_mri_days"]
        .describe()
        .to_string()
    )
    p()

    for yr in [0.5, 1, 2, 3, 5]:
        n = (
            master["abs_years_exam_mri"]
            <= yr
        ).sum()

        p(
            f"within_{yr}y: "
            f"{n}/{len(master)}"
        )

    p()
    p("Core clinical coverage:")

    vars_ = [
        "age_nearest_exam",
        "sex_clinical",
        "BMI_nearest_exam",
        "waist_nearest_exam_in",
        "SBP_nearest_exam",
        "DBP_nearest_exam",
        "htn_med_nearest_exam",
        "lipid_med_nearest_exam",
        "diabetes_history_nearest_exam",
        "current_smoker_nearest_exam"
    ]

    for c in vars_:
        if c in master.columns:
            p(
                f"{c}: "
                f"{master[c].notna().sum()}/{len(master)}"
            )

    p()
    p("Nearest exam by MRI timing direction:")

    p(
        pd.cut(
            master["years_exam_to_mri"],
            bins=[
                -np.inf,
                -3,
                -1,
                0,
                1,
                3,
                np.inf
            ],
            labels=[
                "<-3y",
                "-3 to -1y",
                "-1y to MRI",
                "MRI to +1y",
                "+1 to +3y",
                ">+3y"
            ]
        )
        .value_counts()
        .sort_index()
        .to_string()
    )


print("\nCreated:")
print(OUT_CLIN)
print(OUT_MASTER)
print(OUT_QC)

print("\n" + OUT_QC.read_text())
