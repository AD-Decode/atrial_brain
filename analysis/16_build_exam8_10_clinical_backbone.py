#!/usr/bin/env python3

from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

ROOT = Path("/data/qiallab/Framingham")
RAW  = ROOT / "downloads" / "clinical"
RES  = ROOT / "results"

MASTER = RES / "neurocardiac_metadata_with_cognition_APOE.csv"

OUT_LONG = RES / "clinical_exam8_10_long.csv"
OUT_DATE = RES / "clinical_exam_date_candidates.csv"
OUT_QC   = RES / "clinical_exam8_10_QC.txt"


TABLES = {
    8: "pht000747",
    9: "pht005140",
    10: "pht015104",
}


# ============================================================
# VERIFIED variables from our prior dictionary review
# ============================================================

VARMAP = {

    8: {
        # BP
        "sbp1": "H111",
        "dbp1": "H112",
        "sbp2": "H233",
        "dbp2": "H234",

        # anthropometry
        "weight_lb": "H393",
        "height_in": "H399",
        "waist_umbilicus_in": "H403",
        "waist_iliac_in": "H405",

        # hypertension / lipid / diabetes / CV meds
        "htn_med": "H014",
        "lipid_med": "H015",
        "diabetes_history": "H016",
        "cv_med": "H017",

        # smoking
        "smoked_since_last_exam": "H060",
        "smoked_last_year": "H061",
        "current_smoker": "H062",
        "cigarettes_per_day": "H063",
        "avg_cigarettes_per_day": "H064",

        # cardiovascular history
        "heart_failure_history": "H108",
        "heart_failure_hospitalized": "H109",
        "mi_history": "H131",
        "af_history": "H136",

        # cerebrovascular / cognition
        "memory_dementia_history": "H172",
        "subjective_memory_impairment": "H173",

        # ECG
        "ecg_rhythm": "H309",
        "atrial_enlargement": "H336",
        "lvh": "H338",

        # clinical impression
        "dementia_clinical_impression": "H347",
        "renal_disease_clinical": "H354",

        # physical activity
        "activity_slight_hours": "H482",
        "activity_moderate_hours": "H483",
        "activity_heavy_hours": "H484",
    },

    9: {
        # BP
        "sbp1": "j116",
        "dbp1": "j118",
        "sbp2": "j255",
        "dbp2": "j257",

        # anthropometry
        "height_in": "j472",
        "weight_lb": "j474",
        "waist_umbilicus_in": "j480",

        # clinical history / meds
        "htn_history": "j014",
        "htn_med": "j015",
        "high_cholesterol_history": "j016",
        "lipid_med": "j017",
        "diabetes_history": "j018",
        "diabetes_med": "j019",
        "cv_med": "j020",

        # smoking
        "smoked_since_last_exam": "j062",
        "smoked_last_year": "j063",
        "current_smoker": "j064",
        "cigarettes_per_day": "j065",
        "avg_cigarettes_per_day": "j066",

        # ECG / clinical impression
        "ecg_rhythm": "j364",
        "atrial_enlargement": "j391",
        "lvh": "j393",
        "dementia_clinical_impression": "j404",
        "renal_disease_clinical": "j414",

        # NOTE:
        # broad AF variable deliberately omitted until exact
        # Exam 9 AF definition is verified.
    },

    10: {
        # BP
        "sbp1": "K0385",
        "dbp1": "K0386",
        "sbp2": "K0875",
        "dbp2": "K0876",

        # anthropometry
        "weight_lb": "K1165",
        "height_in": "K1169",
        "waist_umbilicus_in": "K1172",

        # clinical history / meds
        "htn_history": "K0216",
        "htn_med": "K0217",
        "high_cholesterol_history": "K0218",
        "lipid_med": "K0219",
        "diabetes_history": "K0220",
        "diabetes_med": "K0221",
        "cv_med": "K0222",

        # smoking
        "smoked_since_last_exam": "K0302",
        "smoked_last_year": "K0303",
        "current_smoker": "K0304",
        "cigarettes_per_day": "K0305",

        # cognition
        "memory_dementia_history": "K0561",
        "subjective_memory_impairment": "K0562",
        "worsening_memory": "K0563",

        # ECG / clinical impression
        "ecg_rhythm": "K0958",
        "atrial_enlargement": "K0973",
        "lvh": "K0975",
        "dementia_clinical_impression": "K0979",
        "renal_disease_clinical": "K0990",

        # activity
        "activity_slight_hours": "K1236",
        "activity_moderate_hours": "K1237",
        "activity_heavy_hours": "K1238",

        # broad AF deliberately omitted until exact definition verified.
    },
}


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


def find_one(pattern):
    x = sorted(RAW.glob(pattern))
    if not x:
        return None
    return x[0]


def parse_dictionary(xmlfile):
    """
    Return one row per dbGaP <variable>.
    """
    root = ET.parse(xmlfile).getroot()

    rows = []

    for var in root.iter():
        if not str(var.tag).lower().endswith("variable"):
            continue

        rec = {"phv": var.attrib.get("id")}

        for child in list(var):
            tag = str(child.tag).split("}")[-1].lower()
            txt = " ".join(
                t.strip()
                for t in child.itertext()
                if t and t.strip()
            )

            if tag in [
                "name",
                "description",
                "unit",
                "comment",
                "coll_interval"
            ]:
                rec[tag] = txt

        rows.append(rec)

    return pd.DataFrame(rows)


# ============================================================
# 1. Master imaging cohort
# ============================================================

master = pd.read_csv(MASTER)

if "shareid" not in master.columns:
    raise RuntimeError("Master metadata has no shareid.")

master["join_id"] = norm_id(master["shareid"])

cohort_ids = set(master["join_id"].dropna())

print("Master N:", len(master))
print("Unique shareid:", master["join_id"].nunique())


# ============================================================
# 2. Process Exam 8, 9, 10
# ============================================================

long_tables = []
date_candidates_all = []
qc_lines = []

for exam, acc in TABLES.items():

    print("\n" + "=" * 80)
    print(f"EXAM {exam} — {acc}")
    print("=" * 80)

    datafile = find_one(f"*{acc}*.txt.gz")
    dictfile = find_one(f"*{acc}*data_dict.xml")

    if datafile is None:
        print("DATA FILE NOT FOUND")
        continue

    df = read_dbgap(datafile)

    if "shareid" not in df.columns:
        raise RuntimeError(
            f"shareid missing from Exam {exam} table."
        )

    df["join_id"] = norm_id(df["shareid"])

    # restrict immediately to our cohort
    df = df[
        df["join_id"].isin(cohort_ids)
    ].copy()

    print("Rows in imaging cohort:", len(df))
    print("Unique imaging subjects:", df["join_id"].nunique())

    # --------------------------------------------------------
    # Date candidate search
    # --------------------------------------------------------

    date_candidates = []

    # name-based candidates from actual data columns
    for col in df.columns:

        lc = str(col).lower()

        if (
            "date" in lc
            or "examdate" in lc
            or "exam_date" in lc
        ):
            date_candidates.append({
                "exam": exam,
                "accession": acc,
                "variable": col,
                "description": "",
                "source": "column_name"
            })

    # semantic candidates from XML dictionary
    if dictfile is not None:

        dd = parse_dictionary(dictfile)

        for _, r in dd.iterrows():

            name = str(r.get("name", ""))
            desc = str(r.get("description", ""))
            comment = str(r.get("comment", ""))

            text = f"{name} {desc} {comment}".lower()

            # prioritize actual examination / clinic visit dates,
            # not medication dates, hospitalization dates, etc.
            if (
                (
                    "exam date" in text
                    or "date of exam" in text
                    or "examination date" in text
                    or "clinic date" in text
                    or "visit date" in text
                )
                and not any(
                    bad in text
                    for bad in [
                        "medication",
                        "hospital",
                        "stroke",
                        "death",
                        "diagnosis",
                        "surgery",
                        "procedure"
                    ]
                )
            ):
                date_candidates.append({
                    "exam": exam,
                    "accession": acc,
                    "variable": name,
                    "description": desc,
                    "source": "dictionary"
                })

    date_candidates = pd.DataFrame(date_candidates)

    if len(date_candidates):
        date_candidates = date_candidates.drop_duplicates(
            ["exam", "variable", "description"]
        )
        date_candidates_all.append(date_candidates)

        print("\nDATE CANDIDATES:")
        print(
            date_candidates.to_string(index=False)
        )
    else:
        print("\nNo exam-date candidate automatically identified.")

    # --------------------------------------------------------
    # Extract verified clinical variables
    # --------------------------------------------------------

    out = pd.DataFrame({
        "join_id": df["join_id"],
        "shareid": df["shareid"],
        "exam": exam,
        "source_table": acc,
    })

    found = []
    missing = []

    for canonical, rawvar in VARMAP[exam].items():

        if rawvar in df.columns:
            out[canonical] = df[rawvar]
            found.append(f"{canonical}={rawvar}")
        else:
            out[canonical] = np.nan
            missing.append(f"{canonical}={rawvar}")

    # convert quantitative variables
    numeric_vars = [
        "sbp1",
        "dbp1",
        "sbp2",
        "dbp2",
        "weight_lb",
        "height_in",
        "waist_umbilicus_in",
        "waist_iliac_in",
        "cigarettes_per_day",
        "avg_cigarettes_per_day",
        "activity_slight_hours",
        "activity_moderate_hours",
        "activity_heavy_hours",
    ]

    for c in numeric_vars:
        if c in out.columns:
            out[c] = pd.to_numeric(
                out[c],
                errors="coerce"
            )

    # --------------------------------------------------------
    # Derived BMI — only when physiologically plausible
    # --------------------------------------------------------

    valid_wt = out["weight_lb"].between(60, 500)
    valid_ht = out["height_in"].between(45, 85)

    out["BMI"] = np.where(
        valid_wt & valid_ht,
        703.0 * out["weight_lb"] /
        (out["height_in"] ** 2),
        np.nan
    )

    # --------------------------------------------------------
    # Mean BP
    #
    # For now:
    #   if both readings valid -> average
    #   if only one reading valid -> use that reading
    #
    # Keep individual readings for sensitivity analysis.
    # --------------------------------------------------------

    for c in ["sbp1", "sbp2"]:
        out.loc[
            ~out[c].between(60, 280),
            c
        ] = np.nan

    for c in ["dbp1", "dbp2"]:
        out.loc[
            ~out[c].between(30, 180),
            c
        ] = np.nan

    out["SBP_mean"] = out[
        ["sbp1", "sbp2"]
    ].mean(axis=1)

    out["DBP_mean"] = out[
        ["dbp1", "dbp2"]
    ].mean(axis=1)

    out["n_sbp_readings"] = out[
        ["sbp1", "sbp2"]
    ].notna().sum(axis=1)

    out["n_dbp_readings"] = out[
        ["dbp1", "dbp2"]
    ].notna().sum(axis=1)

    long_tables.append(out)

    qc_lines.append(
        f"Exam {exam}: N rows={len(out)}, "
        f"unique subjects={out['join_id'].nunique()}"
    )

    qc_lines.append(
        f"Exam {exam}: BMI available={out['BMI'].notna().sum()}"
    )

    qc_lines.append(
        f"Exam {exam}: SBP available={out['SBP_mean'].notna().sum()}"
    )

    qc_lines.append(
        f"Exam {exam}: DBP available={out['DBP_mean'].notna().sum()}"
    )

    if missing:
        qc_lines.append(
            f"Exam {exam} missing mapped vars: "
            + "; ".join(missing)
        )


# ============================================================
# 3. Combine
# ============================================================

clinical_long = pd.concat(
    long_tables,
    ignore_index=True,
    sort=False
)

clinical_long.to_csv(
    OUT_LONG,
    index=False
)

if date_candidates_all:
    dc = pd.concat(
        date_candidates_all,
        ignore_index=True
    )
else:
    dc = pd.DataFrame(
        columns=[
            "exam",
            "accession",
            "variable",
            "description",
            "source"
        ]
    )

dc.to_csv(
    OUT_DATE,
    index=False
)


# ============================================================
# 4. Coverage summaries
# ============================================================

print("\n" + "=" * 80)
print("CLINICAL LONG TABLE")
print("=" * 80)

print("Rows:", len(clinical_long))
print(
    "Unique subjects:",
    clinical_long["join_id"].nunique()
)

print("\nRows per Exam:")
print(
    clinical_long["exam"]
    .value_counts()
    .sort_index()
)

print("\nSubjects with each number of Exams 8–10:")
n_exam = (
    clinical_long
    .groupby("join_id")["exam"]
    .nunique()
)

print(
    n_exam.value_counts()
    .sort_index()
)

print("\nVariable coverage by exam:")

coverage_vars = [
    "BMI",
    "SBP_mean",
    "DBP_mean",
    "waist_umbilicus_in",
    "htn_med",
    "diabetes_history",
    "current_smoker"
]

for exam in [8, 9, 10]:

    x = clinical_long[
        clinical_long["exam"] == exam
    ]

    print(f"\nExam {exam}")

    for c in coverage_vars:
        if c in x.columns:
            print(
                f"  {c}: "
                f"{x[c].notna().sum()}/{len(x)}"
            )


# ============================================================
# 5. QC report
# ============================================================

with OUT_QC.open("w") as f:

    print(
        "EXAM 8–10 CLINICAL BACKBONE QC",
        file=f
    )
    print(
        "==============================",
        file=f
    )
    print(file=f)

    print(
        f"Master imaging cohort: {len(master)}",
        file=f
    )

    print(
        f"Subjects represented in Exams 8–10: "
        f"{clinical_long['join_id'].nunique()}",
        file=f
    )

    print(file=f)

    for line in qc_lines:
        print(line, file=f)

    print(file=f)
    print(
        "Subjects by number of Exams 8–10:",
        file=f
    )
    print(
        n_exam.value_counts()
        .sort_index()
        .to_string(),
        file=f
    )

    print(file=f)
    print(
        "Exam-date candidates are in:",
        OUT_DATE,
        file=f
    )


print("\nCreated:")
print(OUT_LONG)
print(OUT_DATE)
print(OUT_QC)
