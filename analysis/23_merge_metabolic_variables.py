#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/data/qiallab/Framingham")
RAW  = ROOT / "downloads" / "clinical"
RES  = ROOT / "results"

MASTER = RES / "neurocardiac_metadata_analysis_ready.csv"
OUT    = RES / "neurocardiac_metadata_analysis_ready_metabolic.csv"
QC     = RES / "metabolic_merge_QC.txt"


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
# 1. Load master
# ============================================================

master = pd.read_csv(MASTER)
master["join_id"] = norm_id(master["shareid"])

master["nearest_exam"] = pd.to_numeric(
    master["nearest_exam"],
    errors="coerce"
)


# ============================================================
# 2. Exam 8 insulin — pht003901
# ============================================================

f8 = sorted(
    RAW.glob("*pht003901*.txt.gz")
)[0]

ins8 = read_dbgap(f8)

ins8["join_id"] = norm_id(ins8["shareid"])

ins8["INSULIN"] = pd.to_numeric(
    ins8["INSULIN"],
    errors="coerce"
)

ins8["INSULIN_I"] = pd.to_numeric(
    ins8["INSULIN_I"],
    errors="coerce"
)

ins8 = (
    ins8[
        [
            "join_id",
            "INSULIN",
            "INSULIN_I"
        ]
    ]
    .drop_duplicates("join_id")
    .rename(columns={
        "INSULIN": "insulin_exam8_pmol_L",
        "INSULIN_I": "insulin_exam8_indicator"
    })
)


# ============================================================
# 3. Exam 9 insulin — pht004805
# ============================================================

f9 = sorted(
    RAW.glob("*pht004805*.txt.gz")
)[0]

ins9 = read_dbgap(f9)

ins9["join_id"] = norm_id(ins9["shareid"])

ins9["INSULIN"] = pd.to_numeric(
    ins9["INSULIN"],
    errors="coerce"
)

ins9["INSULIN_I"] = pd.to_numeric(
    ins9["INSULIN_I"],
    errors="coerce"
)

ins9 = (
    ins9[
        [
            "join_id",
            "INSULIN",
            "INSULIN_I"
        ]
    ]
    .drop_duplicates("join_id")
    .rename(columns={
        "INSULIN": "insulin_exam9_pmol_L",
        "INSULIN_I": "insulin_exam9_indicator"
    })
)


# ============================================================
# 4. Historical Exam 7 metabolic biomarkers — pht010725
# ============================================================

f7 = sorted(
    RAW.glob("*pht010725*.txt.gz")
)[0]

met = read_dbgap(f7)

met["join_id"] = norm_id(met["shareid"])
met["EXAM"] = pd.to_numeric(
    met["EXAM"],
    errors="coerce"
)

met7 = met[
    met["EXAM"] == 7
].copy()

historical_vars = {
    "HBA1C": "hba1c_exam7",
    "ADIP": "adiponectin_exam7",
    "INSLN_PF_LNC": "fasting_insulin_exam7",
    "INSLN_2H_LNC": "insulin_2h_exam7",
    "PROINSLN_PF_LNC": "proinsulin_exam7",
    "GLU2H": "glucose_2h_exam7",
    "GLUC2HYN": "glucose_2h_available_exam7",
    "GLUC2HSMPL5": "glucose_2h_sample_exam7"
}

keep = ["join_id"]

for raw, new in historical_vars.items():
    if raw in met7.columns:
        met7[raw] = pd.to_numeric(
            met7[raw],
            errors="coerce"
        )
        keep.append(raw)

met7 = (
    met7[keep]
    .drop_duplicates("join_id")
    .rename(columns=historical_vars)
)


# ============================================================
# 5. Merge all metabolic variables
# ============================================================

df = master.merge(
    ins8,
    on="join_id",
    how="left",
    validate="one_to_one"
)

df = df.merge(
    ins9,
    on="join_id",
    how="left",
    validate="one_to_one"
)

df = df.merge(
    met7,
    on="join_id",
    how="left",
    validate="one_to_one"
)


# ============================================================
# 6. Construct insulin matched to nearest clinical exam
# ============================================================

df["insulin_nearest_exam_pmol_L"] = np.nan
df["insulin_nearest_exam_source"] = np.nan

m8 = df["nearest_exam"].eq(8)

df.loc[
    m8,
    "insulin_nearest_exam_pmol_L"
] = df.loc[
    m8,
    "insulin_exam8_pmol_L"
]

df.loc[
    m8 & df["insulin_exam8_pmol_L"].notna(),
    "insulin_nearest_exam_source"
] = "Exam8"


m9 = df["nearest_exam"].eq(9)

df.loc[
    m9,
    "insulin_nearest_exam_pmol_L"
] = df.loc[
    m9,
    "insulin_exam9_pmol_L"
]

df.loc[
    m9 & df["insulin_exam9_pmol_L"].notna(),
    "insulin_nearest_exam_source"
] = "Exam9"


# no contemporaneous Exam 10 insulin table available
m10 = df["nearest_exam"].eq(10)

df.loc[
    m10,
    "insulin_nearest_exam_source"
] = "Exam10_no_assay"


# ============================================================
# 7. Convert insulin to conventional uU/mL
#
# dbGaP dictionary:
# pmol/L divided by 6.945 = uU/mL
# ============================================================

df["insulin_nearest_exam_uU_mL"] = (
    df["insulin_nearest_exam_pmol_L"] / 6.945
)


# log-transform candidate for modeling
df["log_insulin_nearest_exam"] = np.log(
    df["insulin_nearest_exam_pmol_L"]
)


# ============================================================
# 8. Save
# ============================================================

df.to_csv(
    OUT,
    index=False
)


# ============================================================
# 9. QC
# ============================================================

with QC.open("w") as f:

    def p(*x):
        print(*x, file=f)

    p("METABOLIC MERGE QC")
    p("==================")
    p()

    p(f"N: {len(df)}")
    p()

    p("Nearest exam:")
    p(
        df["nearest_exam"]
        .value_counts(dropna=False)
        .sort_index()
        .to_string()
    )
    p()

    p("Nearest-exam insulin coverage:")
    p(
        f"{df['insulin_nearest_exam_pmol_L'].notna().sum()}/{len(df)}"
    )

    p()
    p("Insulin source:")
    p(
        df["insulin_nearest_exam_source"]
        .value_counts(dropna=False)
        .to_string()
    )

    p()
    p("Insulin pmol/L:")
    p(
        df["insulin_nearest_exam_pmol_L"]
        .describe()
        .to_string()
    )

    p()
    p("Insulin uU/mL:")
    p(
        df["insulin_nearest_exam_uU_mL"]
        .describe()
        .to_string()
    )

    p()
    p("Historical Exam 7 biomarkers:")

    for c in [
        "hba1c_exam7",
        "adiponectin_exam7",
        "fasting_insulin_exam7",
        "proinsulin_exam7",
        "glucose_2h_exam7"
    ]:
        if c in df.columns:
            p(
                f"{c}: "
                f"{df[c].notna().sum()}/{len(df)}"
            )


print("\nCreated:")
print(OUT)
print(QC)

print("\n" + QC.read_text())
