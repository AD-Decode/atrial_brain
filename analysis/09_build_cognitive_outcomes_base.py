#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/data/qiallab/Framingham")
RAW  = ROOT / "downloads" / "clinical"
RES  = ROOT / "results"

OVERLAP = RES / "brain_cardiac_overlap.csv"

DX_ACC = "pht004368"
NP_ACC = "pht004374"
MMSE_ACC = "pht005174"
NPDATES_ACC = "pht003910"

OUT = RES / "neurocardiac_cognitive_outcomes.csv"
NP_LONG = RES / "neuropsych_sessions_long.csv"
QC = RES / "cognitive_outcomes_QC.txt"


def find_one(pattern):
    x = sorted(RAW.glob(pattern))
    if not x:
        return None
    return x[0]


def read_dbgap(path):
    return pd.read_csv(
        path,
        sep="\t",
        compression="gzip",
        comment="#",
        low_memory=False
    )


def norm_id(s):
    """
    Normalize dbGaP/share IDs without converting numeric IDs to floats.
    """
    s = s.astype(str).str.strip()
    s = s.str.replace(r"\.0$", "", regex=True)
    s = s.replace({
        "nan": np.nan,
        "None": np.nan,
        "": np.nan
    })
    return s


# ============================================================
# 1. Imaging overlap cohort
# ============================================================

cohort = pd.read_csv(OVERLAP)

print("Overlap columns:")
print(list(cohort.columns))

# Prefer shareid if present; otherwise use dbGaP_Subject_ID.
possible = [
    "shareid",
    "dbGaP_Subject_ID",
    "dbgap_subject_id",
    "subject_id"
]

cohort_key = next(
    (x for x in possible if x in cohort.columns),
    None
)

if cohort_key is None:
    raise RuntimeError(
        "Could not identify participant key in brain_cardiac_overlap.csv"
    )

cohort["join_id"] = norm_id(cohort[cohort_key])

if cohort["join_id"].duplicated().any():
    print(
        "WARNING: overlap file contains duplicate participant IDs:",
        cohort["join_id"].duplicated().sum()
    )

base = cohort[["join_id"]].drop_duplicates().copy()

print(f"\nOverlap participants: {len(base)}")


# ============================================================
# 2. Dementia consensus / adjudication table
# ============================================================

dxfile = find_one(f"*{DX_ACC}*.txt.gz")

if dxfile is None:
    raise RuntimeError("Could not find dementia adjudication table.")

dx = read_dbgap(dxfile)

if "shareid" not in dx.columns:
    raise RuntimeError("shareid missing from pht004368.")

dx["join_id"] = norm_id(dx["shareid"])

print(f"Dementia table rows: {len(dx)}")
print(f"Dementia unique shareid: {dx['join_id'].nunique()}")


# ------------------------------------------------------------
# Multiple reviews can occur for a participant.
# Use the most recent REVIEW_DATE as the endpoint adjudication.
# Preserve number of records per participant for QC.
# ------------------------------------------------------------

dx["n_dx_records"] = dx.groupby("join_id")["join_id"].transform("size")

if "REVIEW_DATE" in dx.columns:
    dx["REVIEW_DATE_num"] = pd.to_numeric(
        dx["REVIEW_DATE"],
        errors="coerce"
    )

    dx = (
        dx.sort_values(
            ["join_id", "REVIEW_DATE_num"],
            na_position="first"
        )
        .drop_duplicates("join_id", keep="last")
    )
else:
    print("WARNING: REVIEW_DATE absent; keeping last row per shareid.")
    dx = dx.drop_duplicates("join_id", keep="last")


# ============================================================
# 3. Decode endpoint diagnosis
# ============================================================

for col in [
    "DEMRV046",
    "DEMRV103",
    "DEMRV104",
    "DEMRV115",
    "DEMRV116",
    "DEMRV117",
    "DEMRV124",
    "NORMAL_DATE",
    "IMPAIRMENT_DATE",
    "MILD_DATE",
    "MODERATE_DATE",
    "SEVERE_DATE",
    "EDDD",
    "REVIEW_DATE"
]:
    if col in dx.columns:
        dx[col] = pd.to_numeric(dx[col], errors="coerce")


def classify_dx(code):
    if pd.isna(code):
        return np.nan

    code = int(code)

    if code == 0:
        return "CN"

    if code == 9:
        return "MCI"

    if code in [1,2,3,4,5,6,7,8,10]:
        return "Dementia"

    return np.nan


def subtype_label(code):
    if pd.isna(code):
        return np.nan

    labels = {
        0: "None",
        1: "AD_without_stroke",
        2: "AD_with_stroke",
        3: "Vascular_dementia",
        4: "Mixed_AD_vascular",
        5: "Frontotemporal_dementia",
        6: "Dementia_with_Lewy_bodies",
        7: "Other_progressive_dementia",
        8: "Other_nonprogressive_dementia",
        9: "MCI_cognitive_impairment_no_dementia",
        10: "Dementia_uncertain"
    }

    return labels.get(int(code), np.nan)


def ad_class_label(code):
    if pd.isna(code):
        return np.nan

    labels = {
        1: "Probable_AD",
        2: "Possible_AD",
        3: "Definite_AD"
    }

    return labels.get(int(code), np.nan)


dx["dx_ever_adjudicated"] = dx["DEMRV103"].apply(classify_dx)

dx["dementia_subtype_label"] = dx["DEMRV103"].apply(
    subtype_label
)

dx["mci_ever"] = np.where(
    dx["DEMRV103"].eq(9),
    1,
    np.where(dx["DEMRV103"].notna(), 0, np.nan)
)

dx["dementia_ever"] = np.where(
    dx["DEMRV103"].isin([1,2,3,4,5,6,7,8,10]),
    1,
    np.where(dx["DEMRV103"].notna(), 0, np.nan)
)

# AD-containing dementia:
# 1 = AD without stroke
# 2 = AD with stroke
# 4 = mixed AD + vascular dementia
dx["ad_dementia_ever"] = np.where(
    dx["DEMRV103"].isin([1,2,4]),
    1,
    np.where(dx["DEMRV103"].notna(), 0, np.nan)
)

dx["vascular_dementia_ever"] = np.where(
    dx["DEMRV103"].isin([3,4]),
    1,
    np.where(dx["DEMRV103"].notna(), 0, np.nan)
)

if "DEMRV115" in dx.columns:
    dx["ad_nincds_adrda"] = dx["DEMRV115"]

if "DEMRV116" in dx.columns:
    dx["ad_classification_label"] = dx["DEMRV116"].apply(
        ad_class_label
    )


# ------------------------------------------------------------
# Rename time variables so their units are explicit.
# These are NUMBER OF DAYS SINCE EXAM 1, not calendar dates.
# ------------------------------------------------------------

rename_dx = {
    "NORMAL_DATE":       "normal_date_days_from_exam1",
    "IMPAIRMENT_DATE":   "impairment_date_days_from_exam1",
    "MILD_DATE":         "mild_dementia_date_days_from_exam1",
    "MODERATE_DATE":     "moderate_dementia_date_days_from_exam1",
    "SEVERE_DATE":       "severe_dementia_date_days_from_exam1",
    "EDDD":              "earliest_dementia_date_days_from_exam1",
    "REVIEW_DATE":       "review_date_days_from_exam1",
}

dx = dx.rename(columns=rename_dx)


dx_keep = [
    "join_id",
    "n_dx_records",

    "DEMRV046",
    "DEMRV103",
    "DEMRV104",
    "DEMRV115",
    "DEMRV116",
    "DEMRV117",
    "DEMRV124",

    "dx_ever_adjudicated",
    "dementia_subtype_label",
    "mci_ever",
    "dementia_ever",
    "ad_dementia_ever",
    "vascular_dementia_ever",
    "ad_nincds_adrda",
    "ad_classification_label",

    "normal_date_days_from_exam1",
    "impairment_date_days_from_exam1",
    "mild_dementia_date_days_from_exam1",
    "moderate_dementia_date_days_from_exam1",
    "severe_dementia_date_days_from_exam1",
    "earliest_dementia_date_days_from_exam1",
    "review_date_days_from_exam1",
]

dx_keep = [x for x in dx_keep if x in dx.columns]

dx_small = dx[dx_keep].copy()


# ============================================================
# 4. Neuropsychology — preserve LONG format for now
# ============================================================

npfile = find_one(f"*{NP_ACC}*.txt.gz")

if npfile is None:
    raise RuntimeError("Could not find pht004374 neuropsych table.")

npdf = read_dbgap(npfile)

if "shareid" not in npdf.columns:
    raise RuntimeError("shareid missing from pht004374.")

npdf["join_id"] = norm_id(npdf["shareid"])


NP_VARS = [
    # context
    "sex",
    "age",
    "battery",
    "test_language",

    # verbal episodic memory
    "LMi",
    "LMd",
    "LMr",
    "PASi",
    "PASd",
    "PASr",

    # visual memory
    "VRi",
    "VRd",
    "VRr",

    # attention / working memory
    "DSF",
    "DSB",

    # executive / processing speed
    "trailsA",
    "trailsB",

    # language
    "FAS_animal",

    # visuospatial
    "BD",

    # secondary
    "Incidental_free",
    "Math_correct",
]

np_keep = [
    "join_id",
    "shareid"
] + [x for x in NP_VARS if x in npdf.columns]

np_long = npdf[np_keep].copy()

# Restrict to our 785-person imaging cohort
np_long = np_long[
    np_long["join_id"].isin(set(base["join_id"]))
].copy()

np_long.to_csv(NP_LONG, index=False)


# ============================================================
# 5. MMSE — keep raw longitudinal records too
# ============================================================

mmsefile = find_one(f"*{MMSE_ACC}*.txt.gz")

if mmsefile is not None:
    mmse = read_dbgap(mmsefile)

    if "shareid" in mmse.columns:
        mmse["join_id"] = norm_id(mmse["shareid"])

        cols = ["join_id"]

        for x in mmse.columns:
            if x.lower() in [
                "shareid",
                "cogscr",
                "age",
                "sex"
            ]:
                if x not in cols:
                    cols.append(x)

        mmse = mmse[cols]

        mmse = mmse[
            mmse["join_id"].isin(set(base["join_id"]))
        ].copy()

        mmse.to_csv(
            RES / "mmse_sessions_long.csv",
            index=False
        )


# ============================================================
# 6. Build 785-row cognitive outcome backbone
# ============================================================

out = base.merge(
    dx_small,
    on="join_id",
    how="left",
    validate="one_to_one"
)

# Restore original cohort identifier name too
if cohort_key != "join_id":
    idmap = (
        cohort[[cohort_key]]
        .copy()
    )
    idmap["join_id"] = norm_id(idmap[cohort_key])
    idmap = idmap.drop_duplicates("join_id")

    out = out.merge(
        idmap,
        on="join_id",
        how="left",
        validate="one_to_one"
    )

# Put identifiers first
first = [
    x for x in [
        cohort_key,
        "join_id"
    ] if x in out.columns
]

rest = [x for x in out.columns if x not in first]

out = out[first + rest]

out.to_csv(OUT, index=False)


# ============================================================
# 7. QC
# ============================================================

with QC.open("w") as f:

    def p(*args):
        print(*args, file=f)

    p("COGNITIVE OUTCOME QC")
    p("====================")
    p()
    p(f"Imaging overlap N: {len(base)}")
    p(f"Outcome table N: {len(out)}")
    p()

    p("Endpoint adjudication coverage:")
    p(out["dx_ever_adjudicated"].value_counts(
        dropna=False
    ).to_string())
    p()

    p("DEMRV103 raw distribution:")
    p(out["DEMRV103"].value_counts(
        dropna=False
    ).sort_index().to_string())
    p()

    p("Dementia subtype:")
    p(out["dementia_subtype_label"].value_counts(
        dropna=False
    ).to_string())
    p()

    p("AD classification:")
    if "ad_classification_label" in out:
        p(out["ad_classification_label"].value_counts(
            dropna=False
        ).to_string())
    p()

    p("Future timing-variable availability:")
    timing = [
        "normal_date_days_from_exam1",
        "impairment_date_days_from_exam1",
        "mild_dementia_date_days_from_exam1",
        "moderate_dementia_date_days_from_exam1",
        "severe_dementia_date_days_from_exam1",
        "earliest_dementia_date_days_from_exam1"
    ]

    for x in timing:
        if x in out:
            p(f"{x}: {out[x].notna().sum()}")

    p()
    p("Neuropsychology:")
    p(f"NP rows in imaging cohort: {len(np_long)}")
    p(f"NP unique subjects: {np_long['join_id'].nunique()}")
    p(
        "Subjects with >1 NP row: "
        f"{(np_long.groupby('join_id').size() > 1).sum()}"
    )


print("\nCreated:")
print(OUT)
print(NP_LONG)
print(QC)

print("\n===== ENDPOINT DIAGNOSIS IN 785 COHORT =====")
print(
    out["dx_ever_adjudicated"]
    .value_counts(dropna=False)
)

print("\n===== DEMENTIA SUBTYPE =====")
print(
    out["dementia_subtype_label"]
    .value_counts(dropna=False)
)

print("\n===== NEUROPSYCHOLOGY =====")
print("Rows:", len(np_long))
print("Unique subjects:", np_long["join_id"].nunique())
print(
    "Subjects with >1 NP session:",
    (np_long.groupby("join_id").size() > 1).sum()
)
