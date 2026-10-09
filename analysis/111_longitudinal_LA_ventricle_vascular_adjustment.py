#!/usr/bin/env python3

from pathlib import Path
import re
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import norm

# ============================================================
# PATHS
# ============================================================

ROOT = Path("/data/qiallab/Framingham")

DATA = ROOT / "data/longitudinal_20260924"

ANALYSIS_FILE = (
    ROOT / "data"
    / "longitudinal_derived"
    / "E4_E6_repeated_MRI_temporal_analysis.tsv"
)

OUT = (
    ROOT / "results"
    / "longitudinal_heart_brain"
    / "temporal_prediction"
    / "vascular_adjustment"
)
OUT.mkdir(parents=True, exist_ok=True)

print("=" * 100)
print("111: LA -> LATERAL VENTRICLE VASCULAR/METABOLIC ADJUSTMENT")
print("=" * 100)
print("DATA :", DATA)
print("ANALYSIS:", ANALYSIS_FILE)
print("OUT  :", OUT)

# ============================================================
# HELPERS
# ============================================================

def norm_id(s):
    s = s.astype(str).str.strip()
    s = s.str.replace(r"\.0$", "", regex=True)
    return s.replace(
        ["", "nan", "NaN", "NA", "N/A", "."],
        np.nan
    )


def find_pht(pht):

    search_roots = [
        ROOT / "data/longitudinal_20260924",
        ROOT / "downloads",
    ]

    hits = []

    for sr in search_roots:
        if sr.exists():
            hits.extend(
                sorted(
                    sr.glob(
                        f"**/*{pht}*.HMB-IRB-MDS.txt.gz"
                    )
                )
            )

    hits = [
        x for x in hits
        if "data_dict" not in x.name
        and "variable_report" not in x.name
    ]

    return hits[0] if hits else None


def find_dict_or_report(pht):
    hits = sorted(
        DATA.glob(f"**/*{pht}*")
    )

    return [
        x for x in hits
        if (
            "data_dict" in x.name.lower()
            or "variable_report" in x.name.lower()
        )
    ]


def read_pht(pht):
    f = find_pht(pht)

    if f is None:
        return None

    print(f"\nREAD {pht}: {f}")

    d = pd.read_csv(
        f,
        sep="\t",
        dtype=str,
        low_memory=False,
        comment="#"
    )

    d.columns = [c.strip() for c in d.columns]

    if "shareid" in d.columns:
        d["shareid"] = norm_id(d["shareid"])

    return d


def numeric(s):
    return pd.to_numeric(s, errors="coerce")


def candidate_columns(df):
    if df is None:
        return pd.DataFrame()

    patterns = {
        "BMI_weight_height": [
            r"\bbmi\b",
            r"body.?mass",
            r"weight",
            r"height",
            r"wgt",
            r"hgt",
        ],
        "blood_pressure": [
            r"systolic",
            r"diastolic",
            r"\bsbp\b",
            r"\bdbp\b",
            r"blood.?pressure",
            r"hypert",
        ],
        "smoking": [
            r"smok",
            r"cig",
            r"tobac",
        ],
        "diabetes_glucose": [
            r"diabet",
            r"glucose",
            r"fasting",
            r"insulin",
        ],
    }

    rows = []

    for c in df.columns:
        cl = c.lower()

        for group, pats in patterns.items():
            if any(re.search(p, cl) for p in pats):
                rows.append({
                    "group": group,
                    "column": c,
                })

    return pd.DataFrame(rows)


def simple_slope(res, apoe4):
    t2 = "mri_time_years:cardiac_z"
    t3 = "mri_time_years:cardiac_z:APOE4_carrier"

    cov = res.cov_params()

    if apoe4 == 0:
        beta = res.params[t2]
        var = cov.loc[t2, t2]
    else:
        beta = res.params[t2] + res.params[t3]
        var = (
            cov.loc[t2, t2]
            + cov.loc[t3, t3]
            + 2 * cov.loc[t2, t3]
        )

    se = np.sqrt(var)
    p = 2 * norm.sf(abs(beta / se))

    return {
        "beta": beta,
        "SE": se,
        "CI_low": beta - 1.96 * se,
        "CI_high": beta + 1.96 * se,
        "P": p,
    }


# ============================================================
# 1. CHECK EARLY COVARIATE DATASETS
# ============================================================

PHTS = {
    "exam4": "pht000033",
    "exam5": "pht000034",
    "exam6": "pht000035",
    "diabetes": "pht000041",
}

availability = []

candidate_tables = []

loaded = {}

for label, pht in PHTS.items():

    f = find_pht(pht)

    availability.append({
        "dataset": label,
        "pht": pht,
        "present": f is not None,
        "file": str(f) if f else "",
    })

    if f is None:
        print(f"\nMISSING: {label} {pht}")
        loaded[label] = None
        continue

    d = read_pht(pht)
    loaded[label] = d

    cand = candidate_columns(d)

    if len(cand):
        cand.insert(0, "dataset", label)
        cand.insert(1, "pht", pht)
        candidate_tables.append(cand)

availability = pd.DataFrame(availability)

availability.to_csv(
    OUT / "early_covariate_dataset_availability.tsv",
    sep="\t",
    index=False
)

if candidate_tables:
    candidates = pd.concat(
        candidate_tables,
        ignore_index=True
    )
else:
    candidates = pd.DataFrame()

candidates.to_csv(
    OUT / "candidate_early_covariate_columns.tsv",
    sep="\t",
    index=False
)

print("\n" + "=" * 100)
print("DATASET AVAILABILITY")
print("=" * 100)
print(availability.to_string(index=False))

print("\n" + "=" * 100)
print("CANDIDATE EARLY COVARIATE COLUMNS")
print("=" * 100)

if len(candidates):
    print(candidates.to_string(index=False))
else:
    print("No candidate columns found because required datasets are absent.")

# ============================================================
# STOP SAFELY IF CLINIC EXAM DATA ARE NOT AVAILABLE
# ============================================================

if loaded["exam4"] is None or loaded["exam6"] is None:

    print("\n" + "!" * 100)
    print("EARLY CLINIC EXAM DATA ARE NOT YET AVAILABLE LOCALLY.")
    print("Need at minimum:")
    print("  pht000033  Offspring Exam 4 clinic exam")
    print("  pht000035  Offspring Exam 6 clinic exam")
    print("Recommended additionally:")
    print("  pht000034  Offspring Exam 5 clinic exam")
    print("  pht000041  Offspring diabetes status Exams 1-7")
    print()
    print("The script has written an availability report and stops")
    print("rather than substituting later E8 covariates.")
    print("!" * 100)

    raise SystemExit(0)

# ============================================================
# 2. EXPLICIT VARIABLE MAP
#
# IMPORTANT:
# Do NOT guess dbGaP variable names.
#
# After first run, inspect:
# candidate_early_covariate_columns.tsv
#
# Then populate these exact names.
# ============================================================

COVARIATE_MAP = {

    # Exam 4 covariates
    "BMI4": "D443",
    "SBP4_read1": "D192",
    "SBP4_read2": "D288",
    "DBP4_read1": "D193",
    "DBP4_read2": "D289",
    "smoking4": "D093",

    # pht000041 longitudinal diabetes table
    "diabetes4": "CURR_DIAB4",
}

print("\n" + "=" * 100)
print("VARIABLE MAP CHECK")
print("=" * 100)

unset = [
    k for k, v in COVARIATE_MAP.items()
    if v is None
]

if unset:

    print(
        "Exact covariate column names have not yet been assigned:"
    )

    for x in unset:
        print(" ", x)

    print()
    print(
        "This is intentional: dbGaP variable names should be selected "
        "from the actual downloaded data dictionary, not guessed."
    )

    print()
    print(
        "Inspect:"
    )
    print(
        OUT / "candidate_early_covariate_columns.tsv"
    )

    raise SystemExit(0)

# ============================================================
# 3. BUILD EXAM-4 COVARIATE TABLE
# ============================================================

e4 = loaded["exam4"].copy()

need_e4 = [
    "shareid",
    COVARIATE_MAP["BMI4"],
    COVARIATE_MAP["SBP4_read1"],
    COVARIATE_MAP["SBP4_read2"],
    COVARIATE_MAP["DBP4_read1"],
    COVARIATE_MAP["DBP4_read2"],
    COVARIATE_MAP["smoking4"],
]

missing = [
    c for c in need_e4
    if c not in e4.columns
]

if missing:
    raise RuntimeError(
        f"Mapped Exam-4 variables not found: {missing}"
    )

cov = e4[need_e4].copy()

cov["BMI4"] = numeric(cov[COVARIATE_MAP["BMI4"]])

sbp1 = numeric(cov[COVARIATE_MAP["SBP4_read1"]])
sbp2 = numeric(cov[COVARIATE_MAP["SBP4_read2"]])
dbp1 = numeric(cov[COVARIATE_MAP["DBP4_read1"]])
dbp2 = numeric(cov[COVARIATE_MAP["DBP4_read2"]])

cov["SBP4"] = pd.concat([sbp1, sbp2], axis=1).mean(axis=1, skipna=True)
cov["DBP4"] = pd.concat([dbp1, dbp2], axis=1).mean(axis=1, skipna=True)
cov["smoking4"] = numeric(cov[COVARIATE_MAP["smoking4"]])

cov = cov[
    ["shareid", "BMI4", "SBP4", "DBP4", "smoking4"]
].drop_duplicates("shareid")

# ============================================================
# 4. DIABETES
# ============================================================

if loaded["diabetes"] is not None:

    db = loaded["diabetes"].copy()

    diabetes_col = COVARIATE_MAP["diabetes4"]

    if diabetes_col not in db.columns:
        raise RuntimeError(
            f"Mapped diabetes variable not found: {diabetes_col}"
        )

    # If longitudinal diabetes table has EXAM, isolate E4
    exam_cols = [
        c for c in db.columns
        if c.lower() in ["exam", "examnum", "exam_number"]
    ]

    if exam_cols:
        ec = exam_cols[0]
        db[ec] = numeric(db[ec])
        db = db[db[ec] == 4].copy()

    db["diabetes4"] = numeric(
        db[diabetes_col]
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
        validate="one_to_one"
    )

else:

    diabetes_col = COVARIATE_MAP["diabetes4"]

    if diabetes_col not in e4.columns:
        raise RuntimeError(
            "No diabetes dataset and mapped diabetes variable "
            "not present in Exam-4 clinic file."
        )

    temp = e4[
        ["shareid", diabetes_col]
    ].copy()

    temp["diabetes4"] = numeric(
        temp[diabetes_col]
    )

    temp = temp[
        ["shareid", "diabetes4"]
    ].drop_duplicates("shareid")

    cov = cov.merge(
        temp,
        on="shareid",
        how="left"
    )

# ============================================================
# 5. LOAD LONGITUDINAL ANALYSIS DATA
# ============================================================

long = pd.read_csv(
    ANALYSIS_FILE,
    sep="\t",
    dtype={"shareid": str},
    low_memory=False
)

long["shareid"] = norm_id(long["shareid"])

long = long.merge(
    cov,
    on="shareid",
    how="left",
    validate="many_to_one"
)

# ============================================================
# 6. FOCUSED LA -> LATERAL VENTRICLE ANALYSIS
# ============================================================

CARDIAC_BASE = "la_dim"
NCOL = "la_dim_n"
SCOL = "la_dim_slope"
OUTCOME = "Lateralvent"

SUBSETS = {
    "prospective": 0.0,
    "gap_ge2y": 2.0,
}

MODELS = {

    "M0_minimal":
        "brain_z ~ "
        "mri_time_years * cardiac_z * APOE4_carrier "
        "+ age6_c + C(sex)",

    "M1_early_vascular":
        "brain_z ~ "
        "mri_time_years * cardiac_z * APOE4_carrier "
        "+ age6_c + C(sex) "
        "+ BMI4 + SBP4 + smoking4 + diabetes4",

    "M2_early_vascular_plus_DBP":
        "brain_z ~ "
        "mri_time_years * cardiac_z * APOE4_carrier "
        "+ age6_c + C(sex) "
        "+ BMI4 + SBP4 + DBP4 + smoking4 + diabetes4",
}

effect_rows = []
simple_rows = []
qc_rows = []

for subset_name, min_gap in SUBSETS.items():

    base = long[
        (numeric(long[NCOL]) >= 3)
        & long[SCOL].notna()
        & long["APOE4_carrier"].notna()
        & long["age6_c"].notna()
        & long["sex"].notna()
        & (
            numeric(
                long["last_echo_to_first_mri_years"]
            ) >= min_gap
        )
        & long[OUTCOME].notna()
        & long["mri_time_years"].notna()
    ].copy()

    # repeated outcome
    counts = base.groupby("shareid")[OUTCOME].count()
    good = counts[counts >= 2].index

    base = base[
        base["shareid"].isin(good)
    ].copy()

    # standardization exactly as previous analyses
    subj_slopes = (
        base[
            ["shareid", SCOL]
        ]
        .drop_duplicates("shareid")
    )

    cm = subj_slopes[SCOL].mean()
    cs = subj_slopes[SCOL].std()

    base["cardiac_z"] = (
        base[SCOL] - cm
    ) / cs

    bm = base[OUTCOME].mean()
    bs = base[OUTCOME].std()

    base["brain_z"] = (
        base[OUTCOME] - bm
    ) / bs

    for model_name, formula in MODELS.items():

        required = [
            "brain_z",
            "mri_time_years",
            "cardiac_z",
            "APOE4_carrier",
            "age6_c",
            "sex",
        ]

        if model_name != "M0_minimal":
            required += [
                "BMI4",
                "SBP4",
                "smoking4",
                "diabetes4",
            ]

        if model_name == "M2_early_vascular_plus_DBP":
            required += ["DBP4"]

        d = base.dropna(
            subset=required
        ).copy()

        # Make sure repeated MRI survives complete-case filtering
        cc = d.groupby("shareid").size()
        keep = cc[cc >= 2].index
        d = d[
            d["shareid"].isin(keep)
        ].copy()

        nsub = d["shareid"].nunique()

        if nsub < 50:
            continue

        print(
            subset_name,
            model_name,
            "N=", nsub
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            model = smf.mixedlm(
                formula,
                data=d,
                groups=d["shareid"],
                re_formula="1"
            )

            res = model.fit(
                reml=False,
                method="lbfgs",
                maxiter=2000,
                disp=False
            )

            if not res.converged:
                res = model.fit(
                    reml=False,
                    method="powell",
                    maxiter=5000,
                    disp=False
                )

        t2 = "mri_time_years:cardiac_z"
        t3 = (
            "mri_time_years:"
            "cardiac_z:"
            "APOE4_carrier"
        )

        ci = res.conf_int()

        for term, label in [
            (t2, "MRI_time_x_LA"),
            (t3, "MRI_time_x_LA_x_APOE4"),
        ]:

            effect_rows.append({
                "subset": subset_name,
                "model": model_name,
                "term": label,
                "beta": res.params[term],
                "SE": res.bse[term],
                "CI_low": ci.loc[term, 0],
                "CI_high": ci.loc[term, 1],
                "P": res.pvalues[term],
                "N_subjects": nsub,
                "N_observations": len(d),
                "converged": res.converged,
            })

        for apoe in [0, 1]:

            ss = simple_slope(
                res,
                apoe
            )

            simple_rows.append({
                "subset": subset_name,
                "model": model_name,
                "APOE4_carrier": apoe,
                "APOE_group":
                    "APOE4 carrier"
                    if apoe == 1
                    else "APOE4 noncarrier",
                **ss,
                "N_subjects": nsub,
            })

        qc_rows.append({
            "subset": subset_name,
            "model": model_name,
            "N_subjects": nsub,
            "N_observations": len(d),
            "N_with_BMI4":
                d["BMI4"].notna().sum()
                if "BMI4" in d
                else np.nan,
            "N_with_SBP4":
                d["SBP4"].notna().sum()
                if "SBP4" in d
                else np.nan,
            "N_with_smoking4":
                d["smoking4"].notna().sum()
                if "smoking4" in d
                else np.nan,
            "N_with_diabetes4":
                d["diabetes4"].notna().sum()
                if "diabetes4" in d
                else np.nan,
        })

# ============================================================
# 7. SAVE
# ============================================================

effects = pd.DataFrame(effect_rows)
simple = pd.DataFrame(simple_rows)
qc = pd.DataFrame(qc_rows)

effects.to_csv(
    OUT / "LA_ventricle_early_vascular_adjustment.tsv",
    sep="\t",
    index=False
)

simple.to_csv(
    OUT / "LA_ventricle_early_vascular_simple_slopes.tsv",
    sep="\t",
    index=False
)

qc.to_csv(
    OUT / "LA_ventricle_early_vascular_QC.tsv",
    sep="\t",
    index=False
)

# ============================================================
# 8. EFFECT ATTENUATION
# ============================================================

focus = effects[
    effects["term"]
    ==
    "MRI_time_x_LA_x_APOE4"
].copy()

minimal = (
    focus[
        focus["model"] == "M0_minimal"
    ][
        ["subset", "beta"]
    ]
    .rename(
        columns={
            "beta": "beta_minimal"
        }
    )
)

focus = focus.merge(
    minimal,
    on="subset",
    how="left"
)

focus["percent_change_abs_beta"] = (
    (
        np.abs(focus["beta"])
        - np.abs(focus["beta_minimal"])
    )
    /
    np.abs(focus["beta_minimal"])
    * 100
)

focus.to_csv(
    OUT / "LA_APOE4_interaction_attenuation.tsv",
    sep="\t",
    index=False
)

print("\n" + "=" * 100)
print("LA x MRI TIME x APOE4 AFTER EARLY VASCULAR ADJUSTMENT")
print("=" * 100)

print(
    focus[
        [
            "subset",
            "model",
            "beta",
            "SE",
            "CI_low",
            "CI_high",
            "P",
            "N_subjects",
            "percent_change_abs_beta",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.6g}"
    )
)

print("\nSaved to:")
print(OUT)
