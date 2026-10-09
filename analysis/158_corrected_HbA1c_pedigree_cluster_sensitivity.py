#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path("/data/qiallab/Framingham")
RES = ROOT / "results"
RAW = ROOT / "downloads" / "clinical"

INPUT = RES / "neurocardiac_metadata_analysis_ready_metabolic.csv"

OUTDIR = RES / "kinship_sensitivity_corrected"
OUTDIR.mkdir(parents=True, exist_ok=True)

OUTCSV = OUTDIR / "LVEDVi_APOE4_HbA1c_M3_corrected_pedigree_sensitivity.csv"

print("=" * 110)
print("158: CORRECTED PEDIGREE-FAMILY CLUSTER SENSITIVITY")
print("LVEDVi ~ APOE4 × HbA1c — EXACT M3_VASCULAR MODEL")
print("=" * 110)


# ============================================================================
# LOAD
# ============================================================================

df = pd.read_csv(INPUT)

ped_files = sorted(
    RAW.glob("*pht000183*.txt.gz")
)

if not ped_files:
    raise RuntimeError("Pedigree file not found.")

PED = ped_files[0]

ped = pd.read_csv(
    PED,
    sep="\t",
    compression="gzip",
    comment="#",
    low_memory=False,
)

print("\nInput:", INPUT)
print("Pedigree:", PED)
print("Input shape:", df.shape)


# ============================================================================
# PEDIGREE MERGE
# ============================================================================

def norm_id(s):
    s = (
        s.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )

    return s.replace({
        "nan": np.nan,
        "None": np.nan,
        "": np.nan,
    })


df["join_id"] = norm_id(
    df["shareid"]
)

ped["join_id"] = norm_id(
    ped["shareid"]
)

ped = (
    ped[
        [
            "join_id",
            "pedno",
            "fshare",
            "mshare",
            "twinid",
            "idtype",
        ]
    ]
    .drop_duplicates("join_id")
)

df = df.merge(
    ped,
    on="join_id",
    how="left",
)

print(
    "Participants with pedno:",
    df["pedno"].notna().sum()
)
print(
    "Unique pedigrees:",
    df["pedno"].nunique()
)


# ============================================================================
# EXACT VARIABLES FROM SCRIPT 56
# ============================================================================

APOE = "APOE4_carrier"
HBA1C = "hba1c_exam7"

AGE = "age_at_cmr"
SEX = "sex_clinical"

BMI = "BMI_nearest_exam"
SBP = "SBP_nearest_exam"
HTNMED = "htn_med_nearest_exam"

HBA1C_CMR_INTERVAL = "hba1c_exam7_to_cmr_years"


for c in [
    "LVEDV",
    "height_in",
    "weight_lb",
    APOE,
    HBA1C,
    AGE,
    BMI,
    SBP,
    HTNMED,
    HBA1C_CMR_INTERVAL,
]:
    if c not in df.columns:
        raise RuntimeError(
            f"Missing required variable: {c}"
        )

    df[c] = pd.to_numeric(
        df[c],
        errors="coerce",
    )


# ============================================================================
# EXACT LVEDVi DERIVATION
# ============================================================================

df["height_cm"] = (
    df["height_in"] * 2.54
)

df["weight_kg"] = (
    df["weight_lb"] * 0.45359237
)

df["BSA"] = np.sqrt(
    df["height_cm"]
    * df["weight_kg"]
    / 3600.0
)

df["LVEDVi"] = (
    df["LVEDV"]
    / df["BSA"]
)


# ============================================================================
# EXACT STANDARDIZATION FROM SCRIPT 56
#
# IMPORTANT:
# Script 56 standardized these variables in the full dataframe BEFORE
# model-specific complete-case restriction. We reproduce that exactly.
# ============================================================================

def zscore(s):
    s = pd.to_numeric(
        s,
        errors="coerce",
    )

    return (
        (s - s.mean())
        / s.std(ddof=0)
    )


df["HbA1c_z"] = zscore(
    df[HBA1C]
)

df["BMI_z"] = zscore(
    df[BMI]
)

df["SBP_z"] = zscore(
    df[SBP]
)

df["HbA1c_CMR_interval_z"] = zscore(
    df[HBA1C_CMR_INTERVAL]
)


# ============================================================================
# EXACT M3_VASCULAR FORMULA
# ============================================================================

formula = (
    f"LVEDVi ~ {APOE} * HbA1c_z"
    f" + {AGE}"
    f" + C({SEX})"
    f" + HbA1c_CMR_interval_z"
    f" + BMI_z"
    f" + SBP_z"
    f" + {HTNMED}"
)

vars_needed = [
    "LVEDVi",
    APOE,
    "HbA1c_z",
    AGE,
    SEX,
    "HbA1c_CMR_interval_z",
    "BMI_z",
    "SBP_z",
    HTNMED,
]

int_term = f"{APOE}:HbA1c_z"


# ============================================================================
# FITTER
# ============================================================================

def fit_one(d, label, covariance):

    if covariance == "HC3":

        fit = smf.ols(
            formula,
            data=d,
        ).fit(
            cov_type="HC3"
        )

    elif covariance == "PEDIGREE_CLUSTER":

        fit = smf.ols(
            formula,
            data=d,
        ).fit(
            cov_type="cluster",
            cov_kwds={
                "groups":
                    d["pedno"],
            },
        )

    else:
        raise ValueError(covariance)

    term = int_term

    if term not in fit.params.index:
        term = f"HbA1c_z:{APOE}"

    ci = fit.conf_int().loc[
        term
    ]

    return {
        "analysis":
            label,

        "N":
            int(fit.nobs),

        "N_APOE4_noncarrier":
            int(
                (d[APOE] == 0).sum()
            ),

        "N_APOE4_carrier":
            int(
                (d[APOE] == 1).sum()
            ),

        "N_families":
            (
                int(d["pedno"].nunique())
                if "pedno" in d.columns
                else np.nan
            ),

        "beta_interaction":
            float(
                fit.params[term]
            ),

        "SE":
            float(
                fit.bse[term]
            ),

        "CI_low":
            float(
                ci.iloc[0]
            ),

        "CI_high":
            float(
                ci.iloc[1]
            ),

        "P":
            float(
                fit.pvalues[term]
            ),

        "R2":
            float(
                fit.rsquared
            ),

        "covariance":
            covariance,
    }


# ============================================================================
# 1. FULL REFERENCE M3
#
# Must reproduce:
# N = 650
# beta = -2.887308626
# P = 0.014832943
# ============================================================================

d_full = (
    df[
        vars_needed
        + ["pedno"]
    ]
    .dropna(
        subset=vars_needed
    )
    .copy()
)

print("\n" + "=" * 110)
print("FULL M3_VASCULAR REFERENCE")
print("=" * 110)

ref = fit_one(
    d_full,
    "FULL_M3_VASCULAR_HC3",
    "HC3",
)

print(ref)

if ref["N"] != 650:
    raise RuntimeError(
        f"Reference N mismatch: expected 650, got {ref['N']}"
    )

if abs(
    ref["beta_interaction"]
    - (-2.887308626124754)
) > 1e-6:
    raise RuntimeError(
        "Reference beta does not reproduce script 56."
    )


# ============================================================================
# 2. PEDIGREE-COMPLETE M3 — HC3
# ============================================================================

d_ped = (
    df[
        vars_needed
        + ["pedno"]
    ]
    .dropna(
        subset=
        vars_needed
        + ["pedno"]
    )
    .copy()
)

print("\n" + "=" * 110)
print("PEDIGREE-COMPLETE M3_VASCULAR")
print("=" * 110)

print("N:", len(d_ped))
print(
    "Unique pedigrees:",
    d_ped["pedno"].nunique()
)

fam_sizes = (
    d_ped
    .groupby("pedno")
    .size()
)

print(
    "Families with >1 participant:",
    int(
        (fam_sizes > 1).sum()
    )
)

print(
    "Participants in multi-member families:",
    int(
        fam_sizes[
            fam_sizes > 1
        ].sum()
    )
)

ped_hc3 = fit_one(
    d_ped,
    "PEDIGREE_COMPLETE_M3_HC3",
    "HC3",
)


# ============================================================================
# 3. SAME SAMPLE — PEDIGREE-FAMILY CLUSTER
# ============================================================================

ped_cluster = fit_one(
    d_ped,
    "PEDIGREE_COMPLETE_M3_FAMILY_CLUSTER",
    "PEDIGREE_CLUSTER",
)


if not np.isclose(
    ped_hc3["beta_interaction"],
    ped_cluster["beta_interaction"],
    atol=1e-12,
    rtol=0,
):
    raise RuntimeError(
        "Coefficient differs between HC3 and cluster fits."
    )


# ============================================================================
# SAVE
# ============================================================================

res = pd.DataFrame([
    ref,
    ped_hc3,
    ped_cluster,
])

res.to_csv(
    OUTCSV,
    index=False,
)

print("\n" + "=" * 110)
print("CORRECTED HbA1c PEDIGREE SENSITIVITY")
print("=" * 110)

print(
    res.to_string(
        index=False,
        float_format=lambda x: f"{x:.7g}",
    )
)

print("\nSaved:")
print(OUTCSV)
