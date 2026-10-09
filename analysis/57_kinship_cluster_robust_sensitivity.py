#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path("/data/qiallab/Framingham")
RAW  = ROOT / "downloads" / "clinical"
RES  = ROOT / "results"

MAIN = RES / "neurocardiac_metadata_analysis_ready_metabolic.csv"
ATRIA = RES / "neurocardiac_metadata_with_atria.csv"

OUTDIR = RES / "kinship_sensitivity"
OUTDIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# FIND PEDIGREE DATA FILE
# ============================================================

ped_files = sorted(
    RAW.glob("*pht000183*.txt.gz")
)

if not ped_files:
    raise RuntimeError("Could not find pht000183 pedigree data file.")

PED = ped_files[0]

print("Pedigree file:")
print(PED)

# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(MAIN)

ped = pd.read_csv(
    PED,
    sep="\t",
    compression="gzip",
    comment="#",
    low_memory=False
)

print("\nMain:", df.shape)
print("Pedigree:", ped.shape)

# Normalize IDs
def norm_id(s):
    s = s.astype(str).str.strip()
    s = s.str.replace(r"\.0$", "", regex=True)
    return s.replace({
        "nan": np.nan,
        "None": np.nan,
        "": np.nan
    })

df["join_id"] = norm_id(df["shareid"])
ped["join_id"] = norm_id(ped["shareid"])

ped = ped[
    ["join_id", "pedno", "fshare", "mshare", "twinid", "idtype"]
].drop_duplicates("join_id")

df = df.merge(
    ped,
    on="join_id",
    how="left"
)

print("\nAfter pedigree merge:", df.shape)
print("Subjects with pedno:", df["pedno"].notna().sum())
print("Unique pedigrees:", df["pedno"].nunique())

fam_sizes = (
    df.dropna(subset=["pedno"])
      .groupby("pedno")
      .size()
)

print("\nFamily-size distribution:")
print(fam_sizes.describe())

print("\nFamilies with >1 analyzed participant:",
      (fam_sizes > 1).sum())

print("Participants in families with >1 analyzed participant:",
      fam_sizes[fam_sizes > 1].sum())

# ============================================================
# MERGE ATRIAL OUTCOME IF NEEDED
# ============================================================

if ATRIA.exists():
    atr = pd.read_csv(ATRIA)

    if "LAVI_max" in atr.columns:
        atr["join_id"] = norm_id(atr["shareid"])
        atr = atr[
            ["join_id", "LAVI_max"]
        ].drop_duplicates("join_id")

        if "LAVI_max" not in df.columns:
            df = df.merge(
                atr,
                on="join_id",
                how="left"
            )

# ============================================================
# DERIVE INDEXED LV VARIABLES
# ============================================================

for c in [
    "LVEDV", "LVESV", "LV_MASS",
    "height_in", "weight_lb",
    "APOE4_carrier",
    "hba1c_exam7",
    "age_at_mri",
    "age_at_cmr"
]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

height_cm = df["height_in"] * 2.54
weight_kg = df["weight_lb"] * 0.45359237

df["BSA"] = np.sqrt(
    height_cm * weight_kg / 3600.0
)

df["LVEDVi"] = df["LVEDV"] / df["BSA"]
df["LVESVi"] = df["LVESV"] / df["BSA"]
df["LV_MASSi"] = df["LV_MASS"] / df["BSA"]

df["HbA1c_z"] = (
    df["hba1c_exam7"] - df["hba1c_exam7"].mean()
) / df["hba1c_exam7"].std(ddof=0)

# ============================================================
# KEY MODELS
# ============================================================

models = [
    # Primary heart-brain/APOE4 models
    (
        "LVEF_middletemporal",
        "lh_middletemporal_vol",
        "LVEF",
        "age_at_mri"
    ),
    (
        "LVESVi_superiorfrontal",
        "rh_superiorfrontal_area",
        "LVESVi",
        "age_at_mri"
    ),
    (
        "LVMASSi_lingual_vol",
        "rh_lingual_vol",
        "LV_MASSi",
        "age_at_mri"
    ),
    (
        "LVMASSi_lingual_area",
        "rh_lingual_area",
        "LV_MASSi",
        "age_at_mri"
    ),
    (
        "LAVI_occipital_tksd",
        "lh_lateraloccipital_tksd",
        "LAVI_max",
        "age_at_mri"
    ),
]

rows = []

for name, outcome, cardiac, agevar in models:

    needed = [
        outcome,
        cardiac,
        "APOE4_carrier",
        agevar,
        "sex_clinical",
        "pedno"
    ]

    d = df[needed].dropna().copy()

    if len(d) < 50:
        print("Skipping", name, "N too small")
        continue

    formula = (
        f"{outcome} ~ "
        f"{cardiac} * APOE4_carrier"
        f" + {agevar}"
        f" + C(sex_clinical)"
    )

    fit = smf.ols(
        formula,
        data=d
    ).fit(
        cov_type="cluster",
        cov_kwds={
            "groups": d["pedno"]
        }
    )

    term = f"{cardiac}:APOE4_carrier"

    if term not in fit.params.index:
        term = f"APOE4_carrier:{cardiac}"

    ci = fit.conf_int().loc[term]

    rows.append({
        "analysis": name,
        "outcome": outcome,
        "predictor": cardiac,
        "N": int(fit.nobs),
        "N_families": d["pedno"].nunique(),
        "beta_interaction": fit.params[term],
        "SE_cluster": fit.bse[term],
        "CI_low": ci.iloc[0],
        "CI_high": ci.iloc[1],
        "p_cluster": fit.pvalues[term],
        "R2": fit.rsquared
    })

# ============================================================
# LVEDVi × APOE4 × HbA1c
# ============================================================

needed = [
    "LVEDVi",
    "HbA1c_z",
    "APOE4_carrier",
    "age_at_cmr",
    "sex_clinical",
    "pedno"
]

d = df[needed].dropna().copy()

fit = smf.ols(
    "LVEDVi ~ APOE4_carrier * HbA1c_z + age_at_cmr + C(sex_clinical)",
    data=d
).fit(
    cov_type="cluster",
    cov_kwds={
        "groups": d["pedno"]
    }
)

term = "APOE4_carrier:HbA1c_z"

if term not in fit.params.index:
    term = "HbA1c_z:APOE4_carrier"

ci = fit.conf_int().loc[term]

rows.append({
    "analysis": "LVEDVi_APOE4_HbA1c",
    "outcome": "LVEDVi",
    "predictor": "HbA1c_z",
    "N": int(fit.nobs),
    "N_families": d["pedno"].nunique(),
    "beta_interaction": fit.params[term],
    "SE_cluster": fit.bse[term],
    "CI_low": ci.iloc[0],
    "CI_high": ci.iloc[1],
    "p_cluster": fit.pvalues[term],
    "R2": fit.rsquared
})

# ============================================================
# SAVE
# ============================================================

res = pd.DataFrame(rows)

outfile = (
    OUTDIR /
    "primary_models_family_cluster_robust.csv"
)

res.to_csv(
    outfile,
    index=False
)

print("\n" + "="*110)
print("FAMILY-CLUSTER ROBUST SENSITIVITY")
print("="*110)

print(
    res.to_string(
        index=False,
        float_format=lambda x: f"{x:.6g}"
    )
)

print("\nSaved:")
print(outfile)
