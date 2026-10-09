#!/usr/bin/env python3

from pathlib import Path
import gzip
import io
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

ROOT = Path("/data/qiallab/Framingham")

BASEFILE = (
    ROOT / "results" /
    "neurocardiac_metadata_analysis_ready_metabolic.csv"
)

STRAINFILE = (
    ROOT / "data" / "cardiac_cmr" /
    "phs000007.v35.pht015153.v1.p16.c1."
    "t_mrcvstr_2006_1b_1397s.HMB-IRB-MDS.txt.gz"
)

OUTDIR = ROOT / "results" / "LV_RV_3D_strain"
OUTDIR.mkdir(parents=True, exist_ok=True)

MERGEDFILE = OUTDIR / "neurocardiac_with_LV_RV_3D_strain.csv"

# ============================================================
# 1. READ dbGaP STRAIN TABLE
# ============================================================

print("=" * 110)
print("READING VENTRICULAR STRAIN TABLE")
print("=" * 110)

with gzip.open(STRAINFILE, "rt", errors="replace") as fh:
    lines = fh.readlines()

header_idx = None

for i, line in enumerate(lines[:100]):
    low = line.lower()
    if "shareid" in low and "rv_3d_long" in low:
        header_idx = i
        break

if header_idx is None:
    # fallback: shareid + LV_3D_LONG
    for i, line in enumerate(lines[:100]):
        low = line.lower()
        if "shareid" in low and "lv_3d_long" in low:
            header_idx = i
            break

if header_idx is None:
    raise RuntimeError("Could not identify ventricular-strain header.")

print("Header line:", header_idx)
print(lines[header_idx].rstrip())

strain = pd.read_csv(
    io.StringIO("".join(lines[header_idx:])),
    sep="\t",
    low_memory=False
)

print("\nRaw strain shape:", strain.shape)

# ============================================================
# 2. SELECT MATCHED 3D LV/RV MEASURES
# ============================================================

strain_metrics = [
    "LV_3D_LONG",
    "RV_3D_LONG",
    "LV_3D_CIRC",
    "RV_3D_CIRC",
    "LV_3D_RAD",
    "RV_3D_RAD",
]

needed = ["shareid"] + strain_metrics

missing = [c for c in needed if c not in strain.columns]

if missing:
    raise RuntimeError(f"Missing strain variables: {missing}")

s = strain[needed].copy()

s["shareid"] = (
    s["shareid"]
    .astype(str)
    .str.strip()
    .str.replace(r"\.0$", "", regex=True)
)

print("\nDuplicate strain shareids:",
      s["shareid"].duplicated().sum())

if s["shareid"].duplicated().any():
    raise RuntimeError("Duplicate shareids in strain table.")

# ============================================================
# 3. QC RAW 3D STRAIN
# ============================================================

print("\n")
print("=" * 110)
print("RAW 3D STRAIN QC")
print("=" * 110)

for c in strain_metrics:
    x = pd.to_numeric(s[c], errors="coerce")

    print(
        f"{c:15s} "
        f"N={x.notna().sum():5d} "
        f"mean={x.mean():8.3f} "
        f"sd={x.std():8.3f} "
        f"min={x.min():8.3f} "
        f"max={x.max():8.3f}"
    )

# ============================================================
# 4. MERGE INTO NEUROCARDIAC COHORT
# ============================================================

df = pd.read_csv(BASEFILE)

df["shareid"] = (
    df["shareid"]
    .astype(str)
    .str.strip()
    .str.replace(r"\.0$", "", regex=True)
)

# Drop pre-existing versions so we know the values come from
# this authoritative raw ventricular-strain table.
for c in strain_metrics:
    if c in df.columns:
        df = df.drop(columns=[c])

merged = df.merge(
    s,
    on="shareid",
    how="left",
    validate="one_to_one"
)

print("\n")
print("=" * 110)
print("OVERLAP WITH NEUROCARDIAC N=785")
print("=" * 110)

for c in strain_metrics:
    x = pd.to_numeric(merged[c], errors="coerce")

    print(
        f"{c:15s} "
        f"N={x.notna().sum():4d} "
        f"({100*x.notna().mean():5.1f}%) "
        f"mean={x.mean():8.3f} "
        f"sd={x.std():8.3f}"
    )

merged.to_csv(MERGEDFILE, index=False)

print("\nSaved merged file:")
print(MERGEDFILE)

# ============================================================
# 5. BUILD MODEL COVARIATES
# ============================================================

def diabetes_binary(x):
    x = pd.to_numeric(x, errors="coerce")

    out = pd.Series(
        np.nan,
        index=x.index,
        dtype=float
    )

    out.loc[x == 0] = 0
    out.loc[x.isin([1, 2])] = 1

    return out


merged["_sex"] = pd.to_numeric(
    merged["sex_clinical"],
    errors="coerce"
)

merged["diabetes_history_any"] = diabetes_binary(
    merged["diabetes_history_nearest_exam"]
)

# Total cortical surface area
if (
    "lh_WhiteSurfArea_DesKil_area" in merged.columns
    and
    "rh_WhiteSurfArea_DesKil_area" in merged.columns
):
    merged["total_cortical_area"] = (
        pd.to_numeric(
            merged["lh_WhiteSurfArea_DesKil_area"],
            errors="coerce"
        )
        +
        pd.to_numeric(
            merged["rh_WhiteSurfArea_DesKil_area"],
            errors="coerce"
        )
    )
else:
    merged["total_cortical_area"] = np.nan


def zscore(x):
    x = pd.to_numeric(x, errors="coerce")
    return (x - x.mean()) / x.std(ddof=0)

# ============================================================
# 6. IDENTIFY BRAIN OUTCOMES
# ============================================================

brain_cols = []

for c in merged.columns:

    if not (
        c.startswith("lh_")
        or c.startswith("rh_")
        or c.startswith("Left_")
        or c.startswith("Right_")
    ):
        continue

    if any(
        c.endswith(suf)
        for suf in [
            "_vol",
            "_area",
            "_tk",
            "_tksd",
            "_mcv",
        ]
    ):
        brain_cols.append(c)

brain_cols = sorted(set(brain_cols))

print("\nBrain outcomes:", len(brain_cols))

# ============================================================
# 7. MAIN EFFECT MODEL
# ============================================================

def fit_main(outcome, cardiac):

    req = [
        outcome,
        cardiac,
        "age_at_mri",
        "_sex",
        "abs_delta_years",
        "BMI_nearest_exam",
        "SBP_nearest_exam",
        "current_smoker_nearest_exam",
        "diabetes_history_any",
    ]

    if outcome.endswith("_area"):
        req.append("total_cortical_area")

    d = merged[req].copy().dropna()

    if len(d) < 100:
        return None

    d["y_z"] = zscore(d[outcome])
    d["heart_z"] = zscore(d[cardiac])
    d["age_z"] = zscore(d["age_at_mri"])
    d["interval_z"] = zscore(d["abs_delta_years"])
    d["BMI_z"] = zscore(d["BMI_nearest_exam"])
    d["SBP_z"] = zscore(d["SBP_nearest_exam"])

    Xvars = [
        "heart_z",
        "age_z",
        "_sex",
        "interval_z",
        "BMI_z",
        "SBP_z",
        "current_smoker_nearest_exam",
        "diabetes_history_any",
    ]

    if outcome.endswith("_area"):
        d["total_area_z"] = zscore(
            d["total_cortical_area"]
        )
        Xvars.append("total_area_z")

    X = sm.add_constant(
        d[Xvars].astype(float),
        has_constant="add"
    )

    fit = sm.OLS(
        d["y_z"].astype(float),
        X
    ).fit(cov_type="HC3")

    ci = fit.conf_int().loc["heart_z"]

    return {
        "region": outcome,
        "cardiac": cardiac,
        "N": len(d),

        "beta_cardiac": fit.params["heart_z"],
        "SE_cardiac": fit.bse["heart_z"],
        "CI_cardiac_low": ci.iloc[0],
        "CI_cardiac_high": ci.iloc[1],
        "p_cardiac": fit.pvalues["heart_z"],

        "R2": fit.rsquared,
    }

# ============================================================
# 8. APOE4 INTERACTION MODEL
# ============================================================

def fit_interaction(outcome, cardiac):

    req = [
        outcome,
        cardiac,
        "age_at_mri",
        "_sex",
        "abs_delta_years",
        "BMI_nearest_exam",
        "SBP_nearest_exam",
        "current_smoker_nearest_exam",
        "diabetes_history_any",
        "APOE4_carrier",
    ]

    if outcome.endswith("_area"):
        req.append("total_cortical_area")

    d = merged[req].copy().dropna()

    if len(d) < 100:
        return None

    d["y_z"] = zscore(d[outcome])
    d["heart_z"] = zscore(d[cardiac])
    d["age_z"] = zscore(d["age_at_mri"])
    d["interval_z"] = zscore(d["abs_delta_years"])
    d["BMI_z"] = zscore(d["BMI_nearest_exam"])
    d["SBP_z"] = zscore(d["SBP_nearest_exam"])

    d["heart_x_APOE4"] = (
        d["heart_z"] *
        pd.to_numeric(
            d["APOE4_carrier"],
            errors="coerce"
        )
    )

    Xvars = [
        "heart_z",
        "age_z",
        "_sex",
        "interval_z",
        "BMI_z",
        "SBP_z",
        "current_smoker_nearest_exam",
        "diabetes_history_any",
        "APOE4_carrier",
        "heart_x_APOE4",
    ]

    if outcome.endswith("_area"):
        d["total_area_z"] = zscore(
            d["total_cortical_area"]
        )
        Xvars.append("total_area_z")

    X = sm.add_constant(
        d[Xvars].astype(float),
        has_constant="add"
    )

    fit = sm.OLS(
        d["y_z"].astype(float),
        X
    ).fit(cov_type="HC3")

    b0 = fit.params["heart_z"]
    bi = fit.params["heart_x_APOE4"]

    ci = fit.conf_int().loc["heart_x_APOE4"]

    # Correct APOE4-carrier slope SE using covariance
    cov = fit.cov_params()

    var_carrier = (
        cov.loc["heart_z", "heart_z"]
        +
        cov.loc["heart_x_APOE4", "heart_x_APOE4"]
        +
        2 * cov.loc["heart_z", "heart_x_APOE4"]
    )

    se_carrier = np.sqrt(var_carrier)

    carrier_beta = b0 + bi

    carrier_ci_low = carrier_beta - 1.96 * se_carrier
    carrier_ci_high = carrier_beta + 1.96 * se_carrier

    return {
        "region": outcome,
        "cardiac": cardiac,
        "N": len(d),

        "N_APOE4_noncarrier":
            int((d["APOE4_carrier"] == 0).sum()),

        "N_APOE4_carrier":
            int((d["APOE4_carrier"] == 1).sum()),

        "beta_noncarrier": b0,
        "SE_noncarrier": fit.bse["heart_z"],

        "beta_interaction": bi,
        "SE_interaction":
            fit.bse["heart_x_APOE4"],

        "CI_interaction_low": ci.iloc[0],
        "CI_interaction_high": ci.iloc[1],

        "p_interaction":
            fit.pvalues["heart_x_APOE4"],

        "beta_carrier": carrier_beta,
        "SE_carrier": se_carrier,
        "CI_carrier_low": carrier_ci_low,
        "CI_carrier_high": carrier_ci_high,

        "R2": fit.rsquared,
    }

# ============================================================
# 9. RUN MODELS
# ============================================================

main_rows = []
int_rows = []

for cardiac in strain_metrics:

    print("\n" + "=" * 100)
    print("CARDIAC METRIC:", cardiac)
    print("=" * 100)

    for i, region in enumerate(brain_cols, 1):

        r1 = fit_main(region, cardiac)

        if r1 is not None:
            main_rows.append(r1)

        r2 = fit_interaction(region, cardiac)

        if r2 is not None:
            int_rows.append(r2)

        if i % 100 == 0:
            print("Processed", i)

main = pd.DataFrame(main_rows)
inter = pd.DataFrame(int_rows)

# ============================================================
# 10. FDR WITHIN EACH CARDIAC PHENOTYPE
# ============================================================

main["q_cardiac"] = np.nan

for metric, idx in main.groupby("cardiac").groups.items():

    main.loc[idx, "q_cardiac"] = multipletests(
        main.loc[idx, "p_cardiac"].values,
        method="fdr_bh"
    )[1]


inter["q_interaction"] = np.nan

for metric, idx in inter.groupby("cardiac").groups.items():

    inter.loc[idx, "q_interaction"] = multipletests(
        inter.loc[idx, "p_interaction"].values,
        method="fdr_bh"
    )[1]

# ============================================================
# 11. SAVE RESULTS
# ============================================================

main_all = OUTDIR / "LV_RV_3D_main_effects_all.csv"
main_sig = OUTDIR / "LV_RV_3D_main_effects_FDR.csv"

int_all = OUTDIR / "LV_RV_3D_APOE4_interactions_all.csv"
int_sig = OUTDIR / "LV_RV_3D_APOE4_interactions_FDR.csv"

main.to_csv(main_all, index=False)

main_fdr = main[
    main["q_cardiac"] < 0.05
].sort_values(
    ["cardiac", "q_cardiac"]
)

main_fdr.to_csv(main_sig, index=False)

inter.to_csv(int_all, index=False)

int_fdr = inter[
    inter["q_interaction"] < 0.05
].sort_values(
    ["cardiac", "q_interaction"]
)

int_fdr.to_csv(int_sig, index=False)

# ============================================================
# 12. PRINT SIGNIFICANT RESULTS
# ============================================================

print("\n")
print("=" * 110)
print("FDR-SIGNIFICANT MAIN EFFECTS")
print("=" * 110)

if len(main_fdr):

    print(
        main_fdr[
            [
                "region",
                "cardiac",
                "N",
                "beta_cardiac",
                "CI_cardiac_low",
                "CI_cardiac_high",
                "p_cardiac",
                "q_cardiac",
            ]
        ].to_string(index=False)
    )

else:
    print("None at q < 0.05")


print("\n")
print("=" * 110)
print("FDR-SIGNIFICANT APOE4 INTERACTIONS")
print("=" * 110)

if len(int_fdr):

    print(
        int_fdr[
            [
                "region",
                "cardiac",
                "N",
                "N_APOE4_carrier",
                "beta_noncarrier",
                "beta_interaction",
                "beta_carrier",
                "CI_carrier_low",
                "CI_carrier_high",
                "CI_interaction_low",
                "CI_interaction_high",
                "p_interaction",
                "q_interaction",
            ]
        ].to_string(index=False)
    )

else:
    print("None at q < 0.05")


print("\nSaved:")
print(main_all)
print(main_sig)
print(int_all)
print(int_sig)

# ============================================================
# 13. SAVE QC + TOP NOMINAL RESULTS + RUN SUMMARY
# ============================================================

print("\n")
print("=" * 110)
print("SAVING QC AND NOMINAL RESULT SUMMARIES")
print("=" * 110)

# ------------------------------------------------------------
# Raw strain QC
# ------------------------------------------------------------

qc_rows = []

for c in strain_metrics:

    x_raw = pd.to_numeric(
        s[c],
        errors="coerce"
    )

    x_overlap = pd.to_numeric(
        merged[c],
        errors="coerce"
    )

    qc_rows.append({
        "cardiac": c,

        "N_raw": int(x_raw.notna().sum()),
        "mean_raw": x_raw.mean(),
        "sd_raw": x_raw.std(),
        "min_raw": x_raw.min(),
        "max_raw": x_raw.max(),

        "N_neurocardiac_overlap": int(x_overlap.notna().sum()),
        "pct_neurocardiac_overlap":
            100 * x_overlap.notna().mean(),

        "mean_overlap": x_overlap.mean(),
        "sd_overlap": x_overlap.std(),
        "min_overlap": x_overlap.min(),
        "max_overlap": x_overlap.max(),
    })

qc = pd.DataFrame(qc_rows)

qcfile = OUTDIR / "LV_RV_3D_strain_QC.csv"
qc.to_csv(qcfile, index=False)


# ------------------------------------------------------------
# Top 10 main effects per cardiac metric
# ------------------------------------------------------------

top_main = (
    main
    .sort_values(["cardiac", "p_cardiac"])
    .groupby("cardiac", group_keys=False)
    .head(10)
    .copy()
)

top_main_file = (
    OUTDIR /
    "LV_RV_3D_top10_main_effects_per_metric.csv"
)

top_main.to_csv(
    top_main_file,
    index=False
)


# ------------------------------------------------------------
# Top 10 APOE4 interactions per cardiac metric
# ------------------------------------------------------------

top_inter = (
    inter
    .sort_values(["cardiac", "p_interaction"])
    .groupby("cardiac", group_keys=False)
    .head(10)
    .copy()
)

top_inter_file = (
    OUTDIR /
    "LV_RV_3D_top10_APOE4_interactions_per_metric.csv"
)

top_inter.to_csv(
    top_inter_file,
    index=False
)


# ------------------------------------------------------------
# Nominal p<0.01 main effects
# ------------------------------------------------------------

nominal_main = (
    main[
        main["p_cardiac"] < 0.01
    ]
    .sort_values(
        ["cardiac", "p_cardiac"]
    )
    .copy()
)

nominal_main_file = (
    OUTDIR /
    "LV_RV_3D_main_effects_nominal_p001.csv"
)

nominal_main.to_csv(
    nominal_main_file,
    index=False
)


# ------------------------------------------------------------
# Nominal p<0.01 APOE4 interactions
# ------------------------------------------------------------

nominal_inter = (
    inter[
        inter["p_interaction"] < 0.01
    ]
    .sort_values(
        ["cardiac", "p_interaction"]
    )
    .copy()
)

nominal_inter_file = (
    OUTDIR /
    "LV_RV_3D_APOE4_interactions_nominal_p001.csv"
)

nominal_inter.to_csv(
    nominal_inter_file,
    index=False
)


# ------------------------------------------------------------
# Compact per-metric summary
# ------------------------------------------------------------

summary_rows = []

for metric in strain_metrics:

    mm = main[
        main["cardiac"] == metric
    ].copy()

    ii = inter[
        inter["cardiac"] == metric
    ].copy()

    best_main = (
        mm.sort_values("p_cardiac").iloc[0]
        if len(mm)
        else None
    )

    best_int = (
        ii.sort_values("p_interaction").iloc[0]
        if len(ii)
        else None
    )

    summary_rows.append({
        "cardiac": metric,

        "N_main_typical":
            int(mm["N"].median())
            if len(mm) else np.nan,

        "N_interaction_typical":
            int(ii["N"].median())
            if len(ii) else np.nan,

        "N_FDR_main_q05":
            int((mm["q_cardiac"] < 0.05).sum()),

        "N_FDR_interaction_q05":
            int(
                (
                    ii["q_interaction"] < 0.05
                ).sum()
            ),

        "N_nominal_main_p001":
            int(
                (
                    mm["p_cardiac"] < 0.01
                ).sum()
            ),

        "N_nominal_interaction_p001":
            int(
                (
                    ii["p_interaction"] < 0.01
                ).sum()
            ),

        "best_main_region":
            best_main["region"]
            if best_main is not None
            else np.nan,

        "best_main_beta":
            best_main["beta_cardiac"]
            if best_main is not None
            else np.nan,

        "best_main_p":
            best_main["p_cardiac"]
            if best_main is not None
            else np.nan,

        "best_main_q":
            best_main["q_cardiac"]
            if best_main is not None
            else np.nan,

        "best_interaction_region":
            best_int["region"]
            if best_int is not None
            else np.nan,

        "best_interaction_beta":
            best_int["beta_interaction"]
            if best_int is not None
            else np.nan,

        "best_interaction_p":
            best_int["p_interaction"]
            if best_int is not None
            else np.nan,

        "best_interaction_q":
            best_int["q_interaction"]
            if best_int is not None
            else np.nan,
    })

summary = pd.DataFrame(summary_rows)

summary_file = (
    OUTDIR /
    "LV_RV_3D_analysis_summary.csv"
)

summary.to_csv(
    summary_file,
    index=False
)


# ------------------------------------------------------------
# Human-readable run summary
# ------------------------------------------------------------

txtfile = (
    OUTDIR /
    "LV_RV_3D_analysis_summary.txt"
)

with open(txtfile, "w") as f:

    f.write(
        "LV/RV 3D STRAIN ANALYSIS\n"
        "========================\n\n"
    )

    f.write(
        f"Base neurocardiac cohort N = "
        f"{len(merged)}\n"
    )

    f.write(
        f"Brain outcomes tested = "
        f"{len(brain_cols)}\n\n"
    )

    f.write(
        "Primary cardiac metrics:\n"
    )

    for metric in strain_metrics:
        f.write(f"  {metric}\n")

    f.write("\n")

    f.write(
        "Inference:\n"
        "  OLS with HC3 robust covariance\n"
        "  Brain outcomes standardized within analytic subset\n"
        "  Cardiac metrics standardized within analytic subset\n"
        "  Covariates: age, sex, absolute CMR-MRI interval, "
        "BMI, SBP, current smoking, diabetes\n"
        "  Cortical-area outcomes additionally adjusted for "
        "total cortical surface area\n"
        "  APOE4 interaction models additionally include "
        "APOE4 carrier status and cardiac x APOE4\n\n"
    )

    f.write(
        "Multiple-comparison correction:\n"
        "  Benjamini-Hochberg FDR separately within each "
        "cardiac phenotype across brain outcomes.\n\n"
    )

    f.write(
        "Overall result:\n"
        f"  FDR-significant main effects: "
        f"{int((main['q_cardiac'] < 0.05).sum())}\n"
    )

    f.write(
        f"  FDR-significant APOE4 interactions: "
        f"{int((inter['q_interaction'] < 0.05).sum())}\n\n"
    )

    f.write(
        "Per-metric summary:\n"
    )

    f.write(
        summary.to_string(index=False)
    )

    f.write("\n")


print("\nAdditional files saved:")

for p in [
    qcfile,
    top_main_file,
    top_inter_file,
    nominal_main_file,
    nominal_inter_file,
    summary_file,
    txtfile,
]:
    print(p)

