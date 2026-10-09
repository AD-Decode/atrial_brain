#!/usr/bin/env python3

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

# ============================================================
# PATHS
# ============================================================

ROOT = Path("/data/qiallab/Framingham")

DATAFILE = (
    ROOT / "data"
    / "longitudinal_derived"
    / "E4_E6_repeated_MRI_temporal_analysis.tsv"
)

OLD_FILE = (
    ROOT / "results"
    / "longitudinal_heart_brain"
    / "temporal_prediction"
    / "mixedlm_random_intercept"
    / "PRIMARY_APOE4_temporal_interactions.tsv"
)

OUTDIR = (
    ROOT / "results"
    / "longitudinal_heart_brain"
    / "temporal_prediction"
    / "mixedlm_random_time_slope"
)
OUTDIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# FIXED MODEL FAMILY: 4 CARDIAC x 4 MRI OUTCOMES = 16 TESTS
# ============================================================

CARDIAC = {
    "LV_mass": "lv_mass_derived_g",
    "LA_dimension": "la_dim",
    "LVDD": "lvdd",
    "fractional_shortening": "fs_derived_pct",
}

OUTCOMES = {
    "hippocampus": "Hippo",
    "total_brain": "Total_brain",
    "lateral_ventricles": "Lateralvent",
    "WMH": "WMH",
}

TERM = "mri_time_years:cardiac_z:APOE4_carrier"

FORMULA = (
    "brain_z ~ "
    "mri_time_years * cardiac_z * APOE4_carrier "
    "+ age6_c + C(sex) "
    "+ age6_c:mri_time_years "
    "+ C(sex):mri_time_years"
)

# ============================================================
# LOAD DATA
# ============================================================

long = pd.read_csv(
    DATAFILE,
    sep="\t",
    dtype={"shareid": str},
    low_memory=False,
)

# Allow alternative WMH name if needed
if "WMH" not in long.columns and "FLAIR_wmh" in long.columns:
    OUTCOMES["WMH"] = "FLAIR_wmh"

print("=" * 100)
print("155: RANDOM MRI-TIME SLOPE REFIT — ALL 16 PROSPECTIVE APOE4 MODELS")
print("=" * 100)
print("Data:", DATAFILE)
print("Random effects: intercept + MRI time")
print("Target term:", TERM)
print("Formula:", FORMULA)
print()

# Verify columns before fitting
required_common = [
    "shareid",
    "APOE4_carrier",
    "age6_c",
    "sex",
    "mri_time_years",
    "last_echo_to_first_mri_years",
]

missing = [c for c in required_common if c not in long.columns]
if missing:
    raise RuntimeError(f"Missing required common columns: {missing}")

for cardiac_label, base in CARDIAC.items():
    for c in [f"{base}_n", f"{base}_slope"]:
        if c not in long.columns:
            raise RuntimeError(
                f"Missing cardiac column for {cardiac_label}: {c}"
            )

for outcome_label, col in OUTCOMES.items():
    if col not in long.columns:
        raise RuntimeError(
            f"Missing outcome column for {outcome_label}: {col}"
        )

# ============================================================
# FIT 16 PROSPECTIVE MODELS
# ============================================================

rows = []

for cardiac_label, cardiac_base in CARDIAC.items():

    ncol = f"{cardiac_base}_n"
    scol = f"{cardiac_base}_slope"

    for outcome_label, outcome_col in OUTCOMES.items():

        print("-" * 100)
        print(f"{cardiac_label} -> {outcome_label}")

        # ----------------------------------------------------
        # Reproduce prospective eligibility
        # ----------------------------------------------------

        d = long[
            (pd.to_numeric(long[ncol], errors="coerce") >= 3)
            & long[scol].notna()
            & long["APOE4_carrier"].notna()
            & long["age6_c"].notna()
            & long["sex"].notna()
            & long["mri_time_years"].notna()
            & long[outcome_col].notna()
            & (
                pd.to_numeric(
                    long["last_echo_to_first_mri_years"],
                    errors="coerce",
                ) >= 0
            )
        ].copy()

        # Require >=2 observations for this MRI outcome
        counts = d.groupby("shareid")[outcome_col].count()
        good_ids = counts[counts >= 2].index
        d = d[d["shareid"].isin(good_ids)].copy()

        # ----------------------------------------------------
        # Cardiac trajectory scaling:
        # one slope value per subject
        # ----------------------------------------------------

        subject_slopes = (
            d[["shareid", scol]]
            .drop_duplicates("shareid")
        )

        cardiac_mean = subject_slopes[scol].mean()
        cardiac_sd = subject_slopes[scol].std()

        if (
            not np.isfinite(cardiac_sd)
            or cardiac_sd <= 0
        ):
            raise RuntimeError(
                f"Invalid cardiac SD: {cardiac_label} / {outcome_label}"
            )

        d["cardiac_z"] = (
            d[scol] - cardiac_mean
        ) / cardiac_sd

        # ----------------------------------------------------
        # Outcome scaling exactly as script 142
        # ----------------------------------------------------

        brain_mean = d[outcome_col].mean()
        brain_sd = d[outcome_col].std()

        if (
            not np.isfinite(brain_sd)
            or brain_sd <= 0
        ):
            raise RuntimeError(
                f"Invalid outcome SD: {cardiac_label} / {outcome_label}"
            )

        d["brain_z"] = (
            d[outcome_col] - brain_mean
        ) / brain_sd

        # ----------------------------------------------------
        # RANDOM INTERCEPT + RANDOM MRI-TIME SLOPE
        # ----------------------------------------------------

        fit_method = "lbfgs"
        fit_error = ""
        res = None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            try:
                model = smf.mixedlm(
                    FORMULA,
                    data=d,
                    groups=d["shareid"],
                    re_formula="~mri_time_years",
                )

                res = model.fit(
                    reml=False,
                    method="lbfgs",
                    maxiter=2000,
                    disp=False,
                )

            except Exception as e:
                fit_error = f"lbfgs: {type(e).__name__}: {e}"
                res = None

            if res is None or not getattr(res, "converged", False):

                fit_method = "powell"

                try:
                    model = smf.mixedlm(
                        FORMULA,
                        data=d,
                        groups=d["shareid"],
                        re_formula="~mri_time_years",
                    )

                    res = model.fit(
                        reml=False,
                        method="powell",
                        maxiter=5000,
                        disp=False,
                    )

                except Exception as e:
                    if fit_error:
                        fit_error += " | "
                    fit_error += (
                        f"powell: {type(e).__name__}: {e}"
                    )
                    res = None

        if res is None:
            print("FAILED:", fit_error)

            rows.append({
                "subset": "prospective",
                "cardiac": cardiac_label,
                "outcome": outcome_label,
                "term": TERM,
                "beta": np.nan,
                "SE": np.nan,
                "CI_low": np.nan,
                "CI_high": np.nan,
                "P": np.nan,
                "N_subjects": d["shareid"].nunique(),
                "N_observations": len(d),
                "N_APOE4_carrier": (
                    d.loc[
                        d["APOE4_carrier"] == 1,
                        "shareid"
                    ].nunique()
                ),
                "N_APOE4_noncarrier": (
                    d.loc[
                        d["APOE4_carrier"] == 0,
                        "shareid"
                    ].nunique()
                ),
                "converged": False,
                "optimizer": fit_method,
                "fit_error": fit_error,
            })
            continue

        if TERM not in res.params.index:
            raise RuntimeError(
                f"Target term missing for "
                f"{cardiac_label} / {outcome_label}"
            )

        ci = res.conf_int()

        row = {
            "subset": "prospective",
            "cardiac": cardiac_label,
            "outcome": outcome_label,
            "term": TERM,
            "beta": res.params[TERM],
            "SE": res.bse[TERM],
            "CI_low": ci.loc[TERM, 0],
            "CI_high": ci.loc[TERM, 1],
            "P": res.pvalues[TERM],
            "N_subjects": d["shareid"].nunique(),
            "N_observations": len(d),
            "N_APOE4_carrier": (
                d.loc[
                    d["APOE4_carrier"] == 1,
                    "shareid"
                ].nunique()
            ),
            "N_APOE4_noncarrier": (
                d.loc[
                    d["APOE4_carrier"] == 0,
                    "shareid"
                ].nunique()
            ),
            "converged": bool(
                getattr(res, "converged", False)
            ),
            "optimizer": fit_method,
            "fit_error": fit_error,
        }

        rows.append(row)

        print(
            f"N={row['N_subjects']} "
            f"obs={row['N_observations']} "
            f"beta={row['beta']:.6f} "
            f"SE={row['SE']:.6f} "
            f"P={row['P']:.8g} "
            f"converged={row['converged']} "
            f"optimizer={row['optimizer']}"
        )

# ============================================================
# RESULTS + BH FDR ACROSS EXACTLY 16 TESTS
# ============================================================

results = pd.DataFrame(rows)

if len(results) != 16:
    raise RuntimeError(
        f"Expected exactly 16 models; got {len(results)}"
    )

if results["P"].isna().any():
    print("\nWARNING: at least one model failed.")
    print(results.loc[results["P"].isna()].to_string(index=False))
    raise RuntimeError(
        "Cannot compute the prespecified 16-test BH family "
        "because one or more models failed."
    )

results["FDR_q_random_slope"] = multipletests(
    results["P"].values,
    alpha=0.05,
    method="fdr_bh",
)[1]

results = results.sort_values(
    ["P", "cardiac", "outcome"]
).reset_index(drop=True)

# ============================================================
# COMPARE WITH EXISTING RANDOM-INTERCEPT PRIMARY RESULTS
# ============================================================

old = pd.read_csv(OLD_FILE, sep="\t")

old = old[
    old["subset"].eq("prospective")
].copy()

old = old[
    [
        "cardiac",
        "outcome",
        "beta",
        "SE",
        "P",
        "FDR_q",
        "N_subjects",
        "N_observations",
    ]
].rename(
    columns={
        "beta": "beta_random_intercept",
        "SE": "SE_random_intercept",
        "P": "P_random_intercept",
        "FDR_q": "FDR_q_random_intercept",
        "N_subjects": "N_subjects_random_intercept",
        "N_observations": "N_observations_random_intercept",
    }
)

comparison = results.merge(
    old,
    on=["cardiac", "outcome"],
    how="left",
    validate="one_to_one",
)

comparison["delta_beta_random_slope_minus_intercept"] = (
    comparison["beta"]
    - comparison["beta_random_intercept"]
)

# ============================================================
# SAVE
# ============================================================

outfile = (
    OUTDIR
    / "PRIMARY_APOE4_temporal_interactions_random_time_slope.tsv"
)

comparefile = (
    OUTDIR
    / "random_time_slope_vs_random_intercept.tsv"
)

results.to_csv(
    outfile,
    sep="\t",
    index=False,
)

comparison.to_csv(
    comparefile,
    sep="\t",
    index=False,
)

# ============================================================
# PRINT PRIMARY SUMMARY
# ============================================================

print("\n" + "=" * 100)
print("RANDOM MRI-TIME SLOPE: 16-TEST APOE4 FAMILY")
print("=" * 100)

show = results[
    [
        "cardiac",
        "outcome",
        "beta",
        "SE",
        "CI_low",
        "CI_high",
        "P",
        "FDR_q_random_slope",
        "N_subjects",
        "N_observations",
        "converged",
        "optimizer",
    ]
]

print(show.to_string(index=False))

sig = results[
    results["FDR_q_random_slope"] < 0.05
]

print("\n" + "=" * 100)
print("FDR q < 0.05")
print("=" * 100)

if len(sig):
    print(
        sig[
            [
                "cardiac",
                "outcome",
                "beta",
                "SE",
                "P",
                "FDR_q_random_slope",
            ]
        ].to_string(index=False)
    )
else:
    print("NONE")

print("\n" + "=" * 100)
print("LA DIMENSION -> LATERAL VENTRICLES")
print("=" * 100)

focus = comparison[
    (comparison["cardiac"] == "LA_dimension")
    & (
        comparison["outcome"]
        == "lateral_ventricles"
    )
]

print(focus.to_string(index=False))

print("\nSaved:")
print(outfile)
print(comparefile)
print("\n155 COMPLETE.")
