#!/usr/bin/env python3

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import norm
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

OUT = (
    ROOT / "results"
    / "longitudinal_heart_brain"
    / "temporal_prediction"
    / "gee_validation"
)
OUT.mkdir(parents=True, exist_ok=True)

print("=" * 100)
print("108: GEE VALIDATION + APOE-STRATIFIED SIMPLE SLOPES")
print("=" * 100)
print("DATA:", DATAFILE)
print("OUT :", OUT)

# ============================================================
# DEFINITIONS
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

SUBSETS = {
    "prospective": 0.0,
    "gap_ge2y": 2.0,
}

TERM_2WAY = "mri_time_years:cardiac_z"
TERM_3WAY = "mri_time_years:cardiac_z:APOE4_carrier"

FORMULA = (
    "brain_z ~ "
    "mri_time_years * cardiac_z * APOE4_carrier "
    "+ age6_c + C(sex)"
)

# ============================================================
# HELPERS
# ============================================================

def simple_slope(res, apoe4):
    """
    Effect of cardiac_z on longitudinal MRI slope.

    APOE4=0:
        beta_tc

    APOE4=1:
        beta_tc + beta_tca
    """

    params = res.params
    cov = res.cov_params()

    b2 = params[TERM_2WAY]
    b3 = params[TERM_3WAY]

    if apoe4 == 0:
        beta = b2
        var = cov.loc[TERM_2WAY, TERM_2WAY]
    else:
        beta = b2 + b3
        var = (
            cov.loc[TERM_2WAY, TERM_2WAY]
            + cov.loc[TERM_3WAY, TERM_3WAY]
            + 2 * cov.loc[TERM_2WAY, TERM_3WAY]
        )

    se = np.sqrt(var)
    z = beta / se
    p = 2 * norm.sf(abs(z))

    return {
        "beta": beta,
        "SE": se,
        "CI_low": beta - 1.96 * se,
        "CI_high": beta + 1.96 * se,
        "P": p,
    }


def prepare_model_data(long, cardiac_base, outcome_col, min_gap):

    ncol = f"{cardiac_base}_n"
    scol = f"{cardiac_base}_slope"

    d = long[
        (pd.to_numeric(long[ncol], errors="coerce") >= 3)
        & long[scol].notna()
        & long["APOE4_carrier"].notna()
        & long["age6_c"].notna()
        & long["sex"].notna()
        & (
            pd.to_numeric(
                long["last_echo_to_first_mri_years"],
                errors="coerce"
            ) >= min_gap
        )
        & long[outcome_col].notna()
        & long["mri_time_years"].notna()
    ].copy()

    # require >=2 repeated outcome observations
    nobs = d.groupby("shareid")[outcome_col].count()
    good = nobs[nobs >= 2].index
    d = d[d["shareid"].isin(good)].copy()

    # standardized cardiac trajectory using one value per subject
    subject_slopes = (
        d[["shareid", scol]]
        .drop_duplicates("shareid")
    )

    cmean = subject_slopes[scol].mean()
    csd = subject_slopes[scol].std()

    d["cardiac_z"] = (d[scol] - cmean) / csd

    # exactly as 104: standardize outcome across model observations
    om = d[outcome_col].mean()
    osd = d[outcome_col].std()

    d["brain_z"] = (d[outcome_col] - om) / osd

    d = d.dropna(
        subset=[
            "brain_z",
            "mri_time_years",
            "cardiac_z",
            "APOE4_carrier",
            "age6_c",
            "sex",
        ]
    )

    # ensure repeated observations remain
    nn = d.groupby("shareid").size()
    good2 = nn[nn >= 2].index
    d = d[d["shareid"].isin(good2)].copy()

    return d


def extract_terms(res, method, subset, cardiac, outcome, d):

    rows = []

    apoe_subject = (
        d[["shareid", "APOE4_carrier"]]
        .drop_duplicates("shareid")
    )

    meta = {
        "method": method,
        "subset": subset,
        "cardiac": cardiac,
        "outcome": outcome,
        "N_subjects": d["shareid"].nunique(),
        "N_observations": len(d),
        "N_APOE4_carrier":
            int((apoe_subject["APOE4_carrier"] == 1).sum()),
        "N_APOE4_noncarrier":
            int((apoe_subject["APOE4_carrier"] == 0).sum()),
    }

    ci = res.conf_int()

    for term in [TERM_2WAY, TERM_3WAY]:

        rows.append({
            **meta,
            "term": term,
            "beta": res.params[term],
            "SE": res.bse[term],
            "CI_low": ci.loc[term, 0],
            "CI_high": ci.loc[term, 1],
            "P": res.pvalues[term],
        })

    return rows


# ============================================================
# LOAD DERIVED ANALYSIS DATA
# ============================================================

long = pd.read_csv(
    DATAFILE,
    sep="\t",
    dtype={"shareid": str},
    low_memory=False
)

print("Rows:", len(long))
print("Subjects:", long["shareid"].nunique())

# ============================================================
# FIT MODELS
# ============================================================

effect_rows = []
simple_rows = []
fit_rows = []

for subset_name, min_gap in SUBSETS.items():

    print("\n" + "=" * 100)
    print("SUBSET:", subset_name)
    print("=" * 100)

    for cardiac_name, cardiac_base in CARDIAC.items():

        for outcome_name, outcome_col in OUTCOMES.items():

            if outcome_col not in long.columns:
                print("SKIP missing outcome:", outcome_col)
                continue

            d = prepare_model_data(
                long,
                cardiac_base,
                outcome_col,
                min_gap
            )

            nsub = d["shareid"].nunique()

            if nsub < 50:
                continue

            print(
                f"{cardiac_name:24s} "
                f"{outcome_name:22s} "
                f"N={nsub:4d} obs={len(d):5d}"
            )

            # =================================================
            # RANDOM-INTERCEPT MIXEDLM
            # =================================================

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")

                    mlm = smf.mixedlm(
                        FORMULA,
                        data=d,
                        groups=d["shareid"],
                        re_formula="1"
                    )

                    mlm_res = mlm.fit(
                        reml=False,
                        method="lbfgs",
                        maxiter=2000,
                        disp=False
                    )

                    if not getattr(mlm_res, "converged", False):
                        mlm_res = mlm.fit(
                            reml=False,
                            method="powell",
                            maxiter=5000,
                            disp=False
                        )

                fit_rows.append({
                    "method": "MixedLM_random_intercept",
                    "subset": subset_name,
                    "cardiac": cardiac_name,
                    "outcome": outcome_name,
                    "N_subjects": nsub,
                    "N_observations": len(d),
                    "converged":
                        bool(getattr(mlm_res, "converged", False))
                })

                effect_rows.extend(
                    extract_terms(
                        mlm_res,
                        "MixedLM_random_intercept",
                        subset_name,
                        cardiac_name,
                        outcome_name,
                        d
                    )
                )

                for apoe in [0, 1]:
                    s = simple_slope(mlm_res, apoe)

                    simple_rows.append({
                        "method": "MixedLM_random_intercept",
                        "subset": subset_name,
                        "cardiac": cardiac_name,
                        "outcome": outcome_name,
                        "APOE4_carrier": apoe,
                        "APOE_group":
                            "APOE4 carrier"
                            if apoe == 1
                            else "APOE4 noncarrier",
                        **s,
                        "N_subjects": nsub,
                    })

            except Exception as e:

                fit_rows.append({
                    "method": "MixedLM_random_intercept",
                    "subset": subset_name,
                    "cardiac": cardiac_name,
                    "outcome": outcome_name,
                    "N_subjects": nsub,
                    "N_observations": len(d),
                    "converged": False,
                    "error": repr(e),
                })

                print("  MixedLM ERROR:", repr(e))

            # =================================================
            # GEE
            # =================================================

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")

                    gee = smf.gee(
                        FORMULA,
                        groups="shareid",
                        data=d,
                        family=sm.families.Gaussian(),
                        cov_struct=sm.cov_struct.Exchangeable()
                    )

                    # GEE uses robust sandwich covariance by default
                    gee_res = gee.fit(
                        maxiter=200,
                        cov_type="robust"
                    )

                fit_rows.append({
                    "method": "GEE_exchangeable_robust",
                    "subset": subset_name,
                    "cardiac": cardiac_name,
                    "outcome": outcome_name,
                    "N_subjects": nsub,
                    "N_observations": len(d),
                    "converged":
                        bool(getattr(gee_res, "converged", True))
                })

                effect_rows.extend(
                    extract_terms(
                        gee_res,
                        "GEE_exchangeable_robust",
                        subset_name,
                        cardiac_name,
                        outcome_name,
                        d
                    )
                )

                for apoe in [0, 1]:
                    s = simple_slope(gee_res, apoe)

                    simple_rows.append({
                        "method": "GEE_exchangeable_robust",
                        "subset": subset_name,
                        "cardiac": cardiac_name,
                        "outcome": outcome_name,
                        "APOE4_carrier": apoe,
                        "APOE_group":
                            "APOE4 carrier"
                            if apoe == 1
                            else "APOE4 noncarrier",
                        **s,
                        "N_subjects": nsub,
                    })

            except Exception as e:

                fit_rows.append({
                    "method": "GEE_exchangeable_robust",
                    "subset": subset_name,
                    "cardiac": cardiac_name,
                    "outcome": outcome_name,
                    "N_subjects": nsub,
                    "N_observations": len(d),
                    "converged": False,
                    "error": repr(e),
                })

                print("  GEE ERROR:", repr(e))


# ============================================================
# DATA FRAMES
# ============================================================

effects = pd.DataFrame(effect_rows)
simple = pd.DataFrame(simple_rows)
fits = pd.DataFrame(fit_rows)

effects["effect_type"] = np.where(
    effects["term"] == TERM_3WAY,
    "MRI_time_x_cardiac_x_APOE4",
    "MRI_time_x_cardiac"
)

# ============================================================
# FDR
#
# Separate correction within:
# method x subset x effect family
#
# 4 cardiac x 4 brain outcomes = 16 tests
# ============================================================

effects["FDR_q"] = np.nan

for keys, idx in effects.groupby(
    ["method", "subset", "effect_type"]
).groups.items():

    idx = list(idx)
    p = effects.loc[idx, "P"]
    ok = p.notna()

    if ok.sum():
        q = multipletests(
            p.loc[ok],
            method="fdr_bh"
        )[1]

        effects.loc[
            p.loc[ok].index,
            "FDR_q"
        ] = q


# ============================================================
# MIXEDLM vs GEE DIRECT COMPARISON
# ============================================================

comparison = (
    effects.pivot_table(
        index=[
            "subset",
            "cardiac",
            "outcome",
            "effect_type"
        ],
        columns="method",
        values=[
            "beta",
            "SE",
            "P",
            "FDR_q"
        ]
    )
)

comparison.columns = [
    f"{stat}_{method}"
    for stat, method in comparison.columns
]

comparison = comparison.reset_index()

m_beta = "beta_MixedLM_random_intercept"
g_beta = "beta_GEE_exchangeable_robust"

if m_beta in comparison.columns and g_beta in comparison.columns:
    comparison["same_direction"] = (
        np.sign(comparison[m_beta])
        ==
        np.sign(comparison[g_beta])
    )

    comparison["beta_difference_GEE_minus_MixedLM"] = (
        comparison[g_beta]
        - comparison[m_beta]
    )

# ============================================================
# FOCUSED LA -> LATERAL VENTRICLE TABLE
# ============================================================

focus_effects = effects[
    (effects["cardiac"] == "LA_dimension")
    & (effects["outcome"] == "lateral_ventricles")
].copy()

focus_simple = simple[
    (simple["cardiac"] == "LA_dimension")
    & (simple["outcome"] == "lateral_ventricles")
].copy()

# ============================================================
# SAVE
# ============================================================

fits.to_csv(
    OUT / "model_fit_summary.tsv",
    sep="\t",
    index=False
)

effects.to_csv(
    OUT / "MixedLM_GEE_temporal_effects.tsv",
    sep="\t",
    index=False
)

simple.to_csv(
    OUT / "APOE_stratified_simple_slopes.tsv",
    sep="\t",
    index=False
)

comparison.to_csv(
    OUT / "MixedLM_vs_GEE_comparison.tsv",
    sep="\t",
    index=False
)

focus_effects.to_csv(
    OUT / "FOCUS_LA_lateral_ventricle_effects.tsv",
    sep="\t",
    index=False
)

focus_simple.to_csv(
    OUT / "FOCUS_LA_lateral_ventricle_simple_slopes.tsv",
    sep="\t",
    index=False
)

# ============================================================
# PRINT FOCUSED RESULT
# ============================================================

print("\n" + "=" * 100)
print("LA DIMENSION -> LATERAL VENTRICULAR TRAJECTORY")
print("=" * 100)

print(
    focus_effects[
        [
            "method",
            "subset",
            "effect_type",
            "beta",
            "SE",
            "CI_low",
            "CI_high",
            "P",
            "FDR_q",
            "N_subjects"
        ]
    ].sort_values(
        ["subset", "method", "effect_type"]
    ).to_string(
        index=False,
        float_format=lambda x: f"{x:.6g}"
    )
)

print("\n" + "=" * 100)
print("APOE-STRATIFIED SIMPLE SLOPES")
print("=" * 100)

print(
    focus_simple[
        [
            "method",
            "subset",
            "APOE_group",
            "beta",
            "SE",
            "CI_low",
            "CI_high",
            "P",
            "N_subjects"
        ]
    ].sort_values(
        ["subset", "method", "APOE_group"]
    ).to_string(
        index=False,
        float_format=lambda x: f"{x:.6g}"
    )
)

print("\nSaved to:")
print(OUT)
print("\n108 COMPLETE")
