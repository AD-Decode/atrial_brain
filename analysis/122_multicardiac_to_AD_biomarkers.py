#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests


ROOT = Path("/data/qiallab/Framingham")

INDIR = (
    ROOT
    / "results/longitudinal_heart_brain/"
      "plasma_biomarker_prediction"
)

INFILE = (
    INDIR
    / "LA_E4E6_biomarker_merged_analysis_data.tsv"
)

OUTDIR = INDIR / "multicardiac_biomarkers"
OUTDIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD
# ============================================================

d = pd.read_csv(
    INFILE,
    sep="\t",
    low_memory=False
)

print("Input rows:", len(d))
print("Unique subjects:", d["shareid"].nunique())


# ============================================================
# CARDIAC TRAJECTORIES
#
# These are the four E4-E6 longitudinal phenotypes currently
# available in the derived dataset.
# ============================================================

cardiac = {
    "LA dimension": {
        "slope": "la_dim_slope",
        "n": "la_dim_n",
        "short": "LA",
    },

    "LV mass": {
        "slope": "lv_mass_derived_g_slope",
        "n": "lv_mass_derived_g_n",
        "short": "LVmass",
    },

    "Fractional shortening": {
        "slope": "fs_derived_pct_slope",
        "n": "fs_derived_pct_n",
        "short": "FS",
    },

    "LV end-diastolic dimension": {
        "slope": "lvdd_slope",
        "n": "lvdd_n",
        "short": "LVDD",
    },
}


# ============================================================
# BIOMARKERS
# ============================================================

biomarkers = {
    "Aβ42/40": "Abeta42_40_z",
    "p-tau181": "log_pTau181_z",
    "GFAP": "log_GFAP_z",
    "NfL": "log_NFL_z",
}


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required = [
    "shareid",
    "APOE4_carrier",
    "age6",
    "sex",
]

for spec in cardiac.values():
    required.extend([
        spec["slope"],
        spec["n"],
    ])

required.extend(biomarkers.values())

missing = sorted(
    set(required) - set(d.columns)
)

if missing:
    raise RuntimeError(
        "Missing required columns:\n"
        + "\n".join(missing)
    )


# ============================================================
# HELPERS
# ============================================================

def zscore(x):
    x = pd.to_numeric(
        x,
        errors="coerce"
    )

    sd = x.std(ddof=0)

    if not np.isfinite(sd) or sd == 0:
        return pd.Series(
            np.nan,
            index=x.index
        )

    return (x - x.mean()) / sd


def add_fdr(df, pcol, group_cols=None, outcol="q"):
    """
    Add BH-FDR either globally or within supplied grouping columns.
    """

    df = df.copy()
    df[outcol] = np.nan

    if group_cols is None:

        ok = df[pcol].notna()

        if ok.any():
            df.loc[ok, outcol] = multipletests(
                df.loc[ok, pcol],
                method="fdr_bh"
            )[1]

    else:

        for _, idx in df.groupby(
            group_cols,
            dropna=False
        ).groups.items():

            idx = list(idx)

            ok = df.loc[idx, pcol].notna()

            use = np.array(idx)[ok.to_numpy()]

            if len(use):
                df.loc[use, outcol] = multipletests(
                    df.loc[use, pcol],
                    method="fdr_bh"
                )[1]

    return df


# ============================================================
# STANDARDIZE EACH CARDIAC TRAJECTORY
#
# Require all 3 E4/E5/E6 measurements, matching the primary
# longitudinal cardiac trajectory definition.
# ============================================================

for label, spec in cardiac.items():

    slope = spec["slope"]
    ncol = spec["n"]

    valid = (
        pd.to_numeric(
            d[ncol],
            errors="coerce"
        ) == 3
    )

    zcol = f"{spec['short']}_slope_z"

    d[zcol] = np.nan

    d.loc[
        valid,
        zcol
    ] = zscore(
        d.loc[
            valid,
            slope
        ]
    )


# ============================================================
# MODEL SET
#
# M1:
#   biomarker ~ cardiac trajectory + APOE4 + age + sex
#
# Tests overall cardiac association with biomarker.
#
# M2:
#   biomarker ~ cardiac trajectory * APOE4 + age + sex
#
# Tests APOE4 modification.
#
# HC3 robust standard errors.
# ============================================================

main_rows = []
interaction_rows = []
stratified_rows = []

for cardiac_name, spec in cardiac.items():

    zcol = f"{spec['short']}_slope_z"

    for biomarker_name, outcome in biomarkers.items():

        cols = [
            outcome,
            zcol,
            "APOE4_carrier",
            "age6",
            "sex",
        ]

        dx = (
            d[cols]
            .dropna()
            .copy()
        )

        if len(dx) < 50:
            print(
                "SKIP",
                cardiac_name,
                biomarker_name,
                "N=",
                len(dx)
            )
            continue

        n_noncarrier = int(
            (dx["APOE4_carrier"] == 0).sum()
        )

        n_carrier = int(
            (dx["APOE4_carrier"] == 1).sum()
        )

        # ----------------------------------------------------
        # MODEL 1: overall cardiac main effect
        # ----------------------------------------------------

        f_main = (
            f"{outcome} ~ "
            f"{zcol} + APOE4_carrier "
            "+ age6 + C(sex)"
        )

        r_main = smf.ols(
            f_main,
            data=dx
        ).fit(
            cov_type="HC3"
        )

        ci = r_main.conf_int()

        main_rows.append({
            "cardiac": cardiac_name,
            "cardiac_short": spec["short"],
            "biomarker": biomarker_name,

            "N": len(dx),
            "N_APOE4_noncarrier": n_noncarrier,
            "N_APOE4_carrier": n_carrier,

            "beta":
                float(r_main.params[zcol]),

            "SE":
                float(r_main.bse[zcol]),

            "CI_low":
                float(ci.loc[zcol, 0]),

            "CI_high":
                float(ci.loc[zcol, 1]),

            "P":
                float(r_main.pvalues[zcol]),
        })

        # ----------------------------------------------------
        # MODEL 2: cardiac × APOE4
        # ----------------------------------------------------

        f_int = (
            f"{outcome} ~ "
            f"{zcol} * APOE4_carrier "
            "+ age6 + C(sex)"
        )

        r_int = smf.ols(
            f_int,
            data=dx
        ).fit(
            cov_type="HC3"
        )

        term = (
            f"{zcol}:APOE4_carrier"
        )

        ci = r_int.conf_int()

        interaction_rows.append({
            "cardiac": cardiac_name,
            "cardiac_short": spec["short"],
            "biomarker": biomarker_name,

            "N": len(dx),
            "N_APOE4_noncarrier": n_noncarrier,
            "N_APOE4_carrier": n_carrier,

            "interaction_beta":
                float(r_int.params[term]),

            "interaction_SE":
                float(r_int.bse[term]),

            "interaction_CI_low":
                float(ci.loc[term, 0]),

            "interaction_CI_high":
                float(ci.loc[term, 1]),

            "interaction_P":
                float(r_int.pvalues[term]),
        })

        # ----------------------------------------------------
        # APOE-STRATIFIED EFFECTS
        # Descriptive / interpretation aid
        # ----------------------------------------------------

        for apoe in [0, 1]:

            ds = dx[
                dx["APOE4_carrier"] == apoe
            ].copy()

            if len(ds) < 30:
                continue

            rs = smf.ols(
                f"{outcome} ~ "
                f"{zcol} + age6 + C(sex)",
                data=ds
            ).fit(
                cov_type="HC3"
            )

            cis = rs.conf_int()

            stratified_rows.append({
                "cardiac": cardiac_name,
                "cardiac_short": spec["short"],
                "biomarker": biomarker_name,

                "APOE4_carrier": apoe,

                "APOE_group":
                    "APOE ε4 carrier"
                    if apoe == 1
                    else "APOE ε4 noncarrier",

                "N": len(ds),

                "beta":
                    float(rs.params[zcol]),

                "SE":
                    float(rs.bse[zcol]),

                "CI_low":
                    float(cis.loc[zcol, 0]),

                "CI_high":
                    float(cis.loc[zcol, 1]),

                "P":
                    float(rs.pvalues[zcol]),
            })


# ============================================================
# DATAFRAMES
# ============================================================

main = pd.DataFrame(main_rows)
inter = pd.DataFrame(interaction_rows)
strat = pd.DataFrame(stratified_rows)


# ============================================================
# FDR
#
# We report both:
#
# 1. within-cardiac q across 4 biomarkers
#    useful for each prespecified cardiac phenotype
#
# 2. global q across all 16 cardiac × biomarker tests
#    conservative discovery-wide correction
# ============================================================

if not main.empty:

    main = add_fdr(
        main,
        "P",
        group_cols=["cardiac"],
        outcol="q_within_cardiac4"
    )

    main = add_fdr(
        main,
        "P",
        group_cols=None,
        outcol="q_global16"
    )


if not inter.empty:

    inter = add_fdr(
        inter,
        "interaction_P",
        group_cols=["cardiac"],
        outcol="interaction_q_within_cardiac4"
    )

    inter = add_fdr(
        inter,
        "interaction_P",
        group_cols=None,
        outcol="interaction_q_global16"
    )


# ============================================================
# COMBINED SUMMARY
# ============================================================

summary = main.merge(
    inter[
        [
            "cardiac",
            "biomarker",

            "interaction_beta",
            "interaction_SE",
            "interaction_CI_low",
            "interaction_CI_high",
            "interaction_P",
            "interaction_q_within_cardiac4",
            "interaction_q_global16",
        ]
    ],
    on=[
        "cardiac",
        "biomarker",
    ],
    how="left"
)


# ============================================================
# SAVE
# ============================================================

main_file = (
    OUTDIR
    / "cardiac_trajectory_main_effects_biomarkers.tsv"
)

interaction_file = (
    OUTDIR
    / "cardiac_trajectory_APOE4_interactions_biomarkers.tsv"
)

strat_file = (
    OUTDIR
    / "cardiac_trajectory_APOE4_stratified_biomarkers.tsv"
)

summary_file = (
    OUTDIR
    / "cardiac_trajectory_biomarker_summary.tsv"
)

main.to_csv(
    main_file,
    sep="\t",
    index=False
)

inter.to_csv(
    interaction_file,
    sep="\t",
    index=False
)

strat.to_csv(
    strat_file,
    sep="\t",
    index=False
)

summary.to_csv(
    summary_file,
    sep="\t",
    index=False
)


# ============================================================
# RANKED OUTPUT
# ============================================================

print("\n" + "=" * 110)
print("CARDIAC TRAJECTORY MAIN EFFECTS → AD BIOMARKERS")
print("=" * 110)

if not main.empty:
    print(
        main.sort_values(
            "P"
        ).to_string(
            index=False
        )
    )


print("\n" + "=" * 110)
print("CARDIAC TRAJECTORY × APOE4 → AD BIOMARKERS")
print("=" * 110)

if not inter.empty:
    print(
        inter.sort_values(
            "interaction_P"
        ).to_string(
            index=False
        )
    )


print("\n" + "=" * 110)
print("NOMINAL P < 0.05")
print("=" * 110)

if not main.empty:

    sig_main = main[
        main["P"] < 0.05
    ]

    print("\nMAIN EFFECTS:")
    if sig_main.empty:
        print("None")
    else:
        print(
            sig_main.sort_values(
                "P"
            ).to_string(
                index=False
            )
        )


if not inter.empty:

    sig_int = inter[
        inter["interaction_P"] < 0.05
    ]

    print("\nINTERACTIONS:")
    if sig_int.empty:
        print("None")
    else:
        print(
            sig_int.sort_values(
                "interaction_P"
            ).to_string(
                index=False
            )
        )


print("\nSaved:")
print(main_file)
print(interaction_file)
print(strat_file)
print(summary_file)
