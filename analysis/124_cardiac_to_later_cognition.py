#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

ROOT = Path("/data/qiallab/Framingham")

LONG_FILE = (
    ROOT
    / "data/longitudinal_derived/"
      "E4_E6_repeated_MRI_temporal_analysis.tsv"
)

NP_FILE = (
    ROOT
    / "data/longitudinal_20260924/"
      "phs000007.v35.pht004374.v9.p16.c1.vr_npd_2023_a_1528s.HMB-IRB-MDS.txt.gz"
)

OUTDIR = (
    ROOT
    / "results/longitudinal_heart_brain/"
      "cognition_prediction"
)

OUTDIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# HELPERS
# ============================================================

def zscore(x):
    x = pd.to_numeric(x, errors="coerce")
    sd = x.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.nan, index=x.index)
    return (x - x.mean()) / sd

def add_fdr(df, pcol, outcol):
    df = df.copy()
    ok = df[pcol].notna()
    df[outcol] = np.nan
    if ok.any():
        df.loc[ok, outcol] = multipletests(
            df.loc[ok, pcol],
            method="fdr_bh"
        )[1]
    return df

# ============================================================
# LOAD LONGITUDINAL CARDIAC DATA
# ============================================================

long = pd.read_csv(
    LONG_FILE,
    sep="\t",
    low_memory=False
)

# one subject row for cardiac predictors / covariates
subject = (
    long.sort_values("mri_date")
        .drop_duplicates("shareid")
        .copy()
)

subject["last_echo_date"] = pd.to_datetime(
    subject["last_echo_date"],
    errors="coerce"
)

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

for name, spec in cardiac.items():

    valid = (
        pd.to_numeric(
            subject[spec["n"]],
            errors="coerce"
        ) == 3
    )

    zcol = spec["short"] + "_slope_z"

    subject[zcol] = np.nan

    subject.loc[valid, zcol] = zscore(
        subject.loc[
            valid,
            spec["slope"]
        ]
    )

# ============================================================
# LOAD NEUROPSYCH DATA
# ============================================================

npd = pd.read_csv(
    NP_FILE,
    sep="\t",
    compression="gzip",
    low_memory=False,
    comment="#"
)

# normalize ID
if "shareid" not in npd.columns:
    candidates = [
        c for c in npd.columns
        if c.lower() == "shareid"
    ]
    if not candidates:
        raise RuntimeError("No shareid column found")
    npd = npd.rename(
        columns={candidates[0]: "shareid"}
    )

# ============================================================
# CONVERT npdate
#
# Framingham npdate values are consistent with SAS dates:
# days since 1960-01-01.
# ============================================================

npd["np_date"] = (
    pd.Timestamp("1960-01-01")
    + pd.to_timedelta(
        pd.to_numeric(
            npd["npdate"],
            errors="coerce"
        ),
        unit="D"
    )
)

print("\nNeuropsych date range:")
print(npd["np_date"].min(), "to", npd["np_date"].max())

# ============================================================
# COGNITIVE OUTCOMES
#
# Higher z = better cognition for all outcomes.
# ============================================================

# Raw numeric conversion
for c in [
    "LMd",
    "trailsB",
    "FAS_animal",
    "DSB",
]:
    npd[c] = pd.to_numeric(
        npd[c],
        errors="coerce"
    )

# Trails B: lower completion time is better
npd["TrailsB_good"] = np.where(
    npd["trailsB"] > 0,
    -np.log(npd["trailsB"]),
    np.nan
)

cognition = {
    "Logical Memory delayed": "LMd",
    "Trails B": "TrailsB_good",
    "Animal fluency": "FAS_animal",
    "Digit Span Backward": "DSB",
}

# ============================================================
# MERGE CARDIAC + COGNITION
# ============================================================

# Preserve any neuropsych-file sex variable separately and use the
# longitudinal cardiac/clinical sex field as the model covariate.
if "sex" in npd.columns:
    npd = npd.rename(columns={"sex": "sex_np"})

npd = npd.merge(
    subject[
        [
            "shareid",
            "last_echo_date",
            "APOE4_carrier",
            "age6",
            "sex",
        ]
        + [
            spec["short"] + "_slope_z"
            for spec in cardiac.values()
        ]
    ],
    on="shareid",
    how="inner"
)

print("\nPost-merge demographic columns:")
print([
    c for c in npd.columns
    if c.lower().startswith("sex")
    or c.lower().startswith("age")
])

# ============================================================
# TEMPORAL ORDERING
#
# Keep neuropsych visits after final E4-E6 echo.
# ============================================================

npd["echo_to_np_years"] = (
    npd["np_date"]
    - npd["last_echo_date"]
).dt.days / 365.25

prospective = npd[
    npd["echo_to_np_years"] > 0
].copy()

gap2 = npd[
    npd["echo_to_np_years"] >= 2
].copy()

print("\nProspective NP observations:", len(prospective))
print(
    "Prospective unique subjects:",
    prospective["shareid"].nunique()
)

print("\n>=2y-gap NP observations:", len(gap2))
print(
    ">=2y-gap unique subjects:",
    gap2["shareid"].nunique()
)

# ============================================================
# SELECT FIRST COGNITIVE ASSESSMENT AFTER CARDIAC WINDOW
#
# This creates a clean prospective prediction analysis rather
# than modeling sparse repeated cognition.
# ============================================================

def first_post_echo(df):
    return (
        df.sort_values(
            ["shareid", "np_date"]
        )
        .drop_duplicates(
            "shareid",
            keep="first"
        )
        .copy()
    )

samples = {
    "prospective": first_post_echo(prospective),
    "gap_ge2y": first_post_echo(gap2),
}

# ============================================================
# STANDARDIZE COGNITION WITHIN ANALYSIS SAMPLE
# ============================================================

for subset_name, dx in samples.items():

    for cog_name, col in cognition.items():

        dx[col + "_z"] = zscore(
            dx[col]
        )

# ============================================================
# MODELS
#
# Model 1:
# cognition ~ cardiac trajectory + APOE4 + age + sex
#
# Model 2:
# cognition ~ cardiac trajectory * APOE4 + age + sex
#
# HC3 robust SE.
# ============================================================

main_rows = []
interaction_rows = []

for subset_name, sample in samples.items():

    for cardiac_name, spec in cardiac.items():

        cardiac_z = (
            spec["short"]
            + "_slope_z"
        )

        for cog_name, cog_col in cognition.items():

            outcome = cog_col + "_z"

            cols = [
                outcome,
                cardiac_z,
                "APOE4_carrier",
                "age6",
                "sex",
            ]

            dx = (
                sample[cols]
                .dropna()
                .copy()
            )

            if len(dx) < 100:
                continue

            # -----------------------------------------------
            # Main cardiac effect
            # -----------------------------------------------

            formula_main = (
                f"{outcome} ~ "
                f"{cardiac_z} + "
                "APOE4_carrier + "
                "age6 + C(sex)"
            )

            r1 = smf.ols(
                formula_main,
                data=dx
            ).fit(
                cov_type="HC3"
            )

            ci1 = r1.conf_int()

            main_rows.append({
                "subset": subset_name,
                "cardiac": cardiac_name,
                "cognition": cog_name,
                "N": len(dx),
                "N_APOE4_noncarrier":
                    int(
                        (
                            dx["APOE4_carrier"]
                            == 0
                        ).sum()
                    ),
                "N_APOE4_carrier":
                    int(
                        (
                            dx["APOE4_carrier"]
                            == 1
                        ).sum()
                    ),
                "beta":
                    float(
                        r1.params[
                            cardiac_z
                        ]
                    ),
                "SE":
                    float(
                        r1.bse[
                            cardiac_z
                        ]
                    ),
                "CI_low":
                    float(
                        ci1.loc[
                            cardiac_z, 0
                        ]
                    ),
                "CI_high":
                    float(
                        ci1.loc[
                            cardiac_z, 1
                        ]
                    ),
                "P":
                    float(
                        r1.pvalues[
                            cardiac_z
                        ]
                    ),
            })

            # -----------------------------------------------
            # APOE interaction
            # -----------------------------------------------

            formula_int = (
                f"{outcome} ~ "
                f"{cardiac_z} * "
                "APOE4_carrier + "
                "age6 + C(sex)"
            )

            r2 = smf.ols(
                formula_int,
                data=dx
            ).fit(
                cov_type="HC3"
            )

            term = (
                f"{cardiac_z}:"
                "APOE4_carrier"
            )

            ci2 = r2.conf_int()

            interaction_rows.append({
                "subset": subset_name,
                "cardiac": cardiac_name,
                "cognition": cog_name,
                "N": len(dx),
                "interaction_beta":
                    float(
                        r2.params[term]
                    ),
                "interaction_SE":
                    float(
                        r2.bse[term]
                    ),
                "interaction_CI_low":
                    float(
                        ci2.loc[
                            term, 0
                        ]
                    ),
                "interaction_CI_high":
                    float(
                        ci2.loc[
                            term, 1
                        ]
                    ),
                "interaction_P":
                    float(
                        r2.pvalues[
                            term
                        ]
                    ),
            })

# ============================================================
# FDR
#
# 4 cardiac × 4 cognitive outcomes = 16 tests
# per temporal definition.
# ============================================================

main = pd.DataFrame(main_rows)
inter = pd.DataFrame(interaction_rows)

for subset_name in main["subset"].unique():

    idx = (
        main["subset"]
        == subset_name
    )

    main.loc[
        idx,
        "q_global16"
    ] = multipletests(
        main.loc[idx, "P"],
        method="fdr_bh"
    )[1]

for subset_name in inter["subset"].unique():

    idx = (
        inter["subset"]
        == subset_name
    )

    inter.loc[
        idx,
        "interaction_q_global16"
    ] = multipletests(
        inter.loc[
            idx,
            "interaction_P"
        ],
        method="fdr_bh"
    )[1]

# Also within cardiac family, 4 cognitive outcomes
for subset_name in main["subset"].unique():

    for cardiac_name in main["cardiac"].unique():

        idx = (
            (main["subset"] == subset_name)
            &
            (main["cardiac"] == cardiac_name)
        )

        if idx.sum():
            main.loc[
                idx,
                "q_within_cardiac4"
            ] = multipletests(
                main.loc[idx, "P"],
                method="fdr_bh"
            )[1]

for subset_name in inter["subset"].unique():

    for cardiac_name in inter["cardiac"].unique():

        idx = (
            (inter["subset"] == subset_name)
            &
            (inter["cardiac"] == cardiac_name)
        )

        if idx.sum():
            inter.loc[
                idx,
                "interaction_q_within_cardiac4"
            ] = multipletests(
                inter.loc[
                    idx,
                    "interaction_P"
                ],
                method="fdr_bh"
            )[1]

# ============================================================
# SAVE
# ============================================================

main_file = (
    OUTDIR
    / "cardiac_trajectory_later_cognition_main_effects.tsv"
)

inter_file = (
    OUTDIR
    / "cardiac_trajectory_later_cognition_APOE4_interactions.tsv"
)

main.to_csv(
    main_file,
    sep="\t",
    index=False
)

inter.to_csv(
    inter_file,
    sep="\t",
    index=False
)

# Save analysis samples
for name, dx in samples.items():
    dx.to_csv(
        OUTDIR
        / f"cognition_analysis_sample_{name}.tsv",
        sep="\t",
        index=False
    )

# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 110)
print("CARDIAC TRAJECTORIES → LATER COGNITION")
print("=" * 110)

print(
    main.sort_values(
        ["subset", "P"]
    ).to_string(
        index=False
    )
)

print("\n" + "=" * 110)
print("CARDIAC TRAJECTORY × APOE4 → LATER COGNITION")
print("=" * 110)

print(
    inter.sort_values(
        [
            "subset",
            "interaction_P"
        ]
    ).to_string(
        index=False
    )
)

print("\n" + "=" * 110)
print("NOMINAL P < 0.05")
print("=" * 110)

print("\nMAIN EFFECTS:")
sig = main[
    main["P"] < 0.05
]

if sig.empty:
    print("None")
else:
    print(
        sig.sort_values(
            ["subset", "P"]
        ).to_string(
            index=False
        )
    )

print("\nINTERACTIONS:")
sig = inter[
    inter["interaction_P"] < 0.05
]

if sig.empty:
    print("None")
else:
    print(
        sig.sort_values(
            [
                "subset",
                "interaction_P"
            ]
        ).to_string(
            index=False
        )
    )

print("\nSaved:")
print(main_file)
print(inter_file)

print("\n124 COMPLETE")
