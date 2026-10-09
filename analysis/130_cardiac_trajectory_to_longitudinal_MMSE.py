#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

ROOT = Path("/data/qiallab/Framingham")

CARDIAC_FILE = (
    ROOT
    / "data/longitudinal_derived/"
      "E4_E6_repeated_MRI_temporal_analysis.tsv"
)

MMSE_HITS = sorted(
    set(
        list(ROOT.rglob("*pht005174*.txt.gz"))
        + list(ROOT.rglob("*pht005174*.txt"))
    )
)

if not MMSE_HITS:
    raise FileNotFoundError("Could not locate pht005174 MMSE file")

MMSE_FILE = MMSE_HITS[0]

OUTDIR = (
    ROOT
    / "results/longitudinal_heart_brain/"
      "cognition_prediction/"
      "longitudinal_MMSE"
)

OUTDIR.mkdir(parents=True, exist_ok=True)


def zscore(x):
    x = pd.to_numeric(x, errors="coerce")
    sd = x.std(ddof=0)

    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.nan, index=x.index)

    return (x - x.mean()) / sd


# ============================================================
# CARDIAC TRAJECTORY DATA
# ============================================================

long = pd.read_csv(
    CARDIAC_FILE,
    sep="\t",
    low_memory=False
)

# one row / subject for exposure + covariates
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
# MMSE
# ============================================================

mmse = pd.read_csv(
    MMSE_FILE,
    sep="\t",
    compression="infer",
    comment="#",
    low_memory=False
)

for c in [
    "exam",
    "examdate",
    "cogscr",
    "maxcog",
]:
    mmse[c] = pd.to_numeric(
        mmse[c],
        errors="coerce"
    )

# Exam dates use day-count encoding consistent with the other
# Framingham date tables.
mmse["cog_date"] = (
    pd.Timestamp("1960-01-01")
    + pd.to_timedelta(
        mmse["examdate"],
        unit="D"
    )
)

# Normalize for administrations with max score <30
mmse["cog_prop"] = (
    mmse["cogscr"]
    / mmse["maxcog"]
)

# Require sensible scores
mmse.loc[
    (
        mmse["maxcog"] <= 0
    )
    |
    (
        mmse["cogscr"] < 0
    )
    |
    (
        mmse["cogscr"] > mmse["maxcog"]
    ),
    "cog_prop"
] = np.nan


# ============================================================
# MERGE
# ============================================================

keep_subject = [
    "shareid",
    "last_echo_date",
    "APOE4_carrier",
    "age6",
    "sex",
]

for spec in cardiac.values():
    keep_subject.append(
        spec["short"] + "_slope_z"
    )

analysis = mmse.merge(
    subject[keep_subject],
    on="shareid",
    how="inner"
)

analysis["echo_to_cog_years"] = (
    analysis["cog_date"]
    - analysis["last_echo_date"]
).dt.days / 365.25


# ============================================================
# TEMPORAL OUTCOME WINDOWS
#
# Primary:
# cognitive assessments strictly after final E4-E6 echo.
#
# Sensitivity:
# >=2 years after final echo.
# ============================================================

samples = {
    "prospective":
        analysis[
            analysis["echo_to_cog_years"] > 0
        ].copy(),

    "gap_ge2y":
        analysis[
            analysis["echo_to_cog_years"] >= 2
        ].copy(),
}


# ============================================================
# REQUIRE LONGITUDINAL COGNITION
#
# >=2 post-cardiac cognitive assessments per subject.
# ============================================================

for name in list(samples):

    dx = samples[name]

    nvis = (
        dx.dropna(
            subset=["cog_prop"]
        )
        .groupby("shareid")
        .size()
    )

    eligible = set(
        nvis[
            nvis >= 2
        ].index
    )

    dx = dx[
        dx["shareid"].isin(
            eligible
        )
    ].copy()

    # Time zero = first post-cardiac cognitive assessment
    first_date = (
        dx.groupby("shareid")["cog_date"]
        .transform("min")
    )

    dx["cog_time_years"] = (
        dx["cog_date"]
        - first_date
    ).dt.days / 365.25

    # Standardize normalized MMSE ONCE across the entire subset,
    # not separately by visit.
    dx["cog_z"] = zscore(
        dx["cog_prop"]
    )

    samples[name] = dx

    print("\n", "=" * 90)
    print(name)
    print("=" * 90)

    print(
        "Subjects:",
        dx["shareid"].nunique()
    )

    print(
        "Observations:",
        len(dx)
    )

    print(
        "Median visits:",
        dx.groupby("shareid")
          .size()
          .median()
    )

    print(
        "Median cognitive follow-up, years:",
        (
            dx.groupby("shareid")["cog_time_years"]
              .max()
              .median()
        )
    )


# ============================================================
# FIT MODELS
#
# Main:
# cog_z ~ time * cardiac + APOE4 + age6 + sex
#
# APOE:
# cog_z ~ time * cardiac * APOE4 + age6 + sex
#
# Random intercept by subject.
#
# Key coefficients:
# time × cardiac
# time × cardiac × APOE4
# ============================================================

main_rows = []
interaction_rows = []

for subset_name, sample in samples.items():

    for cardiac_name, spec in cardiac.items():

        cz = (
            spec["short"]
            + "_slope_z"
        )

        cols = [
            "shareid",
            "cog_z",
            "cogscr",
            "cog_prop",
            "cog_time_years",
            cz,
            "APOE4_carrier",
            "age6",
            "sex",
        ]

        dx = (
            sample[cols]
            .dropna()
            .copy()
        )

        # require >=2 observations after cardiac data complete
        counts = (
            dx.groupby("shareid")
            .size()
        )

        keep = set(
            counts[
                counts >= 2
            ].index
        )

        dx = dx[
            dx["shareid"].isin(
                keep
            )
        ].copy()

        if (
            dx["shareid"].nunique()
            < 100
        ):
            print(
                "SKIPPING",
                subset_name,
                cardiac_name,
                "N subjects=",
                dx["shareid"].nunique()
            )
            continue

        # ----------------------------------------------------
        # MAIN CARDIAC EFFECT ON COGNITIVE SLOPE
        # ----------------------------------------------------

        f1 = (
            f"cog_z ~ "
            f"cog_time_years * {cz} "
            "+ APOE4_carrier "
            "+ age6 + C(sex)"
        )

        try:
            r1 = smf.mixedlm(
                f1,
                data=dx,
                groups=dx["shareid"]
            ).fit(
                reml=False,
                method="lbfgs",
                maxiter=500,
                disp=False
            )
        except Exception:
            r1 = smf.mixedlm(
                f1,
                data=dx,
                groups=dx["shareid"]
            ).fit(
                reml=False,
                method="powell",
                maxiter=1000,
                disp=False
            )

        term1 = (
            f"cog_time_years:{cz}"
        )

        ci1 = r1.conf_int()

        main_rows.append({
            "subset":
                subset_name,

            "cardiac":
                cardiac_name,

            "N_subjects":
                dx["shareid"].nunique(),

            "N_observations":
                len(dx),

            "time_x_cardiac_beta":
                float(r1.params[term1]),

            "SE":
                float(r1.bse[term1]),

            "CI_low":
                float(ci1.loc[term1, 0]),

            "CI_high":
                float(ci1.loc[term1, 1]),

            "P":
                float(r1.pvalues[term1]),

            "converged":
                bool(r1.converged),
        })

        # ----------------------------------------------------
        # APOE4 MODIFICATION OF COGNITIVE SLOPE
        # ----------------------------------------------------

        f2 = (
            f"cog_z ~ "
            f"cog_time_years * {cz} "
            "* APOE4_carrier "
            "+ age6 + C(sex)"
        )

        try:
            r2 = smf.mixedlm(
                f2,
                data=dx,
                groups=dx["shareid"]
            ).fit(
                reml=False,
                method="lbfgs",
                maxiter=500,
                disp=False
            )
        except Exception:
            r2 = smf.mixedlm(
                f2,
                data=dx,
                groups=dx["shareid"]
            ).fit(
                reml=False,
                method="powell",
                maxiter=1000,
                disp=False
            )

        term2 = (
            f"cog_time_years:"
            f"{cz}:"
            "APOE4_carrier"
        )

        ci2 = r2.conf_int()

        interaction_rows.append({
            "subset":
                subset_name,

            "cardiac":
                cardiac_name,

            "N_subjects":
                dx["shareid"].nunique(),

            "N_observations":
                len(dx),

            "time_x_cardiac_x_APOE4_beta":
                float(r2.params[term2]),

            "SE":
                float(r2.bse[term2]),

            "CI_low":
                float(ci2.loc[term2, 0]),

            "CI_high":
                float(ci2.loc[term2, 1]),

            "P":
                float(r2.pvalues[term2]),

            "converged":
                bool(r2.converged),
        })


# ============================================================
# FDR: 4 cardiac trajectories per temporal subset
# ============================================================

main = pd.DataFrame(main_rows)
inter = pd.DataFrame(interaction_rows)

for subset in main["subset"].unique():

    idx = (
        main["subset"]
        == subset
    )

    main.loc[
        idx,
        "q_FDR4"
    ] = multipletests(
        main.loc[
            idx,
            "P"
        ],
        method="fdr_bh"
    )[1]


for subset in inter["subset"].unique():

    idx = (
        inter["subset"]
        == subset
    )

    inter.loc[
        idx,
        "q_FDR4"
    ] = multipletests(
        inter.loc[
            idx,
            "P"
        ],
        method="fdr_bh"
    )[1]


# ============================================================
# GEE SENSITIVITY
#
# Exchangeable within-subject covariance.
# ============================================================

gee_rows = []

for subset_name, sample in samples.items():

    for cardiac_name, spec in cardiac.items():

        cz = (
            spec["short"]
            + "_slope_z"
        )

        cols = [
            "shareid",
            "cog_z",
            "cog_time_years",
            cz,
            "APOE4_carrier",
            "age6",
            "sex",
        ]

        dx = (
            sample[cols]
            .dropna()
            .copy()
        )

        counts = (
            dx.groupby("shareid")
            .size()
        )

        keep = set(
            counts[
                counts >= 2
            ].index
        )

        dx = dx[
            dx["shareid"].isin(
                keep
            )
        ].copy()

        if dx["shareid"].nunique() < 100:
            continue

        formula = (
            f"cog_z ~ "
            f"cog_time_years * {cz} "
            "* APOE4_carrier "
            "+ age6 + C(sex)"
        )

        rg = smf.gee(
            formula,
            groups="shareid",
            data=dx,
            cov_struct=
                sm.cov_struct.Exchangeable(),
            family=
                sm.families.Gaussian(),
        ).fit()

        term = (
            f"cog_time_years:"
            f"{cz}:"
            "APOE4_carrier"
        )

        ci = rg.conf_int()

        gee_rows.append({
            "subset":
                subset_name,

            "cardiac":
                cardiac_name,

            "N_subjects":
                dx["shareid"].nunique(),

            "N_observations":
                len(dx),

            "three_way_beta":
                float(rg.params[term]),

            "SE":
                float(rg.bse[term]),

            "CI_low":
                float(ci.loc[term, 0]),

            "CI_high":
                float(ci.loc[term, 1]),

            "P":
                float(rg.pvalues[term]),
        })


gee = pd.DataFrame(gee_rows)

for subset in gee["subset"].unique():

    idx = (
        gee["subset"]
        == subset
    )

    gee.loc[
        idx,
        "q_FDR4"
    ] = multipletests(
        gee.loc[
            idx,
            "P"
        ],
        method="fdr_bh"
    )[1]


# ============================================================
# SAVE
# ============================================================

main.to_csv(
    OUTDIR
    / "MMSE_cardiac_trajectory_main_slope_effects.tsv",
    sep="\t",
    index=False
)

inter.to_csv(
    OUTDIR
    / "MMSE_cardiac_trajectory_APOE4_slope_interactions.tsv",
    sep="\t",
    index=False
)

gee.to_csv(
    OUTDIR
    / "MMSE_APOE4_interactions_GEE_sensitivity.tsv",
    sep="\t",
    index=False
)

for name, dx in samples.items():

    dx.to_csv(
        OUTDIR
        / f"MMSE_analysis_sample_{name}.tsv",
        sep="\t",
        index=False
    )


# ============================================================
# PRINT
# ============================================================

print("\n" + "=" * 110)
print("EARLY CARDIAC TRAJECTORY → LATER MMSE CHANGE")
print("=" * 110)

print(
    main.sort_values(
        ["subset", "P"]
    ).to_string(
        index=False
    )
)


print("\n" + "=" * 110)
print("EARLY CARDIAC TRAJECTORY × APOE4 → LATER MMSE CHANGE")
print("=" * 110)

print(
    inter.sort_values(
        ["subset", "P"]
    ).to_string(
        index=False
    )
)


print("\n" + "=" * 110)
print("GEE SENSITIVITY: CARDIAC × APOE4 → MMSE CHANGE")
print("=" * 110)

print(
    gee.sort_values(
        ["subset", "P"]
    ).to_string(
        index=False
    )
)


print("\nSaved to:")
print(OUTDIR)

print("\n130 COMPLETE")
