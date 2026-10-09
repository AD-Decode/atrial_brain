#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

ROOT = Path("/data/qiallab/Framingham")

INFILE = (
    ROOT / "results" /
    "neurocardiac_metadata_analysis_ready_metabolic.csv"
)

OUTDIR = (
    ROOT / "results" /
    "primary_pairs_metabolic_modifiers"
)

OUTDIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Four established primary heart-brain associations
# ============================================================

PAIRS = [
    {
        "pair_id": "LVEF_L_middletemporal_vol",
        "label": "LVEF × APOE4 → L middle temporal volume",
        "cardiac": "LVEF",
        "outcome": "lh_middletemporal_vol",
        "metric_type": "volume",
    },
    {
        "pair_id": "LVESVi_R_superiorfrontal_area",
        "label": "LVESVi × APOE4 → R superior frontal area",
        "cardiac": "LVESVi",
        "outcome": "rh_superiorfrontal_area",
        "metric_type": "surface_area",
    },
    {
        "pair_id": "LVMASSi_R_lingual_vol",
        "label": "LV mass index × APOE4 → R lingual volume",
        "cardiac": "LV_MASSi",
        "outcome": "rh_lingual_vol",
        "metric_type": "volume",
    },
    {
        "pair_id": "LVMASSi_R_lingual_area",
        "label": "LV mass index × APOE4 → R lingual surface area",
        "cardiac": "LV_MASSi",
        "outcome": "rh_lingual_area",
        "metric_type": "surface_area",
    },
]

# ============================================================
# Metabolic modifiers
#
# HbA1c: use raw value then z-score
# Insulin/adiponectin: log-transform first because skewed
# ============================================================

MODIFIERS = [
    {
        "name": "HbA1c",
        "source": "hba1c_exam7",
        "transform": "identity",
        "timing_covariate": "hba1c_exam7_to_cmr_years",
    },
    {
        "name": "Insulin",
        "source": "insulin_nearest_exam_pmol_L",
        "transform": "log",
        "timing_covariate": None,
    },
    {
        "name": "Adiponectin",
        "source": "adiponectin_exam7",
        "transform": "log",
        # Same Exam 7 visit as HbA1c
        "timing_covariate": "hba1c_exam7_to_cmr_years",
    },
]

# ============================================================
# Helpers
# ============================================================

def zscore(x):

    x = pd.to_numeric(
        x,
        errors="coerce"
    ).astype(float)

    sd = x.std(ddof=0)

    if not np.isfinite(sd) or sd == 0:
        return pd.Series(
            np.nan,
            index=x.index
        )

    return (
        x - x.mean()
    ) / sd


def diabetes_binary(x):

    x = pd.to_numeric(
        x,
        errors="coerce"
    )

    out = pd.Series(
        np.nan,
        index=x.index,
        dtype=float
    )

    out.loc[x == 0] = 0
    out.loc[x.isin([1, 2])] = 1

    return out


def prepare_modifier(
    series,
    transform
):

    x = pd.to_numeric(
        series,
        errors="coerce"
    ).astype(float)

    if transform == "log":

        x = x.where(
            x > 0
        )

        x = np.log(x)

    return x


def linear_combo(
    fit,
    terms
):

    beta = sum(
        coef * fit.params[name]
        for name, coef in terms.items()
    )

    cov = fit.cov_params()

    names = list(
        terms.keys()
    )

    var = 0.0

    for i in names:
        for j in names:

            var += (
                terms[i]
                * terms[j]
                * cov.loc[i, j]
            )

    se = np.sqrt(
        max(var, 0)
    )

    z = (
        beta / se
        if se > 0
        else np.nan
    )

    p = (
        2 * norm.sf(abs(z))
        if np.isfinite(z)
        else np.nan
    )

    return {
        "beta": beta,
        "SE": se,
        "CI_low": beta - 1.96 * se,
        "CI_high": beta + 1.96 * se,
        "p": p,
    }


# ============================================================
# Load
# ============================================================

df = pd.read_csv(
    INFILE,
    low_memory=False
)

print(
    "\nLoaded:",
    INFILE
)

print(
    "Shape:",
    df.shape
)

# ============================================================
# Reconstruct indexed cardiac variables exactly as before
# ============================================================

if (
    "height_cm" in df.columns
    and
    "weight_kg" in df.columns
):

    height_cm = pd.to_numeric(
        df["height_cm"],
        errors="coerce"
    )

    weight_kg = pd.to_numeric(
        df["weight_kg"],
        errors="coerce"
    )

elif (
    "height_in" in df.columns
    and
    "weight_lb" in df.columns
):

    height_cm = (
        pd.to_numeric(
            df["height_in"],
            errors="coerce"
        )
        * 2.54
    )

    weight_kg = (
        pd.to_numeric(
            df["weight_lb"],
            errors="coerce"
        )
        * 0.45359237
    )

else:

    raise RuntimeError(
        "Could not find height/weight variables."
    )


df["BSA_m2_exact"] = np.sqrt(
    height_cm
    * weight_kg
    / 3600
)

df["LVESVi"] = (
    pd.to_numeric(
        df["LVESV"],
        errors="coerce"
    )
    /
    df["BSA_m2_exact"]
)

df["LV_MASSi"] = (
    pd.to_numeric(
        df["LV_MASS"],
        errors="coerce"
    )
    /
    df["BSA_m2_exact"]
)

df["LVEF"] = pd.to_numeric(
    df["LVEF"],
    errors="coerce"
)

# ============================================================
# APOE / vascular covariates
# ============================================================

df["APOE4_carrier"] = pd.to_numeric(
    df["APOE4_carrier"],
    errors="coerce"
)

df["_sex"] = pd.to_numeric(
    df["sex_clinical"],
    errors="coerce"
)

df["diabetes_history_any"] = (
    diabetes_binary(
        df[
            "diabetes_history_nearest_exam"
        ]
    )
)

# ============================================================
# Surface-area global adjustment
# ============================================================

if (
    "TotalCorticalSurfaceArea"
    not in df.columns
):

    left = (
        "lh_WhiteSurfArea_DesKil_area"
    )

    right = (
        "rh_WhiteSurfArea_DesKil_area"
    )

    if (
        left in df.columns
        and
        right in df.columns
    ):

        df[
            "TotalCorticalSurfaceArea"
        ] = (
            pd.to_numeric(
                df[left],
                errors="coerce"
            )
            +
            pd.to_numeric(
                df[right],
                errors="coerce"
            )
        )

    else:

        raise RuntimeError(
            "Cannot reconstruct total cortical area."
        )

# ============================================================
# Validate modifier columns
# ============================================================

for m in MODIFIERS:

    if (
        m["source"]
        not in df.columns
    ):

        raise RuntimeError(
            f"Missing metabolic variable: "
            f"{m['source']}"
        )

    if (
        m["timing_covariate"]
        and
        m["timing_covariate"]
        not in df.columns
    ):

        print(
            "WARNING: missing timing variable:",
            m["timing_covariate"]
        )

        m[
            "timing_covariate"
        ] = None


# ============================================================
# Construct matched analysis data
# ============================================================

def make_analysis_data(
    pair,
    modifier=None,
    include_all_modifiers=False
):

    cols = [
        pair["outcome"],
        pair["cardiac"],
        "APOE4_carrier",
        "age_at_mri",
        "_sex",
        "abs_delta_years",
        "BMI_nearest_exam",
        "SBP_nearest_exam",
        "current_smoker_nearest_exam",
        "diabetes_history_any",
    ]

    if (
        pair["metric_type"]
        == "surface_area"
    ):

        cols.append(
            "TotalCorticalSurfaceArea"
        )

    if include_all_modifiers:

        for m in MODIFIERS:

            cols.append(
                m["source"]
            )

            if m[
                "timing_covariate"
            ]:

                cols.append(
                    m[
                        "timing_covariate"
                    ]
                )

    else:

        cols.append(
            modifier["source"]
        )

        if modifier[
            "timing_covariate"
        ]:

            cols.append(
                modifier[
                    "timing_covariate"
                ]
            )

    cols = list(
        dict.fromkeys(cols)
    )

    d = df[
        cols
    ].copy()

    for c in cols:

        d[c] = pd.to_numeric(
            d[c],
            errors="coerce"
        )

    # --------------------------------
    # Transform metabolic variables
    # --------------------------------

    if include_all_modifiers:

        for m in MODIFIERS:

            newcol = (
                m["source"]
                + "_transformed"
            )

            d[newcol] = (
                prepare_modifier(
                    d[m["source"]],
                    m["transform"]
                )
            )

    else:

        d[
            "modifier_transformed"
        ] = prepare_modifier(
            d[modifier["source"]],
            modifier["transform"]
        )

    # --------------------------------
    # Complete-case sample
    # --------------------------------

    required = [
        pair["outcome"],
        pair["cardiac"],
        "APOE4_carrier",
        "age_at_mri",
        "_sex",
        "abs_delta_years",
        "BMI_nearest_exam",
        "SBP_nearest_exam",
        "current_smoker_nearest_exam",
        "diabetes_history_any",
    ]

    if (
        pair["metric_type"]
        == "surface_area"
    ):

        required.append(
            "TotalCorticalSurfaceArea"
        )

    if include_all_modifiers:

        for m in MODIFIERS:

            required.append(
                m["source"]
                + "_transformed"
            )

            if m[
                "timing_covariate"
            ]:

                required.append(
                    m[
                        "timing_covariate"
                    ]
                )

    else:

        required.append(
            "modifier_transformed"
        )

        if modifier[
            "timing_covariate"
        ]:

            required.append(
                modifier[
                    "timing_covariate"
                ]
            )

    required = list(
        dict.fromkeys(required)
    )

    d = d.dropna(
        subset=required
    ).copy()

    if len(d) < 100:
        return d

    # --------------------------------
    # M3 scaling
    # --------------------------------

    d["y_z"] = zscore(
        d[pair["outcome"]]
    )

    d["heart_z"] = zscore(
        d[pair["cardiac"]]
    )

    d["age_z"] = zscore(
        d["age_at_mri"]
    )

    d["interval_z"] = zscore(
        d["abs_delta_years"]
    )

    d["BMI_z"] = zscore(
        d["BMI_nearest_exam"]
    )

    d["SBP_z"] = zscore(
        d["SBP_nearest_exam"]
    )

    if (
        pair["metric_type"]
        == "surface_area"
    ):

        d["total_area_z"] = (
            zscore(
                d[
                    "TotalCorticalSurfaceArea"
                ]
            )
        )

    # --------------------------------
    # Metabolic scaling
    # --------------------------------

    if include_all_modifiers:

        for m in MODIFIERS:

            d[
                m["name"] + "_z"
            ] = zscore(
                d[
                    m["source"]
                    + "_transformed"
                ]
            )

        if (
            "hba1c_exam7_to_cmr_years"
            in d.columns
        ):

            d[
                "exam7_to_cmr_z"
            ] = zscore(
                d[
                    "hba1c_exam7_to_cmr_years"
                ]
            )

    else:

        d[
            "modifier_z"
        ] = zscore(
            d[
                "modifier_transformed"
            ]
        )

        if modifier[
            "timing_covariate"
        ]:

            d[
                "modifier_timing_z"
            ] = zscore(
                d[
                    modifier[
                        "timing_covariate"
                    ]
                ]
            )

    return d


# ============================================================
# Established M3 model
# ============================================================

def base_xvars(pair):

    x = [
        "heart_z",
        "APOE4_carrier",
        "heart_x_APOE4",
        "age_z",
        "_sex",
        "interval_z",
        "BMI_z",
        "SBP_z",
        "current_smoker_nearest_exam",
        "diabetes_history_any",
    ]

    if (
        pair["metric_type"]
        == "surface_area"
    ):

        x.append(
            "total_area_z"
        )

    return x


def fit_base_m3(
    d,
    pair,
    extra_main=None,
    extra_covars=None
):

    dd = d.copy()

    dd[
        "heart_x_APOE4"
    ] = (
        dd["heart_z"]
        *
        dd["APOE4_carrier"]
    )

    xvars = base_xvars(
        pair
    )

    if extra_main:

        xvars += extra_main

    if extra_covars:

        xvars += extra_covars

    xvars = list(
        dict.fromkeys(xvars)
    )

    X = sm.add_constant(
        dd[
            xvars
        ].astype(float),
        has_constant="add"
    )

    fit = sm.OLS(
        dd["y_z"].astype(float),
        X
    ).fit(
        cov_type="HC3"
    )

    return fit


# ============================================================
# Run models
# ============================================================

threeway_rows = []
slope_rows = []
atten_rows = []

for pair in PAIRS:

    for modifier in MODIFIERS:

        d = make_analysis_data(
            pair,
            modifier
        )

        if len(d) < 100:

            print(
                "SKIP N<100:",
                pair["pair_id"],
                modifier["name"],
                len(d)
            )

            continue

        # ====================================================
        # A. Matched-sample attenuation
        #
        # Original M3 and M3+metabolic fitted on EXACTLY
        # the same complete-case subjects.
        # ====================================================

        base_fit = fit_base_m3(
            d,
            pair
        )

        extra_covars = []

        if (
            "modifier_timing_z"
            in d.columns
        ):

            extra_covars.append(
                "modifier_timing_z"
            )

        adjusted_fit = (
            fit_base_m3(
                d,
                pair,
                extra_main=[
                    "modifier_z"
                ],
                extra_covars=
                    extra_covars
            )
        )

        beta0 = (
            base_fit.params[
                "heart_x_APOE4"
            ]
        )

        beta1 = (
            adjusted_fit.params[
                "heart_x_APOE4"
            ]
        )

        ci0 = (
            base_fit.conf_int()
            .loc[
                "heart_x_APOE4"
            ]
        )

        ci1 = (
            adjusted_fit.conf_int()
            .loc[
                "heart_x_APOE4"
            ]
        )

        if beta0 != 0:

            attenuation = (
                (
                    abs(beta0)
                    -
                    abs(beta1)
                )
                /
                abs(beta0)
                * 100
            )

        else:

            attenuation = np.nan

        atten_rows.append({
            "pair_id":
                pair["pair_id"],

            "association":
                pair["label"],

            "modifier":
                modifier["name"],

            "transform":
                modifier["transform"],

            "N_matched":
                int(
                    base_fit.nobs
                ),

            "N_APOE4_noncarrier":
                int(
                    (
                        d[
                            "APOE4_carrier"
                        ] == 0
                    ).sum()
                ),

            "N_APOE4_carrier":
                int(
                    (
                        d[
                            "APOE4_carrier"
                        ] == 1
                    ).sum()
                ),

            "beta_interaction_M3_matched":
                beta0,

            "CI_low_M3_matched":
                ci0.iloc[0],

            "CI_high_M3_matched":
                ci0.iloc[1],

            "p_M3_matched":
                base_fit.pvalues[
                    "heart_x_APOE4"
                ],

            "beta_interaction_plus_metabolic":
                beta1,

            "CI_low_plus_metabolic":
                ci1.iloc[0],

            "CI_high_plus_metabolic":
                ci1.iloc[1],

            "p_plus_metabolic":
                adjusted_fit.pvalues[
                    "heart_x_APOE4"
                ],

            "attenuation_abs_beta_percent":
                attenuation,
        })

        # ====================================================
        # B. Full cardiac × APOE4 × metabolic interaction
        # ====================================================

        dt = d.copy()

        dt[
            "heart_x_APOE4"
        ] = (
            dt["heart_z"]
            *
            dt["APOE4_carrier"]
        )

        dt[
            "heart_x_modifier"
        ] = (
            dt["heart_z"]
            *
            dt["modifier_z"]
        )

        dt[
            "APOE4_x_modifier"
        ] = (
            dt["APOE4_carrier"]
            *
            dt["modifier_z"]
        )

        dt[
            "heart_x_APOE4_x_modifier"
        ] = (
            dt["heart_z"]
            *
            dt["APOE4_carrier"]
            *
            dt["modifier_z"]
        )

        xvars = [
            "heart_z",
            "APOE4_carrier",
            "modifier_z",

            "heart_x_APOE4",
            "heart_x_modifier",
            "APOE4_x_modifier",

            "heart_x_APOE4_x_modifier",

            "age_z",
            "_sex",
            "interval_z",
            "BMI_z",
            "SBP_z",
            "current_smoker_nearest_exam",
            "diabetes_history_any",
        ]

        if (
            pair["metric_type"]
            == "surface_area"
        ):

            xvars.append(
                "total_area_z"
            )

        if (
            "modifier_timing_z"
            in dt.columns
        ):

            xvars.append(
                "modifier_timing_z"
            )

        X = sm.add_constant(
            dt[
                xvars
            ].astype(float),
            has_constant="add"
        )

        fit = sm.OLS(
            dt["y_z"].astype(float),
            X
        ).fit(
            cov_type="HC3"
        )

        term = (
            "heart_x_APOE4_x_modifier"
        )

        ci = (
            fit.conf_int()
            .loc[term]
        )

        threeway_rows.append({
            "pair_id":
                pair["pair_id"],

            "association":
                pair["label"],

            "cardiac":
                pair["cardiac"],

            "outcome":
                pair["outcome"],

            "modifier":
                modifier["name"],

            "modifier_source":
                modifier["source"],

            "transform":
                modifier["transform"],

            "N":
                int(fit.nobs),

            "N_APOE4_noncarrier":
                int(
                    (
                        dt[
                            "APOE4_carrier"
                        ] == 0
                    ).sum()
                ),

            "N_APOE4_carrier":
                int(
                    (
                        dt[
                            "APOE4_carrier"
                        ] == 1
                    ).sum()
                ),

            "beta_threeway":
                fit.params[term],

            "SE_threeway":
                fit.bse[term],

            "CI_low":
                ci.iloc[0],

            "CI_high":
                ci.iloc[1],

            "p_threeway":
                fit.pvalues[term],

            "R2":
                fit.rsquared,
        })

        # ====================================================
        # C. Heart→brain simple slopes
        #
        # metabolic = -1 SD, mean, +1 SD
        # separately APOE4 0 and APOE4 1
        # ====================================================

        for apoe in [0, 1]:

            for mod_z in [
                -1.0,
                0.0,
                1.0
            ]:

                terms = {
                    "heart_z": 1
                }

                if apoe == 1:

                    terms[
                        "heart_x_APOE4"
                    ] = 1

                if mod_z != 0:

                    terms[
                        "heart_x_modifier"
                    ] = mod_z

                if (
                    apoe == 1
                    and
                    mod_z != 0
                ):

                    terms[
                        "heart_x_APOE4_x_modifier"
                    ] = mod_z

                s = linear_combo(
                    fit,
                    terms
                )

                slope_rows.append({
                    "pair_id":
                        pair["pair_id"],

                    "association":
                        pair["label"],

                    "modifier":
                        modifier["name"],

                    "APOE4":
                        apoe,

                    "modifier_SD":
                        mod_z,

                    "heart_to_brain_beta":
                        s["beta"],

                    "SE":
                        s["SE"],

                    "CI_low":
                        s["CI_low"],

                    "CI_high":
                        s["CI_high"],

                    "p":
                        s["p"],

                    "N_model":
                        int(
                            fit.nobs
                        ),
                })


# ============================================================
# FDR across 12 prespecified three-way tests
# ============================================================

three = pd.DataFrame(
    threeway_rows
)

if len(three):

    valid = (
        three[
            "p_threeway"
        ].notna()
    )

    three[
        "q_threeway_FDR12"
    ] = np.nan

    three.loc[
        valid,
        "q_threeway_FDR12"
    ] = multipletests(
        three.loc[
            valid,
            "p_threeway"
        ].values,
        method="fdr_bh"
    )[1]

    three[
        "FDR_significant"
    ] = (
        three[
            "q_threeway_FDR12"
        ]
        < 0.05
    )


slopes = pd.DataFrame(
    slope_rows
)

atten = pd.DataFrame(
    atten_rows
)

# ============================================================
# D. Joint attenuation:
# HbA1c + insulin + adiponectin together
# ============================================================

joint_rows = []

for pair in PAIRS:

    d = make_analysis_data(
        pair,
        include_all_modifiers=True
    )

    if len(d) < 100:
        continue

    base_fit = fit_base_m3(
        d,
        pair
    )

    extra_main = [
        m["name"] + "_z"
        for m in MODIFIERS
    ]

    extra_covars = []

    if (
        "exam7_to_cmr_z"
        in d.columns
    ):

        extra_covars.append(
            "exam7_to_cmr_z"
        )

    adjusted_fit = fit_base_m3(
        d,
        pair,
        extra_main=
            extra_main,
        extra_covars=
            extra_covars
    )

    beta0 = (
        base_fit.params[
            "heart_x_APOE4"
        ]
    )

    beta1 = (
        adjusted_fit.params[
            "heart_x_APOE4"
        ]
    )

    ci0 = (
        base_fit.conf_int()
        .loc[
            "heart_x_APOE4"
        ]
    )

    ci1 = (
        adjusted_fit.conf_int()
        .loc[
            "heart_x_APOE4"
        ]
    )

    attenuation = (
        (
            abs(beta0)
            -
            abs(beta1)
        )
        /
        abs(beta0)
        * 100
        if beta0 != 0
        else np.nan
    )

    joint_rows.append({
        "pair_id":
            pair["pair_id"],

        "association":
            pair["label"],

        "N_common":
            int(
                base_fit.nobs
            ),

        "N_APOE4_noncarrier":
            int(
                (
                    d[
                        "APOE4_carrier"
                    ] == 0
                ).sum()
            ),

        "N_APOE4_carrier":
            int(
                (
                    d[
                        "APOE4_carrier"
                    ] == 1
                ).sum()
            ),

        "beta_interaction_M3_matched":
            beta0,

        "CI_low_M3_matched":
            ci0.iloc[0],

        "CI_high_M3_matched":
            ci0.iloc[1],

        "p_M3_matched":
            base_fit.pvalues[
                "heart_x_APOE4"
            ],

        "beta_interaction_plus_all3":
            beta1,

        "CI_low_plus_all3":
            ci1.iloc[0],

        "CI_high_plus_all3":
            ci1.iloc[1],

        "p_plus_all3":
            adjusted_fit.pvalues[
                "heart_x_APOE4"
            ],

        "attenuation_abs_beta_percent":
            attenuation,
    })


joint = pd.DataFrame(
    joint_rows
)

# ============================================================
# Save outputs
# ============================================================

three_file = (
    OUTDIR /
    "primary_pairs_metabolic_threeway.csv"
)

slope_file = (
    OUTDIR /
    "primary_pairs_metabolic_simple_slopes.csv"
)

atten_file = (
    OUTDIR /
    "primary_pairs_metabolic_attenuation.csv"
)

joint_file = (
    OUTDIR /
    "primary_pairs_metabolic_joint_attenuation.csv"
)

summary_file = (
    OUTDIR /
    "primary_pairs_metabolic_summary.txt"
)


three.to_csv(
    three_file,
    index=False
)

slopes.to_csv(
    slope_file,
    index=False
)

atten.to_csv(
    atten_file,
    index=False
)

joint.to_csv(
    joint_file,
    index=False
)

# ============================================================
# Human-readable summary
# ============================================================

with open(
    summary_file,
    "w"
) as fh:

    fh.write(
        "PRIMARY HEART-BRAIN PAIRS × "
        "METABOLIC MODIFIERS\n"
    )

    fh.write(
        "=" * 80
        + "\n\n"
    )

    fh.write(
        "Modifiers: HbA1c, log(insulin), "
        "log(adiponectin)\n"
    )

    fh.write(
        "FDR family: 4 primary pairs × "
        "3 modifiers = 12 tests\n"
    )

    fh.write(
        "All models use HC3 robust SEs and "
        "the established M3 covariates.\n\n"
    )

    if len(three):

        fh.write(
            "THREE-WAY EFFECT MODIFICATION\n"
        )

        fh.write(
            "-" * 80
            + "\n"
        )

        show = (
            three
            .sort_values(
                "p_threeway"
            )[
                [
                    "association",
                    "modifier",
                    "N",
                    "beta_threeway",
                    "CI_low",
                    "CI_high",
                    "p_threeway",
                    "q_threeway_FDR12",
                ]
            ]
        )

        fh.write(
            show.to_string(
                index=False
            )
        )

        fh.write(
            "\n\n"
        )

        sig = three[
            three[
                "q_threeway_FDR12"
            ] < 0.05
        ]

        fh.write(
            f"FDR-significant: "
            f"{len(sig)}/{len(three)}\n"
        )

    if len(atten):

        fh.write(
            "\nMATCHED-SAMPLE ATTENUATION\n"
        )

        fh.write(
            "-" * 80
            + "\n"
        )

        show = atten[
            [
                "association",
                "modifier",
                "N_matched",
                "beta_interaction_M3_matched",
                "beta_interaction_plus_metabolic",
                "attenuation_abs_beta_percent",
                "p_plus_metabolic",
            ]
        ]

        fh.write(
            show.to_string(
                index=False
            )
        )

        fh.write(
            "\n"
        )

    if len(joint):

        fh.write(
            "\nJOINT HbA1c + INSULIN + "
            "ADIPONECTIN ATTENUATION\n"
        )

        fh.write(
            "-" * 80
            + "\n"
        )

        fh.write(
            joint.to_string(
                index=False
            )
        )

        fh.write(
            "\n"
        )


# ============================================================
# Console output
# ============================================================

print(
    "\n========================================"
)

print(
    "THREE-WAY RESULTS"
)

print(
    "========================================"
)

if len(three):

    print(
        three
        .sort_values(
            "p_threeway"
        )[
            [
                "association",
                "modifier",
                "N",
                "beta_threeway",
                "p_threeway",
                "q_threeway_FDR12",
            ]
        ]
        .to_string(
            index=False
        )
    )


print(
    "\n========================================"
)

print(
    "MATCHED-SAMPLE ATTENUATION"
)

print(
    "========================================"
)

if len(atten):

    print(
        atten[
            [
                "association",
                "modifier",
                "N_matched",
                "beta_interaction_M3_matched",
                "beta_interaction_plus_metabolic",
                "attenuation_abs_beta_percent",
            ]
        ]
        .to_string(
            index=False
        )
    )


print(
    "\n========================================"
)

print(
    "JOINT METABOLIC ATTENUATION"
)

print(
    "========================================"
)

if len(joint):

    print(
        joint[
            [
                "association",
                "N_common",
                "beta_interaction_M3_matched",
                "beta_interaction_plus_all3",
                "attenuation_abs_beta_percent",
            ]
        ]
        .to_string(
            index=False
        )
    )


print(
    "\nSaved:"
)

print(
    three_file
)

print(
    slope_file
)

print(
    atten_file
)

print(
    joint_file
)

print(
    summary_file
)

