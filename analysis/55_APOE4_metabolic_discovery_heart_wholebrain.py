#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

# ============================================================
# PATHS
# ============================================================

ROOT = Path("/data/qiallab/Framingham")
RES  = ROOT / "results"

MAIN = RES / "neurocardiac_metadata_analysis_ready_metabolic.csv"
ATRIA = RES / "neurocardiac_metadata_with_atria.csv"
STRAIN = RES / "LV_RV_3D_strain" / "neurocardiac_with_LV_RV_3D_strain.csv"

OUTDIR = RES / "APOE4_metabolic_discovery_heart_wholebrain"
OUTDIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# LOAD MAIN DATA
# ============================================================

df = pd.read_csv(MAIN)

print("Main:", df.shape)

if "shareid" not in df.columns:
    raise RuntimeError("shareid missing from main table")

# ============================================================
# OPTIONAL MERGE: ATRIA
# ============================================================

if ATRIA.exists():

    atr = pd.read_csv(ATRIA)

    atr_cols = [
        c for c in [
            "shareid",
            "LAVI_max",
            "LAVI_min",
            "LA_total_emptying_fraction"
        ]
        if c in atr.columns
    ]

    if len(atr_cols) > 1:

        atr = atr[atr_cols].drop_duplicates("shareid")

        # avoid duplicate columns
        add_cols = [
            c for c in atr_cols
            if c == "shareid" or c not in df.columns
        ]

        df = df.merge(
            atr[add_cols],
            on="shareid",
            how="left"
        )

        print("After atrial merge:", df.shape)


# ============================================================
# OPTIONAL MERGE: LV/RV STRAIN
# ============================================================

if STRAIN.exists():

    st = pd.read_csv(STRAIN)

    strain_cols = [
        c for c in st.columns
        if (
            c == "shareid"
            or c.startswith("LV_3D_")
            or c.startswith("RV_3D_")
        )
    ]

    if "shareid" in strain_cols and len(strain_cols) > 1:

        st = st[strain_cols].drop_duplicates("shareid")

        add_cols = [
            c for c in strain_cols
            if c == "shareid" or c not in df.columns
        ]

        df = df.merge(
            st[add_cols],
            on="shareid",
            how="left"
        )

        print("After strain merge:", df.shape)


# ============================================================
# VARIABLES
# ============================================================

APOE = "APOE4_carrier"
DIAB = "diabetes_history_nearest_exam"
HBA1C = "hba1c_exam7"

SEX = "sex_clinical"
AGE_BRAIN = "age_at_mri"
AGE_HEART = "age_at_cmr"

for c in [
    APOE,
    DIAB,
    HBA1C,
    AGE_BRAIN,
    AGE_HEART
]:
    df[c] = pd.to_numeric(
        df[c],
        errors="coerce"
    )

# ============================================================
# DIABETES DEFINITION
#
# 0 = no
# 1 = current diabetes / high blood sugar
#
# Exam 8 code 2 = "yes, not now"
# therefore excluded from primary current-diabetes analysis.
# ============================================================

df["DIABETES_CURRENT"] = df[DIAB]

df.loc[
    ~df["DIABETES_CURRENT"].isin([0, 1]),
    "DIABETES_CURRENT"
] = np.nan


# ============================================================
# HbA1c STANDARDIZATION
# ============================================================

df["HbA1c_z"] = (
    df[HBA1C] - df[HBA1C].mean()
) / df[HBA1C].std(ddof=0)


# ============================================================
# DERIVE INDEXED LV VARIABLES
# ============================================================

if all(
    c in df.columns
    for c in ["height_in", "weight_lb"]
):

    height_cm = (
        pd.to_numeric(
            df["height_in"],
            errors="coerce"
        ) * 2.54
    )

    weight_kg = (
        pd.to_numeric(
            df["weight_lb"],
            errors="coerce"
        ) * 0.45359237
    )

    df["BSA"] = np.sqrt(
        height_cm * weight_kg / 3600.0
    )

    if "LVEDV" in df.columns:
        df["LVEDVi_derived"] = (
            pd.to_numeric(
                df["LVEDV"],
                errors="coerce"
            ) / df["BSA"]
        )

    if "LVESV" in df.columns:
        df["LVESVi_derived"] = (
            pd.to_numeric(
                df["LVESV"],
                errors="coerce"
            ) / df["BSA"]
        )

    if "LV_MASS" in df.columns:
        df["LV_MASSi_derived"] = (
            pd.to_numeric(
                df["LV_MASS"],
                errors="coerce"
            ) / df["BSA"]
        )


# ============================================================
# WHOLE-BRAIN OUTCOMES
# ============================================================

def brain_class(c):

    if c.endswith("_area"):
        return "AREA"

    if c.endswith("_vol"):
        return "VOL"

    if c.endswith("_tksd"):
        return "TKSD"

    if c.endswith("_tk"):
        return "TK"

    if c.endswith("_mcv"):
        return "MCV"

    return None


brain_outcomes = []

for c in df.columns:

    pc = brain_class(c)

    if pc is None:
        continue

    # restrict to left/right cortical variables
    if not (
        c.startswith("lh_")
        or c.startswith("rh_")
    ):
        continue

    brain_outcomes.append(c)

brain_outcomes = sorted(set(brain_outcomes))

print("\nWhole-brain outcomes:", len(brain_outcomes))

for pc in ["AREA", "VOL", "TK", "TKSD", "MCV"]:

    n = sum(
        brain_class(c) == pc
        for c in brain_outcomes
    )

    print(pc, n)


# ============================================================
# CARDIAC OUTCOMES
# ============================================================

cardiac_groups = {

    "LV_CHAMBER": [
        "LVEF",
        "LVEDVi_derived",
        "LVESVi_derived",
        "LV_MASSi_derived",
    ],

    "ATRIAL": [
        "LAVI_max",
        "LAVI_min",
        "LA_total_emptying_fraction",
    ],

    "LV_STRAIN": [
        "LV_3D_LONG",
        "LV_3D_CIRC",
        "LV_3D_RAD",
    ],

    "RV_STRAIN": [
        "RV_3D_LONG",
        "RV_3D_CIRC",
        "RV_3D_RAD",
    ],
}

cardiac_outcomes = []

for family, cols in cardiac_groups.items():

    for c in cols:

        if c in df.columns:

            cardiac_outcomes.append(
                (family, c)
            )

print("\nCardiac outcomes:")

for family, c in cardiac_outcomes:
    print(f"  {family:12s} {c}")


# ============================================================
# STANDARDIZATION FUNCTION
# ============================================================

def zscore(s):

    s = pd.to_numeric(
        s,
        errors="coerce"
    )

    sd = s.std(ddof=0)

    if pd.isna(sd) or sd == 0:
        return pd.Series(
            np.nan,
            index=s.index
        )

    return (
        s - s.mean()
    ) / sd


# ============================================================
# MODEL FUNCTION
#
# Outcome is z-standardized.
#
# Therefore beta_interaction is in outcome SD units.
# ============================================================

def run_model(
    outcome,
    modifier,
    domain,
    phenotype_class,
    agevar
):

    needed = [
        outcome,
        APOE,
        modifier,
        agevar,
        SEX,
    ]

    d = df[needed].copy()

    d["Y"] = zscore(
        d[outcome]
    )

    d = d.dropna(
        subset=[
            "Y",
            APOE,
            modifier,
            agevar,
            SEX
        ]
    )

    if len(d) < 50:
        return None

    if d[APOE].nunique() < 2:
        return None

    if d[modifier].nunique() < 2:
        return None

    formula = (
        f"Y ~ "
        f"{APOE} * {modifier}"
        f" + {agevar}"
        f" + C({SEX})"
    )

    try:

        fit = smf.ols(
            formula,
            data=d
        ).fit(
            cov_type="HC3"
        )

        term = f"{APOE}:{modifier}"

        if term not in fit.params.index:
            term = f"{modifier}:{APOE}"

        beta = fit.params[term]
        se = fit.bse[term]
        p = fit.pvalues[term]

        ci = fit.conf_int().loc[term]

        beta_mod = fit.params[modifier]

        out = {

            "domain": domain,
            "phenotype_class": phenotype_class,

            "modifier": modifier,
            "outcome": outcome,

            "N": int(fit.nobs),

            "beta_interaction": beta,
            "SE_interaction": se,

            "CI_low": ci.iloc[0],
            "CI_high": ci.iloc[1],

            "p_interaction": p,

            "beta_modifier_APOE0": beta_mod,
            "beta_modifier_APOE1": (
                beta_mod + beta
            ),

            "R2": fit.rsquared,
            "R2_adj": fit.rsquared_adj,
        }

        if modifier == "DIABETES_CURRENT":

            out.update({

                "N_APOE0_DIAB0": int(
                    (
                        (d[APOE] == 0)
                        &
                        (d[modifier] == 0)
                    ).sum()
                ),

                "N_APOE0_DIAB1": int(
                    (
                        (d[APOE] == 0)
                        &
                        (d[modifier] == 1)
                    ).sum()
                ),

                "N_APOE1_DIAB0": int(
                    (
                        (d[APOE] == 1)
                        &
                        (d[modifier] == 0)
                    ).sum()
                ),

                "N_APOE1_DIAB1": int(
                    (
                        (d[APOE] == 1)
                        &
                        (d[modifier] == 1)
                    ).sum()
                ),
            })

        return out

    except Exception as e:

        print(
            "FAILED:",
            domain,
            phenotype_class,
            outcome,
            modifier,
            e
        )

        return None


# ============================================================
# RUN DISCOVERY
# ============================================================

results = []

MODIFIERS = [
    "DIABETES_CURRENT",
    "HbA1c_z",
]


# ------------------------------------------------------------
# WHOLE BRAIN
# ------------------------------------------------------------

for outcome in brain_outcomes:

    pc = brain_class(outcome)

    for modifier in MODIFIERS:

        r = run_model(
            outcome=outcome,
            modifier=modifier,
            domain="BRAIN",
            phenotype_class=pc,
            agevar=AGE_BRAIN,
        )

        if r is not None:
            results.append(r)


# ------------------------------------------------------------
# HEART
# ------------------------------------------------------------

for family, outcome in cardiac_outcomes:

    for modifier in MODIFIERS:

        r = run_model(
            outcome=outcome,
            modifier=modifier,
            domain="HEART",
            phenotype_class=family,
            agevar=AGE_HEART,
        )

        if r is not None:
            results.append(r)


res = pd.DataFrame(results)

print("\nModels completed:", len(res))


# ============================================================
# FDR
#
# Correct separately within:
#
# modifier × domain × phenotype class
#
# Examples:
# APOE4×HbA1c / BRAIN / VOL
# APOE4×HbA1c / BRAIN / AREA
# APOE4×diabetes / HEART / LV_CHAMBER
# etc.
# ============================================================

res["q_interaction"] = np.nan

for keys, idx in res.groupby(
    [
        "modifier",
        "domain",
        "phenotype_class"
    ]
).groups.items():

    idx = list(idx)

    pvals = res.loc[
        idx,
        "p_interaction"
    ].values

    good = np.isfinite(pvals)

    if good.sum() == 0:
        continue

    q = multipletests(
        pvals[good],
        method="fdr_bh"
    )[1]

    use_idx = np.array(idx)[good]

    res.loc[
        use_idx,
        "q_interaction"
    ] = q


res["FDR_q05"] = (
    res["q_interaction"] < 0.05
)

res["nominal_p05"] = (
    res["p_interaction"] < 0.05
)

res["nominal_p001"] = (
    res["p_interaction"] < 0.001
)


# ============================================================
# SAVE ALL
# ============================================================

all_file = (
    OUTDIR /
    "APOE4_metabolic_discovery_all.csv"
)

res.to_csv(
    all_file,
    index=False
)


# ============================================================
# SEPARATE BRAIN / HEART
# ============================================================

brain = res[
    res["domain"] == "BRAIN"
].copy()

heart = res[
    res["domain"] == "HEART"
].copy()

brain.to_csv(
    OUTDIR /
    "APOE4_metabolic_wholebrain_all.csv",
    index=False
)

heart.to_csv(
    OUTDIR /
    "APOE4_metabolic_cardiac_all.csv",
    index=False
)


# ============================================================
# FDR SIGNIFICANT
# ============================================================

res[
    res["FDR_q05"]
].sort_values(
    "q_interaction"
).to_csv(
    OUTDIR /
    "APOE4_metabolic_discovery_FDR.csv",
    index=False
)


# ============================================================
# TOP 20 PER FAMILY
# ============================================================

top = (
    res
    .sort_values(
        "p_interaction"
    )
    .groupby(
        [
            "modifier",
            "domain",
            "phenotype_class"
        ],
        group_keys=False
    )
    .head(20)
)

top.to_csv(
    OUTDIR /
    "APOE4_metabolic_discovery_top20_per_family.csv",
    index=False
)


# ============================================================
# FAMILY SUMMARY
# ============================================================

summary_rows = []

for (
    modifier,
    domain,
    pc
), g in res.groupby(
    [
        "modifier",
        "domain",
        "phenotype_class"
    ]
):

    g = g.sort_values(
        "p_interaction"
    )

    best = g.iloc[0]

    summary_rows.append({

        "modifier": modifier,
        "domain": domain,
        "phenotype_class": pc,

        "N_tests": len(g),

        "N_FDR_q05": int(
            (g["q_interaction"] < 0.05).sum()
        ),

        "N_nominal_p001": int(
            (g["p_interaction"] < 0.001).sum()
        ),

        "N_nominal_p05": int(
            (g["p_interaction"] < 0.05).sum()
        ),

        "best_outcome":
            best["outcome"],

        "best_beta":
            best["beta_interaction"],

        "best_p":
            best["p_interaction"],

        "best_q":
            best["q_interaction"],
    })


summary = pd.DataFrame(
    summary_rows
)

summary.to_csv(
    OUTDIR /
    "APOE4_metabolic_discovery_family_summary.csv",
    index=False
)


# ============================================================
# PRINT SUMMARY
# ============================================================

pd.set_option(
    "display.max_rows",
    100
)

pd.set_option(
    "display.width",
    220
)

pd.set_option(
    "display.max_columns",
    30
)


print("\n")
print("=" * 110)
print("DISCOVERY FAMILY SUMMARY")
print("=" * 110)

print(
    summary.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.5g}"
    )
)


print("\n")
print("=" * 110)
print("FDR-SIGNIFICANT RESULTS")
print("=" * 110)

sig = res[
    res["q_interaction"] < 0.05
].sort_values(
    "q_interaction"
)

if len(sig) == 0:

    print("None.")

else:

    print(
        sig[
            [
                "modifier",
                "domain",
                "phenotype_class",
                "outcome",
                "N",
                "beta_interaction",
                "CI_low",
                "CI_high",
                "p_interaction",
                "q_interaction"
            ]
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.5g}"
        )
    )


print("\n")
print("=" * 110)
print("TOP NOMINAL RESULTS p < 0.001")
print("=" * 110)

nom = res[
    res["p_interaction"] < 0.001
].sort_values(
    "p_interaction"
)

if len(nom) == 0:

    print("None.")

else:

    print(
        nom[
            [
                "modifier",
                "domain",
                "phenotype_class",
                "outcome",
                "N",
                "beta_interaction",
                "p_interaction",
                "q_interaction"
            ]
        ].head(50).to_string(
            index=False,
            float_format=lambda x:
                f"{x:.5g}"
        )
    )


print("\nSaved to:")
print(OUTDIR)

