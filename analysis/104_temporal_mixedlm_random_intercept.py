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
DATA = ROOT / "data/longitudinal_20260924"

RESULTS = (
    ROOT / "results"
    / "longitudinal_heart_brain"
    / "temporal_prediction"
    / "mixedlm_random_intercept"
)
RESULTS.mkdir(parents=True, exist_ok=True)

ANALYSIS_DATA = DATA / "longitudinal_analysis"

SUBJECT_FILE = ANALYSIS_DATA / "E4_E6_cardiac_MRI_with_APOE.tsv"

print("=" * 100)
print("104 TEMPORAL HEART-BRAIN ANALYSIS")
print("Random-intercept-only MixedLM")
print("RESULTS:", RESULTS)
print("=" * 100)

# ============================================================
# HELPERS
# ============================================================

def norm_id(s):
    s = s.astype(str).str.strip()
    s = s.str.replace(r"\.0$", "", regex=True)
    s = s.replace(["", "nan", "NaN", "NA", "N/A", "."], np.nan)
    return s


def find_pht(pht):
    hits = sorted(DATA.glob(f"**/*{pht}*.HMB-IRB-MDS.txt.gz"))
    hits = [
        x for x in hits
        if "data_dict" not in x.name
        and "variable_report" not in x.name
    ]
    if not hits:
        raise FileNotFoundError(f"No phenotype file found for {pht}")
    print(f"{pht}: {hits[0]}")
    return hits[0]


def read_pht(pht):
    f = find_pht(pht)
    d = pd.read_csv(
        f,
        sep="\t",
        dtype=str,
        low_memory=False,
        comment="#"
    )
    d.columns = [c.strip() for c in d.columns]
    if "shareid" in d.columns:
        d["shareid"] = norm_id(d["shareid"])
    return d


def numeric(d, cols):
    for c in cols:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


def bh_qvalues(df, p_col="P"):
    q = pd.Series(np.nan, index=df.index, dtype=float)
    ok = df[p_col].notna()

    if ok.sum() > 0:
        q.loc[ok] = multipletests(
            df.loc[ok, p_col],
            method="fdr_bh"
        )[1]

    return q


# ============================================================
# SUBJECT-LEVEL CARDIAC TRAJECTORIES + APOE
# ============================================================

subj = pd.read_csv(
    SUBJECT_FILE,
    sep="\t",
    dtype={"shareid": str},
    low_memory=False
)
subj["shareid"] = norm_id(subj["shareid"])

print("\nSubject trajectory file:")
print("rows:", len(subj))
print("subjects:", subj["shareid"].nunique())

# ============================================================
# AGE 6 + SEX
# ============================================================

dates = read_pht("pht003099")

needed = ["shareid", "sex", "age6", "date6"]
missing = [x for x in needed if x not in dates.columns]
if missing:
    raise RuntimeError(f"Missing pht003099 fields: {missing}")

dates = dates[needed].copy()
dates = numeric(dates, ["sex", "age6", "date6"])
dates = dates.drop_duplicates("shareid")

subj = subj.merge(
    dates,
    on="shareid",
    how="left",
    validate="one_to_one"
)

# ============================================================
# STRUCTURAL MRI -- FRAM4
# ============================================================

mri = read_pht("pht015152")

print("\nFRAM4 columns relevant to MRI:")
print([
    c for c in mri.columns
    if any(k in c.lower() for k in [
        "date", "hippo", "brain", "vent"
    ])
])

required_mri = [
    "shareid",
    "mri_date",
    "Hippo",
    "Total_brain",
    "Lateralvent",
]

missing = [x for x in required_mri if x not in mri.columns]
if missing:
    raise RuntimeError(
        f"Missing expected FRAM4 MRI variables: {missing}"
    )

mri = mri[required_mri].copy()

mri = numeric(
    mri,
    [
        "mri_date",
        "Hippo",
        "Total_brain",
        "Lateralvent",
    ]
)

mri = mri.dropna(
    subset=["shareid", "mri_date"]
)

# ============================================================
# WMH
# ============================================================

wmh = read_pht("pht004365")

print("\nPossible WMH/date columns:")
print([
    c for c in wmh.columns
    if any(k in c.lower() for k in [
        "wmh", "white", "hyper", "date"
    ])
])

# Search flexibly for WMH field
WMH_CANDIDATES = [
    "FLAIR_wmh",
    "WMH",
    "wmh",
    "WMH_volume",
    "wmh_volume",
]

wmh_col = next(
    (c for c in WMH_CANDIDATES if c in wmh.columns),
    None
)

if wmh_col is None:
    possible = [
        c for c in wmh.columns
        if "wmh" in c.lower()
    ]
    if len(possible) == 1:
        wmh_col = possible[0]

if wmh_col is None:
    print(
        "\nWARNING: Could not identify a unique WMH column."
        "\n104 will run the 3 structural outcomes only."
    )
else:
    print("Selected WMH column:", wmh_col)

    date_candidates = [
        "mri_date",
        "MRI_date",
        "date",
        "scan_date"
    ]

    wmh_date = next(
        (c for c in date_candidates if c in wmh.columns),
        None
    )

    if wmh_date is None:
        possible_dates = [
            c for c in wmh.columns
            if "date" in c.lower()
        ]
        if len(possible_dates) == 1:
            wmh_date = possible_dates[0]

    if wmh_date is None:
        print(
            "WARNING: WMH date column not uniquely identified."
            " WMH will not be analyzed in this run."
        )
        wmh_col = None
    else:
        print("Selected WMH date column:", wmh_date)

        w = wmh[
            ["shareid", wmh_date, wmh_col]
        ].copy()

        w = numeric(
            w,
            [wmh_date, wmh_col]
        )

        w = w.rename(
            columns={
                wmh_date: "mri_date",
                wmh_col: "WMH"
            }
        )

        w = w.dropna(
            subset=[
                "shareid",
                "mri_date",
                "WMH"
            ]
        )

        w = w.drop_duplicates(
            ["shareid", "mri_date"]
        )

        mri = mri.merge(
            w,
            on=["shareid", "mri_date"],
            how="outer"
        )

# ============================================================
# MERGE MRI WITH SUBJECT-LEVEL EXPOSURES
# ============================================================

long = mri.merge(
    subj,
    on="shareid",
    how="inner",
    validate="many_to_one"
)

long["first_mri_date_model"] = (
    long.groupby("shareid")["mri_date"]
    .transform("min")
)

long["mri_time_years"] = (
    long["mri_date"]
    - long["first_mri_date_model"]
) / 365.25

long["APOE4_carrier"] = pd.to_numeric(
    long["APOE4_carrier"],
    errors="coerce"
)

long["age6"] = pd.to_numeric(
    long["age6"],
    errors="coerce"
)

long["sex"] = pd.to_numeric(
    long["sex"],
    errors="coerce"
)

long["age6_c"] = (
    long["age6"]
    - long["age6"].mean()
)

# ============================================================
# EXPOSURES
# ============================================================

CARDIAC = {
    "LV_mass":
        "lv_mass_derived_g",
    "LA_dimension":
        "la_dim",
    "LVDD":
        "lvdd",
    "fractional_shortening":
        "fs_derived_pct",
}

OUTCOMES = {
    "hippocampus":
        "Hippo",
    "total_brain":
        "Total_brain",
    "lateral_ventricles":
        "Lateralvent",
}

if "WMH" in long.columns:
    OUTCOMES["WMH"] = "WMH"

print("\nOutcomes being modeled:")
for k, v in OUTCOMES.items():
    print(f"  {k}: {v}")

SUBSETS = {
    "prospective": 0.0,
    "gap_ge2y": 2.0,
}

# ============================================================
# RUN MIXED MODELS
# ============================================================

fit_rows = []
coef_rows = []
qc_rows = []

for subset_name, min_gap in SUBSETS.items():

    print("\n" + "=" * 100)
    print("SUBSET:", subset_name)
    print("=" * 100)

    for cardiac_name, cardiac_base in CARDIAC.items():

        ncol = f"{cardiac_base}_n"
        scol = f"{cardiac_base}_slope"

        if ncol not in long.columns:
            raise RuntimeError(f"Missing {ncol}")

        if scol not in long.columns:
            raise RuntimeError(f"Missing {scol}")

        for outcome_name, outcome_col in OUTCOMES.items():

            d = long.copy()

            # PRIMARY CARDIAC REQUIREMENT:
            # all 3 E4/E5/E6 measurements
            d = d[
                (pd.to_numeric(d[ncol], errors="coerce") >= 3)
                & d[scol].notna()
                & d["APOE4_carrier"].notna()
                & d["age6"].notna()
                & d["sex"].notna()
                & (
                    pd.to_numeric(
                        d["last_echo_to_first_mri_years"],
                        errors="coerce"
                    ) >= min_gap
                )
                & d[outcome_col].notna()
                & d["mri_time_years"].notna()
            ].copy()

            # Need repeated outcome
            counts = (
                d.groupby("shareid")[outcome_col]
                .count()
            )
            good_ids = counts[counts >= 2].index
            d = d[
                d["shareid"].isin(good_ids)
            ].copy()

            nsub = d["shareid"].nunique()
            nobs = len(d)

            if nsub < 50:
                print(
                    f"SKIP {subset_name} / {cardiac_name} / "
                    f"{outcome_name}: N={nsub}"
                )
                continue

            # Subject-level cardiac slope mean/SD
            cs = (
                d[["shareid", scol]]
                .drop_duplicates("shareid")
            )

            cmean = cs[scol].mean()
            csd = cs[scol].std()

            if pd.isna(csd) or csd == 0:
                continue

            d["cardiac_z"] = (
                d[scol] - cmean
            ) / csd

            # Standardize outcome across observations
            om = d[outcome_col].mean()
            osd = d[outcome_col].std()

            if pd.isna(osd) or osd == 0:
                continue

            d["brain_z"] = (
                d[outcome_col] - om
            ) / osd

            # APOE subject counts
            apoe = (
                d[
                    ["shareid", "APOE4_carrier"]
                ]
                .drop_duplicates("shareid")
            )

            n_e4_pos = int(
                (apoe["APOE4_carrier"] == 1).sum()
            )

            n_e4_neg = int(
                (apoe["APOE4_carrier"] == 0).sum()
            )

            qc_rows.append({
                "subset": subset_name,
                "cardiac": cardiac_name,
                "outcome": outcome_name,
                "N_subjects": nsub,
                "N_observations": nobs,
                "N_APOE4_carrier": n_e4_pos,
                "N_APOE4_noncarrier": n_e4_neg,
                "median_MRI_visits": (
                    d.groupby("shareid")
                    .size()
                    .median()
                ),
                "median_MRI_followup_y": (
                    d.groupby("shareid")["mri_time_years"]
                    .max()
                    .median()
                ),
            })

            formula = (
                "brain_z ~ "
                "mri_time_years * cardiac_z * APOE4_carrier "
                "+ age6_c + C(sex)"
            )

            print(
                f"{cardiac_name:24s} "
                f"{outcome_name:22s} "
                f"N={nsub:4d}, obs={nobs:5d}, "
                f"E4+={n_e4_pos:3d}"
            )

            try:

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")

                    model = smf.mixedlm(
                        formula,
                        data=d,
                        groups=d["shareid"],
                        re_formula="1"
                    )

                    # Try LBFGS first
                    res = model.fit(
                        reml=False,
                        method="lbfgs",
                        maxiter=2000,
                        disp=False
                    )

                    # If needed, retry Powell
                    if not getattr(
                        res,
                        "converged",
                        False
                    ):
                        res = model.fit(
                            reml=False,
                            method="powell",
                            maxiter=5000,
                            disp=False
                        )

                converged = bool(
                    getattr(
                        res,
                        "converged",
                        False
                    )
                )

                fit_rows.append({
                    "subset": subset_name,
                    "cardiac": cardiac_name,
                    "outcome": outcome_name,
                    "N_subjects": nsub,
                    "N_observations": nobs,
                    "N_APOE4_carrier": n_e4_pos,
                    "N_APOE4_noncarrier": n_e4_neg,
                    "converged": converged,
                    "optimizer": (
                        "converged"
                        if converged
                        else "failed_to_converge"
                    ),
                    "logLik": res.llf,
                    "AIC": res.aic,
                    "BIC": res.bic,
                })

                ci = res.conf_int()

                for term in res.params.index:

                    if "Group Var" in term:
                        continue

                    coef_rows.append({
                        "subset": subset_name,
                        "cardiac": cardiac_name,
                        "outcome": outcome_name,
                        "term": term,
                        "beta":
                            res.params.get(
                                term,
                                np.nan
                            ),
                        "SE":
                            res.bse.get(
                                term,
                                np.nan
                            ),
                        "CI_low":
                            ci.loc[term, 0]
                            if term in ci.index
                            else np.nan,
                        "CI_high":
                            ci.loc[term, 1]
                            if term in ci.index
                            else np.nan,
                        "P":
                            res.pvalues.get(
                                term,
                                np.nan
                            ),
                        "N_subjects": nsub,
                        "N_observations": nobs,
                        "N_APOE4_carrier": n_e4_pos,
                        "N_APOE4_noncarrier": n_e4_neg,
                        "converged": converged,
                    })

            except Exception as e:

                print("ERROR:", repr(e))

                fit_rows.append({
                    "subset": subset_name,
                    "cardiac": cardiac_name,
                    "outcome": outcome_name,
                    "N_subjects": nsub,
                    "N_observations": nobs,
                    "N_APOE4_carrier": n_e4_pos,
                    "N_APOE4_noncarrier": n_e4_neg,
                    "converged": False,
                    "error": repr(e),
                })

# ============================================================
# TABLES
# ============================================================

fits = pd.DataFrame(fit_rows)
coef = pd.DataFrame(coef_rows)
qc = pd.DataFrame(qc_rows)

two_way = "mri_time_years:cardiac_z"
three_way = (
    "mri_time_years:"
    "cardiac_z:"
    "APOE4_carrier"
)

temporal = coef[
    coef["term"].isin(
        [two_way, three_way]
    )
].copy()

temporal["effect_type"] = np.where(
    temporal["term"] == two_way,
    "cardiac_trajectory_x_MRI_time",
    "cardiac_trajectory_x_MRI_time_x_APOE4"
)

# FDR independently within:
# subset × effect type
#
# With WMH present:
# 4 cardiac x 4 outcomes = 16 tests/family
temporal["FDR_q"] = np.nan

for (subset, effect), idx in temporal.groupby(
    ["subset", "effect_type"]
).groups.items():

    ix = list(idx)

    temporal.loc[ix, "FDR_q"] = bh_qvalues(
        temporal.loc[ix]
    )

primary = temporal[
    temporal["effect_type"]
    ==
    "cardiac_trajectory_x_MRI_time_x_APOE4"
].copy()

main_effect = temporal[
    temporal["effect_type"]
    ==
    "cardiac_trajectory_x_MRI_time"
].copy()

primary = primary.sort_values(
    ["subset", "P"]
)

main_effect = main_effect.sort_values(
    ["subset", "P"]
)

# ============================================================
# SAVE ONLY UNDER RESULTS
# ============================================================

qc.to_csv(
    RESULTS / "model_eligibility_QC.tsv",
    sep="\t",
    index=False
)

fits.to_csv(
    RESULTS / "mixed_model_fit_summary.tsv",
    sep="\t",
    index=False
)

coef.to_csv(
    RESULTS / "all_mixed_model_coefficients.tsv",
    sep="\t",
    index=False
)

temporal.to_csv(
    RESULTS / "temporal_interaction_terms.tsv",
    sep="\t",
    index=False
)

primary.to_csv(
    RESULTS / "PRIMARY_APOE4_temporal_interactions.tsv",
    sep="\t",
    index=False
)

main_effect.to_csv(
    RESULTS / "cardiac_trajectory_temporal_effects.tsv",
    sep="\t",
    index=False
)

# Save analysis-ready model rows here because this is
# a derived analysis dataset, not a statistical result.
ANALYSIS_OUT = (
    ROOT / "data"
    / "longitudinal_derived"
)
ANALYSIS_OUT.mkdir(
    parents=True,
    exist_ok=True
)

long.to_csv(
    ANALYSIS_OUT
    / "E4_E6_repeated_MRI_temporal_analysis.tsv",
    sep="\t",
    index=False
)

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 100)
print("CONVERGENCE SUMMARY")
print("=" * 100)

if len(fits):
    print(
        fits.groupby(
            ["subset", "converged"]
        ).size()
    )

print("\n" + "=" * 100)
print("PRIMARY APOE4 TEMPORAL INTERACTIONS")
print("=" * 100)

cols = [
    "subset",
    "cardiac",
    "outcome",
    "beta",
    "SE",
    "P",
    "FDR_q",
    "N_subjects",
    "N_APOE4_carrier",
    "converged",
]

if len(primary):
    print(
        primary[cols].to_string(
            index=False,
            float_format=lambda x: f"{x:.5g}"
        )
    )

print("\n" + "=" * 100)
print("CARDIAC TRAJECTORY x MRI TIME")
print("=" * 100)

if len(main_effect):
    print(
        main_effect[cols].to_string(
            index=False,
            float_format=lambda x: f"{x:.5g}"
        )
    )

print("\nRESULTS SAVED TO:")
print(RESULTS)

print("\nDERIVED ANALYSIS DATA SAVED TO:")
print(ANALYSIS_OUT)

print("\n104 COMPLETE")
