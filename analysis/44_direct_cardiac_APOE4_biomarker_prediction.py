#!/usr/bin/env python3

from pathlib import Path
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests


# ============================================================
# Paths
# ============================================================

ROOT = Path("/data/qiallab/Framingham")
CLIN = ROOT / "downloads/clinical"
RESULTS = ROOT / "results"

DATAFILE = RESULTS / "neurocardiac_metadata_with_atria.csv"

OUTDIR = RESULTS / "direct_biomarker_prediction"
OUTDIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Helpers
# ============================================================

def load_dbgap(pattern, keep):
    files = sorted(CLIN.glob(pattern))

    if not files:
        raise RuntimeError(f"No file found matching: {pattern}")

    path = files[0]

    table = pd.read_csv(
        path,
        sep="\t",
        compression="gzip",
        comment="#",
        low_memory=False,
    )

    missing = [
        column
        for column in keep
        if column not in table.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing columns in {path.name}: {missing}"
        )

    table = table[keep].copy()

    table["shareid"] = pd.to_numeric(
        table["shareid"],
        errors="coerce",
    )

    print(
        f"Loaded {path.name}: "
        f"{len(table):,} rows; "
        f"{table['shareid'].nunique():,} unique shareids"
    )

    duplicate_count = int(
        table["shareid"].duplicated(
            keep=False
        ).sum()
    )

    if duplicate_count:
        print(
            f"WARNING: {duplicate_count} rows have duplicated "
            "shareid values. Numeric values will be averaged."
        )

        numeric_columns = [
            column
            for column in keep
            if column != "shareid"
        ]

        for column in numeric_columns:
            table[column] = pd.to_numeric(
                table[column],
                errors="coerce",
            )

        table = (
            table.groupby(
                "shareid",
                as_index=False,
            )[numeric_columns]
            .mean()
        )

    return table, path


def standardize(series):
    values = pd.to_numeric(
        series,
        errors="coerce",
    )

    sd = values.std(ddof=0)

    if not np.isfinite(sd) or sd == 0:
        return pd.Series(
            np.nan,
            index=values.index,
        )

    return (
        values - values.mean()
    ) / sd


def binary(series, positive_values=None):
    values = pd.to_numeric(
        series,
        errors="coerce",
    )

    if positive_values is not None:
        output = pd.Series(
            np.nan,
            index=series.index,
            dtype=float,
        )

        valid = values.notna()

        output.loc[valid] = (
            values.loc[valid]
            .isin(positive_values)
            .astype(float)
        )

        return output

    unique_values = sorted(
        values.dropna().unique()
    )

    if len(unique_values) != 2:
        raise RuntimeError(
            f"Expected two categories but found: "
            f"{unique_values}"
        )

    return values.map(
        {
            unique_values[0]: 0.0,
            unique_values[1]: 1.0,
        }
    )


# ============================================================
# Load cardiac/APOE/clinical table
# ============================================================

df = pd.read_csv(
    DATAFILE,
    low_memory=False,
)

print(
    f"\nCardiac/APOE input: {DATAFILE}"
)
print(
    f"Rows: {len(df):,}; columns: {len(df.columns):,}"
)

df["shareid"] = pd.to_numeric(
    df["shareid"],
    errors="coerce",
)

if df["shareid"].duplicated().any():
    raise RuntimeError(
        "The cardiac analysis table contains duplicated shareid values."
    )


# ============================================================
# Load biomarkers
# ============================================================

amyloid, amyloid_file = load_dbgap(
    "*pht003309*.HMB-IRB-MDS.txt.gz",
    [
        "shareid",
        "amyloid40",
        "amyloid42",
    ],
)

ptau, ptau_file = load_dbgap(
    "*pht015135*.HMB-IRB-MDS.txt.gz",
    [
        "shareid",
        "pTau_181",
    ],
)

glial, glial_file = load_dbgap(
    "*pht012879*.HMB-IRB-MDS.txt.gz",
    [
        "shareid",
        "gfap",
        "nf_l",
    ],
)

for column in [
    "amyloid40",
    "amyloid42",
]:
    amyloid[column] = pd.to_numeric(
        amyloid[column],
        errors="coerce",
    )

ptau["pTau_181"] = pd.to_numeric(
    ptau["pTau_181"],
    errors="coerce",
)

for column in [
    "gfap",
    "nf_l",
]:
    glial[column] = pd.to_numeric(
        glial[column],
        errors="coerce",
    )


# ============================================================
# Merge
# ============================================================

df = (
    df.merge(
        amyloid,
        on="shareid",
        how="left",
        validate="one_to_one",
    )
    .merge(
        ptau,
        on="shareid",
        how="left",
        validate="one_to_one",
    )
    .merge(
        glial,
        on="shareid",
        how="left",
        validate="one_to_one",
    )
)

df["Abeta42_40"] = (
    df["amyloid42"]
    / df["amyloid40"].replace(0, np.nan)
)

df["log_pTau181"] = np.log1p(
    df["pTau_181"].where(
        df["pTau_181"] >= 0
    )
)

df["log_GFAP"] = np.log1p(
    df["gfap"].where(
        df["gfap"] >= 0
    )
)

df["log_NFL"] = np.log1p(
    df["nf_l"].where(
        df["nf_l"] >= 0
    )
)


# ============================================================
# Derive indexed cardiac variables
# ============================================================

df["BSA_m2"] = pd.to_numeric(
    df["BSA_m2"],
    errors="coerce",
)

bad_bsa = (
    ~np.isfinite(df["BSA_m2"])
    | (df["BSA_m2"] <= 0)
)

df.loc[bad_bsa, "BSA_m2"] = np.nan

df["LVESVi"] = (
    pd.to_numeric(
        df["LVESV"],
        errors="coerce",
    )
    / df["BSA_m2"]
)

df["LV_MASSi"] = (
    pd.to_numeric(
        df["LV_MASS"],
        errors="coerce",
    )
    / df["BSA_m2"]
)

df["LVEF"] = pd.to_numeric(
    df["LVEF"],
    errors="coerce",
)

df["LAVI_max"] = pd.to_numeric(
    df["LAVI_max"],
    errors="coerce",
)


# ============================================================
# Prepare APOE and covariates
# ============================================================

df["APOE4"] = pd.to_numeric(
    df["APOE4_carrier"],
    errors="coerce",
)

valid_apoe = df["APOE4"].isin([0, 1])
df.loc[~valid_apoe, "APOE4"] = np.nan

sex = df["sex_clinical"]

if pd.api.types.is_numeric_dtype(sex):
    df["sex_binary"] = binary(sex)
else:
    text = (
        sex.astype("string")
        .str.strip()
        .str.lower()
    )

    df["sex_binary"] = text.map(
        {
            "female": 0.0,
            "f": 0.0,
            "male": 1.0,
            "m": 1.0,
        }
    )

    if df["sex_binary"].notna().sum() == 0:
        categories = sorted(
            text.dropna().unique()
        )

        if len(categories) != 2:
            raise RuntimeError(
                f"Unexpected sex categories: {categories}"
            )

        df["sex_binary"] = text.map(
            {
                categories[0]: 0.0,
                categories[1]: 1.0,
            }
        )

df["diabetes_any"] = binary(
    df["diabetes_history_nearest_exam"],
    positive_values=[1, 2],
)

df["smoking_binary"] = binary(
    df["current_smoker_nearest_exam"],
    positive_values=[1],
)

continuous_covariates = {
    "age": "age_nearest_exam",
    "BMI": "BMI_nearest_exam",
    "SBP": "SBP_nearest_exam",
}

for new_name, source_name in continuous_covariates.items():
    df[new_name] = pd.to_numeric(
        df[source_name],
        errors="coerce",
    )


# ============================================================
# Model definitions
# ============================================================

cardiac_predictors = {
    "LVEF": "LVEF",
    "LVESVi": "LVESVi",
    "LV mass index": "LV_MASSi",
    "LAVI max": "LAVI_max",
}

biomarkers = {
    "Aβ42/40": "Abeta42_40",
    "p-tau181": "log_pTau181",
    "GFAP": "log_GFAP",
    "NfL": "log_NFL",
}

covariates = [
    "age",
    "sex_binary",
    "BMI",
    "SBP",
    "smoking_binary",
    "diabetes_any",
]


# ============================================================
# Availability summary
# ============================================================

availability_rows = []

for label, column in {
    **cardiac_predictors,
    **biomarkers,
    "APOE4": "APOE4",
    "age": "age",
    "sex": "sex_binary",
    "BMI": "BMI",
    "SBP": "SBP",
    "smoking": "smoking_binary",
    "diabetes": "diabetes_any",
}.items():

    values = pd.to_numeric(
        df[column],
        errors="coerce",
    )

    availability_rows.append(
        {
            "variable": label,
            "column": column,
            "N_available": int(
                values.notna().sum()
            ),
            "N_missing": int(
                values.isna().sum()
            ),
        }
    )

availability = pd.DataFrame(
    availability_rows
)

availability.to_csv(
    OUTDIR / "biomarker_prediction_availability.csv",
    index=False,
)

print("\nVariable availability")
print(
    availability.to_string(
        index=False
    )
)


# ============================================================
# Fit the 16 interaction models
# ============================================================

results = []

for biomarker_label, biomarker_column in biomarkers.items():

    for cardiac_label, cardiac_column in cardiac_predictors.items():

        required = [
            biomarker_column,
            cardiac_column,
            "APOE4",
            *covariates,
        ]

        model_data = (
            df[required]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .dropna()
            .copy()
        )

        n_total = len(model_data)
        n_noncarrier = int(
            (model_data["APOE4"] == 0).sum()
        )
        n_carrier = int(
            (model_data["APOE4"] == 1).sum()
        )

        row = {
            "biomarker": biomarker_label,
            "biomarker_column": biomarker_column,
            "cardiac": cardiac_label,
            "cardiac_column": cardiac_column,
            "N": n_total,
            "N_APOE4_noncarrier": n_noncarrier,
            "N_APOE4_carrier": n_carrier,
        }

        if (
            n_total < 80
            or n_noncarrier < 20
            or n_carrier < 20
        ):
            row["status"] = (
                "insufficient complete cases"
            )

            results.append(row)
            continue

        model_data["outcome_z"] = standardize(
            model_data[biomarker_column]
        )

        model_data["cardiac_z"] = standardize(
            model_data[cardiac_column]
        )

        for covariate in [
            "age",
            "BMI",
            "SBP",
        ]:
            model_data[f"{covariate}_z"] = standardize(
                model_data[covariate]
            )

        model_data["interaction"] = (
            model_data["cardiac_z"]
            * model_data["APOE4"]
        )

        X = pd.DataFrame(
            {
                "cardiac_z": model_data["cardiac_z"],
                "APOE4": model_data["APOE4"],
                "cardiac_x_APOE4": model_data["interaction"],
                "age_z": model_data["age_z"],
                "sex": model_data["sex_binary"],
                "BMI_z": model_data["BMI_z"],
                "SBP_z": model_data["SBP_z"],
                "smoking": model_data["smoking_binary"],
                "diabetes": model_data["diabetes_any"],
            },
            index=model_data.index,
        )

        X = sm.add_constant(
            X,
            has_constant="add",
        )

        y = model_data["outcome_z"]

        fit = sm.OLS(
            y,
            X,
        ).fit(
            cov_type="HC3"
        )

        interaction_name = "cardiac_x_APOE4"

        beta_main = fit.params["cardiac_z"]
        beta_interaction = fit.params[
            interaction_name
        ]

        covariance = fit.cov_params()

        beta_carrier = (
            beta_main
            + beta_interaction
        )

        carrier_variance = (
            covariance.loc[
                "cardiac_z",
                "cardiac_z",
            ]
            + covariance.loc[
                interaction_name,
                interaction_name,
            ]
            + 2
            * covariance.loc[
                "cardiac_z",
                interaction_name,
            ]
        )

        carrier_se = float(
            np.sqrt(
                max(
                    carrier_variance,
                    0,
                )
            )
        )

        if carrier_se > 0:
            carrier_z = (
                beta_carrier
                / carrier_se
            )

            carrier_p = float(
                2
                * norm.sf(
                    abs(carrier_z)
                )
            )
        else:
            carrier_p = np.nan

        confidence_interval = (
            fit.conf_int()
            .loc[interaction_name]
        )

        row.update(
            {
                "status": "ok",
                "interaction_beta": beta_interaction,
                "interaction_SE_HC3": fit.bse[
                    interaction_name
                ],
                "interaction_CI_low": confidence_interval.iloc[0],
                "interaction_CI_high": confidence_interval.iloc[1],
                "interaction_P": fit.pvalues[
                    interaction_name
                ],
                "slope_noncarrier": beta_main,
                "slope_noncarrier_SE": fit.bse[
                    "cardiac_z"
                ],
                "slope_noncarrier_P": fit.pvalues[
                    "cardiac_z"
                ],
                "slope_carrier": beta_carrier,
                "slope_carrier_SE": carrier_se,
                "slope_carrier_P": carrier_p,
                "R2": fit.rsquared,
                "adjusted_R2": fit.rsquared_adj,
            }
        )

        results.append(row)


# ============================================================
# FDR correction across the 16 interaction tests
# ============================================================

results = pd.DataFrame(
    results
)

successful = (
    results["status"] == "ok"
)

results["interaction_q_BH_16"] = np.nan

if successful.any():
    results.loc[
        successful,
        "interaction_q_BH_16",
    ] = multipletests(
        results.loc[
            successful,
            "interaction_P",
        ].to_numpy(),
        method="fdr_bh",
    )[1]

results["FDR_significant"] = (
    results["interaction_q_BH_16"]
    < 0.05
)

results = results.sort_values(
    [
        "interaction_q_BH_16",
        "interaction_P",
    ],
    na_position="last",
)

results.to_csv(
    OUTDIR / "cardiac_APOE4_biomarker_interactions.csv",
    index=False,
)


# ============================================================
# Save merged analysis subset for reproducibility
# ============================================================

save_columns = [
    "shareid",
    "APOE4",
    "APOE_genotype",
    "LVEF",
    "LVESVi",
    "LV_MASSi",
    "LAVI_max",
    "Abeta42_40",
    "log_pTau181",
    "log_GFAP",
    "log_NFL",
    "age",
    "sex_binary",
    "BMI",
    "SBP",
    "smoking_binary",
    "diabetes_any",
]

existing_save_columns = [
    column
    for column in save_columns
    if column in df.columns
]

df[existing_save_columns].to_csv(
    OUTDIR / "cardiac_biomarker_analysis_dataset.csv",
    index=False,
)


# ============================================================
# Save analysis manifest
# ============================================================

manifest = {
    "cardiac_input": str(DATAFILE),
    "amyloid_input": str(amyloid_file),
    "ptau181_input": str(ptau_file),
    "gfap_nfl_input": str(glial_file),
    "merge_identifier": "shareid",
    "cardiac_predictors": cardiac_predictors,
    "biomarker_outcomes": biomarkers,
    "covariates": covariates,
    "transformations": {
        "Abeta42_40": "amyloid42 / amyloid40",
        "log_pTau181": "log1p(pTau_181)",
        "log_GFAP": "log1p(gfap)",
        "log_NFL": "log1p(nf_l)",
        "LVESVi": "LVESV / BSA_m2",
        "LV_MASSi": "LV_MASS / BSA_m2",
    },
    "standardization": (
        "Biomarker outcome and cardiac predictor standardized "
        "within each model-specific complete-case sample."
    ),
    "standard_errors": "HC3",
    "multiple_testing": (
        "Benjamini-Hochberg FDR jointly across all 16 "
        "cardiac x APOE4 interaction tests."
    ),
}

with open(
    OUTDIR / "analysis_manifest.json",
    "w",
) as handle:
    json.dump(
        manifest,
        handle,
        indent=2,
    )


# ============================================================
# Print results
# ============================================================

display_columns = [
    "biomarker",
    "cardiac",
    "N",
    "N_APOE4_noncarrier",
    "N_APOE4_carrier",
    "interaction_beta",
    "interaction_CI_low",
    "interaction_CI_high",
    "interaction_P",
    "interaction_q_BH_16",
    "FDR_significant",
]

print(
    "\nCARDIAC x APOE4 PREDICTION OF PLASMA BIOMARKERS\n"
)

print(
    results.reindex(
        columns=display_columns
    ).to_string(
        index=False
    )
)

print(
    f"\nResults saved to:\n{OUTDIR}"
)
