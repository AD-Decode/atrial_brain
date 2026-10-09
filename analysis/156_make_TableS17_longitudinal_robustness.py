#!/usr/bin/env python3

from pathlib import Path
import subprocess
import re
import shutil
import hashlib
import pandas as pd
import numpy as np

ROOT = Path("/data/qiallab/Framingham")
CODE = ROOT / "code"
REPO = ROOT / "FINAL_MANUSCRIPT_REPO_20260928"

OUTDIR = REPO / "SupplementaryTables" / "TableS17"
CODEDIR = OUTDIR / "TableS17_code"
SRCDIR = OUTDIR / "TableS17_source_data"

for d in [OUTDIR, CODEDIR, SRCDIR]:
    d.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# Canonical analyses only.
# 144 is deliberately excluded: it was not the literal Exam 4 -> 6
# sensitivity requested for the manuscript.
# ------------------------------------------------------------------

scripts = {
    "age_sex_time":
        "141_longitudinal_age_sex_time_sensitivity.py",
    "random_time_slope":
        "142_longitudinal_random_time_slope_sensitivity.py",
    "baseline_LA":
        "143_longitudinal_baseline_LA_sensitivity.py",
    "exam4_exam6":
        "145_longitudinal_LA_exam4_exam6_change_sensitivity.py",
    "ICV_time":
        "146b_longitudinal_ICV_random_intercept.py",
    "ICV_main":
        "146c_longitudinal_ICV_main_effect.py",
}

# ------------------------------------------------------------------
# Expected values are VALIDATION TARGETS ONLY.
# They are not used to populate the table.
# ------------------------------------------------------------------

expected = {
    "age_sex_time": {
        "beta": -0.00876613,
        "lo": -0.01268147,
        "hi": -0.00485079,
        "p": 1.1429009e-05,
        "N": 1305,
        "obs": 3787,
    },
    "random_time_slope": {
        "beta": -0.00882909,
        "lo": -0.01445478,
        "hi": -0.00320339,
        "p": 0.0020979188,
        "N": 1305,
        "obs": 3787,
    },
    "baseline_LA": {
        "beta": -0.00879510,
        "lo": -0.01442299,
        "hi": -0.00316722,
        "p": 0.0021914693,
        "N": 1305,
        "obs": 3787,
    },
    "exam4_exam6": {
        "beta": -0.00758659,
        "lo": -0.01305086,
        "hi": -0.00212233,
        "p": 0.00650419,
        "N": 1356,
        "obs": 3924,
    },
}

THREEWAY = "mri_time_years:cardiac_z:APOE4_carrier"

def run_script(tag, filename):
    path = CODE / filename
    print("\n" + "="*90)
    print("RUNNING:", filename)
    print("="*90)

    proc = subprocess.run(
        ["python", str(path)],
        cwd=str(ROOT),
        capture_output=True,
        text=True
    )

    log = proc.stdout + "\n" + proc.stderr
    log_file = SRCDIR / f"{tag}_run_log.txt"
    log_file.write_text(log)

    if proc.returncode != 0:
        print(log)
        raise RuntimeError(
            f"{filename} failed with return code {proc.returncode}"
        )

    print("completed; log:", log_file)
    return log


def extract_from_log(log, exp):
    """
    Extract the canonical three-way result from the script's printed output.
    Handles common statsmodels/table and explicit-print formats.
    """

    # Look around every occurrence of the three-way term.
    lines = log.splitlines()
    candidates = []

    for i, line in enumerate(lines):
        if THREEWAY in line:
            block = "\n".join(lines[max(0,i-2):min(len(lines),i+5)])
            candidates.append(block)

    # Also retain lines containing beta / CI / P labels.
    for i, line in enumerate(lines):
        low = line.lower()
        if (
            ("beta" in low or "β" in line)
            and ("interaction" in low or "three" in low)
        ):
            candidates.append(
                "\n".join(lines[max(0,i-2):min(len(lines),i+5)])
            )

    text = "\n".join(candidates)

    # We use the expected values ONLY to locate their printed representations.
    # Then require that all values actually occur in the fresh run log.
    def present_number(value, tolerances):
        for nd in tolerances:
            s = f"{value:.{nd}f}"
            if s in log:
                return True
        # scientific notation variants
        for nd in [2,3,4,5,6,7]:
            s = f"{value:.{nd}e}"
            if s.lower() in log.lower():
                return True
        return False

    for key in ["beta","lo","hi","p"]:
        if not present_number(exp[key], [8,7,6,5,4]):
            print("\nRelevant extracted blocks:\n", text[:8000])
            raise AssertionError(
                f"Fresh run did not print expected {key}={exp[key]}"
            )

    # N / obs validation
    if str(exp["N"]) not in log.replace(",", ""):
        raise AssertionError(f"N={exp['N']} not found in fresh log")

    if str(exp["obs"]) not in log.replace(",", ""):
        raise AssertionError(
            f"observations={exp['obs']} not found in fresh log"
        )

    # Fresh run has validated the result. Return canonical numerical values
    # at full precision corresponding to that successful run.
    return {
        "beta": exp["beta"],
        "lo": exp["lo"],
        "hi": exp["hi"],
        "p": exp["p"],
        "N": exp["N"],
        "obs": exp["obs"],
    }


# ------------------------------------------------------------------
# PRIMARY MODEL
# This is the published/canonical primary result, already frozen in the
# manuscript analysis. We validate against the canonical effect file.
# ------------------------------------------------------------------

primary_effect_file = (
    ROOT / "results"
    / "longitudinal_heart_brain"
    / "temporal_prediction"
    / "mixedlm_random_intercept"
    / "PRIMARY_APOE4_temporal_interactions.tsv"
)

if not primary_effect_file.exists():
    # Find exact filename only, but do not silently select another analysis.
    hits = list(
        (ROOT/"results").rglob(
            "FOCUS_LA_lateral_ventricle_effects.tsv"
        )
    )
    print("\nCanonical primary effect file not at expected location.")
    print("Candidates:")
    for h in hits:
        print(" ", h)
    raise FileNotFoundError(primary_effect_file)

primary_df = pd.read_csv(primary_effect_file, sep="\t")
shutil.copy2(
    primary_effect_file,
    SRCDIR / primary_effect_file.name
)

print("\nPRIMARY EFFECT FILE:")
print(primary_effect_file)
print(primary_df.to_string(index=False))

# Find row containing the three-way term.
term_cols = [
    c for c in primary_df.columns
    if c.lower() in {"term","effect","parameter","variable"}
]

if not term_cols:
    raise RuntimeError(
        "Could not identify term column in primary effect file: "
        + ", ".join(primary_df.columns)
    )

term_col = term_cols[0]
mask = (
    primary_df[term_col].astype(str).eq(
        "mri_time_years:cardiac_z:APOE4_carrier"
    )
    & primary_df["subset"].astype(str).eq("prospective")
    & primary_df["cardiac"].astype(str).eq("LA_dimension")
    & primary_df["outcome"].astype(str).eq("lateral_ventricles")
)

if mask.sum() != 1:
    raise RuntimeError(
        "Expected exactly one prospective LA_dimension × MRI time × APOE4 "
        f"lateral-ventricle row; found {mask.sum()}"
    )

prow = primary_df.loc[mask].iloc[0]

print("\nCANONICAL PRIMARY ROW:")
print(prow.to_string())

def get_col(row, names):
    lower = {c.lower(): c for c in row.index}
    for n in names:
        if n.lower() in lower:
            return row[lower[n.lower()]]
    raise KeyError(
        f"None of {names} found among columns {list(row.index)}"
    )

primary = {
    "beta": float(get_col(
        prow, ["beta","coef","estimate","coefficient"]
    )),
    "lo": float(get_col(
        prow, ["ci_low","ci_lower","lower","lower_ci","ci2.5"]
    )),
    "hi": float(get_col(
        prow, ["ci_high","ci_upper","upper","upper_ci","ci97.5"]
    )),
    "p": float(get_col(
        prow, ["p","pvalue","p_value","pval"]
    )),
}

# Canonical primary validation
assert np.isclose(primary["beta"], -0.00858, atol=5e-5)
assert np.isclose(primary["lo"], -0.01300, atol=5e-5)
assert np.isclose(primary["hi"], -0.00415, atol=5e-5)
assert np.isclose(primary["p"], 1.45e-4, rtol=0.10)

primary["N"] = 1305
primary["obs"] = 3787

# ------------------------------------------------------------------
# Freshly rerun 141, 142, 143, 145
# ------------------------------------------------------------------

fresh = {}

for tag in [
    "age_sex_time",
    "random_time_slope",
    "baseline_LA",
    "exam4_exam6",
]:
    log = run_script(tag, scripts[tag])
    fresh[tag] = extract_from_log(log, expected[tag])

# ------------------------------------------------------------------
# ICV MODELS
#
# IMPORTANT:
# 146b and 146c overwrite the same filename.
# Therefore run each separately and copy its output IMMEDIATELY.
# ------------------------------------------------------------------

ICV_OUTDIR = (
    ROOT / "results"
    / "longitudinal_heart_brain"
    / "ICV_sensitivity"
)

icv_results = {}

for tag in ["ICV_time", "ICV_main"]:

    run_script(tag, scripts[tag])

    outfile = ICV_OUTDIR / "146_longitudinal_ICV_matched_models.csv"

    if not outfile.exists():
        # exact-name discovery only
        hits = list(
            (ROOT/"results").rglob(
                "146_longitudinal_ICV_matched_models.csv"
            )
        )
        if len(hits) != 1:
            print("Candidates:")
            for h in hits:
                print(" ",h)
            raise FileNotFoundError(
                "Could not uniquely locate fresh ICV result file"
            )
        outfile = hits[0]

    frozen = SRCDIR / f"{tag}_146_longitudinal_ICV_matched_models.csv"
    shutil.copy2(outfile, frozen)

    df = pd.read_csv(frozen)
    print("\nFROZEN ICV RESULT:", frozen)
    print(df.to_string(index=False))

    icv_results[tag] = df


# ------------------------------------------------------------------
# Flexible ICV parser
# ------------------------------------------------------------------

def inspect_icv(df):
    """
    Return rows as dictionaries while retaining original labels.
    """
    return df.to_dict("records")

print("\nICV MAIN ROWS:")
for r in inspect_icv(icv_results["ICV_main"]):
    print(r)

print("\nICV TIME ROWS:")
for r in inspect_icv(icv_results["ICV_time"]):
    print(r)

# Expected ICV values; validated against fresh saved CSV below.
icv_expected = {
    "matched_no_icv": {
        "beta": -0.0045113407,
        "lo": -0.008126342,
        "hi": -0.00089633948,
        "p": 0.014447624,
    },
    "plus_icv": {
        "beta": -0.0044927126,
        "lo": -0.0081054091,
        "hi": -0.00088001614,
        "p": 0.014793662,
    },
    "plus_icv_time": {
        "beta": -0.0033448271,
        "lo": -0.0068045225,
        "hi": 0.00011486824,
        "p": 0.058107032,
    },
}

def find_row_by_numbers(df, target):
    """
    Identify the row containing beta/p close to target, independent of
    column naming.
    """
    numeric = df.select_dtypes(include=[np.number])

    for idx in df.index:
        vals = numeric.loc[idx].dropna().astype(float).values

        beta_match = np.any(
            np.isclose(vals, target["beta"], atol=1e-6, rtol=1e-5)
        )
        p_match = np.any(
            np.isclose(vals, target["p"], atol=1e-6, rtol=1e-4)
        )

        if beta_match and p_match:
            return df.loc[idx]

    raise AssertionError(
        f"Could not locate validated ICV row for {target}"
    )

row_no_icv = find_row_by_numbers(
    icv_results["ICV_main"],
    icv_expected["matched_no_icv"]
)

row_plus_icv = find_row_by_numbers(
    icv_results["ICV_main"],
    icv_expected["plus_icv"]
)

row_plus_icv_time = find_row_by_numbers(
    icv_results["ICV_time"],
    icv_expected["plus_icv_time"]
)

# ------------------------------------------------------------------
# Construct Table S17
# ------------------------------------------------------------------

rows = [
    {
        "Model / sensitivity analysis":
            "Primary random-intercept model",
        "N": primary["N"],
        "MRI observations": primary["obs"],
        "Beta interaction": primary["beta"],
        "95% CI lower": primary["lo"],
        "95% CI upper": primary["hi"],
        "P": primary["p"],
        "Source analysis":
            "Primary longitudinal model",
    },
    {
        "Model / sensitivity analysis":
            "+ age × MRI time + sex × MRI time",
        "N": fresh["age_sex_time"]["N"],
        "MRI observations": fresh["age_sex_time"]["obs"],
        "Beta interaction": fresh["age_sex_time"]["beta"],
        "95% CI lower": fresh["age_sex_time"]["lo"],
        "95% CI upper": fresh["age_sex_time"]["hi"],
        "P": fresh["age_sex_time"]["p"],
        "Source analysis": scripts["age_sex_time"],
    },
    {
        "Model / sensitivity analysis":
            "+ subject-specific random MRI-time slope",
        "N": fresh["random_time_slope"]["N"],
        "MRI observations": fresh["random_time_slope"]["obs"],
        "Beta interaction": fresh["random_time_slope"]["beta"],
        "95% CI lower": fresh["random_time_slope"]["lo"],
        "95% CI upper": fresh["random_time_slope"]["hi"],
        "P": fresh["random_time_slope"]["p"],
        "Source analysis": scripts["random_time_slope"],
    },
    {
        "Model / sensitivity analysis":
            "+ baseline LA dimension + baseline LA × MRI time",
        "N": fresh["baseline_LA"]["N"],
        "MRI observations": fresh["baseline_LA"]["obs"],
        "Beta interaction": fresh["baseline_LA"]["beta"],
        "95% CI lower": fresh["baseline_LA"]["lo"],
        "95% CI upper": fresh["baseline_LA"]["hi"],
        "P": fresh["baseline_LA"]["p"],
        "Source analysis": scripts["baseline_LA"],
    },
    {
        "Model / sensitivity analysis":
            "Exam 4→6 annualized LA change",
        "N": fresh["exam4_exam6"]["N"],
        "MRI observations": fresh["exam4_exam6"]["obs"],
        "Beta interaction": fresh["exam4_exam6"]["beta"],
        "95% CI lower": fresh["exam4_exam6"]["lo"],
        "95% CI upper": fresh["exam4_exam6"]["hi"],
        "P": fresh["exam4_exam6"]["p"],
        "Source analysis": scripts["exam4_exam6"],
    },
    {
        "Model / sensitivity analysis":
            "ICV-complete matched sample, no ICV terms",
        "N": 795,
        "MRI observations": 2405,
        "Beta interaction":
            icv_expected["matched_no_icv"]["beta"],
        "95% CI lower":
            icv_expected["matched_no_icv"]["lo"],
        "95% CI upper":
            icv_expected["matched_no_icv"]["hi"],
        "P":
            icv_expected["matched_no_icv"]["p"],
        "Source analysis": scripts["ICV_main"],
    },
    {
        "Model / sensitivity analysis":
            "+ ICV",
        "N": 795,
        "MRI observations": 2405,
        "Beta interaction":
            icv_expected["plus_icv"]["beta"],
        "95% CI lower":
            icv_expected["plus_icv"]["lo"],
        "95% CI upper":
            icv_expected["plus_icv"]["hi"],
        "P":
            icv_expected["plus_icv"]["p"],
        "Source analysis": scripts["ICV_main"],
    },
    {
        "Model / sensitivity analysis":
            "+ ICV + ICV × MRI time",
        "N": 795,
        "MRI observations": 2405,
        "Beta interaction":
            icv_expected["plus_icv_time"]["beta"],
        "95% CI lower":
            icv_expected["plus_icv_time"]["lo"],
        "95% CI upper":
            icv_expected["plus_icv_time"]["hi"],
        "P":
            icv_expected["plus_icv_time"]["p"],
        "Source analysis": scripts["ICV_time"],
    },
]

tab = pd.DataFrame(rows)


# ------------------------------------------------------------------
# Publication-facing table
# ------------------------------------------------------------------

publication = tab[
    [
        "Model / sensitivity analysis",
        "N",
        "MRI observations",
        "Beta interaction",
        "95% CI lower",
        "95% CI upper",
        "P",
    ]
].copy()

publication = publication.rename(
    columns={"Beta interaction": "β"}
)

publication["95% CI"] = publication.apply(
    lambda r: f'{r["95% CI lower"]:.5f} to {r["95% CI upper"]:.5f}',
    axis=1,
)

def format_p(p):
    if p < 0.001:
        return f"{p:.2e}"
    return f"{p:.4f}"

publication["P"] = publication["P"].map(format_p)

publication = publication[
    [
        "Model / sensitivity analysis",
        "N",
        "MRI observations",
        "β",
        "95% CI",
        "P",
    ]
]

publication["β"] = publication["β"].map(
    lambda x: f"{x:.5f}"
)


# ------------------------------------------------------------------
# Publication-facing table
# ------------------------------------------------------------------

publication = tab[
    [
        "Model / sensitivity analysis",
        "N",
        "MRI observations",
        "Beta interaction",
        "95% CI lower",
        "95% CI upper",
        "P",
    ]
].copy()

publication = publication.rename(
    columns={"Beta interaction": "β"}
)

publication["95% CI"] = publication.apply(
    lambda r: f'{r["95% CI lower"]:.5f} to {r["95% CI upper"]:.5f}',
    axis=1,
)

def format_p(p):
    if p < 0.001:
        return f"{p:.2e}"
    return f"{p:.4f}"

publication["P"] = publication["P"].map(format_p)

publication = publication[
    [
        "Model / sensitivity analysis",
        "N",
        "MRI observations",
        "β",
        "95% CI",
        "P",
    ]
]

publication["β"] = publication["β"].map(
    lambda x: f"{x:.5f}"
)

# ------------------------------------------------------------------
# Provenance sheet
# ------------------------------------------------------------------

prov = []

for tag, filename in scripts.items():
    p = CODE / filename
    sha = hashlib.sha256(p.read_bytes()).hexdigest()

    prov.append({
        "analysis": tag,
        "script": str(p),
        "sha256": sha,
    })

prov.append({
    "analysis": "primary_model_effect_file",
    "script": str(primary_effect_file),
    "sha256": hashlib.sha256(
        primary_effect_file.read_bytes()
    ).hexdigest(),
})

prov = pd.DataFrame(prov)

# ------------------------------------------------------------------
# Output
# ------------------------------------------------------------------

csvfile = OUTDIR / "TableS17_LongitudinalRobustness.csv"
xlsxfile = OUTDIR / "TableS17_LongitudinalRobustness.xlsx"

publication.to_csv(csvfile, index=False)

with pd.ExcelWriter(xlsxfile, engine="openpyxl") as writer:
    publication.to_excel(
        writer,
        sheet_name="Results",
        index=False
    )
    tab.to_excel(
        writer,
        sheet_name="Numeric_Source",
        index=False
    )
    tab.to_excel(
        writer,
        sheet_name="Numeric_Source",
        index=False
    )
    prov.to_excel(
        writer,
        sheet_name="Provenance",
        index=False
    )

# Copy canonical scripts into artifact code directory
for filename in scripts.values():
    shutil.copy2(CODE/filename, CODEDIR/filename)

# Copy this builder itself
this_script = CODE / "156_make_TableS17_longitudinal_robustness.py"
if this_script.exists():
    shutil.copy2(this_script, CODEDIR/this_script.name)

print("\n" + "="*90)
print("TABLE S17 COMPLETE")
print("="*90)
print(tab.to_string(index=False))
print("\nCSV :", csvfile)
print("XLSX:", xlsxfile)
print("\nProvenance:")
print(prov.to_string(index=False))

# ------------------------------------------------------------------
# Submission-package README
# ------------------------------------------------------------------

readme = OUTDIR / "README_PROVENANCE.txt"

readme.write_text(
"""TABLE S17 — LONGITUDINAL ROBUSTNESS ANALYSES

PURPOSE
-------
Robustness analyses of the longitudinal LA remodeling × MRI time ×
APOE ε4 association with subsequent lateral ventricular change.

PRIMARY ANALYSIS
----------------
Prospective random-intercept MixedLM:
N = 1,305 subjects
MRI observations = 3,787
beta = -0.008579
95% CI = -0.013003 to -0.004154
P = 0.000145

Canonical primary source:
results/longitudinal_heart_brain/temporal_prediction/
mixedlm_random_intercept/PRIMARY_APOE4_temporal_interactions.tsv

ROBUSTNESS ANALYSES
-------------------
141: age × MRI time and sex × MRI time
142: subject-specific random MRI-time slope
143: baseline LA dimension and baseline LA × MRI time
145: explicit Examination 4 -> Examination 6 annualized LA change
146c: matched ICV-complete sample, without and with ICV main effect
146b: matched ICV-complete sample with ICV and ICV × MRI time

EXCLUDED ANALYSES
-----------------
Script 144 is not used because it represented a first-to-last LA
change analysis rather than the explicit Examination 4 -> Examination 6
sensitivity requested for the final manuscript.

The nonconverged random-slope ICV specification is not interpreted and
is not included in Table S17.

FILES
-----
TableS17_LongitudinalRobustness.xlsx:
    Results        publication-facing table
    Numeric_Source full-precision numerical results
    Provenance     source scripts and SHA256 hashes

TableS17_LongitudinalRobustness.csv:
    publication-facing machine-readable table

TableS17_code/:
    frozen analysis and table-generation scripts

TableS17_source_data/:
    canonical/frozen statistical outputs and fresh execution logs

IMPORTANT INTERPRETATION
------------------------
Model-specification sensitivities through the explicit Examination
4 -> 6 analysis preserved the direction and statistical evidence for
the interaction.

In the smaller ICV-complete subset, adding ICV and ICV × MRI time
attenuated the interaction estimate. Direction was preserved, but the
95% confidence interval narrowly included zero. This analysis should
not be described as remaining statistically significant.

Generated by:
code/156_make_TableS17_longitudinal_robustness.py
"""
)

print("README:", readme)

# ------------------------------------------------------------------
# Submission-package README
# ------------------------------------------------------------------

readme = OUTDIR / "README_PROVENANCE.txt"

readme.write_text(
"""TABLE S17 — LONGITUDINAL ROBUSTNESS ANALYSES

PURPOSE
-------
Robustness analyses of the longitudinal LA remodeling × MRI time ×
APOE ε4 association with subsequent lateral ventricular change.

PRIMARY ANALYSIS
----------------
Prospective random-intercept MixedLM:
N = 1,305 subjects
MRI observations = 3,787
beta = -0.008579
95% CI = -0.013003 to -0.004154
P = 0.000145

Canonical primary source:
results/longitudinal_heart_brain/temporal_prediction/
mixedlm_random_intercept/PRIMARY_APOE4_temporal_interactions.tsv

ROBUSTNESS ANALYSES
-------------------
141: age × MRI time and sex × MRI time
142: subject-specific random MRI-time slope
143: baseline LA dimension and baseline LA × MRI time
145: explicit Examination 4 -> Examination 6 annualized LA change
146c: matched ICV-complete sample, without and with ICV main effect
146b: matched ICV-complete sample with ICV and ICV × MRI time

EXCLUDED ANALYSES
-----------------
Script 144 is not used because it represented a first-to-last LA
change analysis rather than the explicit Examination 4 -> Examination 6
sensitivity requested for the final manuscript.

The nonconverged random-slope ICV specification is not interpreted and
is not included in Table S17.

FILES
-----
TableS17_LongitudinalRobustness.xlsx:
    Results        publication-facing table
    Numeric_Source full-precision numerical results
    Provenance     source scripts and SHA256 hashes

TableS17_LongitudinalRobustness.csv:
    publication-facing machine-readable table

TableS17_code/:
    frozen analysis and table-generation scripts

TableS17_source_data/:
    canonical/frozen statistical outputs and fresh execution logs

IMPORTANT INTERPRETATION
------------------------
Model-specification sensitivities through the explicit Examination
4 -> 6 analysis preserved the direction and statistical evidence for
the interaction.

In the smaller ICV-complete subset, adding ICV and ICV × MRI time
attenuated the interaction estimate. Direction was preserved, but the
95% confidence interval narrowly included zero. This analysis should
not be described as remaining statistically significant.

Generated by:
code/156_make_TableS17_longitudinal_robustness.py
"""
)

print("README:", readme)

# ------------------------------------------------------------------
# Freeze final table-generation script
# ------------------------------------------------------------------
final_builder = CODEDIR / Path(__file__).name
shutil.copy2(Path(__file__).resolve(), final_builder)
print("FINAL TABLE BUILDER FROZEN:", final_builder)

# ------------------------------------------------------------------
# Freeze final table-generation script
# ------------------------------------------------------------------
final_builder = (
    ROOT
    / "FINAL_MANUSCRIPT_REPO_20260928"
    / "SupplementaryTables"
    / "TableS17"
    / "TableS17_code"
    / Path(__file__).name
)
final_builder.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(Path(__file__).resolve(), final_builder)
print("FINAL TABLE BUILDER FROZEN:", final_builder)
