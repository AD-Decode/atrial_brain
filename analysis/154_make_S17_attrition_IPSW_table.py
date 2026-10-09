#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("/data/qiallab/Framingham")

SRC = (
    ROOT / "results/longitudinal_heart_brain"
    / "temporal_prediction/attrition_IPSW"
)

OUTDIR = (
    ROOT / "results/longitudinal_heart_brain"
    / "temporal_prediction/attrition_IPSW"
)
OUTDIR.mkdir(parents=True, exist_ok=True)

effects = pd.read_csv(
    SRC / "IPSW_GEE_effect_comparison.tsv",
    sep="\t"
)

selection = pd.read_csv(
    SRC / "selection_model_summary.tsv",
    sep="\t"
)

# ------------------------------------------------------------------
# Panel A: selection model
# ------------------------------------------------------------------

s = selection.iloc[0]

panel_a = pd.DataFrame([
    {
        "Analysis": "At-risk cohort",
        "N": int(s["N_at_risk"]),
        "Estimate": np.nan,
        "95% CI": "",
        "P value": np.nan,
        "Notes": (
            "Participants with valid three-exam LA remodeling trajectory "
            "and classified APOE genotype before conditioning on MRI participation"
        ),
    },
    {
        "Analysis": "Selection-model complete-case cohort",
        "N": int(s["N_selection_complete_case"]),
        "Estimate": np.nan,
        "95% CI": "",
        "P value": np.nan,
        "Notes": (
            f'{int(s["N_included_complete_case"])} included in repeated-MRI analysis'
        ),
    },
    {
        "Analysis": "LA remodeling × APOE ε4 association with inclusion",
        "N": int(s["N_selection_complete_case"]),
        "Estimate": float(s["LAxAPOE_OR"]),
        "95% CI": f'{s["CI_low"]:.3f} to {s["CI_high"]:.3f}',
        "P value": float(s["P"]),
        "Notes": "Odds ratio from multivariable logistic selection model",
    },
    {
        "Analysis": "Stabilized IPSW, 1st percentile",
        "N": int(s["N_selection_complete_case"]),
        "Estimate": float(s["weight_p01"]),
        "95% CI": "",
        "P value": np.nan,
        "Notes": "Lower truncation threshold",
    },
    {
        "Analysis": "Stabilized IPSW, 99th percentile",
        "N": int(s["N_selection_complete_case"]),
        "Estimate": float(s["weight_p99"]),
        "95% CI": "",
        "P value": np.nan,
        "Notes": "Upper truncation threshold",
    },
])

# ------------------------------------------------------------------
# Panel B: longitudinal effect comparison
# ------------------------------------------------------------------

panel_b = effects.copy()

panel_b["95% CI"] = panel_b.apply(
    lambda r: (
        f'{r["CI_low"]:.5f} to {r["CI_high"]:.5f}'
        if pd.notna(r["CI_low"]) and pd.notna(r["CI_high"])
        else ""
    ),
    axis=1
)

panel_b = panel_b[
    [
        "model",
        "N_subjects",
        "N_observations",
        "beta_3way",
        "95% CI",
        "P",
    ]
].rename(
    columns={
        "model": "Model",
        "N_subjects": "N participants",
        "N_observations": "N MRI observations",
        "beta_3way": "MRI time × LA remodeling × APOE ε4 β",
        "P": "P value",
    }
)

# ------------------------------------------------------------------
# Notes
# ------------------------------------------------------------------

notes = pd.DataFrame({
    "Notes": [
        (
            "At-risk cohort: participants with a valid three-exam left-atrial "
            "remodeling trajectory and classified APOE genotype, defined before "
            "conditioning on subsequent MRI participation."
        ),
        (
            "Selection model: inclusion ~ standardized LA remodeling × APOE ε4 "
            "+ age at Exam 6 + sex + Exam 4 BMI + Exam 4 systolic blood pressure "
            "+ Exam 4 diabetes + Exam 4 smoking + baseline LA dimension."
        ),
        (
            "Stabilized inverse-probability-of-selection weights used "
            "P(inclusion | APOE ε4) / P(inclusion | full covariate model) and "
            "were truncated at the 1st and 99th percentiles."
        ),
        (
            "The weighted and unweighted GEE models used exchangeable working "
            "correlation and robust standard errors. The unweighted GEE is fit "
            "to the same complete-case participants as the weighted analysis."
        ),
        (
            "Primary mixed-model estimate is shown for reference. "
            "The principal comparison for attrition sensitivity is between "
            "the like-for-like unweighted and IPSW-weighted GEE estimates."
        ),
    ]
})

# ------------------------------------------------------------------
# Write TSVs
# ------------------------------------------------------------------

panel_a.to_csv(
    OUTDIR / "Table_S17A_attrition_selection_model.tsv",
    sep="\t",
    index=False
)

panel_b.to_csv(
    OUTDIR / "Table_S17B_IPSW_effect_comparison.tsv",
    sep="\t",
    index=False
)

# ------------------------------------------------------------------
# Write formatted Excel workbook
# ------------------------------------------------------------------

xlsx = OUTDIR / "Table_S17_Attrition_IPSW_Sensitivity.xlsx"

with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
    panel_a.to_excel(
        writer,
        sheet_name="S17 Attrition-IPSW",
        index=False,
        startrow=2
    )

    ws = writer.book["S17 Attrition-IPSW"]
    ws["A1"] = (
        "Supplementary Table S17. Attrition and inverse-probability-of-selection "
        "weighted sensitivity analysis"
    )

    start_b = len(panel_a) + 6
    panel_b.to_excel(
        writer,
        sheet_name="S17 Attrition-IPSW",
        index=False,
        startrow=start_b
    )

    start_notes = start_b + len(panel_b) + 4
    notes.to_excel(
        writer,
        sheet_name="S17 Attrition-IPSW",
        index=False,
        startrow=start_notes
    )

    # basic formatting
    ws.freeze_panes = "A4"
    ws.column_dimensions["A"].width = 48
    ws.column_dimensions["B"].width = 16
    ws.column_dimensions["C"].width = 18
    ws.column_dimensions["D"].width = 28
    ws.column_dimensions["E"].width = 14
    ws.column_dimensions["F"].width = 75

    ws["A1"].font = ws["A1"].font.copy(bold=True, size=12)

print("WROTE:")
print(xlsx)
print(OUTDIR / "Table_S17A_attrition_selection_model.tsv")
print(OUTDIR / "Table_S17B_IPSW_effect_comparison.tsv")
