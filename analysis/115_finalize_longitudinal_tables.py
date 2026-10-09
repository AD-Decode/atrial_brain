#!/usr/bin/env python3

from pathlib import Path
import pandas as pd

ROOT = Path("/data/qiallab/Framingham")

TP = (
    ROOT
    / "results/longitudinal_heart_brain/"
      "temporal_prediction"
)

FIG = TP / "figures"
VASC = TP / "vascular_adjustment"
DOSE = TP / "APOE4_dose_sensitivity"

OUT = (
    ROOT
    / "results/paper_figures_tables/"
      "SUBMISSION_PACKAGE_NAMED/05_Supporting_Data"
)

OUT.mkdir(parents=True, exist_ok=True)

# ============================================================
# TABLE S-L1
# Primary interaction + GEE sensitivity
# ============================================================

a = pd.read_csv(
    FIG / "Figure109_panelA_interaction_estimates.tsv",
    sep="\t"
)

a.to_csv(
    OUT / "Table_SL1_Primary_longitudinal_interaction_and_GEE.tsv",
    sep="\t",
    index=False
)

# ============================================================
# TABLE S-L2
# APOE-stratified simple slopes
# ============================================================

b = pd.read_csv(
    FIG / "Figure109_panelB_simple_slopes.tsv",
    sep="\t"
)

b.to_csv(
    OUT / "Table_SL2_APOE4_stratified_simple_slopes.tsv",
    sep="\t",
    index=False
)

# ============================================================
# TABLE S-L3
# Early vascular/metabolic adjustment
# ============================================================

v = pd.read_csv(
    VASC / "LA_ventricle_early_vascular_adjustment.tsv",
    sep="\t"
)

v.to_csv(
    OUT / "Table_SL3_Early_vascular_metabolic_adjustment.tsv",
    sep="\t",
    index=False
)

# ============================================================
# TABLE S-L4
# Ordinal APOE ε4 dose
# ============================================================

ordinal = pd.read_csv(
    DOSE / "APOE4_dose_interaction_results.tsv",
    sep="\t"
)

ordinal_slopes = pd.read_csv(
    DOSE / "APOE4_dose_simple_slopes.tsv",
    sep="\t"
)

ordinal.to_csv(
    OUT / "Table_SL4A_APOE4_ordinal_dose_interaction.tsv",
    sep="\t",
    index=False
)

ordinal_slopes.to_csv(
    OUT / "Table_SL4B_APOE4_ordinal_dose_simple_slopes.tsv",
    sep="\t",
    index=False
)

# ============================================================
# TABLE S-L5
# Categorical APOE ε4 dose
# ============================================================

cat_slopes = pd.read_csv(
    DOSE / "APOE4_categorical_dose_simple_slopes.tsv",
    sep="\t"
)

cat_joint = pd.read_csv(
    DOSE / "APOE4_categorical_dose_joint_test.tsv",
    sep="\t"
)

cat_joint.to_csv(
    OUT / "Table_SL5A_APOE4_categorical_dose_joint_test.tsv",
    sep="\t",
    index=False
)

cat_slopes.to_csv(
    OUT / "Table_SL5B_APOE4_categorical_dose_simple_slopes.tsv",
    sep="\t",
    index=False
)

print("\nFINAL LONGITUDINAL TABLES")
for f in sorted(OUT.glob("Table_SL*.tsv")):
    print(f)
