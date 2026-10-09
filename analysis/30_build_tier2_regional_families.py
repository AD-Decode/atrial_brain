#!/usr/bin/env python3

from pathlib import Path
import re
import pandas as pd
import numpy as np

ROOT = Path("/data/qiallab/Framingham")
RES  = ROOT / "results"

FILE = RES / "neurocardiac_metadata_analysis_ready_metabolic.csv"
OUTDIR = RES / "tier2_regional"
OUTDIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(FILE, nrows=5)

cols = list(df.columns)

# ============================================================
# 1. Identify cortical regional metrics
# ============================================================

def metric_type(c):

    if c.endswith("_vol"):
        return "volume"

    if c.endswith("_tk"):
        return "thickness"

    if c.endswith("_area"):
        return "surface_area"

    return None


# ============================================================
# 2. Anatomical family classification
# ============================================================

def family(c):
    s = c.lower()

    # medial temporal / AD-relevant
    if re.search(
        r"hippocamp|entorh|parahipp|temporal.?pole|pole_temporal",
        s
    ):
        return "medial_temporal_AD"

    # cingulate
    if re.search(
        r"cingul",
        s
    ):
        return "cingulate"

    # insula
    if re.search(
        r"insula",
        s
    ):
        return "insula"

    # frontal
    if re.search(
        r"frontal|frontopol|orbitofrontal|precentral",
        s
    ):
        return "frontal"

    # parietal / precuneus
    if re.search(
        r"parietal|precuneus|postcentral",
        s
    ):
        return "parietal_precuneus"

    # temporal
    if re.search(
        r"temporal|fusiform",
        s
    ):
        return "temporal_other"

    # occipital
    if re.search(
        r"occipital|calcarine|lingual|cuneus",
        s
    ):
        return "occipital"

    return None


rows = []

for c in cols:

    mt = metric_type(c)

    if mt is None:
        continue

    fam = family(c)

    if fam is None:
        continue

    rows.append({
        "variable": c,
        "family": fam,
        "metric_type": mt
    })


# ============================================================
# 3. Add subcortical regions manually
# ============================================================

subcortical = [
    "Left_Hippocampus",
    "Right_Hippocampus",
    "Left_Amygdala",
    "Right_Amygdala",
    "Left_Thalamus_Proper",
    "Right_Thalamus_Proper",
    "Left_Caudate",
    "Right_Caudate",
    "Left_Putamen",
    "Right_Putamen",
    "Left_Pallidum",
    "Right_Pallidum",
    "Left_Accumbens_area",
    "Right_Accumbens_area",
]

for c in subcortical:

    if c in cols:

        fam = (
            "medial_temporal_AD"
            if "Hippocampus" in c or "Amygdala" in c
            else "subcortical_nuclei"
        )

        rows.append({
            "variable": c,
            "family": fam,
            "metric_type": "volume"
        })


out = pd.DataFrame(rows).drop_duplicates()

out = out.sort_values(
    ["metric_type", "family", "variable"]
)

out.to_csv(
    OUTDIR / "tier2_region_family_inventory.csv",
    index=False
)

print("\nTIER 2 REGIONAL INVENTORY")
print("=========================")

print(
    out.groupby(
        ["metric_type", "family"]
    )
    .size()
    .to_string()
)

print("\nTOTAL VARIABLES:", len(out))

print("\nSAMPLE:")
print(
    out.head(100).to_string(index=False)
)

print("\nSaved:")
print(
    OUTDIR / "tier2_region_family_inventory.csv"
)
