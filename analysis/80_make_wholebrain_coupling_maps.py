#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import numpy as np
import re

ROOT = Path("/data/qiallab/Framingham")
SOURCE = ROOT / "results/tier2_corrected/tier2_corrected_all.csv"
OUT = ROOT / "results/transcriptomic_enrichment"
OUT.mkdir(parents=True, exist_ok=True)

print("Using FULL-COHORT Tier-2 table:")
print(" ", SOURCE)

d = pd.read_csv(SOURCE)

print("Input shape:", d.shape)
print("Columns:", list(d.columns))

required = [
    "metric_type",
    "family",
    "region",
    "cardiac",
    "model",
    "N",
    "beta_cardiac",
    "SE_cardiac",
    "p_cardiac",
    "beta_interaction",
    "SE_interaction",
    "p_interaction",
    "q_cardiac",
    "q_interaction",
]

missing = [c for c in required if c not in d.columns]
if missing:
    raise RuntimeError(f"Missing required columns: {missing}")

# ------------------------------------------------------------
# Restrict to manuscript M3
# ------------------------------------------------------------
m3 = d[d["model"].astype(str).str.contains("M3", case=False, na=False)].copy()

print("\nM3 rows:", len(m3))

# ------------------------------------------------------------
# Parse cortical Desikan outcomes
#
# Examples:
# lh_middletemporal_vol
# rh_superiorfrontal_area
# ------------------------------------------------------------
def parse_cortical_region(x):
    x = str(x)

    m = re.match(
        r"^(lh|rh)_(.+)_(vol|volume|area|thickness|thick)$",
        x,
        flags=re.I
    )

    if m is None:
        return pd.Series({
            "hemi": np.nan,
            "parcel": np.nan,
            "parsed_metric": np.nan
        })

    hemi = m.group(1).lower()
    parcel = m.group(2)
    suffix = m.group(3).lower()

    if suffix in ("vol", "volume"):
        metric = "volume"
    elif suffix == "area":
        metric = "surface_area"
    else:
        metric = "thickness"

    return pd.Series({
        "hemi": hemi,
        "parcel": parcel,
        "parsed_metric": metric
    })

parsed = m3["region"].apply(parse_cortical_region)
m3 = pd.concat([m3, parsed], axis=1)

cort = m3.dropna(
    subset=["hemi", "parcel", "parsed_metric"]
).copy()

# Verify parsed suffix agrees with model metric_type
mismatch = (
    cort["metric_type"].astype(str) !=
    cort["parsed_metric"].astype(str)
)

print("\nParsed cortical rows:", len(cort))
print("Metric mismatches:", int(mismatch.sum()))

if mismatch.any():
    print(
        cort.loc[
            mismatch,
            ["region", "metric_type", "parsed_metric"]
        ].head(20).to_string(index=False)
    )

# Use source metric_type as authoritative
cort["metric"] = cort["metric_type"].astype(str)

# Standard Desikan label
cort["desikan_label"] = (
    cort["hemi"] + "_" + cort["parcel"]
)

# ------------------------------------------------------------
# Clean map table
# ------------------------------------------------------------
maps = cort[
    [
        "cardiac",
        "metric",
        "region",
        "hemi",
        "parcel",
        "desikan_label",
        "N",
        "beta_cardiac",
        "SE_cardiac",
        "p_cardiac",
        "q_cardiac",
        "beta_interaction",
        "SE_interaction",
        "p_interaction",
        "q_interaction",
    ]
].copy()

# Ensure one row per cardiac x metric x cortical parcel
dups = maps.duplicated(
    ["cardiac", "metric", "desikan_label"],
    keep=False
)

print(
    "Duplicate cardiac × metric × parcel rows:",
    int(dups.sum())
)

if dups.any():
    print(
        maps.loc[
            dups,
            ["cardiac", "metric", "desikan_label", "region"]
        ].head(30).to_string(index=False)
    )
    raise RuntimeError("Unexpected duplicated cortical map rows.")

# ------------------------------------------------------------
# MASTER MAP
# ------------------------------------------------------------
master_file = OUT / "wholebrain_M3_coupling_maps_FULL.tsv"

maps.to_csv(
    master_file,
    sep="\t",
    index=False
)

# ------------------------------------------------------------
# Separate map files
# ------------------------------------------------------------
for (cardiac, metric), g in maps.groupby(
    ["cardiac", "metric"]
):

    safe_card = (
        str(cardiac)
        .replace("/", "_")
        .replace(" ", "_")
    )

    g = g.sort_values(
        ["hemi", "parcel"]
    ).copy()

    outfile = (
        OUT /
        f"{safe_card}_{metric}_coupling_map_FULL.tsv"
    )

    g.to_csv(
        outfile,
        sep="\t",
        index=False
    )

# ------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------
summary = (
    maps
    .groupby(["cardiac", "metric"])
    .agg(
        N_regions=("desikan_label", "nunique"),

        N_min=("N", "min"),
        N_max=("N", "max"),

        mean_main_beta=("beta_cardiac", "mean"),
        SD_main_beta=("beta_cardiac", "std"),
        min_main_beta=("beta_cardiac", "min"),
        max_main_beta=("beta_cardiac", "max"),

        mean_interaction_beta=("beta_interaction", "mean"),
        SD_interaction_beta=("beta_interaction", "std"),
        min_interaction_beta=("beta_interaction", "min"),
        max_interaction_beta=("beta_interaction", "max"),

        nominal_interaction_regions=(
            "p_interaction",
            lambda x: int((x < 0.05).sum())
        ),

        FDR_interaction_regions=(
            "q_interaction",
            lambda x: int((x < 0.05).sum())
        ),
    )
    .reset_index()
)

summary_file = (
    OUT /
    "wholebrain_coupling_map_summary_FULL.tsv"
)

summary.to_csv(
    summary_file,
    sep="\t",
    index=False
)

print("\n============================================================")
print("FULL-COHORT WHOLE-BRAIN COUPLING MAP SUMMARY")
print("============================================================")
print(summary.round(4).to_string(index=False))

print("\nCardiac measures:")
print(sorted(maps["cardiac"].unique()))

print("\nMetrics:")
print(maps["metric"].value_counts())

print("\nTotal map rows:", len(maps))
print("Unique cortical parcels:", maps["desikan_label"].nunique())

print("\nWrote:")
print(" ", master_file)
print(" ", summary_file)
