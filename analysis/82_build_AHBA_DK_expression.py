#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import numpy as np
import abagen

ROOT = Path("/data/qiallab/Framingham")
OUT = ROOT / "results/transcriptomic_enrichment"
OUT.mkdir(parents=True, exist_ok=True)

MAPFILE = OUT / "wholebrain_M3_coupling_maps_DesikanKilliany.tsv"
GENEPANEL = ROOT / "results/genetics_pathways/gene_panel_v1.tsv"

# ------------------------------------------------------------
# Exact 56 DK regions used in Framingham
# ------------------------------------------------------------
maps = pd.read_csv(MAPFILE, sep="\t")

regions56 = (
    maps[["hemi", "parcel", "desikan_label"]]
    .drop_duplicates()
    .sort_values(["hemi", "parcel"])
)

print("Framingham DK parcels:", len(regions56))
assert len(regions56) == 56

# ------------------------------------------------------------
# Fetch standard DK atlas distributed with abagen
# ------------------------------------------------------------
atlas = abagen.fetch_desikan_killiany()

print("Atlas image:", atlas["image"])
print("Atlas info:", atlas["info"])

atlas_info = pd.read_csv(atlas["info"])
print("\nAtlas info columns:", list(atlas_info.columns))
print("Atlas regions:", len(atlas_info))
print(atlas_info.head())

# ------------------------------------------------------------
# Generate AHBA regional expression
#
# Common recommended processing choices:
# - intensity-based probe filtering
# - differential stability probe selection
# - scaled robust sigmoid normalization
# - normalization within structures
# ------------------------------------------------------------
expression = abagen.get_expression_data(
    atlas["image"],
    atlas_info=atlas["info"],
    probe_selection="diff_stability",
    donor_probes="aggregate",
    lr_mirror="bidirectional",
    missing="interpolate",
    sample_norm="srs",
    gene_norm="srs",
    norm_structures=True,
    region_agg="donors",
    agg_metric="mean",
    return_donors=False,
    verbose=1,
)

print("\nAHBA expression shape:", expression.shape)
print("Index example:", expression.index[:10].tolist())
print("Gene example:", expression.columns[:10].tolist())

expression.to_csv(
    OUT / "AHBA_DesikanKilliany_expression_FULL.tsv",
    sep="\t"
)

# ------------------------------------------------------------
# Inspect atlas labels and map to our FreeSurfer names
# ------------------------------------------------------------
info = atlas_info.copy()

print("\nFULL ATLAS INFO")
print(info.to_string(index=False))

info.to_csv(
    OUT / "AHBA_DesikanKilliany_atlas_info.tsv",
    sep="\t",
    index=False
)

# ------------------------------------------------------------
# Save the Framingham target parcel list
# ------------------------------------------------------------
regions56.to_csv(
    OUT / "Framingham_DK56_target_regions.tsv",
    sep="\t",
    index=False
)

# ------------------------------------------------------------
# Check our prespecified genes against AHBA
# ------------------------------------------------------------
panel = pd.read_csv(GENEPANEL, sep="\t")

available = set(expression.columns.astype(str))

genecheck = (
    panel[["pathway", "gene"]]
    .drop_duplicates()
    .assign(
        present_in_AHBA=lambda x:
            x["gene"].astype(str).isin(available)
    )
)

print("\nGENE AVAILABILITY BY PATHWAY")
summary = (
    genecheck.groupby("pathway")
    .agg(
        genes_requested=("gene", "nunique"),
        genes_available=("present_in_AHBA", "sum")
    )
)

summary["pct_available"] = (
    100 * summary["genes_available"] /
    summary["genes_requested"]
)

print(summary.round(1).to_string())

missing = genecheck[~genecheck["present_in_AHBA"]]

print("\nMissing genes:")
if len(missing):
    print(missing.to_string(index=False))
else:
    print("None")

genecheck.to_csv(
    OUT / "AHBA_gene_panel_availability.tsv",
    sep="\t",
    index=False
)

print("\nWrote:")
print(" ", OUT / "AHBA_DesikanKilliany_expression_FULL.tsv")
print(" ", OUT / "AHBA_DesikanKilliany_atlas_info.tsv")
print(" ", OUT / "Framingham_DK56_target_regions.tsv")
print(" ", OUT / "AHBA_gene_panel_availability.tsv")
