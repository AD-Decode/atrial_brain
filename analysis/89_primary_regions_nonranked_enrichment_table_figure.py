#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path("/data/qiallab")
SRC = (
    ROOT /
    "reference_data/transcriptomics/AHBA/regional_atlas/"
    "DK83_region_process_enrichment_FDR05.tsv"
)

OUT = (
    ROOT /
    "Framingham/results/transcriptomic_enrichment"
)
OUT.mkdir(parents=True, exist_ok=True)

REGIONS = {
    "lh_middletemporal":
        "Left middle temporal",
    "rh_superiorfrontal":
        "Right superior frontal",
    "rh_lingual":
        "Right lingual",
}

FINDINGS = {
    "lh_middletemporal":
        "LVEF × APOE4 → volume",
    "rh_superiorfrontal":
        "LVESVi × APOE4 → surface area",
    "rh_lingual":
        "LV mass index × APOE4 → volume / surface area",
}

df = pd.read_csv(SRC, sep="\t")

x = df[
    df["region"].isin(REGIONS)
].copy()

# For the present result only GO_BP survives,
# but retain library in the table.
x["Region"] = x["region"].map(REGIONS)
x["Heart–brain association"] = x["region"].map(FINDINGS)

x["Enrichment ratio"] = pd.to_numeric(
    x["enrichment_ratio"], errors="coerce"
)
x["P"] = pd.to_numeric(
    x["P"], errors="coerce"
)
x["FDR q"] = pd.to_numeric(
    x["FDR_q_region"], errors="coerce"
)

# Clean GO IDs out of display label, while retaining original term.
x["Process"] = (
    x["term"]
    .str.replace(
        r"\s*\(GO:\d+\)$",
        "",
        regex=True
    )
)

# ---------------------------------------------
# Publication-ready table
# ---------------------------------------------
table = x[
    [
        "Region",
        "Heart–brain association",
        "library",
        "term",
        "Enrichment ratio",
        "P",
        "FDR q",
        "overlap_genes",
    ]
].copy()

table.columns = [
    "Region",
    "Heart–brain association",
    "Gene-set library",
    "Enriched biological process",
    "Enrichment ratio",
    "P",
    "FDR q",
    "Overlap genes",
]

table = table.sort_values(
    ["Region", "FDR q", "P"]
)

table_out = (
    OUT /
    "Table_AHBA_primary_regions_nonranked_enrichment.tsv"
)

table.to_csv(
    table_out,
    sep="\t",
    index=False
)

# Also Excel-friendly CSV
table.to_csv(
    OUT /
    "Table_AHBA_primary_regions_nonranked_enrichment.csv",
    index=False
)

# ---------------------------------------------
# Figure data
# ---------------------------------------------
plot = x.copy()

plot["minus_log10_q"] = (
    -np.log10(plot["FDR q"])
)

# order by region, then significance
region_order = [
    "Left middle temporal",
    "Right superior frontal",
]

plot["Region"] = pd.Categorical(
    plot["Region"],
    categories=region_order,
    ordered=True,
)

plot = plot.sort_values(
    ["Region", "FDR q"],
    ascending=[True, False]
)

# More compact labels
label_map = {
    "Regulation Of Epithelial To Mesenchymal Transition":
        "Regulation of epithelial-to-mesenchymal transition",
    "Nervous System Development":
        "Nervous system development",
    "Calcium-Ion Regulated Exocytosis":
        "Calcium-ion regulated exocytosis",
    "Regulation Of Regulated Secretory Pathway":
        "Regulation of regulated secretory pathway",
    "Regulation Of Heterotypic Cell-Cell Adhesion":
        "Regulation of heterotypic cell-cell adhesion",
    "Cholesterol Biosynthetic Process":
        "Cholesterol biosynthetic process",
    "Cellular Response To Calcium Ion":
        "Cellular response to calcium ion",
    "Diterpenoid Metabolic Process":
        "Diterpenoid metabolic process",
}

plot["display_process"] = plot["Process"].map(
    lambda z: label_map.get(z, z)
)

# create combined y label to make region membership explicit
plot["y_label"] = (
    plot["display_process"]
    + "  ["
    + plot["Region"].astype(str)
    + "]"
)

fig_h = max(
    5.0,
    0.55 * len(plot) + 1.6
)

fig, ax = plt.subplots(
    figsize=(10.5, fig_h)
)

y = np.arange(len(plot))

# point size encodes enrichment ratio
sizes = 40 + 28 * plot["Enrichment ratio"].values

ax.scatter(
    plot["minus_log10_q"],
    y,
    s=sizes,
    alpha=0.8
)

ax.set_yticks(y)
ax.set_yticklabels(
    plot["y_label"],
    fontsize=9
)

ax.invert_yaxis()

ax.set_xlabel(
    "−log10(FDR q)"
)

ax.set_title(
    "Regional transcriptomic enrichment of Framingham "
    "heart–brain loci"
)

# q=.05 reference
ax.axvline(
    -np.log10(0.05),
    linestyle="--",
    linewidth=1
)

# Annotate enrichment ratio
for yi, (_, row) in enumerate(plot.iterrows()):
    ax.text(
        row["minus_log10_q"] + 0.025,
        yi,
        f"{row['Enrichment ratio']:.1f}×",
        va="center",
        fontsize=8,
    )

# Explicit note about lingual result
ax.text(
    0.01,
    -0.10,
    "Right lingual cortex: no GO Biological Process or "
    "Reactome term reached FDR q < 0.05.",
    transform=ax.transAxes,
    fontsize=9,
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()

for ext in ["png", "pdf", "svg"]:
    fig.savefig(
        OUT /
        f"Figure_AHBA_primary_regions_nonranked_enrichment.{ext}",
        dpi=300,
        bbox_inches="tight",
    )

plt.close(fig)

# ---------------------------------------------
# Concise console summary
# ---------------------------------------------
print("\nPUBLICATION TABLE")
print("=" * 80)

print(
    table[
        [
            "Region",
            "Enriched biological process",
            "Enrichment ratio",
            "FDR q",
        ]
    ]
    .round({
        "Enrichment ratio": 2,
        "FDR q": 4,
    })
    .to_string(index=False)
)

print("\nRight lingual:")
print("No FDR-significant GO BP or Reactome enrichment.")

print("\nWrote:")
print(table_out)
print(
    OUT /
    "Figure_AHBA_primary_regions_nonranked_enrichment.pdf"
)
