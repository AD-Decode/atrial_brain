#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr

ROOT = Path("/data/qiallab/Framingham")
OUT  = ROOT / "results/transcriptomic_enrichment"

EXPR = OUT / "AHBA_DesikanKilliany_expression_FULL.tsv"
INFO = OUT / "AHBA_DesikanKilliany_atlas_info.tsv"
TARGET = OUT / "Framingham_DK56_target_regions.tsv"
PANEL = ROOT / "results/genetics_pathways/gene_panel_v1.tsv"
MAPS = OUT / "wholebrain_M3_coupling_maps_DesikanKilliany.tsv"

expr = pd.read_csv(EXPR, sep="\t", index_col=0)
info = pd.read_csv(INFO, sep="\t")
target = pd.read_csv(TARGET, sep="\t")
panel = pd.read_csv(PANEL, sep="\t")
maps = pd.read_csv(MAPS, sep="\t")

# ---------------------------------------------------------
# Build AHBA IDs corresponding exactly to Framingham DK56
# ---------------------------------------------------------
info["hemi"] = info["hemisphere"].map({"L":"lh", "R":"rh"})
info["desikan_label"] = info["hemi"].astype(str) + "_" + info["label"].astype(str)

target_labels = set(target["desikan_label"])

match = info[
    (info["structure"] == "cortex") &
    (info["desikan_label"].isin(target_labels))
].copy()

print("Matched AHBA cortical regions:", len(match))
print("Unique Framingham target labels:", len(target_labels))

missing_regions = sorted(target_labels - set(match["desikan_label"]))

if missing_regions:
    print("\nMISSING TARGET REGIONS:")
    print("\n".join(missing_regions))
    raise RuntimeError("Not all 56 Framingham regions matched AHBA atlas.")

assert len(match) == 56

# Expression rows indexed by atlas ID
expr.index = pd.to_numeric(expr.index)
match = match.sort_values("id")

expr56 = expr.loc[match["id"].values].copy()
expr56.index = match["desikan_label"].values

print("Matched expression matrix:", expr56.shape)

expr56.to_csv(
    OUT / "AHBA_DK56_expression.tsv",
    sep="\t"
)

# ---------------------------------------------------------
# Gene-wise z-score ACROSS REGIONS
# ---------------------------------------------------------
mu = expr56.mean(axis=0)
sd = expr56.std(axis=0, ddof=1)

valid = sd > 0
exprz = (expr56.loc[:, valid] - mu[valid]) / sd[valid]

# ---------------------------------------------------------
# Pathway regional scores
# ---------------------------------------------------------
pathway_scores = pd.DataFrame(index=expr56.index)

coverage_rows = []

for pathway, g in panel.groupby("pathway"):

    requested = sorted(set(g["gene"].astype(str)))
    available = [x for x in requested if x in exprz.columns]

    coverage_rows.append({
        "pathway": pathway,
        "genes_requested": len(requested),
        "genes_available": len(available),
        "pct_available": 100 * len(available) / len(requested)
    })

    if len(available) < 3:
        pathway_scores[pathway] = np.nan
        continue

    # Mean standardized expression across genes
    pathway_scores[pathway] = exprz[available].mean(axis=1)

coverage = pd.DataFrame(coverage_rows)

pathway_scores.to_csv(
    OUT / "AHBA_DK56_pathway_expression_scores.tsv",
    sep="\t"
)

coverage.to_csv(
    OUT / "AHBA_DK56_pathway_coverage.tsv",
    sep="\t",
    index=False
)

print("\nPATHWAY COVERAGE")
print(coverage.round(1).to_string(index=False))

# ---------------------------------------------------------
# Four manuscript interaction maps
# ---------------------------------------------------------
primary = [
    ("LVEF", "volume"),
    ("LVESVi", "surface_area"),
    ("LV_MASSi", "volume"),
    ("LV_MASSi", "surface_area"),
]

results = []

for cardiac, metric in primary:

    m = maps[
        (maps["cardiac"] == cardiac) &
        (maps["metric"] == metric)
    ].copy()

    m = m.set_index("desikan_label")

    common = sorted(
        set(m.index).intersection(pathway_scores.index)
    )

    if len(common) != 56:
        raise RuntimeError(
            f"{cardiac}/{metric}: expected 56 matched regions, got {len(common)}"
        )

    beta = m.loc[common, "beta_interaction"].astype(float)

    for pathway in pathway_scores.columns:

        score = pathway_scores.loc[common, pathway].astype(float)

        ok = beta.notna() & score.notna()

        rho, p_spear = spearmanr(beta[ok], score[ok])
        r, p_pear = pearsonr(beta[ok], score[ok])

        results.append({
            "cardiac": cardiac,
            "metric": metric,
            "pathway": pathway,
            "N_regions": int(ok.sum()),
            "spearman_rho": rho,
            "spearman_P_naive": p_spear,
            "pearson_r": r,
            "pearson_P_naive": p_pear,
        })

res = pd.DataFrame(results)

# BH FDR helper
def bh(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * len(p) / np.arange(1, len(p)+1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.minimum(q, 1)
    out = np.empty_like(q)
    out[order] = q
    return out

res["spearman_q_naive28"] = bh(res["spearman_P_naive"].values)

res = res.sort_values(
    ["spearman_P_naive", "cardiac", "metric", "pathway"]
)

outfile = OUT / "pathway_expression_vs_primary_interaction_maps_NAIVE.tsv"
res.to_csv(outfile, sep="\t", index=False)

print("\n==========================================================")
print("DESCRIPTIVE PATHWAY EXPRESSION × APOE4 COUPLING")
print("==========================================================")
print(
    res[
        [
            "cardiac",
            "metric",
            "pathway",
            "N_regions",
            "spearman_rho",
            "spearman_P_naive",
            "spearman_q_naive28",
        ]
    ]
    .round(5)
    .to_string(index=False)
)

print("\nIMPORTANT: these P values are descriptive only.")
print("They do NOT yet account for cortical spatial autocorrelation.")

print("\nWrote:", outfile)
