#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/data/qiallab/Framingham")
OUT = ROOT / "results"
TX = OUT / "transcriptomic_enrichment"
GEN = OUT / "genetics_pathways"
FINAL = OUT / "final_validation"
TABLES = OUT / "paper_figures_tables"
PKG = TABLES / "SUBMISSION_PACKAGE" / "05_Source_Data"

TABLES.mkdir(parents=True, exist_ok=True)
PKG.mkdir(parents=True, exist_ok=True)

# ============================================================
# INPUTS
# ============================================================

CORE = FINAL / "FINAL_master_core4_interactions.csv"

WGS = (
    GEN
    / "pathway_heart_brain_models"
    / "pathway_omnibus_primary28.tsv"
)

AHBA_TARGETED = (
    TX
    / "pathway_expression_vs_primary_interaction_maps_SPATIAL_NULL_FAST.tsv"
)

DISCOVERY = (
    TX
    / "GO_Reactome_spatial_coupling_FAST_N5000_ALL.tsv"
)

# ============================================================
# DEFINITIONS FOR FOUR FROZEN ASSOCIATIONS
# ============================================================

pairs = [
    {
        "pair_order": 1,
        "association": "LVEF × APOE ε4 → L middle temporal volume",
        "cardiac": "LVEF",
        "metric": "volume",
        "outcome": "lh_middletemporal_vol",
    },
    {
        "pair_order": 2,
        "association": "LVESVi × APOE ε4 → R superior frontal area",
        "cardiac": "LVESVi",
        "metric": "surface_area",
        "outcome": "rh_superiorfrontal_area",
    },
    {
        "pair_order": 3,
        "association": "LV mass index × APOE ε4 → R lingual volume",
        "cardiac": "LV_MASSi",
        "metric": "volume",
        "outcome": "rh_lingual_vol",
    },
    {
        "pair_order": 4,
        "association": "LV mass index × APOE ε4 → R lingual area",
        "cardiac": "LV_MASSi",
        "metric": "surface_area",
        "outcome": "rh_lingual_area",
    },
]

# ============================================================
# LOAD
# ============================================================

core = pd.read_csv(CORE)
wgs = pd.read_csv(WGS, sep="\t")
ahba = pd.read_csv(AHBA_TARGETED, sep="\t")
disc = pd.read_csv(DISCOVERY, sep="\t")

print("CORE columns:")
print(core.columns.tolist())
print("\nWGS columns:")
print(wgs.columns.tolist())
print("\nAHBA columns:")
print(ahba.columns.tolist())
print("\nDISCOVERY columns:")
print(disc.columns.tolist())

# ============================================================
# HELPERS
# ============================================================

def first_existing(df, names):
    for x in names:
        if x in df.columns:
            return x
    return None

def fmt_p(x):
    if pd.isna(x):
        return ""
    if x < 0.0001:
        return "<0.0001"
    return f"{x:.4f}"

def fmt_num(x, nd=3):
    if pd.isna(x):
        return ""
    return f"{x:.{nd}f}"

# Primary-effect column detection
core_cardiac = first_existing(core, ["cardiac_metric", "cardiac", "cardiac_predictor"])
core_outcome = first_existing(core, ["brain_metric", "outcome", "brain_outcome"])
core_beta = first_existing(
    core,
    ["interaction_beta", "beta_interaction", "Interaction beta"]
)
core_p = first_existing(
    core,
    ["interaction_P", "interaction_p", "P", "p"]
)
core_q = first_existing(
    core,
    ["interaction_q", "q", "FDR q", "interaction_fdr_q"]
)
core_ci_lo = first_existing(
    core,
    ["interaction_CI_low", "CI_low", "ci_low"]
)
core_ci_hi = first_existing(
    core,
    ["interaction_CI_high", "CI_high", "ci_high"]
)

# WGS column detection
wgs_cardiac = first_existing(wgs, ["cardiac", "cardiac_predictor"])
wgs_outcome = first_existing(wgs, ["outcome", "brain_outcome"])
wgs_pathway = first_existing(wgs, ["pathway", "pathway_name"])

# Prefer cluster-robust inferential columns
wgs_p = first_existing(
    wgs,
    [
        "omnibus_P_cluster",
        "omnibus_p_cluster",
        "cluster_robust_P",
        "omnibus_P",
        "P_cluster",
        "P"
    ]
)

wgs_q = first_existing(
    wgs,
    [
        "q_cluster_primary28",
        "omnibus_q_cluster",
        "q_cluster",
        "FDR_q_cluster",
        "q",
    ]
)

# AHBA targeted
ahba_p = "spatial_empirical_P"
ahba_q = "spatial_q_primary28"
ahba_rho = "observed_spearman_rho"

# Discovery
disc_p = "spatial_empirical_P"
disc_q_within = "q_within_map_library"
disc_q_global = "q_global"
disc_rho = "observed_spearman_rho"

# ============================================================
# BUILD TABLE
# ============================================================

rows = []

for spec in pairs:

    cardiac = spec["cardiac"]
    metric = spec["metric"]
    outcome = spec["outcome"]

    # --------------------------------------------------------
    # PRIMARY PHENOTYPE
    # --------------------------------------------------------

    c = core.copy()

    if core_cardiac is not None:
        c = c[c[core_cardiac] == cardiac]

    if core_outcome is not None:
        c = c[c[core_outcome] == outcome]

    if len(c) != 1:
        print(
            f"WARNING: core match for {cardiac}/{outcome}: {len(c)} rows"
        )

    if len(c):
        cr = c.iloc[0]
        beta = cr[core_beta] if core_beta else np.nan
        p_primary = cr[core_p] if core_p else np.nan
        q_primary = cr[core_q] if core_q else np.nan
        lo = cr[core_ci_lo] if core_ci_lo else np.nan
        hi = cr[core_ci_hi] if core_ci_hi else np.nan
    else:
        beta = p_primary = q_primary = lo = hi = np.nan

    # --------------------------------------------------------
    # WGS: strongest omnibus pathway for this association
    # --------------------------------------------------------

    g = wgs.copy()

    if wgs_cardiac is not None:
        g = g[g[wgs_cardiac] == cardiac]

    if wgs_outcome is not None:
        g = g[g[wgs_outcome] == outcome]

    if len(g) and wgs_p is not None:
        g = g.sort_values(wgs_p)
        gr = g.iloc[0]

        best_wgs_pathway = (
            gr[wgs_pathway]
            if wgs_pathway is not None
            else ""
        )
        best_wgs_p = gr[wgs_p]
        best_wgs_q = (
            gr[wgs_q]
            if wgs_q is not None
            else np.nan
        )
        n_wgs_pathways = len(g)
    else:
        best_wgs_pathway = ""
        best_wgs_p = np.nan
        best_wgs_q = np.nan
        n_wgs_pathways = 0

    # --------------------------------------------------------
    # TARGETED AHBA: strongest of 7 pathways
    # --------------------------------------------------------

    a = ahba[
        (ahba["cardiac"] == cardiac)
        &
        (ahba["metric"] == metric)
    ].copy()

    if len(a):
        a = a.sort_values(ahba_p)
        ar = a.iloc[0]

        best_ahba_pathway = ar["pathway"]
        best_ahba_rho = ar[ahba_rho]
        best_ahba_p = ar[ahba_p]
        best_ahba_q = ar[ahba_q]
        n_ahba_targeted = len(a)
    else:
        best_ahba_pathway = ""
        best_ahba_rho = np.nan
        best_ahba_p = np.nan
        best_ahba_q = np.nan
        n_ahba_targeted = 0

    # --------------------------------------------------------
    # UNBIASED DISCOVERY: strongest GO/Reactome term
    # --------------------------------------------------------

    d = disc[
        (disc["cardiac"] == cardiac)
        &
        (disc["metric"] == metric)
    ].copy()

    if len(d):
        d = d.sort_values(
            [disc_q_global, disc_q_within, disc_p]
        )
        dr = d.iloc[0]

        best_disc_library = dr["library"]
        best_disc_term = dr["term"]
        best_disc_rho = dr[disc_rho]
        best_disc_p = dr[disc_p]
        best_disc_q_within = dr[disc_q_within]
        best_disc_q_global = dr[disc_q_global]

        n_disc_tests = len(d)
        n_disc_global_sig = int(
            (d[disc_q_global] < 0.05).sum()
        )
    else:
        best_disc_library = ""
        best_disc_term = ""
        best_disc_rho = np.nan
        best_disc_p = np.nan
        best_disc_q_within = np.nan
        best_disc_q_global = np.nan
        n_disc_tests = 0
        n_disc_global_sig = 0

    # --------------------------------------------------------
    # INTERPRETATION
    # --------------------------------------------------------

    wgs_positive = (
        pd.notna(best_wgs_q)
        and best_wgs_q < 0.05
    )

    ahba_positive = (
        pd.notna(best_ahba_q)
        and best_ahba_q < 0.05
    )

    discovery_positive = (
        n_disc_global_sig > 0
    )

    if not wgs_positive and not ahba_positive and not discovery_positive:
        conclusion = (
            "No significant mechanistic convergence across "
            "participant-level WGS, targeted AHBA spatial testing, "
            "or unbiased GO/Reactome spatial discovery."
        )
    else:
        hits = []
        if wgs_positive:
            hits.append("WGS")
        if ahba_positive:
            hits.append("targeted AHBA")
        if discovery_positive:
            hits.append("GO/Reactome discovery")
        conclusion = "Signal detected in: " + ", ".join(hits)

    rows.append({
        "Association":
            spec["association"],

        "Primary interaction β":
            beta,

        "Primary 95% CI":
            (
                f"{fmt_num(lo)} to {fmt_num(hi)}"
                if pd.notna(lo) and pd.notna(hi)
                else ""
            ),

        "Primary P":
            p_primary,

        "Primary FDR q":
            q_primary,

        "WGS pathways tested":
            n_wgs_pathways,

        "Top-ranked WGS pathway":
            best_wgs_pathway,

        "Top-ranked WGS omnibus P":
            best_wgs_p,

        "Top-ranked WGS FDR q":
            best_wgs_q,

        "Targeted AHBA pathways tested":
            n_ahba_targeted,

        "Top-ranked targeted AHBA pathway":
            best_ahba_pathway,

        "Top-ranked targeted AHBA Spearman ρ":
            best_ahba_rho,

        "Top-ranked targeted AHBA spatial P":
            best_ahba_p,

        "Targeted AHBA FDR q":
            best_ahba_q,

        "GO/Reactome discovery tests":
            n_disc_tests,

        "Top-ranked discovery library":
            best_disc_library,

        "Top-ranked discovery term":
            best_disc_term,

        "Top-ranked discovery Spearman ρ":
            best_disc_rho,

        "Top-ranked discovery spatial P":
            best_disc_p,

        "Top-ranked discovery within-map/library q":
            best_disc_q_within,

        "Top-ranked discovery global q":
            best_disc_q_global,

        "Global FDR-significant discovery terms":
            n_disc_global_sig,

        "Mechanistic triangulation conclusion":
            conclusion,
    })


table = pd.DataFrame(rows)

# ============================================================
# ROUND FOR PUBLICATION TABLE
# ============================================================

display = table.copy()

for col in [
    "Primary interaction β",
    "Top-ranked targeted AHBA Spearman ρ",
    "Top-ranked discovery Spearman ρ",
]:
    display[col] = display[col].map(
        lambda x: "" if pd.isna(x) else f"{x:.3f}"
    )

for col in [
    "Primary P",
    "Primary FDR q",
    "Top-ranked WGS omnibus P",
    "Top-ranked WGS FDR q",
    "Top-ranked targeted AHBA spatial P",
    "Targeted AHBA FDR q",
    "Top-ranked discovery spatial P",
    "Top-ranked discovery within-map/library q",
    "Top-ranked discovery global q",
]:
    display[col] = display[col].map(fmt_p)

# ============================================================
# SAVE
# ============================================================

rawfile = TABLES / "TableS8_mechanistic_triangulation_RAW.tsv"
pubfile = TABLES / "TableS8_mechanistic_triangulation.tsv"
csvfile = TABLES / "TableS8_mechanistic_triangulation.csv"

table.to_csv(rawfile, sep="\t", index=False)
display.to_csv(pubfile, sep="\t", index=False)
display.to_csv(csvfile, index=False)

# Submission-package source copy
display.to_csv(
    PKG / "TableS8_mechanistic_triangulation.tsv",
    sep="\t",
    index=False
)

print("\n============================================================")
print("TABLE S8 — MECHANISTIC TRIANGULATION")
print("============================================================\n")

print(display.to_string(index=False))

print("\nSaved:")
print(rawfile)
print(pubfile)
print(csvfile)
print(PKG / "TableS8_mechanistic_triangulation.tsv")
