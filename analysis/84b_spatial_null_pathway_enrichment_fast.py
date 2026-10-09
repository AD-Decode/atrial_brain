from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from statsmodels.stats.multitest import multipletests

ROOT = Path("/data/qiallab/Framingham")
OUT = ROOT / "results/transcriptomic_enrichment"
CACHE = OUT / "spatial_null_cache"
CACHE.mkdir(parents=True, exist_ok=True)

N_PERM = 5000
SEED = 20260923

# Inputs
COUPLING = OUT / "wholebrain_M3_coupling_maps_DesikanKilliany.tsv"
PATHWAYS = OUT / "AHBA_DK56_pathway_expression_scores.tsv"

# ------------------------------------------------------------------
# IMPORTANT:
# Reuse the exact Moran-null generation function / adjacency definition
# from script 84. This fast script assumes a function:
#
#     generate_moran_nulls(values, n_perm, seed)
#
# returning shape (n_perm, n_regions).
#
# Paste the exact helper and spatial-weight construction from script 84
# below this comment. Do NOT change the spatial-null method.
# ------------------------------------------------------------------

def rank_rows(x):
    """Rank each row independently."""
    return np.apply_along_axis(rankdata, 1, x)

def corr_vec_with_matrix(x, Y):
    """
    Pearson correlation between vector x and every row of Y.
    Applied to ranks, this is Spearman correlation.
    """
    x = np.asarray(x, float)
    Y = np.asarray(Y, float)

    x = x - x.mean()
    Y = Y - Y.mean(axis=1, keepdims=True)

    num = Y @ x
    den = np.sqrt((Y * Y).sum(axis=1) * (x * x).sum())

    return num / den

def empirical_two_sided_p(obs, null):
    return (1 + np.sum(np.abs(null) >= abs(obs))) / (len(null) + 1)

# ================================================================
# Load data
# ================================================================
coup = pd.read_csv(COUPLING, sep="\t")
pway = pd.read_csv(PATHWAYS, sep="\t")

print("Coupling columns:", coup.columns.tolist(), flush=True)
print("Pathway columns:", pway.columns.tolist(), flush=True)

# Adapt these if needed after inspecting current files.
region_col_c = "region"
region_col_p = "region"

# Find pathway columns: everything numeric except region metadata
meta_candidates = {
    "region", "hemi", "hemisphere", "label", "atlas",
    "structure", "network"
}
pathway_cols = [
    c for c in pway.columns
    if c not in meta_candidates
    and pd.api.types.is_numeric_dtype(pway[c])
]

# Find coupling map columns: numeric columns excluding region metadata
map_cols = [
    c for c in coup.columns
    if c not in meta_candidates
    and pd.api.types.is_numeric_dtype(coup[c])
]

# Merge once and preserve exact region order
d = coup.merge(
    pway,
    left_on=region_col_c,
    right_on=region_col_p,
    how="inner",
    suffixes=("_map", "_path")
)

print(f"N merged regions: {len(d)}", flush=True)
print(f"N coupling maps: {len(map_cols)}", flush=True)
print(f"N pathways: {len(pathway_cols)}", flush=True)

results = []

for mi, map_col in enumerate(map_cols, 1):
    print(
        f"\n[{mi}/{len(map_cols)}] MAP: {map_col}",
        flush=True
    )

    beta = pd.to_numeric(d[map_col], errors="coerce").to_numpy(float)

    good_beta = np.isfinite(beta)
    if good_beta.sum() < 10:
        print("  skipped: insufficient regions", flush=True)
        continue

    cache_file = CACHE / f"{map_col}_moran_nulls_{N_PERM}.npy"

    if cache_file.exists():
        null_maps = np.load(cache_file)
        print(
            f"  loaded cached nulls: {null_maps.shape}",
            flush=True
        )
    else:
        print("  generating Moran nulls...", flush=True)

        # ----------------------------------------------------------
        # Replace this line using the EXACT function from script 84:
        # null_maps = generate_moran_nulls(
        #     beta,
        #     n_perm=N_PERM,
        #     seed=SEED + mi
        # )
        # ----------------------------------------------------------
        raise RuntimeError(
            "Paste exact Moran-null generator from script 84 "
            "before running."
        )

        np.save(cache_file, null_maps)
        print(
            f"  cached nulls to {cache_file}",
            flush=True
        )

    # Rank observed and null maps once
    beta_rank = rankdata(beta)
    null_rank = rank_rows(null_maps)

    for pathway in pathway_cols:
        expr = pd.to_numeric(d[pathway], errors="coerce").to_numpy(float)

        good = np.isfinite(beta) & np.isfinite(expr)

        if good.sum() < 10:
            continue

        # Fast path: complete data across all regions
        if good.all():
            expr_rank = rankdata(expr)

            obs_rho = np.corrcoef(beta_rank, expr_rank)[0, 1]

            null_rho = corr_vec_with_matrix(
                expr_rank,
                null_rank
            )

        # Fallback for missing pathway values
        else:
            br = rankdata(beta[good])
            er = rankdata(expr[good])

            obs_rho = np.corrcoef(br, er)[0, 1]

            nr = rank_rows(null_maps[:, good])
            null_rho = corr_vec_with_matrix(
                er,
                nr
            )

        p_spatial = empirical_two_sided_p(
            obs_rho,
            null_rho
        )

        results.append({
            "cardiac_map": map_col,
            "pathway": pathway,
            "N_regions": int(good.sum()),
            "spearman_rho": float(obs_rho),
            "spatial_P": float(p_spatial),
            "null_mean": float(np.mean(null_rho)),
            "null_sd": float(np.std(null_rho, ddof=1)),
        })

    # checkpoint after every map
    tmp = pd.DataFrame(results)

    checkpoint = OUT / (
        "pathway_expression_vs_primary_interaction_maps_"
        "SPATIAL_NULL_FAST_CHECKPOINT.tsv"
    )
    tmp.to_csv(checkpoint, sep="\t", index=False)

    print(
        f"  checkpoint rows: {len(tmp)}",
        flush=True
    )

res = pd.DataFrame(results)

# BH across all targeted tests
res["q_spatial_global"] = multipletests(
    res["spatial_P"],
    method="fdr_bh"
)[1]

outfile = OUT / (
    "pathway_expression_vs_primary_interaction_maps_"
    "SPATIAL_NULL_FAST.tsv"
)

res.to_csv(outfile, sep="\t", index=False)

print("\nDONE", flush=True)
print(outfile, flush=True)
print(
    res.sort_values("spatial_P")
       .head(20)
       .to_string(index=False),
    flush=True
)
