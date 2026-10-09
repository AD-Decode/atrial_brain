#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from statsmodels.stats.multitest import multipletests

ROOT = Path("/data/qiallab/Framingham")
TX = ROOT / "results/transcriptomic_enrichment"
CACHE = TX / "spatial_null_cache"

EXPR = TX / "AHBA_DesikanKilliany_expression_FULL.tsv"
ATLAS = TX / "AHBA_DesikanKilliany_atlas_info.tsv"
ORDER = CACHE / "DK56_region_order.tsv"

GMT = (
    ROOT / "reference_data/msigdb/"
    "h.all.v2026.1.Hs.symbols.gmt"
)

MAPFILE = TX / "wholebrain_M3_coupling_maps_DesikanKilliany.tsv"

OUT_ALL = TX / "MSigDB_Hallmark_spatial_N5000_ALL.tsv"
OUT_SIG = TX / "MSigDB_Hallmark_spatial_N5000_GLOBAL_FDR05.tsv"
OUT_TOP = TX / "MSigDB_Hallmark_spatial_N5000_TOP20.tsv"
OUT_SCORES = TX / "MSigDB_Hallmark_DK56_regional_scores.tsv"

N_PERM = 5000
MIN_GENES = 10


# ============================================================
# HELPERS
# ============================================================

def clean_gene(x):
    return str(x).strip().upper()


def read_gmt(path):
    out = {}
    with open(path) as f:
        for line in f:
            x = line.rstrip("\n").split("\t")
            if len(x) < 3:
                continue
            name = x[0]
            genes = {clean_gene(g) for g in x[2:] if g.strip()}
            out[name] = genes
    return out


def rank_columns(A):
    """
    Rank each column of matrix A.
    A: regions x permutations
    """
    A = np.asarray(A, dtype=float)
    R = np.empty_like(A, dtype=float)

    for j in range(A.shape[1]):
        R[:, j] = rankdata(A[:, j])

    return R


def spearman_observed(x, y):
    rx = rankdata(np.asarray(x, float))
    ry = rankdata(np.asarray(y, float))

    rx -= rx.mean()
    ry -= ry.mean()

    return np.sum(rx * ry) / np.sqrt(
        np.sum(rx**2) * np.sum(ry**2)
    )


def spearman_against_nulls(x, nulls):
    """
    x: 56-vector pathway score
    nulls: 56 x N_PERM
    """
    rx = rankdata(np.asarray(x, float))
    rx -= rx.mean()

    rn = rank_columns(nulls)
    rn -= rn.mean(axis=0, keepdims=True)

    numerator = np.sum(rx[:, None] * rn, axis=0)

    denominator = np.sqrt(
        np.sum(rx**2) *
        np.sum(rn**2, axis=0)
    )

    return numerator / denominator


# ============================================================
# CHECK FILES
# ============================================================

required = [EXPR, ATLAS, ORDER, GMT, MAPFILE]

for p in required:
    if not p.exists():
        raise SystemExit(f"Missing required file: {p}")

print("Expression:", EXPR)
print("Atlas info:", ATLAS)
print("DK56 order:", ORDER)
print("Hallmark GMT:", GMT)
print("Coupling maps:", MAPFILE)


# ============================================================
# LOAD AHBA
# ============================================================

expr = pd.read_csv(EXPR, sep="\t", index_col=0)
atlas = pd.read_csv(ATLAS, sep="\t")
order = pd.read_csv(ORDER, sep="\t")

# Expression row labels are atlas IDs
expr.index = pd.to_numeric(expr.index)

print("\nAHBA full expression:", expr.shape)
print("Atlas rows:", atlas.shape)
print("DK56 rows:", order.shape)


# ============================================================
# BUILD EXACT DESIKAN LABEL -> ATLAS ID MAP
# ============================================================

atlas["desikan_label"] = np.where(
    atlas["hemisphere"].eq("L"),
    "lh_" + atlas["label"].astype(str),
    np.where(
        atlas["hemisphere"].eq("R"),
        "rh_" + atlas["label"].astype(str),
        ""
    )
)

lookup = dict(
    zip(
        atlas["desikan_label"],
        atlas["id"]
    )
)

dk_labels = order["desikan_label"].astype(str).tolist()

missing_labels = [
    x for x in dk_labels
    if x not in lookup
]

if missing_labels:
    raise RuntimeError(
        "DK56 labels missing from atlas info:\n" +
        "\n".join(missing_labels)
    )

dk_ids = [lookup[x] for x in dk_labels]

missing_ids = [
    x for x in dk_ids
    if x not in expr.index
]

if missing_ids:
    raise RuntimeError(
        f"Atlas IDs missing from expression matrix: {missing_ids}"
    )

# IMPORTANT:
# subset in exact DK56 spatial-null order
X = expr.loc[dk_ids].copy()
X.index = dk_labels

print("\nDK56 expression:", X.shape)
print("First 10 DK56 labels:")
print(X.index[:10].tolist())


# ============================================================
# GENE CLEANUP
# ============================================================

X.columns = [clean_gene(x) for x in X.columns]

# Collapse duplicate gene symbols if any
X = X.T.groupby(level=0).mean().T

# Keep genes observed in at least 50/56 regions
X = X.loc[:, X.notna().sum(axis=0) >= 50]

# Fill any residual missing values with regional mean
X = X.apply(lambda s: s.fillna(s.mean()), axis=0)

# Z-score each gene across the 56 regions
mu = X.mean(axis=0)
sd = X.std(axis=0, ddof=1)

keep = sd > 0
X = X.loc[:, keep]
Z = (X - mu[keep]) / sd[keep]

print("Genes retained:", Z.shape[1])


# ============================================================
# HALLMARK REGIONAL SCORES
# ============================================================

sets = read_gmt(GMT)

print("\nHallmark gene sets in GMT:", len(sets))

if len(sets) != 50:
    print(
        "WARNING: expected 50 Hallmark signatures, "
        f"found {len(sets)}"
    )

available = set(Z.columns)

score_data = {}
n_genes = {}

for name, genes in sets.items():

    overlap = sorted(genes & available)

    if len(overlap) < MIN_GENES:
        print(
            f"SKIP {name}: only {len(overlap)} genes represented"
        )
        continue

    score_data[name] = Z[overlap].mean(axis=1)
    n_genes[name] = len(overlap)


scores = pd.DataFrame(score_data, index=dk_labels)
scores.index.name = "desikan_label"

scores.to_csv(
    OUT_SCORES,
    sep="\t"
)

print(
    "Hallmark signatures retained:",
    scores.shape[1]
)


# ============================================================
# LOAD CARDIAC × APOE4 COUPLING MAPS
# ============================================================

maps = pd.read_csv(MAPFILE, sep="\t")

print("\nCoupling-map columns:")
print(maps.columns.tolist())


# Identify region column
region_col = None

for c in [
    "desikan_label",
    "region",
    "region_name",
    "parcel",
    "label",
]:
    if c in maps.columns:
        region_col = c
        break

if region_col is None:
    raise RuntimeError(
        "Could not identify region column in coupling-map file."
    )

maps = maps.set_index(region_col)

missing = [
    x for x in dk_labels
    if x not in maps.index
]

if missing:
    raise RuntimeError(
        "Coupling map missing DK56 labels:\n"
        + "\n".join(missing)
    )

maps = maps.loc[dk_labels]


# ============================================================
# MAP DEFINITIONS + EXISTING NULLS
# ============================================================

specs = [
    (
        "LVEF",
        "volume",
        "00_LVEF_volume_N5000_seed20260922.npy",
    ),
    (
        "LVESVi",
        "surface_area",
        "01_LVESVi_surface_area_N5000_seed20260923.npy",
    ),
    (
        "LV_MASSi",
        "volume",
        "02_LV_MASSi_volume_N5000_seed20260924.npy",
    ),
    (
        "LV_MASSi",
        "surface_area",
        "03_LV_MASSi_surface_area_N5000_seed20260925.npy",
    ),
]



# ============================================================
# LONG-FORM COUPLING MAP HANDLING
# ============================================================

required_map_cols = {
    "cardiac",
    "metric",
    "beta_interaction",
}

missing_cols = required_map_cols - set(maps.columns)

if missing_cols:
    raise RuntimeError(
        f"Missing required coupling-map columns: {sorted(missing_cols)}"
    )


# ============================================================
# TEST HALLMARK SIGNATURES
# ============================================================


rows = []

for cardiac, metric, cache_name in specs:

    m = maps[
        (maps["cardiac"] == cardiac)
        &
        (maps["metric"] == metric)
    ].copy()

    if len(m) != 56:
        raise RuntimeError(
            f"{cardiac}/{metric}: expected 56 rows, found {len(m)}"
        )

    # maps already uses desikan_label as its index
    missing = [
        x for x in dk_labels
        if x not in m.index
    ]

    if missing:
        raise RuntimeError(
            f"{cardiac}/{metric}: missing DK56 regions:\n"
            + "\n".join(missing)
        )

    # exact same regional order as spatial-null cache
    m = m.loc[dk_labels]

    beta = pd.to_numeric(
        m["beta_interaction"],
        errors="raise"
    ).values

    cache_file = CACHE / cache_name

    if not cache_file.exists():
        raise RuntimeError(
            f"Missing cached N5000 nulls: {cache_file}"
        )

    nulls = np.load(
        cache_file,
        mmap_mode="r"
    )

    # normalize orientation to 56 x 5000
    if nulls.shape == (N_PERM, 56):
        nulls = nulls.T

    if nulls.shape != (56, N_PERM):
        raise RuntimeError(
            f"{cache_name}: unexpected shape {nulls.shape}"
        )

    print(
        f"\nTesting {cardiac} / {metric}"
    )
    print("Using beta_interaction from long-format coupling table")
    print("Null shape:", nulls.shape)

    # Rank the 5000 spatial-null maps ONCE for this cardiac map.
    # This is much faster than re-ranking for all 50 Hallmark sets.
    ranked_nulls = rank_columns(nulls)
    ranked_nulls -= ranked_nulls.mean(axis=0, keepdims=True)

    ranked_null_denom = np.sqrt(
        np.sum(ranked_nulls ** 2, axis=0)
    )

    for hallmark in scores.columns:

        s = scores[hallmark].values

        rho = spearman_observed(
            s,
            beta
        )

        rs = rankdata(np.asarray(s, float))
        rs -= rs.mean()

        null_rho = (
            np.sum(rs[:, None] * ranked_nulls, axis=0)
            /
            (
                np.sqrt(np.sum(rs ** 2))
                * ranked_null_denom
            )
        )

        p_spatial = (
            1
            + np.sum(
                np.abs(null_rho)
                >= abs(rho)
            )
        ) / (N_PERM + 1)

        rows.append({
            "cardiac": cardiac,
            "metric": metric,
            "hallmark": hallmark,
            "N_regions": 56,
            "N_genes_AHBA": n_genes[hallmark],
            "observed_spearman_rho": rho,
            "spatial_empirical_P": p_spatial,
            "null_mean_rho": np.mean(null_rho),
            "null_SD_rho": np.std(
                null_rho,
                ddof=1
            ),
            "N_permutations": N_PERM,
        })


res = pd.DataFrame(rows)


# ============================================================
# FDR
# ============================================================

res["q_within_map"] = np.nan

for _, idx in res.groupby(
    ["cardiac", "metric"]
).groups.items():

    res.loc[idx, "q_within_map"] = (
        multipletests(
            res.loc[
                idx,
                "spatial_empirical_P"
            ],
            method="fdr_bh"
        )[1]
    )


res["q_global"] = multipletests(
    res["spatial_empirical_P"],
    method="fdr_bh"
)[1]


res = res.sort_values(
    [
        "q_global",
        "q_within_map",
        "spatial_empirical_P",
    ]
).reset_index(drop=True)


# ============================================================
# SAVE
# ============================================================

res.to_csv(
    OUT_ALL,
    sep="\t",
    index=False
)

res[
    res["q_global"] < 0.05
].to_csv(
    OUT_SIG,
    sep="\t",
    index=False
)

(
    res.groupby(
        ["cardiac", "metric"],
        group_keys=False
    )
    .head(5)
    .to_csv(
        OUT_TOP,
        sep="\t",
        index=False
    )
)


# ============================================================
# SUMMARY
# ============================================================

print("\n============================================================")
print("MSIGDB HALLMARK — SPATIAL NULL RESULTS")
print("============================================================")

print("Total tests:", len(res))
print(
    "Spatial P < 0.05:",
    int(
        (res["spatial_empirical_P"] < .05).sum()
    )
)
print(
    "Within-map FDR q < 0.05:",
    int(
        (res["q_within_map"] < .05).sum()
    )
)
print(
    "Global FDR q < 0.05:",
    int(
        (res["q_global"] < .05).sum()
    )
)

print("\nMinimum values:")
print(
    "min spatial P =",
    res["spatial_empirical_P"].min()
)
print(
    "min within-map q =",
    res["q_within_map"].min()
)
print(
    "min global q =",
    res["q_global"].min()
)

print("\nTOP 20:")
print(
    res[
        [
            "cardiac",
            "metric",
            "hallmark",
            "N_genes_AHBA",
            "observed_spearman_rho",
            "spatial_empirical_P",
            "q_within_map",
            "q_global",
        ]
    ]
    .head(20)
    .to_string(index=False)
)

print("\nSaved:")
print(OUT_ALL)
print(OUT_SIG)
print(OUT_TOP)
print(OUT_SCORES)

print("\nDONE")
