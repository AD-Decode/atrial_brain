#!/usr/bin/env python3

from pathlib import Path
import os
import time
import json
import hashlib

import numpy as np
import pandas as pd
import nibabel as nib
from scipy.stats import rankdata
from neuromaps import nulls


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path("/data/qiallab/Framingham")
OUT = ROOT / "results/transcriptomic_enrichment"
CACHE = OUT / "spatial_null_cache"
CACHE.mkdir(parents=True, exist_ok=True)

EXPR_SCORES = OUT / "AHBA_DK56_pathway_expression_scores.tsv"
MAPS = OUT / "wholebrain_M3_coupling_maps_DesikanKilliany.tsv"
INFO = OUT / "AHBA_DesikanKilliany_atlas_info.tsv"

N_PERM = int(os.environ.get("N_PERM", "5000"))
SEED = int(os.environ.get("MORAN_SEED", "20260922"))

N_PROC = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))

PRIMARY = [
    ("LVEF", "volume"),
    ("LVESVi", "surface_area"),
    ("LV_MASSi", "volume"),
    ("LV_MASSi", "surface_area"),
]


# ============================================================
# HELPERS
# ============================================================

def bh(p):
    """Benjamini-Hochberg, matching original script."""
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    ranked = p[order]

    q = ranked * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.minimum(q, 1)

    out = np.empty_like(q)
    out[order] = q
    return out


def centered_ranks_vector(x):
    r = rankdata(np.asarray(x, float)).astype(np.float64)
    return r - r.mean()


def centered_ranks_columns(X):
    """
    X shape = n_regions x n_maps
    Each COLUMN gets ranked independently.
    """
    R = rankdata(np.asarray(X, float), axis=0).astype(np.float64)
    R -= R.mean(axis=0, keepdims=True)
    return R


def corr_centered(xc, yc):
    return float(
        np.dot(xc, yc) /
        np.sqrt(
            np.dot(xc, xc) *
            np.dot(yc, yc)
        )
    )


def correlations_against_nulls(pathway_centered_rank, null_centered_ranks):
    """
    pathway_centered_rank: (56,)
    null_centered_ranks:    (56, 5000)

    Returns 5000 Spearman correlations simultaneously.
    """
    x = pathway_centered_rank
    Y = null_centered_ranks

    numerator = x @ Y

    denominator = np.sqrt(
        np.dot(x, x) *
        np.sum(Y * Y, axis=0)
    )

    return numerator / denominator


def md5_array(x):
    x = np.ascontiguousarray(x)
    return hashlib.md5(x.tobytes()).hexdigest()


# ============================================================
# STARTUP
# ============================================================

print("=" * 72, flush=True)
print("FAST AHBA TARGETED SPATIAL-NULL ANALYSIS", flush=True)
print("=" * 72, flush=True)
print(f"N permutations: {N_PERM}", flush=True)
print(f"CPUs:           {N_PROC}", flush=True)
print(f"Seed base:      {SEED}", flush=True)
print(f"Cache:          {CACHE}", flush=True)


# ============================================================
# EXACT SAME DESIKAN-KILLIANY PARCELLATION AS SCRIPT 84
# ============================================================

try:
    import abagen
    atlas = abagen.fetch_desikan_killiany()
    ATLAS_IMG = Path(atlas["image"])
except Exception as e:
    raise RuntimeError(f"Could not locate abagen DK atlas: {e}")


scores = pd.read_csv(EXPR_SCORES, sep="\t", index_col=0)
maps = pd.read_csv(MAPS, sep="\t")
info = pd.read_csv(INFO, sep="\t")

info["hemi"] = info["hemisphere"].map({"L": "lh", "R": "rh"})
info["desikan_label"] = (
    info["hemi"].astype(str)
    + "_"
    + info["label"].astype(str)
)

labels56 = list(scores.index)

if len(labels56) != 56:
    raise RuntimeError(
        f"Expected 56 pathway-score regions, found {len(labels56)}"
    )

print(f"Regions:        {len(labels56)}", flush=True)
print(f"Pathways:       {len(scores.columns)}", flush=True)
print("Pathway names:", list(scores.columns), flush=True)


img = nib.load(str(ATLAS_IMG))
arr = np.asanyarray(img.dataobj).copy()

label_to_id = dict(zip(info["desikan_label"], info["id"]))

missing = [x for x in labels56 if x not in label_to_id]
if missing:
    raise RuntimeError(f"Missing atlas labels: {missing}")

ids56 = [int(label_to_id[x]) for x in labels56]

partial = np.zeros(arr.shape, dtype=np.int16)

for new_id, old_id in enumerate(ids56, start=1):
    partial[arr == old_id] = new_id

partial_img = nib.Nifti1Image(
    partial,
    affine=img.affine,
    header=img.header
)

partial_file = OUT / "Framingham_DK56_partial_parcellation.nii.gz"
nib.save(partial_img, partial_file)

print("Parcellation:", partial_file, flush=True)


# ============================================================
# SAVE REGION ORDER — CRITICAL FOR REUSING NULLS
# ============================================================

region_order_file = CACHE / "DK56_region_order.tsv"

pd.DataFrame({
    "index0": np.arange(56),
    "index1": np.arange(1, 57),
    "desikan_label": labels56,
}).to_csv(region_order_file, sep="\t", index=False)

print("Region order:", region_order_file, flush=True)


# ============================================================
# RUN FOUR PRIMARY MAPS
# ============================================================

results = []
manifest = []

for map_index, (cardiac, metric) in enumerate(PRIMARY):

    t_map = time.time()

    print("\n" + "=" * 72, flush=True)
    print(
        f"MAP {map_index + 1}/4: {cardiac} / {metric}",
        flush=True
    )
    print("=" * 72, flush=True)

    m = maps[
        (maps["cardiac"] == cardiac) &
        (maps["metric"] == metric)
    ].copy()

    m = m.set_index("desikan_label")

    missing_map = [x for x in labels56 if x not in m.index]
    if missing_map:
        raise RuntimeError(
            f"Missing coupling regions for {cardiac}/{metric}: "
            f"{missing_map}"
        )

    beta = (
        m.loc[labels56, "beta_interaction"]
        .astype(float)
        .values
    )

    if not np.all(np.isfinite(beta)):
        raise RuntimeError(
            f"Nonfinite beta values for {cardiac}/{metric}"
        )

    beta_hash = md5_array(beta)

    cache_stem = (
        f"{map_index:02d}_{cardiac}_{metric}"
        f"_N{N_PERM}_seed{SEED + map_index}"
    )

    null_file = CACHE / f"{cache_stem}.npy"
    meta_file = CACHE / f"{cache_stem}.json"

    # --------------------------------------------------------
    # LOAD PRECOMPUTED PARCEL DISTANCE MATRIX
    # --------------------------------------------------------

    DISTMAT_FILE = (
        CACHE /
        "DK56_MNI152_parcel_distance.npy"
    )

    if not DISTMAT_FILE.exists():
        raise RuntimeError(
            f"Missing cached distance matrix: {DISTMAT_FILE}\n"
            "Run 83b_build_DK56_distance_cache.py first."
        )

    distmat = np.load(DISTMAT_FILE)

    if distmat.shape != (56, 56):
        raise RuntimeError(
            f"Unexpected distance matrix shape: {distmat.shape}"
        )

    # --------------------------------------------------------
    # LOAD OR GENERATE MORAN NULLS
    # --------------------------------------------------------

    use_cache = False

    if null_file.exists() and meta_file.exists():
        with open(meta_file) as f:
            meta = json.load(f)

        if (
            meta.get("beta_md5") == beta_hash
            and meta.get("N_PERM") == N_PERM
            and meta.get("seed") == SEED + map_index
            and meta.get("labels56") == labels56
        ):
            use_cache = True

    if use_cache:
        print("Loading cached nulls:", null_file, flush=True)
        surrogate = np.load(null_file)

    else:
        print(
            f"Generating {N_PERM} Moran null maps "
            f"using n_proc={N_PROC}...",
            flush=True
        )

        t_null = time.time()

        # EXACT SAME NULL MODEL AS ORIGINAL SCRIPT 84
        surrogate = nulls.moran(
            beta,
            atlas="MNI152",
            density="2mm",
            parcellation=str(partial_file),
            distmat=distmat,
            n_perm=N_PERM,
            seed=SEED + map_index,
            n_proc=N_PROC
        )

        print(
            "Moran generation: "
            f"{(time.time() - t_null) / 60:.2f} min",
            flush=True
        )

        np.save(null_file, surrogate)

        meta = {
            "cardiac": cardiac,
            "metric": metric,
            "map_index": map_index,
            "N_PERM": N_PERM,
            "seed": SEED + map_index,
            "beta_md5": beta_hash,
            "labels56": labels56,
            "shape": list(surrogate.shape),
        }

        with open(meta_file, "w") as f:
            json.dump(meta, f, indent=2)

        print("Saved null cache:", null_file, flush=True)

    if surrogate.shape != (56, N_PERM):
        raise RuntimeError(
            f"Unexpected surrogate shape for "
            f"{cardiac}/{metric}: {surrogate.shape}"
        )

    # --------------------------------------------------------
    # VECTORIZE ALL SPEARMAN NULL CORRELATIONS
    # --------------------------------------------------------

    t_corr = time.time()

    beta_rank = centered_ranks_vector(beta)

    # Each COLUMN = one surrogate map
    null_rank = centered_ranks_columns(surrogate)

    for pathway in scores.columns:

        pv = (
            scores.loc[labels56, pathway]
            .astype(float)
            .values
        )

        if not np.all(np.isfinite(pv)):
            raise RuntimeError(
                f"Nonfinite values for pathway {pathway}"
            )

        pathway_rank = centered_ranks_vector(pv)

        observed = corr_centered(
            beta_rank,
            pathway_rank
        )

        null_r = correlations_against_nulls(
            pathway_rank,
            null_rank
        )

        empirical_p = (
            1
            + np.sum(np.abs(null_r) >= abs(observed))
        ) / (N_PERM + 1)

        results.append({
            "cardiac": cardiac,
            "metric": metric,
            "pathway": pathway,
            "N_regions": 56,
            "observed_spearman_rho": observed,
            "spatial_empirical_P": empirical_p,
            "null_mean_rho": float(np.mean(null_r)),
            "null_SD_rho": float(np.std(null_r, ddof=1)),
            "N_permutations": N_PERM,
        })

    print(
        "Seven pathway correlations: "
        f"{time.time() - t_corr:.3f} sec",
        flush=True
    )

    # --------------------------------------------------------
    # CHECKPOINT AFTER EVERY MAP
    # --------------------------------------------------------

    checkpoint = (
        OUT /
        "pathway_expression_vs_primary_interaction_maps_"
        "SPATIAL_NULL_FAST_CHECKPOINT.tsv"
    )

    pd.DataFrame(results).to_csv(
        checkpoint,
        sep="\t",
        index=False
    )

    manifest.append({
        "map_index": map_index,
        "cardiac": cardiac,
        "metric": metric,
        "seed": SEED + map_index,
        "null_file": str(null_file),
        "beta_md5": beta_hash,
        "elapsed_min": (time.time() - t_map) / 60,
    })

    pd.DataFrame(manifest).to_csv(
        CACHE / "spatial_null_manifest.tsv",
        sep="\t",
        index=False
    )

    print(
        f"Checkpoint: {len(results)} / 28 tests",
        flush=True
    )
    print(
        f"Map total: {(time.time() - t_map) / 60:.2f} min",
        flush=True
    )


# ============================================================
# FINAL MULTIPLICITY CORRECTION
# ============================================================

res = pd.DataFrame(results)

if len(res) != 28:
    raise RuntimeError(
        f"Expected exactly 28 tests, got {len(res)}"
    )

res["spatial_q_primary28"] = bh(
    res["spatial_empirical_P"].values
)

res = res.sort_values(
    [
        "spatial_empirical_P",
        "cardiac",
        "metric",
        "pathway"
    ]
).reset_index(drop=True)


outfile = (
    OUT /
    "pathway_expression_vs_primary_interaction_maps_"
    "SPATIAL_NULL_FAST.tsv"
)

res.to_csv(
    outfile,
    sep="\t",
    index=False
)


# ============================================================
# REPORT
# ============================================================

print("\n" + "=" * 72, flush=True)
print("FINAL FAST SPATIAL RESULTS", flush=True)
print("=" * 72, flush=True)

show = [
    "cardiac",
    "metric",
    "pathway",
    "observed_spearman_rho",
    "spatial_empirical_P",
    "spatial_q_primary28",
    "null_mean_rho",
    "null_SD_rho",
]

print(
    res[show]
    .round(6)
    .to_string(index=False),
    flush=True
)

print("\nSpatial P < 0.05:", flush=True)
x = res[res["spatial_empirical_P"] < 0.05]

if len(x):
    print(
        x[show]
        .round(6)
        .to_string(index=False),
        flush=True
    )
else:
    print("None.", flush=True)

print("\nSpatial FDR q < 0.05:", flush=True)
x = res[res["spatial_q_primary28"] < 0.05]

if len(x):
    print(
        x[show]
        .round(6)
        .to_string(index=False),
        flush=True
    )
else:
    print("None.", flush=True)

print("\nSaved:", outfile, flush=True)
print("Cache:", CACHE, flush=True)
print("DONE", flush=True)
