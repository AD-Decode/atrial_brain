#!/usr/bin/env python3

from pathlib import Path
import os
import re
import json
import time
import hashlib

import numpy as np
import pandas as pd
from scipy.stats import rankdata


# =====================================================================
# CONFIG
# =====================================================================

ROOT = Path("/data/qiallab/Framingham")
OUT = ROOT / "results/transcriptomic_enrichment"
CACHE = OUT / "spatial_null_cache"

GSEA_DIR = Path(
    "/data/qiallab/reference_data/transcriptomics/AHBA/"
    "regional_atlas/rank_based_array"
)

ATLAS_INFO = OUT / "AHBA_DesikanKilliany_atlas_info.tsv"
MAPS = OUT / "wholebrain_M3_coupling_maps_DesikanKilliany.tsv"
REGION_ORDER = CACHE / "DK56_region_order.tsv"

N_PERM = int(os.environ.get("N_PERM", "5000"))
BLOCK_TERMS = int(os.environ.get("BLOCK_TERMS", "250"))

# Recover MIN_REGIONS from original script 90 when possible.
OLD90 = ROOT / "code/90_GO_Reactome_spatial_coupling_enrichment.py"

MIN_REGIONS = None

if OLD90.exists():
    txt = OLD90.read_text()
    m = re.search(r"MIN_REGIONS\s*=\s*(\d+)", txt)
    if m:
        MIN_REGIONS = int(m.group(1))

if MIN_REGIONS is None:
    MIN_REGIONS = int(os.environ.get("MIN_REGIONS", "40"))

PRIMARY = [
    ("LVEF", "volume"),
    ("LVESVi", "surface_area"),
    ("LV_MASSi", "volume"),
    ("LV_MASSi", "surface_area"),
]

SEED_BASE = 20260922


# =====================================================================
# HELPERS
# =====================================================================

def bh(p):
    p = np.asarray(p, dtype=float)

    valid = np.isfinite(p)

    out = np.full(len(p), np.nan)

    if not valid.any():
        return out

    pv = p[valid]

    order = np.argsort(pv)
    ranked = pv[order]

    q = ranked * len(pv) / np.arange(1, len(pv) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.minimum(q, 1)

    tmp = np.empty_like(q)
    tmp[order] = q

    out[valid] = tmp

    return out


def md5_array(x):
    x = np.ascontiguousarray(x)
    return hashlib.md5(x.tobytes()).hexdigest()


def centered_rank_columns(X):
    """
    Rank each column independently, then center.
    X contains no NaNs.
    """
    R = rankdata(
        np.asarray(X, float),
        axis=0
    ).astype(np.float64)

    R -= R.mean(
        axis=0,
        keepdims=True
    )

    return R


def detect_column(df, candidates):
    lower = {
        str(c).lower(): c
        for c in df.columns
    }

    for x in candidates:
        if x.lower() in lower:
            return lower[x.lower()]

    return None


def canonical_library(x):
    s = str(x).strip()
    lo = s.lower()

    if "reactome" in lo:
        return "Reactome"

    if (
        "go" in lo
        or "biological_process" in lo
        or "gene_ontology" in lo
    ):
        return "GO_BP"

    return s


# =====================================================================
# REGION ORDER
# =====================================================================

if not REGION_ORDER.exists():
    raise RuntimeError(
        f"Missing region-order file: {REGION_ORDER}\n"
        "Run script 84b first."
    )

regions = pd.read_csv(
    REGION_ORDER,
    sep="\t"
)["desikan_label"].astype(str).tolist()

if len(regions) != 56:
    raise RuntimeError(
        f"Expected 56 regions, got {len(regions)}"
    )

region_set = set(regions)


# =====================================================================
# ATLAS INFO — used for robust label identification
# =====================================================================

atlas_info = pd.read_csv(
    ATLAS_INFO,
    sep="\t"
)

atlas_info["hemi"] = (
    atlas_info["hemisphere"]
    .map({"L": "lh", "R": "rh"})
)

atlas_info["desikan_label"] = (
    atlas_info["hemi"].astype(str)
    + "_"
    + atlas_info["label"].astype(str)
)

known_labels = set(
    atlas_info["desikan_label"].astype(str)
)


# =====================================================================
# LOAD REGIONAL GSEA FILES
# =====================================================================

gsea_files = sorted(
    GSEA_DIR.glob("*_GSEA.tsv")
)

if not gsea_files:
    raise RuntimeError(
        f"No *_GSEA.tsv files found in {GSEA_DIR}"
    )

print("=" * 78, flush=True)
print("FAST GO/REACTOME SPATIAL COUPLING DISCOVERY", flush=True)
print("=" * 78, flush=True)

print("GSEA files:", len(gsea_files), flush=True)
print("N_PERM:", N_PERM, flush=True)
print("MIN_REGIONS:", MIN_REGIONS, flush=True)
print("BLOCK_TERMS:", BLOCK_TERMS, flush=True)
print("DK regions:", len(regions), flush=True)


# =====================================================================
# DISCOVER FILE FORMAT
# =====================================================================

first = pd.read_csv(
    gsea_files[0],
    sep="\t"
)

print("\nExample GSEA columns:", list(first.columns), flush=True)

TERM_CANDIDATES = [
    "term",
    "pathway",
    "name",
    "gene_set",
    "geneset",
    "Term",
]

NES_CANDIDATES = [
    "NES",
    "nes",
    "normalized_enrichment_score",
    "normalized enrichment score",
]

LIB_CANDIDATES = [
    "library",
    "source",
    "database",
    "collection",
    "gene_set_library",
]

REGION_CANDIDATES = [
    "desikan_label",
    "region",
    "region_label",
    "parcel",
    "parcel_label",
]

term_col = detect_column(
    first,
    TERM_CANDIDATES
)

nes_col = detect_column(
    first,
    NES_CANDIDATES
)

library_col = detect_column(
    first,
    LIB_CANDIDATES
)

region_col = detect_column(
    first,
    REGION_CANDIDATES
)

if term_col is None or nes_col is None:
    raise RuntimeError(
        "Could not identify term/NES columns.\n"
        f"Columns present: {list(first.columns)}"
    )

print("Detected term column:", term_col, flush=True)
print("Detected NES column:", nes_col, flush=True)
print("Detected library column:", library_col, flush=True)
print("Detected region column:", region_col, flush=True)


# =====================================================================
# MAP EACH FILE TO A REGION
# =====================================================================

records = []
unmapped_files = []

for f in gsea_files:

    df = pd.read_csv(
        f,
        sep="\t"
    )

    # Re-detect in case columns differ slightly.
    tc = detect_column(df, TERM_CANDIDATES)
    nc = detect_column(df, NES_CANDIDATES)
    lc = detect_column(df, LIB_CANDIDATES)
    rc = detect_column(df, REGION_CANDIDATES)

    if tc is None or nc is None:
        continue

    region = None

    # ---------------------------------------------------------
    # Preferred: region label stored directly in the GSEA file.
    # ---------------------------------------------------------

    if rc is not None:

        vals = (
            df[rc]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        exact = [
            x for x in vals
            if x in known_labels
        ]

        if len(exact) == 1:
            region = exact[0]

    # ---------------------------------------------------------
    # Fallback:
    # search every constant string column for a known DK label.
    # ---------------------------------------------------------

    if region is None:

        for c in df.columns:

            vals = (
                df[c]
                .dropna()
                .astype(str)
                .unique()
            )

            if len(vals) == 1 and vals[0] in known_labels:
                region = vals[0]
                break

    if region is None:
        unmapped_files.append(str(f))
        continue

    # We only need exact Framingham DK56 cortex.
    if region not in region_set:
        continue

    for _, row in df.iterrows():

        term = row.get(tc)
        nes = row.get(nc)

        if pd.isna(term) or pd.isna(nes):
            continue

        try:
            nes = float(nes)
        except Exception:
            continue

        if lc is not None:
            library = canonical_library(
                row.get(lc)
            )
        else:
            t = str(term).lower()

            if "reactome" in t:
                library = "Reactome"
            else:
                library = "GO_BP"

        records.append({
            "region": region,
            "library": library,
            "term": str(term),
            "NES": nes,
        })


if len(records) == 0:
    print("\nCould not map any GSEA files to DK labels.", flush=True)
    print(
        "First 10 unmapped files:",
        unmapped_files[:10],
        flush=True
    )

    raise RuntimeError(
        "Regional GSEA files do not contain recognizable "
        "DK region labels. Inspect one file and the original "
        "script 90 mapping logic."
    )


long = pd.DataFrame(records)

print("\nMapped GSEA rows:", len(long), flush=True)
print(
    "Mapped DK56 regions:",
    long["region"].nunique(),
    flush=True
)

print(
    "Libraries:",
    long["library"]
    .value_counts()
    .to_dict(),
    flush=True
)


# =====================================================================
# BUILD REGION × TERM NES MATRICES
# =====================================================================

nes_matrices = {}

for library, d in long.groupby("library"):

    mat = d.pivot_table(
        index="region",
        columns="term",
        values="NES",
        aggfunc="mean"
    )

    mat = mat.reindex(
        regions
    )

    n_available = mat.notna().sum(axis=0)

    keep = n_available[
        n_available >= MIN_REGIONS
    ].index

    mat = mat.loc[:, keep]

    if mat.shape[1] == 0:
        continue

    nes_matrices[library] = mat

    print(
        f"{library}: "
        f"{mat.shape[1]} terms retained "
        f"(>= {MIN_REGIONS} regions)",
        flush=True
    )


if not nes_matrices:
    raise RuntimeError(
        "No GO/Reactome terms passed MIN_REGIONS."
    )


# =====================================================================
# COUPLING MAPS
# =====================================================================

maps = pd.read_csv(
    MAPS,
    sep="\t"
)


# =====================================================================
# DISCOVERY
# =====================================================================

all_results = []

for map_index, (cardiac, metric) in enumerate(PRIMARY):

    map_start = time.time()

    print("\n" + "=" * 78, flush=True)
    print(
        f"MAP {map_index + 1}/4: "
        f"{cardiac} / {metric}",
        flush=True
    )
    print("=" * 78, flush=True)

    m = maps[
        (maps["cardiac"] == cardiac)
        &
        (maps["metric"] == metric)
    ].copy()

    m = m.set_index(
        "desikan_label"
    )

    beta = (
        m.loc[
            regions,
            "beta_interaction"
        ]
        .astype(float)
        .values
    )

    beta_hash = md5_array(beta)

    seed = SEED_BASE + map_index

    cache_file = (
        CACHE /
        f"{map_index:02d}_{cardiac}_{metric}"
        f"_N{N_PERM}_seed{seed}.npy"
    )

    meta_file = (
        CACHE /
        f"{map_index:02d}_{cardiac}_{metric}"
        f"_N{N_PERM}_seed{seed}.json"
    )

    if not cache_file.exists():
        raise RuntimeError(
            f"Missing Moran cache:\n{cache_file}\n"
            "Run 84b with the same N_PERM first."
        )

    if not meta_file.exists():
        raise RuntimeError(
            f"Missing Moran cache metadata:\n{meta_file}"
        )

    with open(meta_file) as f:
        meta = json.load(f)

    if meta["labels56"] != regions:
        raise RuntimeError(
            f"Region order mismatch for {cardiac}/{metric}"
        )

    if meta["beta_md5"] != beta_hash:
        raise RuntimeError(
            f"Coupling-map hash mismatch for {cardiac}/{metric}"
        )

    null_maps = np.load(
        cache_file,
        mmap_mode="r"
    )

    if null_maps.shape != (56, N_PERM):
        raise RuntimeError(
            f"Unexpected null shape: {null_maps.shape}"
        )

    print(
        "Loaded cached nulls:",
        cache_file.name,
        flush=True
    )

    # ---------------------------------------------------------
    # Each library separately.
    # ---------------------------------------------------------

    for library, mat in nes_matrices.items():

        print(
            f"\n  {library}: {mat.shape[1]} terms",
            flush=True
        )

        values = mat.to_numpy(
            dtype=float
        )

        terms = np.asarray(
            mat.columns.astype(str)
        )

        # -----------------------------------------------------
        # Group terms by identical regional availability mask.
        # This allows exact Spearman ranking on the appropriate
        # regional subset while remaining vectorized.
        # -----------------------------------------------------

        groups = {}

        for j in range(values.shape[1]):

            mask = np.isfinite(
                values[:, j]
            )

            if mask.sum() < MIN_REGIONS:
                continue

            key = mask.tobytes()

            if key not in groups:
                groups[key] = {
                    "mask": mask,
                    "indices": []
                }

            groups[key]["indices"].append(j)

        print(
            "  availability-mask groups:",
            len(groups),
            flush=True
        )

        for g in groups.values():

            mask = g["mask"]
            indices = g["indices"]

            nreg = int(mask.sum())

            beta_sub = beta[mask]

            null_sub = np.asarray(
                null_maps[mask, :],
                dtype=float
            )

            # Rank coupling map on this exact subset.
            beta_rank = (
                rankdata(beta_sub)
                .astype(float)
            )

            beta_rank -= beta_rank.mean()

            # Rank each null map on same regional subset.
            null_rank = rankdata(
                null_sub,
                axis=0
            ).astype(float)

            null_rank -= null_rank.mean(
                axis=0,
                keepdims=True
            )

            beta_norm = np.sqrt(
                np.sum(beta_rank ** 2)
            )

            null_norm = np.sqrt(
                np.sum(
                    null_rank ** 2,
                    axis=0
                )
            )

            # -----------------------------------------------
            # Term blocks to bound RAM.
            # -----------------------------------------------

            for start in range(
                0,
                len(indices),
                BLOCK_TERMS
            ):

                idx = indices[
                    start:start + BLOCK_TERMS
                ]

                X = values[
                    mask,
                    :
                ][:, idx]

                # Since all terms share this exact mask,
                # X has no missing values.
                Xrank = centered_rank_columns(
                    X
                )

                term_norm = np.sqrt(
                    np.sum(
                        Xrank ** 2,
                        axis=0
                    )
                )

                # Observed Spearman rho:
                # terms x 1
                obs = (
                    Xrank.T @ beta_rank
                ) / (
                    term_norm * beta_norm
                )

                # Null Spearman rho:
                # terms x permutations
                null_r = (
                    Xrank.T @ null_rank
                ) / (
                    term_norm[:, None]
                    *
                    null_norm[None, :]
                )

                p_emp = (
                    1
                    +
                    np.sum(
                        np.abs(null_r)
                        >=
                        np.abs(obs)[:, None],
                        axis=1
                    )
                ) / (
                    N_PERM + 1
                )

                null_mean = np.mean(
                    null_r,
                    axis=1
                )

                null_sd = np.std(
                    null_r,
                    axis=1,
                    ddof=1
                )

                for k, col_idx in enumerate(idx):

                    all_results.append({
                        "cardiac":
                            cardiac,
                        "metric":
                            metric,
                        "library":
                            library,
                        "term":
                            terms[col_idx],
                        "N_regions":
                            nreg,
                        "observed_spearman_rho":
                            float(obs[k]),
                        "spatial_empirical_P":
                            float(p_emp[k]),
                        "null_mean_rho":
                            float(null_mean[k]),
                        "null_SD_rho":
                            float(null_sd[k]),
                        "N_permutations":
                            N_PERM,
                    })

    # ---------------------------------------------------------
    # CHECKPOINT AFTER EACH CARDIAC MAP
    # ---------------------------------------------------------

    checkpoint = pd.DataFrame(
        all_results
    )

    checkpoint_file = (
        OUT /
        f"GO_Reactome_spatial_coupling_FAST_"
        f"N{N_PERM}_CHECKPOINT.tsv"
    )

    checkpoint.to_csv(
        checkpoint_file,
        sep="\t",
        index=False
    )

    print(
        f"\nCheckpoint: {len(checkpoint):,} tests",
        flush=True
    )

    print(
        "Map elapsed:",
        f"{(time.time() - map_start) / 60:.2f} min",
        flush=True
    )


# =====================================================================
# MULTIPLE-COMPARISON CORRECTION
# =====================================================================

res = pd.DataFrame(
    all_results
)

if len(res) == 0:
    raise RuntimeError(
        "Discovery returned zero tests."
    )

# FDR within map × library.
res["q_within_map_library"] = np.nan

for _, idx in res.groupby(
    ["cardiac", "metric", "library"]
).groups.items():

    idx = list(idx)

    res.loc[
        idx,
        "q_within_map_library"
    ] = bh(
        res.loc[
            idx,
            "spatial_empirical_P"
        ].values
    )


# Global FDR across all discovery tests.
res["q_global"] = bh(
    res["spatial_empirical_P"].values
)


# =====================================================================
# SAVE FULL RESULTS
# =====================================================================

res = res.sort_values(
    [
        "q_global",
        "spatial_empirical_P",
        "cardiac",
        "metric",
        "library"
    ]
).reset_index(drop=True)


all_file = (
    OUT /
    f"GO_Reactome_spatial_coupling_FAST_"
    f"N{N_PERM}_ALL.tsv"
)

res.to_csv(
    all_file,
    sep="\t",
    index=False
)


# =====================================================================
# FDR < .05
# =====================================================================

sig = res[
    res["q_global"] < 0.05
].copy()

sig_file = (
    OUT /
    f"GO_Reactome_spatial_coupling_FAST_"
    f"N{N_PERM}_GLOBAL_FDR05.tsv"
)

sig.to_csv(
    sig_file,
    sep="\t",
    index=False
)


# =====================================================================
# TOP 25 PER MAP × LIBRARY
# =====================================================================

top25 = (
    res
    .sort_values(
        [
            "cardiac",
            "metric",
            "library",
            "q_within_map_library",
            "spatial_empirical_P",
        ]
    )
    .groupby(
        ["cardiac", "metric", "library"],
        group_keys=False
    )
    .head(25)
    .copy()
)

top_file = (
    OUT /
    f"GO_Reactome_spatial_coupling_FAST_"
    f"N{N_PERM}_TOP25.tsv"
)

top25.to_csv(
    top_file,
    sep="\t",
    index=False
)


# =====================================================================
# SUMMARY
# =====================================================================

print("\n" + "=" * 78, flush=True)
print("DISCOVERY COMPLETE", flush=True)
print("=" * 78, flush=True)

print(
    "Total tests:",
    f"{len(res):,}",
    flush=True
)

print(
    "Nominal spatial P < .05:",
    int(
        (res["spatial_empirical_P"] < 0.05).sum()
    ),
    flush=True
)

print(
    "Within-map/library FDR < .05:",
    int(
        (res["q_within_map_library"] < 0.05).sum()
    ),
    flush=True
)

print(
    "Global FDR < .05:",
    len(sig),
    flush=True
)

print("\nTop 30 discovery results:", flush=True)

print(
    res[
        [
            "cardiac",
            "metric",
            "library",
            "term",
            "N_regions",
            "observed_spearman_rho",
            "spatial_empirical_P",
            "q_within_map_library",
            "q_global",
        ]
    ]
    .head(30)
    .to_string(index=False),
    flush=True
)

print("\nSaved:", flush=True)
print(all_file, flush=True)
print(sig_file, flush=True)
print(top_file, flush=True)

print("\nDONE", flush=True)
