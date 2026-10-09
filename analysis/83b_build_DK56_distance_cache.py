#!/usr/bin/env python3

from pathlib import Path
import time
import numpy as np
import nibabel as nib
from scipy.spatial.distance import cdist
from scipy import ndimage

ROOT = Path("/data/qiallab/Framingham")
OUT = ROOT / "results/transcriptomic_enrichment"
CACHE = OUT / "spatial_null_cache"
CACHE.mkdir(parents=True, exist_ok=True)

PARC = OUT / "Framingham_DK56_partial_parcellation.nii.gz"
OUTFILE = CACHE / "DK56_MNI152_parcel_distance.npy"

print("=" * 72, flush=True)
print("BUILDING DK56 PARCEL DISTANCE MATRIX ONCE", flush=True)
print("=" * 72, flush=True)

if OUTFILE.exists():
    x = np.load(OUTFILE)
    print("Cache already exists:", OUTFILE, flush=True)
    print("Shape:", x.shape, flush=True)
    raise SystemExit(0)

img = nib.load(str(PARC))
darr = np.asanyarray(img.dataobj)

labels = np.trim_zeros(
    np.unique(darr)
).astype(int)

if len(labels) != 56:
    raise RuntimeError(
        f"Expected 56 labels; got {len(labels)}"
    )

mask = np.logical_not(
    np.logical_or(
        np.isclose(darr, 0),
        np.isnan(darr)
    )
)

xyz = nib.affines.apply_affine(
    img.affine,
    np.column_stack(
        np.where(mask)
    )
)

parcellation = darr[mask].astype(int)

print("Voxels:", len(xyz), flush=True)
print("Parcels:", len(labels), flush=True)

# ---------------------------------------------------------
# Reproduce neuromaps._vol_surrogates parcel-distance logic
# exactly, but do it ONCE and save the result.
# ---------------------------------------------------------

t0 = time.time()

row_dist = np.zeros(
    (len(xyz), len(labels)),
    dtype=np.float32
)

for n, row in enumerate(xyz):

    xyz_dist = cdist(
        row[None],
        xyz
    ).astype(np.float32)

    row_dist[n] = ndimage.mean(
        xyz_dist,
        parcellation,
        labels
    )

    if (n + 1) % 5000 == 0:
        print(
            f"Voxel {n+1:,}/{len(xyz):,} "
            f"elapsed {(time.time()-t0)/60:.2f} min",
            flush=True
        )

dist = np.zeros(
    (len(labels), len(labels)),
    dtype=np.float32
)

for n in range(len(labels)):
    dist[n] = ndimage.mean(
        row_dist[:, n],
        parcellation,
        labels
    )

np.save(
    OUTFILE,
    dist
)

print("\nSaved:", OUTFILE, flush=True)
print("Shape:", dist.shape, flush=True)
print(
    "Total minutes:",
    round(
        (time.time() - t0) / 60,
        2
    ),
    flush=True
)

print("\nDistance matrix:")
print(dist)
