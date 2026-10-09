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
MAPFILE = TX / "wholebrain_M3_coupling_maps_DesikanKilliany.tsv"

GMT = ROOT / "reference_data/msigdb/c8.all.v2026.1.Hs.symbols.gmt"

SELECTED = TX / "MSigDB_C8_brain_specific_signatures.tsv"

OUT_SIG_ALL = TX / "MSigDB_C8_brain_individual_N5000_ALL.tsv"
OUT_SIG_FDR = TX / "MSigDB_C8_brain_individual_N5000_GLOBAL_FDR05.tsv"

OUT_CLASS_ALL = TX / "MSigDB_C8_cellclass_N5000_ALL.tsv"
OUT_CLASS_FDR = TX / "MSigDB_C8_cellclass_N5000_GLOBAL_FDR05.tsv"

OUT_SCORES = TX / "MSigDB_C8_DK56_signature_scores.tsv"
OUT_CLASS_SCORES = TX / "MSigDB_C8_DK56_cellclass_scores.tsv"

N_PERM = 5000
MIN_GENES = 10


def clean_gene(x):
    return str(x).strip().upper()


def read_gmt(path):
    d = {}
    with open(path) as f:
        for line in f:
            x = line.rstrip("\n").split("\t")
            if len(x) >= 3:
                d[x[0]] = {
                    clean_gene(g)
                    for g in x[2:]
                    if g.strip()
                }
    return d


def rho_obs(x, y):
    rx = rankdata(np.asarray(x, float))
    ry = rankdata(np.asarray(y, float))
    rx -= rx.mean()
    ry -= ry.mean()
    return np.sum(rx * ry) / np.sqrt(
        np.sum(rx**2) * np.sum(ry**2)
    )


def rank_nulls(nulls):
    R = np.empty_like(nulls, dtype=float)
    for j in range(nulls.shape[1]):
        R[:, j] = rankdata(nulls[:, j])
    R -= R.mean(axis=0, keepdims=True)
    denom = np.sqrt(np.sum(R**2, axis=0))
    return R, denom


def rho_null(x, R, denom):
    rx = rankdata(np.asarray(x, float))
    rx -= rx.mean()

    return (
        np.sum(rx[:, None] * R, axis=0)
        /
        (
            np.sqrt(np.sum(rx**2))
            * denom
        )
    )


def classify(name):
    u = name.upper()

    if "ASTROCY" in u:
        return "Astrocyte"

    if "MICROGL" in u:
        return "Microglia"

    if "OLIGODENDRO" in u or "_OPC" in u:
        return "Oligodendrocyte_OPC"

    if "ENDOTHEL" in u or "PERICY" in u:
        return "Brain_vascular"

    if (
        "NEURON" in u
        or "INTERNEURON" in u
        or "PURKINJE" in u
    ):
        return "Neuron"

    if "GLIA" in u or "GLIAL" in u:
        return "Other_glia"

    return "Other"


# ============================================================
# LOAD AHBA AND EXACT DK56 ORDER
# ============================================================

expr = pd.read_csv(EXPR, sep="\t", index_col=0)
expr.index = pd.to_numeric(expr.index)

atlas = pd.read_csv(ATLAS, sep="\t")
order = pd.read_csv(ORDER, sep="\t")

atlas["desikan_label"] = np.where(
    atlas["hemisphere"].eq("L"),
    "lh_" + atlas["label"].astype(str),
    np.where(
        atlas["hemisphere"].eq("R"),
        "rh_" + atlas["label"].astype(str),
        ""
    )
)

lookup = dict(zip(atlas["desikan_label"], atlas["id"]))

dk_labels = order["desikan_label"].astype(str).tolist()
dk_ids = [lookup[x] for x in dk_labels]

X = expr.loc[dk_ids].copy()
X.index = dk_labels

X.columns = [clean_gene(x) for x in X.columns]
X = X.T.groupby(level=0).mean().T

X = X.loc[:, X.notna().sum(axis=0) >= 50]
X = X.apply(lambda s: s.fillna(s.mean()), axis=0)

sd = X.std(axis=0, ddof=1)
X = X.loc[:, sd > 0]

Z = (
    X - X.mean(axis=0)
) / X.std(axis=0, ddof=1)

print("DK56 expression:", Z.shape)


# ============================================================
# SELECTED C8 SIGNATURES
# ============================================================

sel = pd.read_csv(SELECTED, sep="\t")

selected_names = set(sel["signature"].astype(str))
gmt = read_gmt(GMT)

available = set(Z.columns)

score_data = {}
meta = []

for name in sorted(selected_names):

    if name not in gmt:
        print("WARNING missing from GMT:", name)
        continue

    genes = sorted(gmt[name] & available)

    if len(genes) < MIN_GENES:
        print(
            f"EXCLUDE sparse signature: {name} "
            f"({len(genes)} AHBA genes)"
        )
        continue

    score_data[name] = Z[genes].mean(axis=1)

    meta.append({
        "signature": name,
        "cell_class": classify(name),
        "N_genes_AHBA": len(genes),
    })

meta = pd.DataFrame(meta)

scores = pd.DataFrame(
    score_data,
    index=dk_labels
)

scores.index.name = "desikan_label"
scores.to_csv(OUT_SCORES, sep="\t")

print("\nIndividual signatures retained:", scores.shape[1])

print(
    meta.groupby("cell_class")
        .size()
        .sort_values(ascending=False)
)


# ============================================================
# CELL-CLASS CONSENSUS SCORES
#
# First z-standardize each signature across regions,
# then average signatures belonging to the same cell class.
# ============================================================

sig_z = (
    scores - scores.mean(axis=0)
) / scores.std(axis=0, ddof=1)

class_scores = {}

PRIMARY_CLASSES = [
    "Astrocyte",
    "Microglia",
    "Oligodendrocyte_OPC",
    "Neuron",
    "Brain_vascular",
]

for cls in PRIMARY_CLASSES:

    names = meta.loc[
        meta["cell_class"] == cls,
        "signature"
    ].tolist()

    if not names:
        continue

    class_scores[cls] = sig_z[names].mean(axis=1)

    print(
        f"{cls}: {len(names)} signatures"
    )

class_scores = pd.DataFrame(
    class_scores,
    index=dk_labels
)

class_scores.index.name = "desikan_label"
class_scores.to_csv(OUT_CLASS_SCORES, sep="\t")


# ============================================================
# COUPLING MAPS
# ============================================================

maps = pd.read_csv(MAPFILE, sep="\t")
maps = maps.set_index("desikan_label")

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


def test_scores(score_df, label_name):

    rows = []

    for cardiac, metric, cache_name in specs:

        m = maps[
            (maps["cardiac"] == cardiac)
            &
            (maps["metric"] == metric)
        ].copy()

        if len(m) != 56:
            raise RuntimeError(
                f"{cardiac}/{metric}: "
                f"expected 56 rows, got {len(m)}"
            )

        m = m.loc[dk_labels]

        beta = pd.to_numeric(
            m["beta_interaction"],
            errors="raise"
        ).values

        nulls = np.load(
            CACHE / cache_name,
            mmap_mode="r"
        )

        if nulls.shape == (N_PERM, 56):
            nulls = nulls.T

        if nulls.shape != (56, N_PERM):
            raise RuntimeError(
                f"{cache_name}: unexpected {nulls.shape}"
            )

        R, denom = rank_nulls(nulls)

        print(
            f"Testing {cardiac}/{metric}: "
            f"{score_df.shape[1]} scores"
        )

        for name in score_df.columns:

            x = score_df[name].values

            rho = rho_obs(x, beta)
            nr = rho_null(x, R, denom)

            p = (
                1
                + np.sum(np.abs(nr) >= abs(rho))
            ) / (N_PERM + 1)

            row = {
                "cardiac": cardiac,
                "metric": metric,
                label_name: name,
                "N_regions": 56,
                "observed_spearman_rho": rho,
                "spatial_empirical_P": p,
                "null_mean_rho": np.mean(nr),
                "null_SD_rho": np.std(nr, ddof=1),
                "N_permutations": N_PERM,
            }

            rows.append(row)

    d = pd.DataFrame(rows)

    d["q_within_map"] = np.nan

    for _, idx in d.groupby(
        ["cardiac", "metric"]
    ).groups.items():

        d.loc[idx, "q_within_map"] = multipletests(
            d.loc[idx, "spatial_empirical_P"],
            method="fdr_bh"
        )[1]

    d["q_global"] = multipletests(
        d["spatial_empirical_P"],
        method="fdr_bh"
    )[1]

    return d.sort_values(
        [
            "q_global",
            "q_within_map",
            "spatial_empirical_P"
        ]
    ).reset_index(drop=True)


# ============================================================
# PRIMARY: FIVE CELL CLASSES
# ============================================================

class_res = test_scores(
    class_scores,
    "cell_class"
)

class_res.to_csv(
    OUT_CLASS_ALL,
    sep="\t",
    index=False
)

class_res[
    class_res["q_global"] < .05
].to_csv(
    OUT_CLASS_FDR,
    sep="\t",
    index=False
)


# ============================================================
# SECONDARY: INDIVIDUAL C8 SIGNATURES
# ============================================================

sig_res = test_scores(
    scores,
    "signature"
)

sig_res = sig_res.merge(
    meta,
    on="signature",
    how="left"
)

sig_res.to_csv(
    OUT_SIG_ALL,
    sep="\t",
    index=False
)

sig_res[
    sig_res["q_global"] < .05
].to_csv(
    OUT_SIG_FDR,
    sep="\t",
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n============================================================")
print("PRIMARY C8 CELL-CLASS RESULTS")
print("============================================================")

print("Tests:", len(class_res))
print(
    "Spatial P < .05:",
    (class_res["spatial_empirical_P"] < .05).sum()
)
print(
    "Within-map q < .05:",
    (class_res["q_within_map"] < .05).sum()
)
print(
    "Global q < .05:",
    (class_res["q_global"] < .05).sum()
)

print("\nCELL-CLASS RESULTS:")
print(
    class_res[
        [
            "cardiac",
            "metric",
            "cell_class",
            "observed_spearman_rho",
            "spatial_empirical_P",
            "q_within_map",
            "q_global",
        ]
    ].to_string(index=False)
)


print("\n============================================================")
print("SECONDARY INDIVIDUAL C8 RESULTS")
print("============================================================")

print("Tests:", len(sig_res))
print(
    "Spatial P < .05:",
    (sig_res["spatial_empirical_P"] < .05).sum()
)
print(
    "Within-map q < .05:",
    (sig_res["q_within_map"] < .05).sum()
)
print(
    "Global q < .05:",
    (sig_res["q_global"] < .05).sum()
)

print("\nTOP 25:")
print(
    sig_res[
        [
            "cardiac",
            "metric",
            "cell_class",
            "signature",
            "N_genes_AHBA",
            "observed_spearman_rho",
            "spatial_empirical_P",
            "q_within_map",
            "q_global",
        ]
    ].head(25).to_string(index=False)
)

print("\nSaved:")
print(OUT_CLASS_ALL)
print(OUT_CLASS_FDR)
print(OUT_SIG_ALL)
print(OUT_SIG_FDR)
print(OUT_SCORES)
print(OUT_CLASS_SCORES)

print("\nDONE")
