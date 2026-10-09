#!/usr/bin/env python3

from pathlib import Path
import pandas as pd

ROOT = Path("/data/qiallab/Framingham")
INFILE = ROOT / "results/transcriptomic_enrichment/wholebrain_M3_coupling_maps_FULL.tsv"
OUTDIR = ROOT / "results/transcriptomic_enrichment"
OUTDIR.mkdir(parents=True, exist_ok=True)

d = pd.read_csv(INFILE, sep="\t")

def classify(parcel):
    p = str(parcel)

    # Destrieux/aparc.a2009s naming convention
    if p.startswith(("G_", "S_", "Pole_")):
        return "Destrieux"

    # Remaining cortical labels are Desikan-Killiany/aparc
    return "DesikanKilliany"

d["atlas"] = d["parcel"].apply(classify)

print("\nAtlas counts by unique parcel:")
print(
    d[["hemi","parcel","atlas"]]
    .drop_duplicates()
    .groupby(["atlas","hemi"])
    .size()
)

print("\nAtlas counts by cardiac × metric:")
print(
    d.groupby(["atlas","cardiac","metric"])
    ["desikan_label"]
    .nunique()
)

# Master separated tables
for atlas, g in d.groupby("atlas"):

    out = OUTDIR / f"wholebrain_M3_coupling_maps_{atlas}.tsv"
    g.to_csv(out, sep="\t", index=False)

    print("\nWrote:", out)
    print("Unique parcels:", g["desikan_label"].nunique())

# Individual maps for transcriptomic work
for (atlas, cardiac, metric), g in d.groupby(
    ["atlas","cardiac","metric"]
):
    safe = f"{atlas}_{cardiac}_{metric}"
    out = OUTDIR / f"{safe}_interaction_map.tsv"

    g = g.sort_values(["hemi","parcel"])

    g[
        [
            "hemi",
            "parcel",
            "desikan_label",
            "N",
            "beta_interaction",
            "SE_interaction",
            "p_interaction",
            "q_interaction",
            "beta_cardiac",
        ]
    ].to_csv(out, sep="\t", index=False)

print("\nPRIMARY DK MAP CHECK")
dk = d[d["atlas"] == "DesikanKilliany"]

print(
    dk.groupby(["cardiac","metric"])
    ["desikan_label"]
    .nunique()
)

print("\nDK parcels:")
print(
    "\n".join(
        sorted(
            dk["desikan_label"].unique()
        )
    )
)
