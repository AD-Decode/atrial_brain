#!/usr/bin/env python3

from pathlib import Path
import pandas as pd

ROOT = Path("/data/qiallab/Framingham")

GMT = (
    ROOT
    / "reference_data"
    / "msigdb"
    / "c8.all.v2026.1.Hs.symbols.gmt"
)

OUT = (
    ROOT
    / "results"
    / "transcriptomic_enrichment"
    / "MSigDB_C8_brain_signature_selection.tsv"
)

if not GMT.exists():
    raise SystemExit(f"Missing C8 GMT:\n{GMT}")

# Broad first-pass neural/cerebrovascular vocabulary.
# We will inspect this list before doing statistics.
keywords = [
    "BRAIN",
    "CORTEX",
    "CORTICAL",
    "HIPPOCAMP",
    "NEURON",
    "NEURONAL",
    "ASTROCY",
    "MICROGL",
    "OLIGODENDRO",
    "OPC",
    "GLIA",
    "GLIAL",
    "PERICY",
    "ENDOTHEL",
    "VASCULAR",
    "NEURO",
]

rows = []

with open(GMT) as f:
    for line in f:
        x = line.rstrip("\n").split("\t")

        if len(x) < 3:
            continue

        name = x[0]
        description = x[1]
        genes = [g for g in x[2:] if g]

        u = name.upper()

        hits = [
            k for k in keywords
            if k in u
        ]

        rows.append({
            "signature": name,
            "description": description,
            "N_genes": len(genes),
            "matched_keywords": ",".join(hits),
            "brain_keyword_selected": len(hits) > 0,
        })

d = pd.DataFrame(rows)

selected = d[d["brain_keyword_selected"]].copy()

selected.to_csv(
    OUT,
    sep="\t",
    index=False
)

print("Total C8 signatures:", len(d))
print("Brain/neural candidate signatures:", len(selected))

print("\nFIRST 100 CANDIDATES:")
print(
    selected[
        ["signature", "N_genes", "matched_keywords"]
    ].head(100).to_string(index=False)
)

print("\nSaved:")
print(OUT)
