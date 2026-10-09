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
    / "MSigDB_C8_brain_specific_signatures.tsv"
)

brain_context = [
    "BRAIN",
    "_CTX_",
    "CORTEX",
    "CORTICAL",
    "CEREBRUM",
    "CEREBELLUM",
    "HIPPOCAMP",
]

cell_terms = [
    "NEURON",
    "ASTROCY",
    "MICROGL",
    "OLIGODENDRO",
    "_OPC",
    "GLIA",
    "GLIAL",
    "ENDOTHEL",
    "PERICY",
]

exclude_terms = [
    "EYE",
    "RETINA",
    "OLFACTORY",
    "NEUROEPITHELIUM",
    "ENS",
    "INTESTINE",
    "STOMACH",
    "PANCREAS",
    "HEART",
    "LIVER",
    "KIDNEY",
    "LUNG",
    "PLACENTA",
    "ADRENAL",
    "MUSCLE",
    "SPLEEN",
    "THYMUS",
]

rows = []

with open(GMT) as f:
    for line in f:
        x = line.rstrip("\n").split("\t")

        if len(x) < 3:
            continue

        name = x[0]
        genes = [g for g in x[2:] if g]

        u = name.upper()

        has_brain_context = any(k in u for k in brain_context)
        has_cell_term = any(k in u for k in cell_terms)
        excluded = any(k in u for k in exclude_terms)

        keep = has_brain_context and has_cell_term and not excluded

        if keep:
            rows.append({
                "signature": name,
                "N_genes": len(genes),
            })

d = pd.DataFrame(rows)

d.to_csv(
    OUT,
    sep="\t",
    index=False
)

print("Final brain-specific C8 signatures:", len(d))
print()
print(d.to_string(index=False))
print()
print("Saved:")
print(OUT)
