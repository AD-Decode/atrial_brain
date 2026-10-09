#!/usr/bin/env python3

import sys
import gzip
import re
import pandas as pd
from pathlib import Path

ROOT = Path("/data/qiallab/Framingham")
PANEL = ROOT / "results/genetics_pathways/gene_panel_v1.tsv"
OUT = ROOT / "results/genetics_pathways"

if len(sys.argv) != 2:
    raise SystemExit(
        "Usage: python 72_make_pathway_bed.py /path/to/GRCh38.annotation.gtf[.gz]"
    )

gtf = Path(sys.argv[1])

panel = pd.read_csv(PANEL, sep="\t")
wanted = set(panel["gene"])

opener = gzip.open if str(gtf).endswith(".gz") else open

records = []

with opener(gtf, "rt", errors="ignore") as f:
    for line in f:
        if line.startswith("#"):
            continue

        x = line.rstrip("\n").split("\t")

        if len(x) < 9:
            continue

        chrom, source, feature, start, end, score, strand, frame, attrs = x

        if feature != "gene":
            continue

        m = re.search(r'gene_name "([^"]+)"', attrs)

        if not m:
            m = re.search(r'gene=([^;]+)', attrs)

        if not m:
            continue

        gene = m.group(1)

        if gene not in wanted:
            continue

        # gene body +/- 50 kb
        start0 = max(0, int(start) - 1 - 50000)
        end0 = int(end) + 50000

        records.append({
            "chrom": chrom,
            "start": start0,
            "end": end0,
            "gene": gene
        })

genes = pd.DataFrame(records)

if genes.empty:
    raise RuntimeError(
        "None of the panel genes were found. Check that the annotation is GRCh38/hg38."
    )

genes = (
    genes.sort_values(["chrom", "start", "end"])
    .drop_duplicates("gene")
)

missing = sorted(wanted - set(genes["gene"]))

print("Panel genes:", len(wanted))
print("Genes located:", len(genes))
print("Missing:", len(missing))

if missing:
    print("\nMissing genes:")
    print("\n".join(missing))

genes.to_csv(
    OUT / "gene_panel_v1_hg38_50kb.bed",
    sep="\t",
    header=False,
    index=False
)

annot = panel.merge(
    genes,
    on="gene",
    how="left"
)

annot.to_csv(
    OUT / "gene_panel_v1_hg38_coordinates.tsv",
    sep="\t",
    index=False
)

print("\nChromosomes needed:")
print(
    genes["chrom"]
    .drop_duplicates()
    .sort_values()
    .to_string(index=False)
)

print("\nSaved:")
print(OUT / "gene_panel_v1_hg38_50kb.bed")
print(OUT / "gene_panel_v1_hg38_coordinates.tsv")
