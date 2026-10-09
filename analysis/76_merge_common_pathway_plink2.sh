#!/bin/bash
set -euo pipefail

ROOT=/data/qiallab/Framingham
VCFDIR=$ROOT/results/genetics_pathways/targeted_autosomes
OUTDIR=$ROOT/results/genetics_pathways/plink_common

mkdir -p "$OUTDIR"

# ------------------------------------------------------------
# 1. Make ordered VCF list
# ------------------------------------------------------------

LIST=$OUTDIR/common_vcfs.list
: > "$LIST"

for chr in $(seq 1 21); do
    VCF=$VCFDIR/chr${chr}_541_panel_common.vcf.gz

    if [[ ! -f "$VCF" ]]; then
        echo "ERROR: missing $VCF"
        exit 1
    fi

    echo "$VCF" >> "$LIST"
done

echo "VCFs:"
cat "$LIST"

# ------------------------------------------------------------
# 2. Concatenate chromosomes
# ------------------------------------------------------------

MERGED=$OUTDIR/pathway_common_541_autosomes.vcf.gz

bcftools concat \
    -f "$LIST" \
    -Oz \
    -o "$MERGED"

bcftools index -f "$MERGED"

echo
echo "Merged variant count:"
bcftools index -n "$MERGED"

echo "Merged sample count:"
bcftools query -l "$MERGED" | wc -l

# ------------------------------------------------------------
# 3. Convert to PLINK2
# ------------------------------------------------------------

plink2 \
    --vcf "$MERGED" \
    --set-all-var-ids '@:#:$r:$a' \
    --new-id-max-allele-len 1000 \
    --make-pgen \
    --out "$OUTDIR/pathway_common_541_autosomes"

echo
echo "PLINK2 dataset:"
ls -lh "$OUTDIR"/pathway_common_541_autosomes.*
