#!/bin/bash
#SBATCH --job-name=topmed_path
#SBATCH --account=qiallab
#SBATCH --partition=compute
#SBATCH --array=1-21%6
#SBATCH --time=04:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --output=/data/qiallab/Framingham/results/genetics_pathways/logs/extract_chr%A_%a.out
#SBATCH --error=/data/qiallab/Framingham/results/genetics_pathways/logs/extract_chr%A_%a.err

set -euo pipefail

BCFTOOLS=$HOME/.conda/envs/framingham/bin/bcftools

CHR=${SLURM_ARRAY_TASK_ID}

ROOT=/data/qiallab/Framingham
VCFDIR=$ROOT/data/genetics/wgs/c1_unpacked/phg002135.v1.TOPMed_WGS_Framingham_v6_frz10_SNPs.genotype-calls-vcf.c1.HMB-IRB-MDS

VCF=$VCFDIR/FRAMINGHAM_phs000974_TOPMed_WGS_freeze.10b.chr${CHR}.hg38.c1.vcf.gz

BED=$ROOT/results/genetics_pathways/bed_by_chr/chr${CHR}.bed
SAMPLES=$ROOT/results/genetics_pathways/WGS_541_NWD_samples.txt

OUTDIR=$ROOT/results/genetics_pathways/targeted_autosomes
mkdir -p "$OUTDIR"
mkdir -p $ROOT/results/genetics_pathways/logs

RAW=$OUTDIR/chr${CHR}_541_panel_raw.vcf.gz
TAGGED=$OUTDIR/chr${CHR}_541_panel_tags.vcf.gz
COMMON=$OUTDIR/chr${CHR}_541_panel_common.vcf.gz

echo "===== chr${CHR} ====="
date

if [[ ! -f "$VCF" ]]; then
    echo "ERROR: missing VCF: $VCF"
    exit 1
fi

if [[ ! -f "${VCF}.csi" ]]; then
    echo "ERROR: missing CSI: ${VCF}.csi"
    exit 1
fi

if [[ ! -s "$BED" ]]; then
    echo "No panel BED intervals for chr${CHR}; exiting."
    exit 0
fi

echo "Source:"
ls -lh "$VCF"

echo
echo "Regions:"
cat "$BED"

# ----------------------------------------------------------
# 1. 541 subjects + targeted regions + biallelic SNPs
# ----------------------------------------------------------

"$BCFTOOLS" view \
    --threads 2 \
    -S "$SAMPLES" \
    -R "$BED" \
    -m2 -M2 \
    -v snps \
    -Oz \
    -o "$RAW" \
    "$VCF"

"$BCFTOOLS" index -f "$RAW"

# ----------------------------------------------------------
# 2. Recalculate AF/MAF/missingness in OUR 541 subjects
# ----------------------------------------------------------

"$BCFTOOLS" +fill-tags "$RAW" \
    -Oz \
    -o "$TAGGED" \
    -- -t AC,AN,AF,MAF,F_MISSING,HWE

"$BCFTOOLS" index -f "$TAGGED"

# ----------------------------------------------------------
# 3. Common-variant branch
# MAF >=1%, genotype missingness <=2%
# ----------------------------------------------------------

"$BCFTOOLS" view \
    --threads 2 \
    -i 'INFO/MAF>=0.01 && INFO/F_MISSING<=0.02' \
    -Oz \
    -o "$COMMON" \
    "$TAGGED"

"$BCFTOOLS" index -f "$COMMON"

# ----------------------------------------------------------
# Summary
# ----------------------------------------------------------

NRAW=$("$BCFTOOLS" index -n "$RAW")
NCOMMON=$("$BCFTOOLS" index -n "$COMMON")
NSAMP=$("$BCFTOOLS" query -l "$COMMON" | wc -l)

echo
echo "SUMMARY"
echo "chr=${CHR}"
echo "raw_targeted_SNPs=${NRAW}"
echo "common_QC_SNPs=${NCOMMON}"
echo "samples=${NSAMP}"

# RAW is redundant after TAGGED has been verified
rm -f "$RAW" "$RAW.csi"

echo
echo "Finished:"
date
