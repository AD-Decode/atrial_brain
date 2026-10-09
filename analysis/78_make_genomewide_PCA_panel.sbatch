#!/bin/bash
#SBATCH --job-name=wgs_pca
#SBATCH --account=qiallab
#SBATCH --partition=compute
#SBATCH --array=1-22%6
#SBATCH --time=08:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --output=/data/qiallab/Framingham/results/genetics_pathways/pca_genomewide/logs/pca_chr%A_%a.out
#SBATCH --error=/data/qiallab/Framingham/results/genetics_pathways/pca_genomewide/logs/pca_chr%A_%a.err

set -euo pipefail

BCFTOOLS=$HOME/.conda/envs/framingham/bin/bcftools
PLINK2=$HOME/.conda/envs/framingham/bin/plink2

CHR=${SLURM_ARRAY_TASK_ID}

ROOT=/data/qiallab/Framingham
VCFDIR=$ROOT/data/genetics/wgs/c1_unpacked/phg002135.v1.TOPMed_WGS_Framingham_v6_frz10_SNPs.genotype-calls-vcf.c1.HMB-IRB-MDS
SAMPLES=$ROOT/results/genetics_pathways/WGS_541_NWD_samples.txt

OUTDIR=$ROOT/results/genetics_pathways/pca_genomewide
mkdir -p "$OUTDIR" "$OUTDIR/logs"

VCF=$VCFDIR/FRAMINGHAM_phs000974_TOPMed_WGS_freeze.10b.chr${CHR}.hg38.c1.vcf.gz

TMP=$OUTDIR/chr${CHR}_541_common.vcf.gz
BASE=$OUTDIR/chr${CHR}_common
PCA=$OUTDIR/chr${CHR}_pca

echo "===== chr${CHR} ====="
date
ls -lh "$VCF"

# ---------------------------------------------------------
# 1. Restrict full WGS to our 541 subjects
#    PASS, biallelic SNPs only
# ---------------------------------------------------------

"$BCFTOOLS" view \
    --threads 2 \
    -S "$SAMPLES" \
    -f PASS \
    -m2 -M2 \
    -v snps \
    -Ou \
    "$VCF" \
| "$BCFTOOLS" +fill-tags \
    -Ou \
    -- -t AC,AN,AF,MAF,F_MISSING \
| "$BCFTOOLS" view \
    --threads 2 \
    -i 'INFO/MAF>=0.05 && INFO/F_MISSING<=0.02' \
    -Oz \
    -o "$TMP"

"$BCFTOOLS" index -f "$TMP"

echo "Common variants:"
"$BCFTOOLS" index -n "$TMP"

# ---------------------------------------------------------
# 2. PLINK2
# ---------------------------------------------------------

"$PLINK2" \
    --vcf "$TMP" \
    --set-all-var-ids '@:#:$r:$a' \
    --new-id-max-allele-len 1000 \
    --make-pgen \
    --threads 2 \
    --out "$BASE"

# ---------------------------------------------------------
# 3. LD pruning for ancestry PCA
# ---------------------------------------------------------

"$PLINK2" \
    --pfile "$BASE" \
    --indep-pairwise 1000kb 0.1 \
    --threads 2 \
    --out "${BASE}_prune"

"$PLINK2" \
    --pfile "$BASE" \
    --extract "${BASE}_prune.prune.in" \
    --make-pgen \
    --threads 2 \
    --out "$PCA"

echo
echo "Final chr${CHR} PCA SNP count:"
grep -vc '^#' "${PCA}.pvar"

# temporary large files no longer needed
rm -f "$TMP" "$TMP.csi"
rm -f "${BASE}.pgen" "${BASE}.pvar" "${BASE}.psam" "${BASE}.log"

date
