#!/bin/bash
#SBATCH --job-name=unpack_topmed
#SBATCH --account=qiallab
#SBATCH --output=/data/qiallab/Framingham/results/genetics_pathways/unpack_topmed_%j.out
#SBATCH --error=/data/qiallab/Framingham/results/genetics_pathways/unpack_topmed_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2

set -euo pipefail

TAR=/data/qiallab/Framingham/data/genetics/wgs/phg002135.v1.TOPMed_WGS_Framingham_v6_frz10_SNPs.genotype-calls-vcf.c1.HMB-IRB-MDS.tar.gz
DEST=/data/qiallab/Framingham/data/genetics/wgs/c1_unpacked

mkdir -p "$DEST"

echo "Started: $(date)"
echo "Source:"
ls -lh "$TAR"

tar -xzf "$TAR" -C "$DEST"

echo
echo "Finished: $(date)"
echo
echo "Extracted VCFs:"
find "$DEST" -type f -name "*.vcf.gz" | sort

echo
echo "Counts:"
echo -n "VCF = "
find "$DEST" -type f -name "*.vcf.gz" | wc -l

echo -n "CSI = "
find "$DEST" -type f -name "*.vcf.gz.csi" | wc -l

echo
du -sh "$DEST"
