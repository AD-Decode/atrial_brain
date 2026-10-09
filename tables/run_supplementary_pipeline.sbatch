#!/bin/bash
#SBATCH --job-name=supp_tables
#SBATCH --account=qiallab
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=/data/qiallab/Framingham/rebuild_workspace/pipeline_%j.log
# add  #SBATCH --partition=<name>  if your cluster requires one
source ~/.bashrc
conda activate framingham
mkdir -p /data/qiallab/Framingham/rebuild_workspace
cd /data/qiallab/Framingham/CODE_REPOSITORY_20261007/tables
python -u run_supplementary_pipeline.py --supplement Framingham_Supplementary_Materials_100826_clean.docx --keep-going
# for S19-S25 as well (hours):  add --molecular
