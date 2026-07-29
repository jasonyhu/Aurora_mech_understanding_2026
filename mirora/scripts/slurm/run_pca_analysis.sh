#!/bin/bash
#SBATCH --job-name=pca_emb
#SBATCH --output=logs/pca_emb_%j.out
#SBATCH --error=logs/pca_emb_%j.err
#SBATCH --partition=serc,normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mem=256GB

echo "Starting at $(date)"
START_TIME=$(date +%s)

source spinup.sh
python3 mirora/scripts/python/run_pca_analysis.py

ELAPSED=$(($(date +%s) - START_TIME))
echo "Ending at $(date)"
printf "Elapsed time: %s\n\n" "$(date -d@$ELAPSED -u +%H\ h\ %M\ m\ %S\ s)"
