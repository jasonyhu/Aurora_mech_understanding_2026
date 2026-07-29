#!/bin/bash
#SBATCH --job-name=sae_grid
#SBATCH --output=logs/sae_grid_%j.out
#SBATCH --error=logs/sae_grid_%j.err
#SBATCH --partition=serc,gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=1
#SBATCH --time=5:00
#SBATCH --mem-per-gpu=32GB
#SBATCH --gpus=1
#SBATCH --constraint="GPU_MEM:40GB|GPU_MEM:48GB|GPU_MEM:80GB"

echo "Starting at $(date)"
START_TIME=$(date +%s)

source spinup.sh
nvidia-smi
python3 mirora/scripts/python/plot_sae_thesis_grid.py

ELAPSED=$(($(date +%s) - START_TIME))
echo "Ending at $(date)"
printf "Elapsed time: %s\n\n" "$(date -d@$ELAPSED -u +%H\ h\ %M\ m\ %S\ s)"
