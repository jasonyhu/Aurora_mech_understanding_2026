#!/bin/bash
#SBATCH --job-name=fve_topk
#SBATCH --output=logs/fve_topk_%j.out
#SBATCH --error=logs/fve_topk_%j.err
#SBATCH --partition=serc,gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=4
#SBATCH --time=1:00:00
#SBATCH --mem-per-gpu=128GB
#SBATCH --gpus=1
#SBATCH --constraint="GPU_MEM:40GB|GPU_MEM:48GB|GPU_MEM:80GB"

echo "Starting at $(date)"
START_TIME=$(date +%s)

source spinup.sh
nvidia-smi
python3 mirora/scripts/python/compute_fve_topk.py

ELAPSED=$(($(date +%s) - START_TIME))
echo "Ending at $(date)"
printf "Elapsed time: %s\n\n" "$(date -d@$ELAPSED -u +%H\ h\ %M\ m\ %S\ s)"
