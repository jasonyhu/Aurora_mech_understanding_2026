#!/bin/bash
#SBATCH --job-name=train_sae
#SBATCH --output=logs/train_sae_%A_%a.out
#SBATCH --error=logs/train_sae_%A_%a.err
#SBATCH --partition=serc,gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=4:00:00
#SBATCH --mem-per-gpu=128GB
#SBATCH --gpus=1
#SBATCH --constraint="GPU_MEM:40GB|GPU_MEM:48GB|GPU_MEM:80GB"
#SBATCH --array=0-47

echo "Starting task $SLURM_ARRAY_TASK_ID / $SLURM_ARRAY_TASK_COUNT at $(date)"
START_TIME=$(date +%s)

source spinup.sh
nvidia-smi
python3 mirora/scripts/python/train_sae.py $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_COUNT ${EXTRA_ARGS}

ELAPSED=$(($(date +%s) - START_TIME))
echo "Ending task at $(date)"
printf "Elapsed time: %s\n\n" "$(date -d@$ELAPSED -u +%H\ h\ %M\ m\ %S\ s)"
