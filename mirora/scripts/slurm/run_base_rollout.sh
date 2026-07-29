#!/bin/bash
#SBATCH --job-name=aap
#SBATCH --output=logs/aap_%A_%a.out
#SBATCH --error=logs/aap_%A_%a.err
#SBATCH --partition=serc,gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=4
#SBATCH --time=2:30:00
#SBATCH --mem-per-gpu=64GB
#SBATCH --gpus=1
#SBATCH --constraint="GPU_MEM:40GB|GPU_MEM:48GB|GPU_MEM:80GB"
#SBATCH --array=0-8

NUM_TASKS=$SLURM_ARRAY_TASK_COUNT

echo "Starting task $SLURM_ARRAY_TASK_ID / $NUM_TASKS at $(date)"
START_TIME=$(date +%s)

source spinup.sh
nvidia-smi
python3 mirora/scripts/python/run_base_rollout.py $SLURM_ARRAY_TASK_ID $NUM_TASKS

ELAPSED=$(($(date +%s) - START_TIME))
echo "Ending task at $(date)"
printf "Elapsed time: %s\n\n" "$(date -d@$ELAPSED -u +%H\ h\ %M\ m\ %S\ s)"