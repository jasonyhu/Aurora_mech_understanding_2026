#!/bin/bash
#SBATCH --job-name=download_cams
#SBATCH --output=logs/download_cams_%A_%a.out
#SBATCH --error=logs/download_cams_%A_%a.err
#SBATCH --partition=service
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem=8GB
#SBATCH --array=0-1

# Starts two workers: one downloads all data at 00z, other downloads all data at 12z
echo "Starting task at $(date)"
START_TIME=$(date +%s)

source spinup.sh
python3 mirora/scripts/python/download_cams.py $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_COUNT

ELAPSED=$(($(date +%s) - START_TIME))
echo "Ending task at $(date)"
printf "Elapsed time: %s\n\n" "$(date -d@$ELAPSED -u +%H\ h\ %M\ m\ %S\ s)"
