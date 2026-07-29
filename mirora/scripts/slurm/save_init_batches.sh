#!/bin/bash
#SBATCH --job-name=save_init_batches
#SBATCH --output=logs/save_init_batches_%A_%a.out
#SBATCH --error=logs/save_init_batches_%A_%a.err
#SBATCH --partition=normal,serc
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --mem=8GB
#SBATCH --array=0-7

echo "Starting task at $(date)"
START_TIME=$(date +%s)

source spinup.sh
python3 mirora/scripts/python/save_init_batches.py --thread_id $SLURM_ARRAY_TASK_ID --thread_ct $SLURM_ARRAY_TASK_COUNT

ELAPSED=$(($(date +%s) - START_TIME))
echo "Ending task at $(date)"
printf "Elapsed time: %s\n\n" "$(date -d@$ELAPSED -u +%H\ h\ %M\ m\ %S\ s)"
