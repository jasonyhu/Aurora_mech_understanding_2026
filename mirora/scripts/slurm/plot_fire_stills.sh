#!/bin/bash
#SBATCH --job-name=fire_stills
#SBATCH --output=logs/fire_stills_%j.out
#SBATCH --error=logs/fire_stills_%j.err
#SBATCH --partition=serc,normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=20:00
#SBATCH --mem=64GB

echo "Starting at $(date)"
START_TIME=$(date +%s)

source spinup.sh
python3 mirora/scripts/python/plot_fire_stills.py "$@"

ELAPSED=$(($(date +%s) - START_TIME))
echo "Ending at $(date)"
printf "Elapsed time: %s\n\n" "$(date -d@$ELAPSED -u +%H\ h\ %M\ m\ %S\ s)"
