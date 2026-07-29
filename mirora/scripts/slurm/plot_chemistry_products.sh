#!/bin/bash
#SBATCH --job-name=chem_plots
#SBATCH --output=logs/chem_plots_%j.out
#SBATCH --error=logs/chem_plots_%j.err
#SBATCH --partition=serc,normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --mem=64GB

echo "Starting at $(date)"
START_TIME=$(date +%s)

source spinup.sh
python3 mirora/scripts/python/plot_chemistry_products.py

ELAPSED=$(($(date +%s) - START_TIME))
echo "Ending at $(date)"
printf "Elapsed time: %s\n\n" "$(date -d@$ELAPSED -u +%H\ h\ %M\ m\ %S\ s)"