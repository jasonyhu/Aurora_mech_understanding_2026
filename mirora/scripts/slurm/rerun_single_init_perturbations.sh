#!/bin/bash
#SBATCH --job-name=rerun_sept01
#SBATCH --output=logs/rerun_sept01_%j.out
#SBATCH --error=logs/rerun_sept01_%j.err
#SBATCH --partition=gpu,owners,serc
#SBATCH -G 1
#SBATCH -C "GPU_MEM:32GB|GPU_MEM:40GB|GPU_MEM:48GB|GPU_MEM:80GB"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --mem=64GB

echo "Starting at $(date)"
START_TIME=$(date +%s)

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source spinup.sh

# Model checkpoint may have been purged from $SCRATCH (90-day inactivity).
# Re-download from HuggingFace microsoft/aurora (public) if missing before rollout.
CKPT="$SCRATCH/mirora/model/aurora-0.4-air-pollution.ckpt"
if [ ! -f "$CKPT" ]; then
    echo "Checkpoint missing, downloading from HuggingFace..."
    python3 -c "from pathlib import Path; from mirora.setup.download_model import download_model_checkpoint; download_model_checkpoint(Path('$SCRATCH/mirora/model'))"
fi

python3 mirora/scripts/python/rerun_single_init_perturbations.py

ELAPSED=$(($(date +%s) - START_TIME))
echo "Ending at $(date)"
printf "Elapsed time: %s\n\n" "$(date -d@$ELAPSED -u +%H\ h\ %M\ m\ %S\ s)"
