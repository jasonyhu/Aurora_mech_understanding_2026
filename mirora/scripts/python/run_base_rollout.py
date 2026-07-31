"""Run the base Aurora Air Pollution rollout for each init batch and save
predictions + pre/post-backbone embeddings.

Shards the inits across `task_ct` parallel tasks; run task `task_id` with:

    python3 run_base_rollout.py <task_id> <task_ct>
"""

import sys
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from mirora.data.aurora_batches import AuroraInitBatchesDataset
from mirora.inference.inference import aap_rollout
import consts

assert len(sys.argv) == 3, "INTENDED USAGE: python3 run_base_rollout.py <task_id> <task_ct>"
task_id, task_ct = int(sys.argv[1]), int(sys.argv[2])

def aurora_collate_fn(batch):
    return batch[0]

EXPERIMENT_DIR = consts.EXPERIMENTS_PATH / "base"

dataset = AuroraInitBatchesDataset(consts.BATCH_INITS_PATH, consts.val_start_time, consts.val_end_time, skip_batch_check=True)
indices = list(range(task_id, len(dataset), task_ct))
batch_data = DataLoader(Subset(dataset, indices), collate_fn=aurora_collate_fn, num_workers=4)

for batch in tqdm(batch_data, desc=f"Task {task_id}/{task_ct}"):
    batch_time = batch.metadata.time[0]
    result_path = aap_rollout(batch, batch_time, consts.CHECKPOINT_PATH, EXPERIMENT_DIR, save_backbone_activations=True)
