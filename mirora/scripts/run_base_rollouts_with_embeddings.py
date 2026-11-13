"""Run the base Aurora Air Pollution rollout for each validation init batch and save
predictions + pre/post-backbone embeddings (sharded over inits by task id/count)."""

import sys

from tqdm import tqdm

from mirora.data.loading import load_batch
from mirora.inference.inference import aap_rollout, load_aurora
import consts


def main():
    assert len(sys.argv) == 3, "INTENDED USAGE: python3 run_base_rollouts_with_embeddings.py <task_id> <task_ct>"
    task_id, task_ct = int(sys.argv[1]), int(sys.argv[2])

    model, _ = load_aurora(consts.CHECKPOINT_PATH)
    for init_time in tqdm(consts.val_init_times()[task_id::task_ct], desc=f"Task {task_id}/{task_ct}"):
        batch = load_batch(consts.BATCH_INITS_PATH, init_time)
        aap_rollout(batch, batch.metadata.time[0], consts.CHECKPOINT_PATH, consts.EXPERIMENTS_PATH / "base",
                    save_backbone_activations=True, embedding_steps=1, model=model)


if __name__ == "__main__":
    main()
