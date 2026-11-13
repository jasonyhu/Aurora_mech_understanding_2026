"""
Helpers for addressing blocks of the Aurora Swin3D backbone.
"""

from contextlib import contextmanager


def get_backbone_block(model, component: str, stage_idx: int, block_idx: int):
    """The backbone block `component` ('encoder' | 'decoder'), stage `stage_idx`, block `block_idx`."""
    layers = model.backbone.decoder_layers if component == "decoder" else model.backbone.encoder_layers
    return layers[stage_idx].blocks[block_idx]


def get_spatial_resolution(model, batch, component: str, stage_idx: int) -> tuple[int, int, int]:
    """(C, H, W) token grid at a backbone stage. Decoder stage i runs at encoder stage (n - 1 - i)'s resolution."""
    patch_res = (model.encoder.latent_levels,
                 batch.metadata.lat.shape[0] // model.patch_size,
                 batch.metadata.lon.shape[0] // model.patch_size)
    all_res, _ = model.backbone.get_encoder_specs(patch_res)
    res_idx = len(all_res) - stage_idx - 1 if component == "decoder" else stage_idx
    return all_res[res_idx]


@contextmanager
def capture_blocks(model, keys):
    """Record the output of each backbone block in `keys` ((component, stage, block) tuples).

    Yields a dict filled with detached outputs on every forward pass inside the block:
        with capture_blocks(model, [("encoder", 0, 0)]) as acts:
            model(batch)
        x = acts[("encoder", 0, 0)]
    """
    captured = {}

    def make_hook(key):
        def hook(_m, _i, output):
            captured[key] = (output[0] if isinstance(output, tuple) else output).detach()
        return hook

    handles = [get_backbone_block(model, *key).register_forward_hook(make_hook(key)) for key in keys]
    try:
        yield captured
    finally:
        for h in handles:
            h.remove()
