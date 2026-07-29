"""One-time setup helpers: download the Aurora checkpoint and CAMS forcing data.

The submodules pull in optional dependencies (``huggingface_hub``, ``cdsapi``),
so they are intentionally not re-exported here. Import them directly::

    from mirora.setup.download_model import download_model_checkpoint
    from mirora.setup.download_cams import download_cams
"""
