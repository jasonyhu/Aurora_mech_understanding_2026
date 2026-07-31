#!/usr/bin/env python3
"""One entry point to regenerate every manuscript figure.

Reviewers should normally only ever need this script:

    python3 mirora/scripts/reproduce_figures.py --list     # what exists
    python3 mirora/scripts/reproduce_figures.py --check    # are inputs present?
    python3 mirora/scripts/reproduce_figures.py --all      # render everything runnable
    python3 mirora/scripts/reproduce_figures.py --figure ozone_perturbation

It does not reimplement any plotting: it is a manifest plus a runner that calls
the existing per-figure scripts in this directory, in dependency order.

See REPRODUCE.md (repo root) for the data tiers.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import consts

HERE = Path(__file__).resolve().parent

# Rollout initialisation used by almost every manuscript figure.
MAIN_INIT = "20220901_0000"

# Wildfire case studies: (event name, rollout init) used by the wildfire figure.
FIRE_EVENTS = [
    ("central_russia_aug2022", "20220819_1200"),
    ("siberia_yakutia_jul2022", "20220710_1200"),
    ("alaska_jul2022", "20220703_0000"),
    ("amazon_sep2022", "20220906_0000"),
    ("south_america_jun2022", "20220615_1200"),
]


def _base(init: str) -> Path:
    return consts.EXPERIMENTS_PATH / "base" / init / "predictions.pickle"


def _exp(name: str, init: str = MAIN_INIT) -> Path:
    return consts.EXPERIMENTS_PATH / name / init / "predictions.pickle"


def _cams(init: str, kind: str) -> Path:
    return consts.CAMS_DATA_PATH / f"cams_{init}" / f"data_{kind}.nc"


# ---------------------------------------------------------------------------
# Figure manifest
#
# tier 1 = renders from the published figure-inputs bundle, CPU only
# tier 2 = needs a GPU (runs the Aurora model live)
# tier 3 = needs the full pipeline output (terabytes of backbone embeddings);
#          not reproducible from any practical download
#
# `paper` is the manuscript figure label. Confirm against the final manuscript
# before release -- numbering may shift during revision.
# ---------------------------------------------------------------------------
FIGURES = [
    {
        "id": "architecture",
        "paper": "Fig 1",
        "title": "Aurora Air Pollution architecture diagram",
        "script": "plot_architecture_diagram.py",
        "args": [],
        "tier": 1,
        "gpu": False,
        "inputs": [],  # fully self-contained
        "output": consts.FIGURES_PATH / "architecture",
    },
    {
        "id": "chemistry_products",
        "paper": "Fig 2",
        "title": "Leighton ratio + NOx ratio, CAMS vs Aurora",
        "script": "plot_chemistry_products.py",
        "args": [],
        "tier": 1,
        "gpu": False,
        "inputs": [_base(MAIN_INIT), _cams(MAIN_INIT, "plev")],
        "output": consts.FIGURES_PATH / "chemistry_products",
    },
    {
        "id": "ozone_perturbation",
        "paper": "Fig 3",
        "title": "NOx perturbation effect on ozone (composite + KDEs)",
        "script": "plot_ozone_perturbation.py",
        "args": [],
        "tier": 1,
        "gpu": False,
        "inputs": [
            _base(MAIN_INIT),
            _exp("NOx_0.5ppb_increase"),
            _exp("NOx_5ppb_increase"),
            _exp("NOx_20ppb_increase"),
        ],
        "output": consts.FIGURES_PATH / "ozone_perturbation",
    },
    {
        "id": "fire_stills",
        "paper": "Fig 4",
        "title": "Wildfire case studies, CAMS vs Aurora (5 events)",
        "script": "plot_fire_stills.py",
        "args": [],
        "tier": 1,
        "gpu": False,
        "inputs": (
            [_base(init) for _, init in FIRE_EVENTS]
            + [_cams(init, "sfc") for _, init in FIRE_EVENTS]
        ),
        "output": consts.FIGURES_PATH / "fire_stills",
    },
    {
        "id": "pca_embeddings",
        "paper": "Fig 5",
        "title": "PCA of backbone embeddings (single day / monthly / yearly)",
        "script": "run_pca_analysis.py",
        "args": [],
        "tier": 3,
        "gpu": False,
        # Reads pre/post backbone embeddings for EVERY rollout in the range:
        # ~12.8 GB per init, 730 inits for the yearly panel. Not shippable.
        "inputs": [consts.EXPERIMENTS_PATH / "base" / MAIN_INIT / "pre_backbone_embeddings.pickle"],
        "output": consts.FIGURES_PATH / "pca_embeddings",
    },
    {
        "id": "sae_feature_grid",
        "paper": "Fig 6",
        "title": "SAE feature gallery (per-feature maps)",
        "script": "plot_sae_feature_grid.py",
        "args": [],
        "tier": 2,
        "gpu": True,
        "inputs": [
            consts.CHECKPOINT_PATH,
            consts.BATCH_INITS_PATH / f"init_{MAIN_INIT}",
            consts.MODEL_PATH / "sae",
        ],
        "output": consts.FIGURES_PATH / "sae_feature_grid",
    },
    {
        "id": "sae_examples",
        "paper": "Fig 7 / SI",
        "title": "SAE activation examples, 6x2 gallery",
        "script": "plot_sae_examples.py",
        "args": [],
        "tier": 2,
        "gpu": True,
        "inputs": [
            consts.CHECKPOINT_PATH,
            consts.BATCH_INITS_PATH / "init_20221101_0000",
            consts.MODEL_PATH / "sae",
        ],
        "output": consts.FIGURES_PATH / "sae_examples",
    },
    # ----- Supporting information -----
    {
        "id": "si_single_cell_nox",
        "paper": "SI",
        "title": "Single-grid-cell NOx spike, spatial reach",
        "script": "plot_single_cell_nox.py",
        "args": [],
        "tier": 1,
        "gpu": False,
        "inputs": [_exp("single_cell_nox_beijing")],
        "output": consts.FIGURES_PATH / "single_cell_nox",
    },
    {
        "id": "si_validate_nox",
        "paper": "SI stats",
        "title": "Validate the reported NOx perturbation statistics",
        "script": "validate_nox_claims.py",
        "args": [],
        "tier": 1,
        "gpu": False,
        "inputs": [
            _base(MAIN_INIT),
            _exp("NOx_5ppb_increase"),
            _exp("NOx_20ppb_increase"),
        ],
        "output": None,  # prints numbers to stdout
    },
]

BY_ID = {f["id"]: f for f in FIGURES}


def missing_inputs(fig: dict) -> list[Path]:
    return [p for p in fig["inputs"] if not Path(p).exists()]


def cmd_list() -> None:
    print(f"{'ID':<22} {'PAPER':<10} {'TIER':<5} {'GPU':<4} TITLE")
    print("-" * 100)
    for f in FIGURES:
        print(f"{f['id']:<22} {f['paper']:<10} {f['tier']:<5} "
              f"{'yes' if f['gpu'] else '-':<4} {f['title']}")
    print("\nTier 1 = CPU, renders from the figure-inputs bundle")
    print("Tier 2 = needs a GPU (runs Aurora live)")
    print("Tier 3 = needs full-pipeline outputs (terabytes); see REPRODUCE.md")


def cmd_check() -> int:
    print(f"Storage roots:\n  SCRATCH_PATH = {consts.SCRATCH_PATH}\n"
          f"  OAK_PATH     = {consts.OAK_PATH}\n")
    n_ready = 0
    for f in FIGURES:
        miss = missing_inputs(f)
        if not miss:
            n_ready += 1
            print(f"[ready]   {f['id']}")
        else:
            print(f"[MISSING] {f['id']}  ({len(miss)} input(s))")
            for p in miss[:4]:
                print(f"             - {p}")
            if len(miss) > 4:
                print(f"             ... and {len(miss) - 4} more")
    print(f"\n{n_ready}/{len(FIGURES)} figures have all inputs present.")
    return 0 if n_ready == len(FIGURES) else 1


def run_figure(fig: dict, dry_run: bool = False) -> bool:
    miss = missing_inputs(fig)
    if miss:
        print(f"!! skipping {fig['id']}: {len(miss)} missing input(s); "
              f"first: {miss[0]}")
        return False
    cmd = [sys.executable, str(HERE / fig["script"]), *fig["args"]]
    print(f"\n=== {fig['id']} ({fig['paper']}) — {fig['title']} ===")
    print("    $ " + " ".join(cmd))
    if dry_run:
        return True
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"!! {fig['id']} FAILED (exit {result.returncode})")
        return False
    if fig["output"]:
        print(f"    -> {fig['output']}")
    return True


def main() -> int:
    p = argparse.ArgumentParser(
        description="Regenerate manuscript figures.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--list", action="store_true", help="list all figures and exit")
    p.add_argument("--check", action="store_true",
                   help="report which figures have their inputs present")
    p.add_argument("--figure", nargs="+", metavar="ID",
                   help="render only these figure IDs")
    p.add_argument("--all", action="store_true", help="render every figure")
    p.add_argument("--tier", type=int, choices=[1, 2, 3],
                   help="only render figures at or below this tier")
    p.add_argument("--dry-run", action="store_true",
                   help="print the commands without running them")
    args = p.parse_args()

    if args.list:
        cmd_list()
        return 0
    if args.check:
        return cmd_check()

    if args.figure:
        unknown = [i for i in args.figure if i not in BY_ID]
        if unknown:
            print(f"Unknown figure id(s): {', '.join(unknown)}\n"
                  f"Valid ids: {', '.join(BY_ID)}")
            return 2
        selected = [BY_ID[i] for i in args.figure]
    elif args.all:
        selected = list(FIGURES)
    else:
        p.print_help()
        return 0

    if args.tier:
        selected = [f for f in selected if f["tier"] <= args.tier]

    ok = sum(run_figure(f, args.dry_run) for f in selected)
    print(f"\n{ok}/{len(selected)} figure job(s) completed.")
    return 0 if ok == len(selected) else 1


if __name__ == "__main__":
    sys.exit(main())
