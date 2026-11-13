"""
Cross-init consistency of the AuroraScope feature ranking and ablations
(Tables S5 and S6), from tableS5_S6_rank_and_ablate_top_sae_features.py run once per
monthly init.

Table S5: for every feature that reaches the top 10 in any init, the number of
          inits where it is in the top 10 and its fraction of active tokens
          (mean +/- sd over the inits where it was ablated).
Table S6: global mean |dO3| and |dNO2| (ppb, raw and area-weighted) after
          zero-ablating each case-study feature, mean +/- sd over inits, plus
          the steering-figure fraction of cells beyond the colour-scale saturation.
"""

import logging

import pandas as pd

from mirora.utils import setup_logging

import consts

logger = logging.getLogger(__name__)

ROOT = consts.EXPERIMENTS_PATH / "sae_top_activating_features"
N_TOP = 10
REFERENCE_INIT = consts.MAIN_INIT_STR


def main():
    setup_logging()
    paths = sorted(ROOT.glob("*/top_features_results.csv"))
    df = pd.concat([pd.read_csv(p).assign(init=p.parent.name) for p in paths], ignore_index=True)
    logger.info(f"{len(paths)} inits: {', '.join(p.parent.name for p in paths)}")
    pd.set_option("display.width", 250)

    top = df[df["rank"] <= N_TOP]
    table_sy = (top.groupby("name")
                .agg(component=("component", "first"), stage=("stage", "first"), block=("block", "first"),
                     inits_in_top10=("init", "nunique"), mean_rank=("rank", "mean"),
                     frac_active_mean=("frac_tokens_active", "mean"), frac_active_sd=("frac_tokens_active", "std"))
                .sort_values(["inits_in_top10", "mean_rank"], ascending=[False, True]))
    table_sy.to_csv(ROOT / "table_SY_feature_recurrence.csv")
    per_init = top.groupby("init")["name"].apply(set)
    ref = REFERENCE_INIT if REFERENCE_INIT in per_init.index else per_init.index[0]
    others = per_init.drop(ref)
    recurring = [len(s & per_init[ref]) for s in others]  # overlap with the running example's top 10
    logger.info(f"Table S5 ({len(table_sy)} distinct features reached a top 10):\n{table_sy.round(4).to_string()}")
    logger.info(f"Mean overlap of the other {len(others)} inits' top 10 with the {ref} top 10: "
                f"{sum(recurring) / max(len(recurring), 1):.1f} of {N_TOP}")

    names = [t["name"] for t in consts.CASE_STUDY_FEATURES]
    metrics = ["mean_abs_dO3_ppb", "mean_abs_dO3_ppb_areaw", "mean_abs_dNO2_ppb", "mean_abs_dNO2_ppb_areaw",
               "frac_cells_dO3_beyond_sat", "frac_cells_dNO2_beyond_sat", "rank"]
    table_sz = df[df["name"].isin(names)].groupby("name")[metrics].agg(["mean", "std", "count"])
    table_sz.to_csv(ROOT / "table_SZ_case_study_ablations.csv")
    logger.info(f"Table SZ:\n{table_sz.round(4).to_string()}")


if __name__ == "__main__":
    main()
