"""
Generate the SI table of the number of SAE features (dictionary size =
expansion_factor x d_model) for each AuroraScope SAE stage. Deterministic;
no model or activations required. Writes a LaTeX tabular.
"""

from pathlib import Path
import consts

EXPANSION = 8
# (component, stage, d_model, n_blocks)
STAGES = [
    ("Encoder", 0, 512, 6),
    ("Encoder", 1, 1024, 10),
    ("Encoder", 2, 2048, 8),
    ("Decoder", 0, 2048, 8),
    ("Decoder", 1, 1024, 10),
    ("Decoder", 2, 512, 6),
]

OUT = consts.FIGURES_PATH / "si" / "sae_feature_counts.tex"
OUT.parent.mkdir(parents=True, exist_ok=True)

lines = [
    r"\begin{table}[t]",
    r"\caption{Number of learned features (dictionary size $= 8\times d_{\mathrm{model}}$) "
    r"per \textit{AuroraScope} SAE, by U-Net operator stage.}",
    r"\label{tab:feature_counts}",
    r"\centering",
    r"\begin{tabular}{llrrr}",
    r"\toprule",
    r"Component & Stage & $d_{\mathrm{model}}$ & \# SAEs (blocks) & Features per SAE \\",
    r"\midrule",
]
total = 0
for comp, stage, dim, nblk in STAGES:
    feats = EXPANSION * dim
    total += feats * nblk
    if comp == "Decoder" and stage == 0:
        lines.append(r"\addlinespace")
    lines.append(f"{comp} & {stage} & {dim} & {nblk} & {feats:,} \\\\")
lines += [
    r"\midrule",
    rf"\multicolumn{{4}}{{l}}{{Total across all 48 SAEs}} & {total:,} \\",
    r"\bottomrule",
    r"\end{tabular}",
    r"\end{table}",
]
OUT.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"\nWrote {OUT}")
