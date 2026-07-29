"""
Generate a publication-ready architecture diagram of the Aurora Air Pollution
model's Swin3D backbone, showing encoder/decoder stages, skip connections,
patch merging/splitting, and embedding dimensions.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np

plt.rcParams.update({
    "mathtext.fontset": "stix",
    "font.family": "STIXGeneral",
    "font.size": 10,
    "figure.facecolor": "white",
})

# Colors
C_ENC = "#4878CF"      # encoder blue
C_DEC = "#D65F5F"      # decoder red
C_SKIP = "#6AB572"     # skip green
C_MERGE = "#F5A623"    # merge/split orange
C_IO = "#9B59B6"       # input/output purple
C_BG = "#F8F9FA"       # light background


def draw_block(ax, x, y, w, h, label, sublabel, color, alpha=0.85):
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02",
        facecolor=color, edgecolor="black", linewidth=0.8, alpha=alpha,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h * 0.62, label, ha="center", va="center",
            fontsize=9, fontweight="bold", color="white")
    ax.text(x + w / 2, y + h * 0.3, sublabel, ha="center", va="center",
            fontsize=7, color="white", alpha=0.9)


def draw_arrow(ax, x1, y1, x2, y2, color="black", style="-|>", lw=1.2, ls="-"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw, ls=ls))


def draw_skip(ax, x1, y1, x2, y2, label, color=C_SKIP):
    mid_y = (y1 + y2) / 2
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5,
                                connectionstyle="arc3,rad=0.3", ls="--"))
    ax.text((x1 + x2) / 2 + 0.15, mid_y, label, fontsize=7, color=color,
            ha="center", va="center", fontstyle="italic")


def main():
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 6.5)
    ax.axis("off")

    bw, bh = 1.3, 0.7  # block width, height
    mw, mh = 0.8, 0.4  # merge/split width, height

    # ===== INPUT =====
    draw_block(ax, 0, 2.5, bw, bh, "Input", "451×900\n13 levels", C_IO)
    draw_arrow(ax, bw + 0.05, 2.85, 1.6, 2.85)

    # ===== PERCEIVER ENCODER =====
    draw_block(ax, 1.6, 2.5, bw, bh, "Perceiver\nEncoder", "→ 4 latent levels", C_IO, alpha=0.7)
    draw_arrow(ax, 1.6 + bw + 0.05, 2.85, 3.2, 2.85)

    # ===== PATCH EMBED =====
    ax.text(3.35, 3.35, "Patch embed (3×3)", fontsize=7, ha="center", color="#666", fontstyle="italic")
    draw_arrow(ax, 3.5, 2.85, 3.5, 4.0, color="#888", lw=0.8)

    # ===== ENCODER STAGES =====
    enc_x = [3.0, 4.8, 6.6]
    enc_y = [4.2, 4.2, 4.2]
    enc_labels = [
        ("Enc Stage 0", "6 blocks\n512 dim"),
        ("Enc Stage 1", "10 blocks\n1024 dim"),
        ("Enc Stage 2", "8 blocks\n2048 dim"),
    ]
    enc_res = ["4×150×300", "4×75×150", "4×38×75"]

    for i, (x, y) in enumerate(zip(enc_x, enc_y)):
        draw_block(ax, x, y, bw, bh, enc_labels[i][0], enc_labels[i][1], C_ENC)
        ax.text(x + bw / 2, y + bh + 0.12, enc_res[i], fontsize=7, ha="center", color="#555")
        if i < 2:
            # Patch merging arrow
            mx = x + bw + 0.1
            draw_block(ax, mx, y + 0.15, mw, mh, "Merge", "÷4 tokens", C_MERGE, alpha=0.7)
            draw_arrow(ax, mx + mw + 0.05, y + 0.35, enc_x[i + 1] - 0.05, y + 0.35)

    # ===== BOTTLENECK ARROW (encoder → decoder) =====
    draw_arrow(ax, enc_x[2] + bw / 2, enc_y[2], enc_x[2] + bw / 2, 2.0, color=C_ENC, lw=1.5)

    # ===== DECODER STAGES =====
    dec_x = [6.6, 4.8, 3.0]
    dec_y = [1.0, 1.0, 1.0]
    dec_labels = [
        ("Dec Stage 0", "8 blocks\n2048 dim"),
        ("Dec Stage 1", "10 blocks\n1024 dim"),
        ("Dec Stage 2", "6 blocks\n512→1024 dim"),
    ]
    dec_res = ["4×38×75", "4×75×150", "4×150×300"]

    for i, (x, y) in enumerate(zip(dec_x, dec_y)):
        draw_block(ax, x, y, bw, bh, dec_labels[i][0], dec_labels[i][1], C_DEC)
        ax.text(x + bw / 2, y - 0.15, dec_res[i], fontsize=7, ha="center", color="#555")
        if i < 2:
            # Patch splitting arrow
            sx = dec_x[i + 1] + bw + 0.1
            draw_block(ax, sx, y + 0.15, mw, mh, "Split", "×4 tokens", C_MERGE, alpha=0.7)
            draw_arrow(ax, x - 0.05, y + 0.35, sx + mw + 0.05, y + 0.35, color=C_DEC)

    # ===== SKIP CONNECTIONS =====
    # Enc 1 → Dec 1 (additive)
    draw_skip(ax, enc_x[1] + bw / 2, enc_y[1], dec_x[1] + bw / 2, dec_y[1] + bh,
              "additive\nskip", C_SKIP)

    # Enc 0 → Dec 2 (concatenative)
    draw_skip(ax, enc_x[0] + bw * 0.3, enc_y[0], dec_x[2] + bw * 0.3, dec_y[2] + bh,
              "concat\nskip", "#E67E22")

    # ===== UNPATCH + PERCEIVER DECODER =====
    draw_arrow(ax, dec_x[2], 1.35, 1.6 + bw + 0.05, 1.35, color="#888", lw=0.8)
    ax.text(2.4, 1.65, "Unpatchify", fontsize=7, ha="center", color="#666", fontstyle="italic")

    draw_block(ax, 0.2, 1.0, bw, bh, "Perceiver\nDecoder", "4 → 13 levels", C_IO, alpha=0.7)
    draw_arrow(ax, 1.6, 1.35, 0.2 + bw + 0.05, 1.35)

    # ===== OUTPUT =====
    draw_block(ax, -1.2, 1.0, bw - 0.2, bh, "Output", "451×900\n13 levels", C_IO)
    draw_arrow(ax, 0.15, 1.35, -1.2 + bw - 0.2 + 0.05, 1.35)

    # ===== TITLE =====
    ax.text(5.0, 6.1, "Aurora Air Pollution — Swin3D Backbone Architecture",
            fontsize=14, fontweight="bold", ha="center")
    ax.text(5.0, 5.7,
            "Encoder: 3 stages (6,10,8 blocks) at dim (512,1024,2048)  |  "
            "Decoder: 3 stages (8,10,6 blocks) at dim (2048,1024,512)",
            fontsize=8, ha="center", color="#666")

    # ===== LEGEND =====
    legend_items = [
        mpatches.Patch(color=C_ENC, label="Encoder stages"),
        mpatches.Patch(color=C_DEC, label="Decoder stages"),
        mpatches.Patch(color=C_MERGE, label="Patch merge/split"),
        mpatches.Patch(color=C_SKIP, label="Additive skip"),
        mpatches.Patch(color="#E67E22", label="Concatenative skip"),
        mpatches.Patch(color=C_IO, label="Input/Output"),
    ]
    ax.legend(handles=legend_items, loc="upper right", fontsize=8, framealpha=0.9,
              edgecolor="#ccc", ncol=2)

    # ===== SAE ANNOTATION =====
    ax.annotate("SAE hooks at\neach block", xy=(enc_x[1] + bw / 2, enc_y[1] + bh),
                xytext=(enc_x[1] + bw / 2 + 0.3, 5.5),
                fontsize=8, color="#888", fontstyle="italic", ha="center",
                arrowprops=dict(arrowstyle="->", color="#aaa", lw=0.8))

    save_dir = consts.FIGURES_PATH
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / "aurora_architecture.png", dpi=300, bbox_inches="tight")
    plt.savefig(save_dir / "aurora_architecture.pdf", bbox_inches="tight")
    plt.close()
    logger.info(f"Architecture diagram saved to {save_dir}")


if __name__ == "__main__":
    import logging
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
    import consts
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    main()
