"""Shared plot styling so every figure in the project looks consistent.

The palette is built around a dark slate base with a warm accent. Nothing
here depends on the data, it only configures matplotlib and exposes a few
helpers used across the analysis scripts.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl

# Core palette
INK = "#22313f"        # near-black slate, used for text and primary bars
ACCENT = "#e2725b"     # terracotta, used to highlight the thing that matters
SOFT = "#9db4c0"       # muted steel blue for secondary elements
NEUTRAL = "#cfd8dc"    # light gray for context elements
GOOD = "#3a7d6b"       # green for positive deltas
BAD = "#b85042"        # red for negative deltas

CATEGORY_COLORS = ["#22313f", "#e2725b", "#9db4c0", "#3a7d6b", "#c9a227", "#7d5a7a"]


def apply_style():
    """Set global matplotlib defaults for the whole project."""
    mpl.rcParams.update({
        "figure.facecolor": "white",
        "figure.dpi": 110,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
        "axes.facecolor": "white",
        "axes.edgecolor": "#b0bec5",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": "#eceff1",
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlepad": 12,
        "axes.labelsize": 11,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "font.family": "DejaVu Sans",
        "text.color": INK,
        "legend.frameon": False,
        "legend.fontsize": 10,
    })


def annotate_bars(ax, fmt="{:.1f}", suffix="", fontsize=9, offset=0.5):
    """Write the value of each bar just above it."""
    for patch in ax.patches:
        height = patch.get_height()
        if height == 0:
            continue
        ax.annotate(
            fmt.format(height) + suffix,
            (patch.get_x() + patch.get_width() / 2, height),
            ha="center", va="bottom",
            fontsize=fontsize, color=INK,
            xytext=(0, offset), textcoords="offset points",
        )


def tag_source(fig, text="Source: Google Ads Sales Dataset (Kaggle)"):
    """Small footnote in the bottom-left corner of a figure."""
    fig.text(0.01, 0.005, text, fontsize=7.5, color="#90a4ae", ha="left")
