"""Exploratory analysis. Produces the figures saved under reports/figures.

Each function takes the cleaned frame and writes one figure. run() calls all
of them in order.

Usage:
    python -m src.eda
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .plotting import (ACCENT, CATEGORY_COLORS, INK, NEUTRAL, SOFT,
                       annotate_bars, apply_style, tag_source)

FIG_DIR = Path("reports/figures")
TABLE_DIR = Path("reports/tables")
PROCESSED_PATH = Path("data/processed/cleaned_google_ads.csv")


def load_processed(path: Path = PROCESSED_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["ad_date"])
    return df


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ("impressions", "clicks", "cost", "conversions",
                        "ctr", "cvr", "cpc", "roas") if c in df.columns]
    desc = df[cols].describe().T.round(4)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    desc.to_csv(TABLE_DIR / "descriptive_stats.csv")
    return desc


def fig_rate_distributions(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, col, label in zip(axes, ("ctr", "cvr"),
                              ("Click-through rate", "Conversion rate")):
        data = df[col].dropna() * 100
        ax.hist(data, bins=45, color=SOFT, edgecolor="white", linewidth=0.4)
        med = data.median()
        ax.axvline(med, color=ACCENT, linewidth=1.6)
        ax.annotate(f"median {med:.2f}%", xy=(med, ax.get_ylim()[1] * 0.92),
                    xytext=(6, 0), textcoords="offset points",
                    color=ACCENT, fontsize=9, fontweight="bold")
        ax.set_title(f"{label} across ads")
        ax.set_xlabel(f"{col.upper()} (%)")
        ax.set_ylabel("Number of ads")
    tag_source(fig)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_rate_distributions.png")
    plt.close(fig)


def fig_device_split(df: pd.DataFrame):
    grp = df.groupby("device").agg(
        ctr=("ctr", "mean"), cvr=("cvr", "mean"), ads=("ctr", "size"))
    grp = grp.sort_values("ctr", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, col, label in zip(axes, ("ctr", "cvr"),
                              ("Average CTR by device", "Average CVR by device")):
        vals = grp[col] * 100
        colors = [ACCENT if v == vals.max() else INK for v in vals]
        ax.bar(vals.index, vals.values, color=colors, width=0.55)
        annotate_bars(ax, fmt="{:.2f}", suffix="%")
        ax.set_title(label)
        ax.set_ylabel("Rate (%)")
        ax.set_ylim(0, vals.max() * 1.18)
    tag_source(fig)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_device_split.png")
    plt.close(fig)


def fig_location_split(df: pd.DataFrame):
    grp = (df.groupby("location")[["ctr", "cvr"]].mean() * 100)
    grp = grp.sort_values("ctr", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    y = np.arange(len(grp))
    ax.barh(y - 0.18, grp["ctr"], height=0.36, color=INK, label="CTR")
    ax.barh(y + 0.18, grp["cvr"], height=0.36, color=ACCENT, label="CVR")
    ax.set_yticks(y, grp.index)
    ax.set_xlabel("Rate (%)")
    ax.set_title("CTR and CVR by location")
    ax.legend(loc="lower right")
    for yi, (c, v) in enumerate(zip(grp["ctr"], grp["cvr"])):
        ax.text(c + 0.05, yi - 0.18, f"{c:.2f}", va="center", fontsize=8.5)
        ax.text(v + 0.05, yi + 0.18, f"{v:.2f}", va="center", fontsize=8.5,
                color=ACCENT)
    tag_source(fig)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_location_split.png")
    plt.close(fig)


def fig_branded_split(df: pd.DataFrame):
    if "is_branded" not in df.columns:
        return
    grp = (df.groupby("is_branded")[["ctr", "cvr"]].mean() * 100)
    grp.index = grp.index.map({0: "Non-branded", 1: "Branded"})

    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(len(grp))
    ax.bar(x - 0.18, grp["ctr"], width=0.36, color=INK, label="CTR")
    ax.bar(x + 0.18, grp["cvr"], width=0.36, color=ACCENT, label="CVR")
    ax.set_xticks(x, grp.index)
    annotate_bars(ax, fmt="{:.2f}", suffix="%")
    ax.set_ylabel("Rate (%)")
    ax.set_title("Branded keywords vs everything else")
    ax.legend()
    tag_source(fig)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_branded_split.png")
    plt.close(fig)


def fig_correlation(df: pd.DataFrame):
    cols = [c for c in ("impressions", "clicks", "cost", "conversions",
                        "sale_amount", "ctr", "cvr", "cpc",
                        "keyword_length", "keyword_word_count",
                        "is_branded", "is_weekend") if c in df.columns]
    corr = df[cols].corr(method="spearman")
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, square=True, linewidths=0.6,
                annot_kws={"size": 8}, cbar_kws={"shrink": 0.75}, ax=ax)
    ax.set_title("Spearman correlation between campaign variables")
    tag_source(fig)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_correlation_heatmap.png")
    plt.close(fig)


def fig_cost_vs_rates(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    sample = df.sample(min(len(df), 1500), random_state=0)
    for ax, col, label in zip(axes, ("ctr", "cvr"), ("CTR", "CVR")):
        for i, dev in enumerate(sorted(sample["device"].dropna().unique())):
            sub = sample[sample["device"] == dev]
            ax.scatter(sub["cpc"], sub[col] * 100, s=14, alpha=0.45,
                       color=CATEGORY_COLORS[i], label=dev, edgecolors="none")
        ax.set_xlabel("Cost per click")
        ax.set_ylabel(f"{label} (%)")
        ax.set_title(f"Does paying more per click buy {label}?")
        ax.legend(markerscale=1.6)
    tag_source(fig)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_cpc_vs_rates.png")
    plt.close(fig)


def fig_keyword_quadrants(df: pd.DataFrame):
    """The portfolio view: every keyword placed on a CTR x CVR grid, bubble
    size proportional to spend. The four quadrants tell you where money is
    working and where it is leaking."""
    kw = df.groupby("keyword").agg(
        ctr=("ctr", "mean"), cvr=("cvr", "mean"), cost=("cost", "sum"),
        branded=("is_branded", "max")).dropna()
    kw[["ctr", "cvr"]] *= 100
    x_mid, y_mid = kw["ctr"].median(), kw["cvr"].median()

    fig, ax = plt.subplots(figsize=(10, 7))
    sizes = 40 + 900 * (kw["cost"] / kw["cost"].max())
    colors = np.where(kw["branded"] == 1, ACCENT, SOFT)
    ax.scatter(kw["ctr"], kw["cvr"], s=sizes, c=colors, alpha=0.75,
               edgecolors="white", linewidths=0.8)
    ax.axvline(x_mid, color=NEUTRAL, linewidth=1)
    ax.axhline(y_mid, color=NEUTRAL, linewidth=1)

    pad_x = (kw["ctr"].max() - kw["ctr"].min()) * 0.03
    pad_y = (kw["cvr"].max() - kw["cvr"].min()) * 0.03
    quadrant_labels = [
        (kw["ctr"].max() - pad_x, kw["cvr"].max() - pad_y, "Stars", "right", "top"),
        (kw["ctr"].min() + pad_x, kw["cvr"].max() - pad_y, "Hidden gems\n(low CTR, high CVR)", "left", "top"),
        (kw["ctr"].max() - pad_x, kw["cvr"].min() + pad_y, "Curiosity clicks\n(high CTR, low CVR)", "right", "bottom"),
        (kw["ctr"].min() + pad_x, kw["cvr"].min() + pad_y, "Budget drains", "left", "bottom"),
    ]
    for x, y, text, ha, va in quadrant_labels:
        ax.text(x, y, text, ha=ha, va=va, fontsize=9.5, color="#78909c",
                fontstyle="italic")

    # Name the biggest spenders so the chart is actionable
    for name, row in kw.nlargest(5, "cost").iterrows():
        ax.annotate(name, (row["ctr"], row["cvr"]),
                    xytext=(0, 10), textcoords="offset points",
                    ha="center", fontsize=8, color=INK)

    ax.set_xlabel("Average CTR (%)")
    ax.set_ylabel("Average CVR (%)")
    ax.set_title("Keyword portfolio map (bubble size = total spend, "
                 "orange = branded)")
    tag_source(fig)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "07_keyword_quadrants.png")
    plt.close(fig)


def fig_monthly_trend(df: pd.DataFrame):
    if "ad_date" not in df.columns:
        return
    monthly = (df.set_index("ad_date")
                 .resample("ME")[["ctr", "cvr"]].mean() * 100)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(monthly.index, monthly["ctr"], color=INK, linewidth=2,
            marker="o", markersize=4, label="CTR")
    ax2 = ax.twinx()
    ax2.plot(monthly.index, monthly["cvr"], color=ACCENT, linewidth=2,
             marker="o", markersize=4, label="CVR")
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(ACCENT)
    ax2.tick_params(axis="y", colors=ACCENT)
    ax.set_ylabel("CTR (%)")
    ax2.set_ylabel("CVR (%)", color=ACCENT)
    ax.set_title("Monthly average CTR and CVR")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], loc="upper left")
    tag_source(fig)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "08_monthly_trend.png")
    plt.close(fig)


def run(df: pd.DataFrame | None = None):
    apply_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    if df is None:
        df = load_processed()

    print(summary_table(df))
    fig_rate_distributions(df)
    fig_device_split(df)
    fig_location_split(df)
    fig_branded_split(df)
    fig_correlation(df)
    fig_cost_vs_rates(df)
    fig_keyword_quadrants(df)
    fig_monthly_trend(df)
    print(f"Figures written to {FIG_DIR}/")


if __name__ == "__main__":
    run()
