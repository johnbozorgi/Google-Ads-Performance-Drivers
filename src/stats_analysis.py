"""Statistical driver analysis.

Two weighted least squares models, one for CTR and one for CVR, plus the
hypothesis tests that back up the headline claims. Rates estimated from few
impressions or clicks are noisy, so each observation is weighted by its
denominator (impressions for CTR, clicks for CVR). Standard errors are
heteroskedasticity robust (HC3).

Usage:
    python -m src.stats_analysis
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from .eda import load_processed
from .plotting import ACCENT, INK, NEUTRAL, apply_style, tag_source

FIG_DIR = Path("reports/figures")
TABLE_DIR = Path("reports/tables")

CTR_FORMULA = ("ctr_pct ~ C(device, Treatment('Desktop')) + C(location) "
               "+ is_branded + keyword_word_count + is_weekend + cpc")
CVR_FORMULA = ("cvr_pct ~ C(device, Treatment('Desktop')) + C(location) "
               "+ is_branded + keyword_word_count + is_weekend + cpc")


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ctr_pct"] = out["ctr"] * 100
    out["cvr_pct"] = out["cvr"] * 100
    return out.dropna(subset=["ctr_pct", "device", "location", "cpc"])


def fit_models(df: pd.DataFrame):
    data = _prepare(df)

    ctr_model = smf.wls(CTR_FORMULA, data=data,
                        weights=data["impressions"]).fit(cov_type="HC3")

    cvr_data = data.dropna(subset=["cvr_pct"])
    cvr_data = cvr_data[cvr_data["clicks"] > 0]
    cvr_model = smf.wls(CVR_FORMULA, data=cvr_data,
                        weights=cvr_data["clicks"]).fit(cov_type="HC3")

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    for name, model in (("ctr", ctr_model), ("cvr", cvr_model)):
        with open(TABLE_DIR / f"ols_{name}_summary.txt", "w") as fh:
            fh.write(model.summary().as_text())

    return ctr_model, cvr_model


def _tidy_params(model) -> pd.DataFrame:
    frame = pd.DataFrame({
        "coef": model.params,
        "low": model.conf_int()[0],
        "high": model.conf_int()[1],
        "pvalue": model.pvalues,
    })
    frame = frame.drop(index="Intercept", errors="ignore")
    frame.index = (frame.index
                   .str.replace(r"C\(device, Treatment\('Desktop'\)\)\[T\.", "Device: ", regex=True)
                   .str.replace(r"C\(location\)\[T\.", "Location: ", regex=True)
                   .str.replace(r"\]", "", regex=True)
                   .str.replace("is_branded", "Branded keyword")
                   .str.replace("keyword_word_count", "Keyword word count")
                   .str.replace("is_weekend", "Weekend")
                   .str.replace("cpc", "Cost per click"))
    return frame.sort_values("coef")


def fig_coefficients(ctr_model, cvr_model):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=False)
    for ax, model, title in zip(
            axes, (ctr_model, cvr_model),
            ("What moves CTR (percentage points)",
             "What moves CVR (percentage points)")):
        params = _tidy_params(model)
        y = np.arange(len(params))
        colors = [ACCENT if p < 0.05 else NEUTRAL for p in params["pvalue"]]
        ax.hlines(y, params["low"], params["high"], color=colors, linewidth=2)
        ax.scatter(params["coef"], y, color=colors, s=42, zorder=3)
        ax.axvline(0, color=INK, linewidth=0.8)
        ax.set_yticks(y, params.index)
        ax.set_title(title)
        ax.set_xlabel("Effect vs baseline (Desktop, Bangalore)")
    fig.text(0.5, -0.01,
             "Orange: significant at p < 0.05 with HC3 robust errors. "
             "Gray: not significant.",
             ha="center", fontsize=9, color="#78909c")
    tag_source(fig)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "09_regression_coefficients.png")
    plt.close(fig)


def hypothesis_tests(df: pd.DataFrame) -> pd.DataFrame:
    """A few direct two-sample comparisons reported alongside the models."""
    rows = []

    mob = df.loc[df["device"] == "Mobile", "ctr"].dropna()
    desk = df.loc[df["device"] == "Desktop", "ctr"].dropna()
    t, p = stats.mannwhitneyu(mob, desk, alternative="greater")
    rows.append({
        "test": "Mobile CTR > Desktop CTR (Mann-Whitney U)",
        "statistic": t, "p_value": p,
        "group_a_mean": mob.mean(), "group_b_mean": desk.mean(),
    })

    if "is_branded" in df.columns:
        br = df.loc[df["is_branded"] == 1, "cvr"].dropna()
        nb = df.loc[df["is_branded"] == 0, "cvr"].dropna()
        t, p = stats.mannwhitneyu(br, nb, alternative="greater")
        rows.append({
            "test": "Branded CVR > Non-branded CVR (Mann-Whitney U)",
            "statistic": t, "p_value": p,
            "group_a_mean": br.mean(), "group_b_mean": nb.mean(),
        })

    if "is_weekend" in df.columns:
        we = df.loc[df["is_weekend"] == 1, "ctr"].dropna()
        wd = df.loc[df["is_weekend"] == 0, "ctr"].dropna()
        t, p = stats.mannwhitneyu(we, wd, alternative="two-sided")
        rows.append({
            "test": "Weekend CTR differs from weekday CTR (Mann-Whitney U)",
            "statistic": t, "p_value": p,
            "group_a_mean": we.mean(), "group_b_mean": wd.mean(),
        })

    result = pd.DataFrame(rows)
    result.to_csv(TABLE_DIR / "hypothesis_tests.csv", index=False)
    return result


def run(df: pd.DataFrame | None = None):
    apply_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    if df is None:
        df = load_processed()

    ctr_model, cvr_model = fit_models(df)
    print(f"CTR model  R2 = {ctr_model.rsquared:.3f}")
    print(f"CVR model  R2 = {cvr_model.rsquared:.3f}")
    fig_coefficients(ctr_model, cvr_model)

    tests = hypothesis_tests(df)
    print(tests.to_string(index=False))
    print(f"Outputs written to {TABLE_DIR}/ and {FIG_DIR}/")
    return ctr_model, cvr_model, tests


if __name__ == "__main__":
    run()
