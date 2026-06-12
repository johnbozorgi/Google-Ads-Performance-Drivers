"""Predictive modeling for CTR and CVR.

Three regressors are compared per target: ridge regression as the linear
baseline, a random forest, and gradient boosting. The point is less to chase
the last decimal of R squared and more to check whether the drivers the
statistical models flagged also dominate the importance rankings of models
that can pick up non-linear structure.

Usage:
    python -m src.modeling
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .eda import load_processed
from .plotting import ACCENT, INK, SOFT, annotate_bars, apply_style, tag_source

FIG_DIR = Path("reports/figures")
TABLE_DIR = Path("reports/tables")

CATEGORICAL = ["device", "location"]
NUMERIC = ["cpc", "keyword_word_count", "keyword_length",
           "is_branded", "is_weekend", "month"]
SEED = 42


def _build_models():
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("num", StandardScaler(), NUMERIC),
    ])
    return {
        "Ridge": Pipeline([("pre", pre),
                           ("model", Ridge(alpha=1.0))]),
        "Random Forest": Pipeline([("pre", pre),
                                   ("model", RandomForestRegressor(
                                       n_estimators=400, min_samples_leaf=5,
                                       random_state=SEED, n_jobs=-1))]),
        "Gradient Boosting": Pipeline([("pre", pre),
                                       ("model", HistGradientBoostingRegressor(
                                           max_depth=4, learning_rate=0.08,
                                           max_iter=350, random_state=SEED))]),
    }


def _frame_for(df: pd.DataFrame, target: str) -> pd.DataFrame:
    cols = CATEGORICAL + NUMERIC + [target]
    data = df[cols].dropna().copy()
    data[NUMERIC] = data[NUMERIC].astype(float)
    data[target] = data[target] * 100
    return data


def evaluate(df: pd.DataFrame, target: str):
    data = _frame_for(df, target)
    X, y = data[CATEGORICAL + NUMERIC], data[target]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25,
                                              random_state=SEED)
    rows, fitted = [], {}
    for name, pipe in _build_models().items():
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        rows.append({
            "target": target.upper(),
            "model": name,
            "r2": r2_score(y_te, pred),
            "rmse": np.sqrt(mean_squared_error(y_te, pred)),
            "mae": mean_absolute_error(y_te, pred),
        })
        fitted[name] = pipe
    scores = pd.DataFrame(rows)
    best_name = scores.sort_values("r2", ascending=False).iloc[0]["model"]
    return scores, fitted[best_name], best_name, (X_te, y_te)


def fig_model_comparison(all_scores: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 4.6))
    pivot = all_scores.pivot(index="model", columns="target", values="r2")
    pivot = pivot.loc[["Ridge", "Random Forest", "Gradient Boosting"]]
    x = np.arange(len(pivot))
    ax.bar(x - 0.18, pivot["CTR"], width=0.36, color=INK, label="CTR model")
    ax.bar(x + 0.18, pivot["CVR"], width=0.36, color=ACCENT, label="CVR model")
    ax.set_xticks(x, pivot.index)
    annotate_bars(ax, fmt="{:.2f}")
    ax.set_ylabel("R squared on the held-out test set")
    ax.set_title("Model comparison")
    ax.legend()
    tag_source(fig)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "10_model_comparison.png")
    plt.close(fig)


def fig_permutation_importance(model, X_te, y_te, target: str, index: int):
    result = permutation_importance(model, X_te, y_te, n_repeats=12,
                                    random_state=SEED, n_jobs=-1)
    imp = (pd.Series(result.importances_mean, index=X_te.columns)
           .sort_values())
    err = pd.Series(result.importances_std, index=X_te.columns)[imp.index]

    labels = {
        "device": "Device", "location": "Location", "cpc": "Cost per click",
        "keyword_word_count": "Keyword word count",
        "keyword_length": "Keyword length",
        "is_branded": "Branded keyword", "is_weekend": "Weekend",
        "month": "Month",
    }
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    colors = [ACCENT if i >= len(imp) - 3 else SOFT for i in range(len(imp))]
    ax.barh([labels.get(i, i) for i in imp.index], imp.values,
            xerr=err.values, color=colors, error_kw={"ecolor": "#90a4ae"})
    ax.set_xlabel("Drop in R squared when the feature is shuffled")
    ax.set_title(f"Permutation importance, {target.upper()} model")
    tag_source(fig)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{index}_importance_{target}.png")
    plt.close(fig)
    return imp.sort_values(ascending=False)


def fig_partial_dependence(model, X_te, target: str, index: int):
    features = ["cpc", "keyword_word_count", "is_branded"]
    fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.8))
    PartialDependenceDisplay.from_estimator(
        model, X_te, features, ax=ax,
        line_kw={"color": ACCENT, "linewidth": 2})
    for a in np.ravel(ax):
        a.set_ylabel(f"Predicted {target.upper()} (%)")
    fig.suptitle(f"Partial dependence, {target.upper()} model",
                 fontweight="bold", fontsize=13)
    tag_source(fig)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{index}_pdp_{target}.png")
    plt.close(fig)


def run(df: pd.DataFrame | None = None):
    apply_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    if df is None:
        df = load_processed()

    all_scores, importances = [], {}
    for i, target in enumerate(("ctr", "cvr")):
        scores, best, best_name, (X_te, y_te) = evaluate(df, target)
        all_scores.append(scores)
        print(f"{target.upper()}: best model is {best_name}")
        importances[target] = fig_permutation_importance(
            best, X_te, y_te, target, index=11 + i * 2)
        fig_partial_dependence(best, X_te, target, index=12 + i * 2)

    scores = pd.concat(all_scores, ignore_index=True).round(4)
    scores.to_csv(TABLE_DIR / "model_scores.csv", index=False)
    print(scores.to_string(index=False))
    fig_model_comparison(scores)
    print(f"Outputs written to {TABLE_DIR}/ and {FIG_DIR}/")
    return scores, importances


if __name__ == "__main__":
    run()
