"""Data cleaning and feature engineering for the Google Ads dataset.

The raw Kaggle file is intentionally messy: currency symbols inside numeric
columns, inconsistent date formats, duplicate ad ids, missing values and the
occasional row where clicks exceed impressions. Everything in this module is
written to survive that file, but it also works on any export that follows
the same schema.

Usage:
    python -m src.data_prep --input data/raw/google_ads_sales.csv
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DEFAULT = Path("data/raw/google_ads_sales.csv")
PROCESSED_PATH = Path("data/processed/cleaned_google_ads.csv")

# Tokens that mark a keyword as branded. The dataset simulates a single
# advertiser, so the brand vocabulary is short. Extend this list when you
# point the pipeline at your own account export.
BRAND_TOKENS = ("brand", "official", "store", "shop online", "buy from us")

CURRENCY_RE = re.compile(r"[^0-9.\-]")


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase, strip and underscore the column names."""
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"[ \-/]+", "_", regex=True)
    )
    return df


def _to_number(series: pd.Series) -> pd.Series:
    """Coerce a column to float, stripping currency symbols, commas and
    percent signs along the way."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    cleaned = (
        series.astype(str)
        .str.strip()
        .replace({"": np.nan, "nan": np.nan, "None": np.nan, "-": np.nan})
        .apply(lambda x: CURRENCY_RE.sub("", x) if isinstance(x, str) else x)
        .replace({"": np.nan})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _parse_dates(series: pd.Series) -> pd.Series:
    """Try a couple of common formats before falling back to pandas guessing."""
    parsed = pd.to_datetime(series, errors="coerce", format="%Y-%m-%d")
    mask = parsed.isna()
    if mask.any():
        fallback = pd.to_datetime(series[mask], errors="coerce", dayfirst=True)
        parsed.loc[mask] = fallback
    return parsed


def load_raw(path: Path = RAW_DEFAULT) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Raw file not found at {path}. Download the dataset from Kaggle "
            "(nayakganesh007/google-ads-sales-dataset) and place the CSV "
            "there, or run scripts/make_sample_data.py to generate a "
            "synthetic stand-in with the same schema."
        )
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a tidy frame with consistent types and no impossible rows."""
    df = _normalize_columns(df)

    # Tolerate small naming differences between dataset versions
    renames = {
        "conversion_rate_(%)": "conversion_rate",
        "cost_(inr)": "cost",
        "sale_amount_(inr)": "sale_amount",
    }
    df = df.rename(columns={k: v for k, v in renames.items() if k in df.columns})

    numeric_cols = [c for c in
                    ("impressions", "clicks", "cost", "leads",
                     "conversions", "conversion_rate", "sale_amount")
                    if c in df.columns]
    for col in numeric_cols:
        df[col] = _to_number(df[col])

    if "ad_date" in df.columns:
        df["ad_date"] = _parse_dates(df["ad_date"])

    for col in ("device", "location", "keyword", "campaign_name"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()
            df[col] = df[col].replace({"Nan": np.nan, "": np.nan})

    before = len(df)
    df = df.drop_duplicates()

    # Rows without the basics cannot be used for rate analysis
    df = df.dropna(subset=["impressions", "clicks"])

    # Physical sanity checks
    df = df[df["impressions"] > 0]
    df = df[df["clicks"] >= 0]
    df = df[df["clicks"] <= df["impressions"]]
    if "conversions" in df.columns:
        df.loc[df["conversions"] > df["clicks"], "conversions"] = np.nan

    df = df.reset_index(drop=True)
    df.attrs["rows_dropped"] = before - len(df)
    return df


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Add the derived metrics the analysis is actually about."""
    df = df.copy()

    df["ctr"] = df["clicks"] / df["impressions"]

    if "conversions" in df.columns:
        df["cvr"] = np.where(df["clicks"] > 0,
                             df["conversions"] / df["clicks"], np.nan)
        df["cvr"] = df["cvr"].clip(upper=1.0)

    if "cost" in df.columns:
        df["cpc"] = np.where(df["clicks"] > 0, df["cost"] / df["clicks"], np.nan)
        if "conversions" in df.columns:
            df["cpa"] = np.where(df["conversions"] > 0,
                                 df["cost"] / df["conversions"], np.nan)

    if "sale_amount" in df.columns and "cost" in df.columns:
        df["roas"] = np.where(df["cost"] > 0,
                              df["sale_amount"] / df["cost"], np.nan)
        df["revenue_per_click"] = np.where(df["clicks"] > 0,
                                           df["sale_amount"] / df["clicks"],
                                           np.nan)

    if "keyword" in df.columns:
        kw = df["keyword"].fillna("")
        df["keyword_length"] = kw.str.len()
        df["keyword_word_count"] = kw.str.split().str.len()
        pattern = "|".join(BRAND_TOKENS)
        df["is_branded"] = kw.str.lower().str.contains(pattern).astype(int)

    if "ad_date" in df.columns:
        df["month"] = df["ad_date"].dt.month
        df["day_of_week"] = df["ad_date"].dt.day_name()
        df["is_weekend"] = df["ad_date"].dt.dayofweek.isin([5, 6]).astype(int)

    return df


def run(input_path: Path = RAW_DEFAULT, output_path: Path = PROCESSED_PATH) -> pd.DataFrame:
    raw = load_raw(input_path)
    tidy = engineer(clean(raw))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tidy.to_csv(output_path, index=False)
    print(f"Raw rows: {len(raw)}")
    print(f"Clean rows: {len(tidy)} (dropped {len(raw) - len(tidy)})")
    print(f"Saved to {output_path}")
    return tidy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=RAW_DEFAULT)
    parser.add_argument("--output", type=Path, default=PROCESSED_PATH)
    args = parser.parse_args()
    run(args.input, args.output)
