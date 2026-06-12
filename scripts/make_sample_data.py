"""Generate a synthetic stand-in for the Kaggle Google Ads Sales dataset.

The real dataset lives on Kaggle and is not committed to this repository.
This script produces a file with the same schema and the same kind of mess
(currency symbols, mixed date formats, a few impossible rows) so the whole
pipeline can be tested end to end without downloading anything.

The signal baked into the data is deliberate and documented, so the analysis
has something real to find:
  - branded keywords get a large CTR and CVR lift
  - mobile gets a CTR lift but a small CVR penalty versus desktop
  - longer keywords (more words) convert slightly better
  - locations differ moderately
  - weekends dip a little

Usage:
    python scripts/make_sample_data.py --rows 3500 --seed 7
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

OUT_PATH = Path("data/raw/google_ads_sales.csv")

GENERIC_KEYWORDS = [
    "running shoes", "buy sneakers online", "cheap sports shoes",
    "best trail running shoes", "mens gym shoes", "womens walking shoes",
    "shoe sale", "leather boots", "kids school shoes", "waterproof hiking boots",
    "white sneakers under 2000", "comfortable office shoes", "marathon shoes",
    "tennis shoes", "slip on shoes", "canvas shoes", "formal shoes for men",
    "sports sandals", "shoe store near me", "orthopedic walking shoes",
]
BRANDED_KEYWORDS = [
    "stride official store", "stride brand shoes", "stride shop online",
    "buy from us stride", "stride official website", "stride brand sneakers",
]
DEVICES = ["Mobile", "Desktop", "Tablet"]
DEVICE_P = [0.58, 0.32, 0.10]
LOCATIONS = ["Hyderabad", "Mumbai", "Bangalore", "Chennai", "Delhi", "Pune"]
LOCATION_P = [0.20, 0.22, 0.18, 0.13, 0.17, 0.10]

DEVICE_CTR_LIFT = {"Mobile": 0.35, "Desktop": 0.0, "Tablet": -0.15}
DEVICE_CVR_LIFT = {"Mobile": -0.12, "Desktop": 0.0, "Tablet": -0.20}
LOCATION_CTR_LIFT = {"Hyderabad": 0.10, "Mumbai": 0.05, "Bangalore": 0.18,
                     "Chennai": -0.08, "Delhi": 0.0, "Pune": -0.05}
LOCATION_CVR_LIFT = {"Hyderabad": 0.05, "Mumbai": 0.15, "Bangalore": 0.10,
                     "Chennai": -0.05, "Delhi": 0.0, "Pune": 0.08}


def build(rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    branded_mask = rng.random(rows) < 0.18
    keywords = np.where(
        branded_mask,
        rng.choice(BRANDED_KEYWORDS, rows),
        rng.choice(GENERIC_KEYWORDS, rows),
    )
    devices = rng.choice(DEVICES, rows, p=DEVICE_P)
    locations = rng.choice(LOCATIONS, rows, p=LOCATION_P)
    dates = pd.to_datetime("2024-01-01") + pd.to_timedelta(
        rng.integers(0, 365, rows), unit="D")
    weekend = pd.Series(dates).dt.dayofweek.isin([5, 6]).to_numpy()

    word_count = pd.Series(keywords).str.split().str.len().to_numpy()

    impressions = np.maximum(rng.lognormal(mean=6.8, sigma=0.9, size=rows), 30)
    impressions = impressions.astype(int)

    # Latent CTR
    base_ctr = 0.032
    ctr_mu = base_ctr * (
        1
        + 0.85 * branded_mask
        + np.vectorize(DEVICE_CTR_LIFT.get)(devices)
        + np.vectorize(LOCATION_CTR_LIFT.get)(locations)
        - 0.06 * weekend
    )
    ctr_true = np.clip(rng.normal(ctr_mu, 0.006), 0.002, 0.35)
    clicks = rng.binomial(impressions, ctr_true)

    # Latent CVR
    base_cvr = 0.055
    cvr_mu = base_cvr * (
        1
        + 1.1 * branded_mask
        + np.vectorize(DEVICE_CVR_LIFT.get)(devices)
        + np.vectorize(LOCATION_CVR_LIFT.get)(locations)
        + 0.05 * (word_count - 2)
    )
    cvr_true = np.clip(rng.normal(cvr_mu, 0.012), 0.002, 0.6)
    conversions = rng.binomial(np.maximum(clicks, 0), cvr_true)

    cpc = np.clip(rng.normal(14 - 4.5 * branded_mask, 3.0), 2, None)
    cost = clicks * cpc * rng.uniform(0.9, 1.1, rows)
    order_value = rng.normal(2600, 700, rows).clip(400)
    sale_amount = conversions * order_value
    leads = conversions + rng.binomial(np.maximum(clicks - conversions, 0), 0.04)

    df = pd.DataFrame({
        "Ad_ID": [f"AD{100000 + i}" for i in range(rows)],
        "Ad_Date": dates.strftime("%Y-%m-%d"),
        "Campaign_Name": "Search_Shoes_" + pd.Series(locations),
        "Keyword": keywords,
        "Device": devices,
        "Location": locations,
        "Impressions": impressions,
        "Clicks": clicks,
        "Cost": cost.round(2),
        "Leads": leads,
        "Conversions": conversions,
        "Conversion Rate": np.where(clicks > 0,
                                    (conversions / np.maximum(clicks, 1)) * 100,
                                    0).round(2),
        "Sale_Amount": sale_amount.round(2),
    })

    # Inject the kind of mess found in the original file
    df["Cost"] = df["Cost"].astype(object)
    df["Sale_Amount"] = df["Sale_Amount"].astype(object)
    dirty = rng.choice(rows, size=int(rows * 0.04), replace=False)
    half = len(dirty) // 2
    df.loc[dirty[:half], "Cost"] = df.loc[dirty[:half], "Cost"].map(
        lambda v: f"Rs.{v:,.2f}")
    df.loc[dirty[half:], "Sale_Amount"] = df.loc[dirty[half:], "Sale_Amount"].map(
        lambda v: f"{v:,.2f} INR")

    odd_dates = rng.choice(rows, size=int(rows * 0.03), replace=False)
    df.loc[odd_dates, "Ad_Date"] = pd.to_datetime(
        df.loc[odd_dates, "Ad_Date"]).dt.strftime("%d-%m-%Y")

    missing = rng.choice(rows, size=int(rows * 0.02), replace=False)
    df.loc[missing, "Conversions"] = np.nan

    dupes = df.sample(n=int(rows * 0.01), random_state=seed)
    df = pd.concat([df, dupes], ignore_index=True)

    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=3500)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()

    frame = build(args.rows, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(f"Wrote {len(frame)} rows to {args.out}")
