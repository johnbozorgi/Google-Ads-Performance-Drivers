"""Run the full pipeline: cleaning, EDA, statistics, modeling.

Usage:
    python run_pipeline.py
    python run_pipeline.py --input data/raw/google_ads_sales.csv
"""

import argparse
from pathlib import Path

from src import data_prep, eda, modeling, stats_analysis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=data_prep.RAW_DEFAULT,
                        help="Path to the raw Kaggle CSV")
    args = parser.parse_args()

    print("=" * 60)
    print("Phase 1: data preparation")
    print("=" * 60)
    df = data_prep.run(args.input)

    print("\n" + "=" * 60)
    print("Phase 2: exploratory analysis")
    print("=" * 60)
    eda.run(df)

    print("\n" + "=" * 60)
    print("Phase 3: statistical driver analysis")
    print("=" * 60)
    stats_analysis.run(df)

    print("\n" + "=" * 60)
    print("Phase 4: predictive modeling")
    print("=" * 60)
    modeling.run(df)

    print("\nDone. See reports/figures and reports/tables.")


if __name__ == "__main__":
    main()
