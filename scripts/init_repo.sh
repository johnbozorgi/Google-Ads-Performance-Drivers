#!/usr/bin/env bash
# Build the repository git history as a sequence of incremental commits, the
# way the project was actually developed: scaffold first, then data work,
# then analysis, then polish.
#
# Run once from the repository root, with your git identity configured:
#     bash scripts/init_repo.sh
#
# Afterwards:
#     git remote add origin git@github.com:<you>/google-ads-performance-drivers.git
#     git push -u origin main

set -euo pipefail

if [ -d .git ]; then
    echo "A .git directory already exists, refusing to overwrite it."
    exit 1
fi

git init -b main

git add .gitignore LICENSE requirements.txt data/raw/.gitkeep data/processed/.gitkeep
git commit -m "Initial commit: project scaffold, license, requirements"

git add scripts/make_sample_data.py
git commit -m "Add synthetic data generator matching the Kaggle schema"

git add src/__init__.py src/data_prep.py
git commit -m "Data cleaning and feature engineering pipeline

Handles currency strings, mixed date formats, duplicates and rows that
fail physical sanity checks. Adds CTR, CVR, CPC, ROAS, keyword features
and a branded flag."

git add src/plotting.py
git commit -m "Shared figure styling with a custom palette"

git add src/eda.py
git commit -m "EDA module: distributions, splits, correlation and the keyword portfolio map"

git add src/stats_analysis.py
git commit -m "Weighted OLS driver models with HC3 errors and hypothesis tests"

git add src/modeling.py run_pipeline.py
git commit -m "Model comparison, permutation importance and partial dependence

Ridge, random forest and gradient boosting on a 25 percent holdout,
plus a single entry point that runs all four phases."

git add app/dashboard.py
git commit -m "Streamlit dashboard with device, location and keyword filters"

git add scripts/build_notebooks.py notebooks/
git commit -m "Split the analysis into four executed notebooks"

git add .github/workflows/pipeline.yml
git commit -m "CI: run the pipeline on generated sample data"

git add reports/ README.md
git add -A
git commit -m "Update README with findings, figures and method notes"

echo
echo "History created:"
git log --oneline
