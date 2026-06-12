"""Build the four analysis notebooks from cell definitions.

The notebook sources live here as plain Python so they stay reviewable in
version control and can be regenerated at any time:

    python scripts/build_notebooks.py
"""

from pathlib import Path

import nbformat as nbf

OUT_DIR = Path("notebooks")

SETUP = """
import sys
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
import os
os.chdir(ROOT)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, display

from src import data_prep, eda, stats_analysis, modeling
from src.plotting import apply_style

apply_style()
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")
"""


def build(name, cells):
    nb_cells = []
    for kind, src in cells:
        if kind == "md":
            nb_cells.append(nbf.v4.new_markdown_cell(src.strip()))
        else:
            nb_cells.append(nbf.v4.new_code_cell(src.strip()))
    nb = nbf.v4.new_notebook(cells=nb_cells)
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3", "language": "python", "name": "python3"}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    nbf.write(nb, path)
    print(f"Wrote {path} with {len(nb_cells)} cells")


# ---------------------------------------------------------------- notebook 1
nb1 = [
    ("md", """
# 01. Data Preparation

First notebook of four. The raw Kaggle file is messy on purpose: currency
strings inside numeric columns, two date formats, duplicate ad ids and a few
rows where clicks exceed impressions. This notebook cleans it, engineers the
metrics the rest of the analysis depends on, and saves the processed file.

If you have not placed the Kaggle CSV in `data/raw/`, run
`python scripts/make_sample_data.py` first to generate a synthetic stand-in
with the same schema.
"""),
    ("code", SETUP),
    ("md", """
## Raw file inspection

A quick look before touching anything. Note the mixed types in `Cost` and
`Sale_Amount` and the two date formats in `Ad_Date`.
"""),
    ("code", """
raw = data_prep.load_raw()
raw.sample(8, random_state=1)
"""),
    ("code", """
raw.dtypes
"""),
    ("md", """
## Cleaning

`data_prep.clean` normalizes column names, strips currency symbols, parses
both date formats, drops duplicates and removes rows that fail physical
sanity checks (clicks above impressions, conversions above clicks).
"""),
    ("code", """
tidy = data_prep.clean(raw)
print(f"Raw rows: {len(raw)}")
print(f"Clean rows: {len(tidy)}")
tidy.dtypes
"""),
    ("md", """
## Feature engineering

The derived metrics the project is actually about:

- `ctr` clicks / impressions and `cvr` conversions / clicks
- `cpc`, `cpa`, `roas`, `revenue_per_click`
- `keyword_length`, `keyword_word_count`, and `is_branded`, flagged when the
  query contains brand tokens
- calendar features: `month`, `day_of_week`, `is_weekend`
"""),
    ("code", """
df = data_prep.engineer(tidy)
df.to_csv("data/processed/cleaned_google_ads.csv", index=False)
df[["keyword", "device", "location", "ctr", "cvr", "cpc",
    "is_branded", "keyword_word_count"]].head(8)
"""),
    ("code", """
print(f"Date range: {df['ad_date'].min().date()} to {df['ad_date'].max().date()}")
print(f"Branded share of ads: {df['is_branded'].mean():.1%}")
print(f"Missing CVR (zero-click ads): {df['cvr'].isna().mean():.1%}")
"""),
    ("md", """
The processed file is saved to `data/processed/cleaned_google_ads.csv`.
Continue with `02_exploratory_analysis.ipynb`.
"""),
]

# ---------------------------------------------------------------- notebook 2
nb2 = [
    ("md", """
# 02. Exploratory Analysis

Where the dataset gets a face. The goal here is not decoration: each figure
either answers a research question directly or sets up a control the
regression notebooks will need. All figures are also saved to
`reports/figures/`.
"""),
    ("code", SETUP),
    ("code", """
df = eda.load_processed()
eda.FIG_DIR.mkdir(parents=True, exist_ok=True)
eda.summary_table(df)
"""),
    ("md", """
## How the two targets are distributed
"""),
    ("code", """
eda.fig_rate_distributions(df)
display(Image("reports/figures/01_rate_distributions.png"))
"""),
    ("md", """
Both rates are right skewed, which is normal for ads data: most ads sit near
the account average and a small group performs far above it. The skew is why
the hypothesis tests later use rank based methods instead of plain t tests.
"""),
    ("md", """
## The three splits that matter
"""),
    ("code", """
eda.fig_device_split(df)
display(Image("reports/figures/02_device_split.png"))
"""),
    ("code", """
eda.fig_location_split(df)
display(Image("reports/figures/03_location_split.png"))
"""),
    ("code", """
eda.fig_branded_split(df)
display(Image("reports/figures/04_branded_split.png"))
"""),
    ("md", """
The branded split is the loudest signal so far. Branded queries both click
and convert at a much higher rate, which matches what practitioners see in
real accounts: someone searching for a brand is already late in the funnel.
The important consequence is that any analysis ignoring the branded flag will
attribute this lift to whatever happens to correlate with it.
"""),
    ("md", """
## Correlations and spend
"""),
    ("code", """
eda.fig_correlation(df)
display(Image("reports/figures/05_correlation_heatmap.png"))
"""),
    ("code", """
eda.fig_cost_vs_rates(df)
display(Image("reports/figures/06_cpc_vs_rates.png"))
"""),
    ("md", """
## The keyword portfolio map

The chart I would put in front of a client. Each bubble is one keyword,
sized by total spend, split into four quadrants at the median CTR and median
CVR:

* **Stars** earn their budget on both metrics.
* **Hidden gems** convert well but rarely get clicked. Usually an ad copy or
  position problem, not a targeting problem.
* **Curiosity clicks** attract clicks that go nowhere, often a sign the ad
  promises something the landing page does not deliver.
* **Budget drains** underperform on both and are the first place to cut.
"""),
    ("code", """
eda.fig_keyword_quadrants(df)
display(Image("reports/figures/07_keyword_quadrants.png"))
"""),
    ("code", """
eda.fig_monthly_trend(df)
display(Image("reports/figures/08_monthly_trend.png"))
"""),
    ("md", """
Continue with `03_statistical_drivers.ipynb`, where these visual impressions
get coefficients and p-values.
"""),
]

# ---------------------------------------------------------------- notebook 3
nb3 = [
    ("md", """
# 03. Statistical Driver Analysis

The EDA suggested branded intent and device matter most. This notebook makes
those claims precise with two weighted least squares models, one per target,
plus direct hypothesis tests.

Two methodological choices worth calling out:

1. **Weighting.** A CTR computed from 40 impressions is far noisier than one
   computed from 4,000, so each observation is weighted by its denominator
   (impressions for the CTR model, clicks for the CVR model).
2. **Robust errors.** Standard errors are HC3 heteroskedasticity robust, so
   the p-values do not depend on the constant-variance assumption the raw
   rates clearly violate.

Coefficients are in percentage points and read directly: "switching from
desktop to mobile is worth about X points of CTR, all else equal".
"""),
    ("code", SETUP),
    ("code", """
df = eda.load_processed()
ctr_model, cvr_model = stats_analysis.fit_models(df)
print(f"CTR model R2: {ctr_model.rsquared:.3f}")
print(f"CVR model R2: {cvr_model.rsquared:.3f}")
"""),
    ("code", """
stats_analysis.fig_coefficients(ctr_model, cvr_model)
display(Image("reports/figures/09_regression_coefficients.png"))
"""),
    ("md", """
Reading the chart: orange intervals exclude zero at the 5 percent level.
Branded intent dominates both models. Device is the second CTR driver and
flips sign for CVR, the classic browse-on-mobile, buy-on-desktop pattern.
Cost per click is flat for CTR once intent is controlled, meaning expensive
clicks are not better clicks.

Full regression tables below, also saved under `reports/tables/`.
"""),
    ("code", """
ctr_model.summary()
"""),
    ("code", """
cvr_model.summary()
"""),
    ("md", """
## Hypothesis tests

Three direct comparisons with the Mann-Whitney U test, chosen because the
rate distributions are right skewed:
"""),
    ("code", """
tests = stats_analysis.hypothesis_tests(df)
tests
"""),
    ("md", """
All three headline differences hold up: mobile CTR exceeds desktop CTR,
branded CVR exceeds non-branded CVR, and the weekend dip in CTR is small but
real. Continue with `04_predictive_modeling.ipynb`.
"""),
]

# ---------------------------------------------------------------- notebook 4
nb4 = [
    ("md", """
# 04. Predictive Modeling

The regressions identified the drivers under a linear lens. This notebook
asks two further questions:

1. How much of CTR and CVR can be predicted from pre-click variables at all?
2. Do models that can pick up non-linear structure agree with the regression
   about which features matter?

Three regressors per target: ridge as the linear baseline, a random forest,
and histogram gradient boosting. The split is 75/25 and all metrics are on
the held-out test set. Importance is measured by permutation on the test
set, which avoids the bias impurity-based importances have toward
high cardinality features.
"""),
    ("code", SETUP),
    ("code", """
df = eda.load_processed()
scores, importances = modeling.run(df)
scores
"""),
    ("code", """
display(Image("reports/figures/10_model_comparison.png"))
"""),
    ("md", """
## Which features carry the prediction
"""),
    ("code", """
display(Image("reports/figures/11_importance_ctr.png"))
display(Image("reports/figures/13_importance_cvr.png"))
"""),
    ("code", """
display(Image("reports/figures/12_pdp_ctr.png"))
display(Image("reports/figures/14_pdp_cvr.png"))
"""),
    ("md", """
## Reading the results

CTR is considerably more predictable than CVR from this feature set. That
gap is itself a finding: clicking happens inside the search results page,
where the variables we observe (device, query type, brand affinity) live.
Converting happens after the click, on a landing page we have no features
for. With only pre-click variables, a low CVR R squared is the honest
result, not a modeling failure.

The permutation rankings agree with the regression: the branded flag and
device dominate CTR, the branded flag dominates CVR, and location plays a
secondary role. When a linear model and a boosted ensemble rank the same
drivers at the top, that agreement is worth more than either model alone.

## Findings and recommendations

**What moves CTR.** Branded intent first, device second (mobile ahead of
desktop, tablet behind), location third. A higher CPC does not buy CTR.

**What moves CVR.** Branded intent by a wide margin. Device flips: mobile
clicks convert slightly worse than desktop clicks. Longer, more specific
queries convert a little better.

**What a growth team should do with this.**

1. Report branded and non-branded performance separately. Blending them
   flatters every average and hides the true cost of growth.
2. Bid device aware: mobile earns clicks cheaply, desktop closes them.
3. Work the hidden gems quadrant first. Keywords that convert well but get
   few clicks need better copy or positions, which is cheaper than buying
   new traffic.
4. Audit the curiosity clicks quadrant for ad-to-page message match before
   renewing their budget.
5. Invest in post-click data collection. Landing page and audience features
   are what a useful CVR model is missing, not more campaign metadata.

**Limitations.** The dataset is simulated, covers a single advertiser and
one year, and aggregates at the ad level. Effects are associations under a
linear model, not causal estimates. The value of the project is the method:
the same pipeline runs unchanged on a real account export with these
columns.
"""),
]

if __name__ == "__main__":
    build("01_data_preparation.ipynb", nb1)
    build("02_exploratory_analysis.ipynb", nb2)
    build("03_statistical_drivers.ipynb", nb3)
    build("04_predictive_modeling.ipynb", nb4)
