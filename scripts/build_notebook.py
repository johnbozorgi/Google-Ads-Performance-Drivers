"""Build notebooks/analysis.ipynb from cell definitions.

Kept as a script so the notebook source lives in version control as plain
Python and can be regenerated at any time:
    python scripts/build_notebook.py
"""

from pathlib import Path

import nbformat as nbf

OUT = Path("notebooks/analysis.ipynb")

md = []
code = []
cells = []


def m(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def c(src):
    cells.append(nbf.v4.new_code_cell(src.strip()))


m("""
# Identifying Key Drivers of CTR and Conversion Rate in Google Ads Campaigns

This notebook walks through the full analysis behind the project. It answers
four questions:

1. Which factors (device, location, keyword characteristics) move CTR the most?
2. Which factors move conversion rate the most?
3. Can CTR and CVR be predicted to a useful degree from the available variables?
4. What should a growth team actually do with these findings?

The dataset is the [Google Ads Sales Dataset](https://www.kaggle.com/datasets/nayakganesh007/google-ads-sales-dataset)
from Kaggle. If you have not downloaded it, the repository ships a generator
that produces a synthetic file with the same schema, so the notebook runs
either way. The reusable logic lives in `src/`, this notebook calls it and
adds commentary.
""")

c("""
import sys
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
import os
os.chdir(ROOT)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src import data_prep, eda, stats_analysis, modeling
from src.plotting import apply_style

apply_style()
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")
""")

m("""
## Phase 1: data preparation

The raw file is messy on purpose: currency strings inside numeric columns,
two date formats, duplicate ad ids and a few rows where clicks exceed
impressions. `data_prep.clean` handles all of that, then `data_prep.engineer`
adds the derived metrics the analysis is about: CTR, CVR, CPC, ROAS, keyword
length and word count, a branded flag, and calendar features.
""")

c("""
df = data_prep.run()
df.head()
""")

c("""
print(f"Rows: {len(df)}")
print(f"Date range: {df['ad_date'].min().date()} to {df['ad_date'].max().date()}")
print(f"Devices: {sorted(df['device'].dropna().unique())}")
print(f"Locations: {sorted(df['location'].dropna().unique())}")
print(f"Branded share: {df['is_branded'].mean():.1%}")
""")

m("""
## Phase 2: exploratory analysis

A quick look at the shape of the two target variables, then the splits that
matter for the research questions. Every figure is also saved under
`reports/figures/`.
""")

c("""
eda.summary_table(df)
""")

c("""
eda.FIG_DIR.mkdir(parents=True, exist_ok=True)
eda.fig_rate_distributions(df)
from IPython.display import Image, display
display(Image("reports/figures/01_rate_distributions.png"))
""")

m("""
Both rates are right skewed, which is normal for ads data: most ads sit near
the account average and a small group performs far above it. The skew is one
reason the later hypothesis tests use rank based methods instead of plain
t tests.
""")

c("""
eda.fig_device_split(df)
display(Image("reports/figures/02_device_split.png"))
""")

c("""
eda.fig_location_split(df)
display(Image("reports/figures/03_location_split.png"))
""")

c("""
eda.fig_branded_split(df)
display(Image("reports/figures/04_branded_split.png"))
""")

m("""
The branded split is the loudest signal so far. Branded queries both click
and convert at a much higher rate, which matches what practitioners see in
real accounts: people searching for a brand are already late in the funnel.
The important consequence is that any analysis that ignores the branded flag
risks attributing this lift to whatever variable happens to correlate with it.
""")

c("""
eda.fig_correlation(df)
display(Image("reports/figures/05_correlation_heatmap.png"))
""")

c("""
eda.fig_keyword_quadrants(df)
display(Image("reports/figures/07_keyword_quadrants.png"))
""")

m("""
The portfolio map is the chart I would put in front of a client. Each bubble
is one keyword, sized by total spend, split into four quadrants at the median
CTR and median CVR:

* **Stars** earn their budget on both metrics.
* **Hidden gems** convert well but rarely get clicked, usually an ad copy
  or position problem, not a targeting problem.
* **Curiosity clicks** attract clicks that go nowhere, often a sign the ad
  promises something the landing page does not deliver.
* **Budget drains** underperform on both and are the first place to cut.
""")

c("""
eda.fig_monthly_trend(df)
display(Image("reports/figures/08_monthly_trend.png"))
""")

m("""
## Phase 3: statistical driver analysis

Two weighted least squares models, one per target. Each observation is
weighted by its denominator (impressions for CTR, clicks for CVR), because a
rate computed from 40 impressions is far noisier than one computed from
4,000. Standard errors are HC3 robust. The coefficients are in percentage
points, so they read directly as "switching from desktop to mobile is worth
about X points of CTR, all else equal".
""")

c("""
ctr_model, cvr_model = stats_analysis.fit_models(df)
print(f"CTR model R2: {ctr_model.rsquared:.3f}")
print(f"CVR model R2: {cvr_model.rsquared:.3f}")
""")

c("""
stats_analysis.fig_coefficients(ctr_model, cvr_model)
display(Image("reports/figures/09_regression_coefficients.png"))
""")

c("""
ctr_model.summary()
""")

c("""
cvr_model.summary()
""")

m("""
### Hypothesis tests

Three direct comparisons, run with the Mann-Whitney U test because the rate
distributions are skewed:
""")

c("""
tests = stats_analysis.hypothesis_tests(df)
tests
""")

m("""
## Phase 4: predictive modeling

Three regressors per target: ridge as the linear baseline, a random forest,
and histogram gradient boosting. The split is 75/25 and the metrics below are
on the held out test set. Importance is measured by permutation on the test
set, which avoids the bias tree impurity importances have toward
high cardinality features.
""")

c("""
scores, importances = modeling.run(df)
scores
""")

c("""
display(Image("reports/figures/10_model_comparison.png"))
""")

c("""
display(Image("reports/figures/11_importance_ctr.png"))
display(Image("reports/figures/13_importance_cvr.png"))
""")

c("""
display(Image("reports/figures/12_pdp_ctr.png"))
display(Image("reports/figures/14_pdp_cvr.png"))
""")

m("""
CTR is considerably more predictable than CVR from this feature set. That gap
is itself a finding: clicking happens inside the search results page, where
the variables we observe (device, query type, brand affinity) live. Converting
happens after the click, on a landing page we have no features for. With only
pre-click variables, a low CVR R squared is the honest result, not a modeling
failure.

The permutation rankings agree with the regression: the branded flag and
device dominate CTR, the branded flag dominates CVR, and location plays a
secondary role.
""")

m("""
## Phase 5: findings and recommendations

**What moves CTR.** The branded keyword flag is the single largest driver,
followed by device, with mobile clearly ahead of desktop and tablet behind.
Location effects are real but smaller. Paying a higher CPC does not, by
itself, buy a better CTR.

**What moves CVR.** Branded intent again leads by a wide margin. Device flips
direction here: mobile clicks convert slightly worse than desktop clicks,
the classic browse-on-mobile, buy-on-desktop pattern. Longer, more specific
queries convert a little better.

**What a growth team should do with this.**

1. Report branded and non-branded performance separately. Blending them
   flatters every average and hides the true cost of growth.
2. Bid device aware: mobile earns clicks cheaply, desktop closes them.
   If the platform allows it, a positive mobile adjustment on awareness
   campaigns and a positive desktop adjustment on conversion campaigns
   follows directly from the coefficients.
3. Work the hidden gems quadrant first. Keywords that convert well but get
   few clicks need better ad copy or higher positions, which is cheaper than
   finding new keywords.
4. Audit the curiosity clicks quadrant for message match between ad and
   landing page before spending another month on them.
5. Treat CVR prediction with humility. Without landing page and audience
   features, models cannot see most of what decides a conversion. The next
   data to collect is post-click, not more of the same.

**Limitations.** The dataset is simulated, covers a single advertiser and one
year, and aggregates at the ad level rather than the auction level. Effects
here are associations under a linear model, not causal estimates. The value
of the project is the method: the same pipeline runs unchanged on a real
account export with these columns.
""")

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {
    "display_name": "Python 3", "language": "python", "name": "python3"}
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(f"Wrote {OUT} with {len(cells)} cells")
