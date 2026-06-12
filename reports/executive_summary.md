# Executive Summary

**Project:** Identifying Key Drivers of Click-Through Rate and Conversion
Rate in Google Ads Campaigns

**Data:** Google Ads Sales Dataset (Kaggle), ad-level records covering one
year of a search account: keyword, device, location, date, impressions,
clicks, cost, conversions, sale amount.

**Method:** Cleaning of an intentionally messy export, feature engineering
(CTR, CVR, CPC, ROAS, branded flag, keyword characteristics, calendar
features), weighted least squares regression with robust standard errors,
Mann-Whitney hypothesis tests, and a comparison of ridge, random forest and
gradient boosting models with permutation importance and partial dependence.

## What drives CTR

| Driver | Direction | Size |
| --- | --- | --- |
| Branded keyword | up | about +2.1 pp vs non-branded |
| Mobile device | up | about +1.1 pp vs desktop |
| Tablet device | down | about -0.5 pp vs desktop |
| Location | mixed | up to 0.9 pp between cities |
| Cost per click | flat | near zero once intent is controlled |

## What drives CVR

| Driver | Direction | Size |
| --- | --- | --- |
| Branded keyword | up | about +5.1 pp vs non-branded |
| Mobile device | down | about -0.7 pp vs desktop |
| Keyword specificity | up | small lift per extra word |
| Location | mixed | roughly 1 pp spread |

## Predictability

The best CTR model explains roughly half the variance on held-out data
(R squared near 0.50). CVR tops out near 0.13. The gap is the finding:
clicks are decided on the results page, where these features live;
conversions are decided on the landing page, where they do not.

## Recommendations

1. Separate branded and non-branded reporting. Blended averages overstate
   how well acquisition is really going.
2. Bid by device and by goal: mobile for cheap attention, desktop for
   closing. The opposite signs of the device coefficients justify split
   adjustments.
3. Prioritize "hidden gem" keywords (high CVR, low CTR) for ad copy and
   position work before buying new traffic.
4. Audit "curiosity click" keywords (high CTR, low CVR) for ad-to-page
   message match before renewing their budget.
5. Invest in post-click data collection. No amount of campaign metadata will
   predict conversions well; landing page and audience features will.

Full detail: `notebooks/analysis.ipynb` and `reports/`.
