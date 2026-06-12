# Publishing checklist

Steps to get this repository on GitHub looking the way it should.

## 1. Build the git history

From the repository root, with your git name and email already configured:

```bash
bash scripts/init_repo.sh
```

This creates eleven commits in development order (scaffold, data generator,
cleaning, EDA, statistics, modeling, dashboard, notebooks, CI, README), so
the history reads like the project was built, because that is the order it
was built in.

## 2. Create the repository and push

```bash
git remote add origin git@github.com:<your-username>/google-ads-performance-drivers.git
git push -u origin main
```

## 3. Fix the badge

In `README.md`, replace `USERNAME` in the workflow badge URL with your
GitHub username, commit and push. The badge turns green after the first
Actions run finishes.

## 4. Fill the About box

Suggested description:

> Weighted regression and ML driver analysis of CTR and conversion rate in
> Google Ads campaigns, with permutation importance, partial dependence and
> a Streamlit dashboard

Suggested topics:

```
python  machine-learning  google-ads  digital-marketing  ppc
data-analysis  statsmodels  scikit-learn  streamlit  jupyter-notebook
```

## 5. Optional polish

- Pin the repository on your profile.
- Set `reports/figures/07_keyword_quadrants.png` as the social preview image
  (Settings, then Social preview).
- Delete this file before or after pushing, it is a checklist, not part of
  the project.
