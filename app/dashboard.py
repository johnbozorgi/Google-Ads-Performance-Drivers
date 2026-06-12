"""Interactive dashboard for the Google Ads driver analysis.

Run from the repository root:
    streamlit run app/dashboard.py
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

DATA_PATH = ROOT / "data" / "processed" / "cleaned_google_ads.csv"

PALETTE = ["#22313f", "#e2725b", "#9db4c0", "#3a7d6b", "#c9a227", "#7d5a7a"]

st.set_page_config(page_title="Google Ads Performance Drivers",
                   layout="wide")


@st.cache_data
def load() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH, parse_dates=["ad_date"])


if not DATA_PATH.exists():
    st.error("Processed data not found. Run `python run_pipeline.py` first.")
    st.stop()

df = load()

st.title("Google Ads Performance Drivers")
st.caption("Explore what moves click-through and conversion rates. "
           "Filter on the left, every chart reacts.")

with st.sidebar:
    st.header("Filters")
    devices = st.multiselect("Device", sorted(df["device"].dropna().unique()),
                             default=sorted(df["device"].dropna().unique()))
    locations = st.multiselect("Location",
                               sorted(df["location"].dropna().unique()),
                               default=sorted(df["location"].dropna().unique()))
    branded = st.radio("Keyword type", ["All", "Branded only", "Non-branded only"])

mask = df["device"].isin(devices) & df["location"].isin(locations)
if branded == "Branded only":
    mask &= df["is_branded"] == 1
elif branded == "Non-branded only":
    mask &= df["is_branded"] == 0
view = df[mask]

if view.empty:
    st.warning("No rows match the current filters.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Ads", f"{len(view):,}")
c2.metric("Average CTR", f"{view['ctr'].mean() * 100:.2f}%")
c3.metric("Average CVR", f"{view['cvr'].mean() * 100:.2f}%")
c4.metric("Total spend", f"{view['cost'].sum():,.0f}")

left, right = st.columns(2)

with left:
    grp = (view.groupby("device")[["ctr", "cvr"]].mean() * 100).reset_index()
    melted = grp.melt(id_vars="device", var_name="metric", value_name="rate")
    fig = px.bar(melted, x="device", y="rate", color="metric", barmode="group",
                 color_discrete_sequence=PALETTE[:2],
                 labels={"rate": "Rate (%)", "device": ""},
                 title="CTR and CVR by device")
    st.plotly_chart(fig, use_container_width=True)

with right:
    grp = (view.groupby("location")[["ctr", "cvr"]].mean() * 100).reset_index()
    melted = grp.melt(id_vars="location", var_name="metric", value_name="rate")
    fig = px.bar(melted, x="location", y="rate", color="metric",
                 barmode="group", color_discrete_sequence=PALETTE[:2],
                 labels={"rate": "Rate (%)", "location": ""},
                 title="CTR and CVR by location")
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Keyword portfolio map")
kw = (view.groupby("keyword")
      .agg(ctr=("ctr", "mean"), cvr=("cvr", "mean"),
           spend=("cost", "sum"), branded=("is_branded", "max"))
      .reset_index().dropna())
kw[["ctr", "cvr"]] *= 100
kw["type"] = kw["branded"].map({1: "Branded", 0: "Non-branded"})
fig = px.scatter(kw, x="ctr", y="cvr", size="spend", color="type",
                 hover_name="keyword",
                 color_discrete_map={"Branded": PALETTE[1],
                                     "Non-branded": PALETTE[2]},
                 labels={"ctr": "Average CTR (%)", "cvr": "Average CVR (%)"},
                 size_max=45)
fig.add_vline(x=kw["ctr"].median(), line_color="#cfd8dc")
fig.add_hline(y=kw["cvr"].median(), line_color="#cfd8dc")
st.plotly_chart(fig, use_container_width=True)
st.caption("Bubble size is total spend. Top-right keywords earn their budget, "
           "bottom-left keywords deserve a review.")

st.subheader("Monthly trend")
monthly = (view.set_index("ad_date").resample("ME")[["ctr", "cvr"]]
           .mean() * 100).reset_index()
fig = px.line(monthly.melt(id_vars="ad_date", var_name="metric",
                           value_name="rate"),
              x="ad_date", y="rate", color="metric", markers=True,
              color_discrete_sequence=PALETTE[:2],
              labels={"ad_date": "", "rate": "Rate (%)"})
st.plotly_chart(fig, use_container_width=True)
