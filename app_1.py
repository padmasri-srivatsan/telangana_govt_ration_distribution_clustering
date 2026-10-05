"""
Streamlit Web Application: Telangana PDS Analytics Platform.

Provides interactive dashboards for executive spatial analysis, cluster persona deep-dives,
and automated shop anomaly profiling.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from pathlib import Path

# Page Layout Setup
st.set_page_config(
    page_title="Telangana PDS Analytics Platform",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_processed_data() -> pd.DataFrame:
    """Loads clustered dataset from parquet store."""
    file_path = Path("/Users/sundharamurali/Documents/Code/py-projects/marketing_analysis/shop_features_modeled.parquet")
    if file_path.exists():
        return pd.read_parquet(file_path)
    else:
        # Fallback simulation generator for local testing
        np.random.seed(42)
        n = 500
        return pd.DataFrame({
            "dist_code": np.random.choice([1, 2, 3], n),
            "shop_no": np.arange(1000, 1000 + n),
            "dist_name": np.random.choice(["HYDERABAD", "WARANGAL", "KARIMNAGAR"], n),
            "latitude": np.random.uniform(17.0, 18.5, n),
            "longitude": np.random.uniform(78.0, 79.5, n),
            "avg_transactions": np.random.exponential(400, n),
            "avg_entitled_cards": np.random.normal(500, 100, n),
            "avg_utilization_ratio": np.random.uniform(0.4, 0.95, n),
            "avg_portability_rate": np.random.uniform(0.01, 0.35, n),
            "pc1": np.random.normal(0, 1, n),
            "pc2": np.random.normal(0, 1, n),
            "cluster_id": np.random.choice([0, 1, 2, 3], n),
            "is_anomaly": np.random.choice([0, 1], n, p=[0.93, 0.07]),
            "dbscan_label": np.random.choice([0, -1], n, p=[0.93, 0.07])
        })

# Load Data
df = load_processed_data()

# Executive Dashboard Header
st.title("🌾 Telangana PDS Analytics Platform")
st.markdown("### Multi-Dimensional Shop Performance Clustering & Anomaly Profiling")

# Sidebar Controls
st.sidebar.header("🕹️ Executive Control Panel")

all_districts = sorted(df["dist_name"].dropna().unique().tolist())
selected_districts = st.sidebar.multiselect(
    "Filter District(s)",
    options=all_districts,
    default=all_districts
)

cluster_filter = st.sidebar.multiselect(
    "Filter Behavioral Personas",
    options=sorted(df["cluster_id"].unique().tolist()),
    default=sorted(df["cluster_id"].unique().tolist())
)

show_anomalies_only = st.sidebar.checkbox("Show Flagged Anomalies Only (-1)", value=False)

# Apply Filter Cascade
filtered_df = df[
    (df["dist_name"].isin(selected_districts)) &
    (df["cluster_id"].isin(cluster_filter))
]

if show_anomalies_only:
    filtered_df = filtered_df[filtered_df["is_anomaly"] == 1]

# Top KPI Metric Cards
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Active Shops", f"{len(filtered_df):,}")
c2.metric("Districts Monitored", f"{filtered_df['dist_name'].nunique()}")
c3.metric("Flagged Anomalies", f"{filtered_df['is_anomaly'].sum():,}", delta_color="inverse")
c4.metric("Avg Utilization", f"{filtered_df['avg_utilization_ratio'].mean():.1%}")
c5.metric("Portability Traffic", f"{filtered_df['avg_portability_rate'].mean():.1%}")

st.markdown("---")

# Tab Layout Navigation
tab_clusters, tab_lookup, tab_reports = st.tabs([
    "📊 PCA Cluster Spaces",
    "🔍 Shop Deep-Dive Lookup",
    "📄 Executive Anomaly Report"
])

with tab_clusters:
    st.subheader("High-Dimensional Behavioral Separation (PCA Projections)")

    col_left, col_right = st.columns([2, 1])

    with col_left:
        fig_pca = px.scatter(
            filtered_df,
            x="pc1",
            y="pc2",
            color=filtered_df["cluster_id"].astype(str),
            symbol=filtered_df["is_anomaly"].map({0: "Normal", 1: "Anomaly"}),
            labels={"color": "Cluster Persona", "symbol": "Status"},
            title="PCA Dimension Projection (PC1 vs PC2)",
            hover_data=["shop_no", "dist_name", "avg_transactions"]
        )
        st.plotly_chart(fig_pca, use_container_width=True)

    with col_right:
        st.subheader("Cluster Persona Breakdown")
        profile_df = filtered_df.groupby("cluster_id").agg(
            Shops=("shop_no", "count"),
            Avg_Trans=("avg_transactions", "mean"),
            Utilization=("avg_utilization_ratio", "mean"),
            Portability=("avg_portability_rate", "mean")
        ).round(2)
        st.dataframe(profile_df, use_container_width=True)

with tab_lookup:
    st.subheader("Individual Fair Price Shop Auditor Lookup")

    search_shop = st.text_input("Enter Shop Number (e.g. 1001):")
    if search_shop:
        result = df[df["shop_no"].astype(str) == search_shop.strip()]
        if not result.empty:
            shop_data = result.iloc[0]

            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Shop Number", shop_data["shop_no"])
            col_b.metric("District", shop_data["dist_name"])
            col_c.metric("Cluster Assignment", f"Cluster {shop_data['cluster_id']}")

            st.markdown("#### Operational Metrics vs State Averages")
            metrics_comparison = pd.DataFrame({
                "Metric": ["Avg Transactions", "Capacity Utilization", "Portability Rate"],
                "Shop Value": [
                    f"{shop_data['avg_transactions']:.1f}",
                    f"{shop_data['avg_utilization_ratio']:.1%}",
                    f"{shop_data['avg_portability_rate']:.1%}"
                ],
                "State Benchmark": [
                    f"{df['avg_transactions'].mean():.1f}",
                    f"{df['avg_utilization_ratio'].mean():.1%}",
                    f"{df['avg_portability_rate'].mean():.1%}"
                ]
            })
            st.table(metrics_comparison)

            if shop_data["is_anomaly"] == 1:
                st.error("⚠️ ALERT: This shop has been flagged as an anomalous operational profile by DBSCAN.")
            else:
                st.success("✅ Normal Profile: Shop metrics fall within expected cluster bounds.")
        else:
            st.warning("Shop ID not found in database.")

with tab_reports:
    st.subheader("Anomalous Outlier Priority Queue")
    anomaly_queue = df[df["is_anomaly"] == 1][[
        "dist_name", "shop_no", "cluster_id", "avg_transactions",
        "avg_entitled_cards", "avg_utilization_ratio", "avg_portability_rate"
    ]].sort_values("avg_utilization_ratio", ascending=False)

    st.dataframe(anomaly_queue, use_container_width=True)

    st.download_button(
        label="📥 Download Anomaly Audit CSV Report",
        data=anomaly_queue.to_csv(index=False),
        file_name="telangana_pds_anomalies_audit_report.csv",
        mime="text/csv"
    )