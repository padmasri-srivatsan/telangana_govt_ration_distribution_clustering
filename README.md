# Telangana PDS Analytics: Multi-Dimensional Shop Performance Clustering & Anomaly Profiling

An end-to-end machine learning pipeline for analyzing 7.5 years of transactional, card status, and geospatial data from the Telangana State Civil Supplies Department. This project constructs shop-level behavioral profiles, segments Fair Price Shops (FPS) using K-Means clustering, and detects anomalous transaction patterns with DBSCAN.

---

## 📌 Table of Contents
- [Project Overview](#-project-overview)
- [Key Features & Architecture](#-key-features--architecture)
- [Repository Structure](#-repository-structure)
- [Installation & Setup](#-installation--setup)
- [Dataset Requirements](#-dataset-requirements)
- [Pipeline Execution & Workflow](#-pipeline-execution--workflow)
- [Outputs & Artifacts](#-outputs--artifacts)
- [License](#-license)

---

## 📸 Project Overview

The Public Distribution System (PDS) in Telangana manages food grain distribution across thousands of Fair Price Shops. This project automates the consolidation and feature engineering of multi-year transactional records to:
1. **Track Portability Trends:** Evaluate the impact of One Nation One Ration Card (ONORC) policy implementations.
2. **Cluster Fair Price Shops:** Segment shops into behavioral clusters (e.g., high-volume urban hubs, low-utilization rural outlets, high-portability hotspots) using **K-Means**.
3. **Flag Anomalies:** Identify potential leakage, structural outliers, or inventory discrepancies using **DBSCAN** density-based clustering.

---

## 🚀 Key Features & Architecture

* **Automated File Ingestion & Data Hygiene:** Safely parses heterogeneous CSV files with encoding fallback (`latin1`/`utf-8`), standardizes column naming schemes, strips text whitespaces, and enforces strong numeric typing.
* **Master Schema Integration:** Merges transactions, ration card entitlements, and spatial location metadata into a consolidated monthly panel dataset stored efficiently in Parquet format.
* **Advanced Feature Engineering:** Calculates shop-level metrics including:
  * Utilization rates ($\text{Transactions} / \text{Entitled Cards}$)
  * Portability ratios ($\text{Other Shop Transactions} / \text{Total Transactions}$)
  * Commodity intensity (Rice vs. Wheat proportions)
  * Multi-year transaction growth and volatility metrics
* **Leakage-Free Preprocessing Pipeline:** Integrates `SimpleImputer` (median imputation) and `StandardScaler` inside `sklearn.pipeline.Pipeline` objects to prevent data leakage during scaling and modeling.
* **Unsupervised Modeling & Diagnostics:**
  * **K-Means:** Evaluated via Elbow curves, Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Score.
  * **PCA:** Dimensionality reduction for visualization and cumulative variance analysis.
  * **DBSCAN:** Automated $k$-distance analysis for optimal $\epsilon$ selection and outlier extraction.

---

## 📂 Repository Structure

```text
.
├── Telangana_PDS_Analytics.ipynb   # Main end-to-end Jupyter Notebook
├── README.md                       # Project documentation
└── data/ (Google Drive Path)
    ├── raw/                        # Primary input CSV folders
    │   ├── table1_locations/       # Shop geospatial coordinates & district mappings
    │   ├── table2_card_status/     # Monthly active card tallies
    │   └── table3_transactions/    # Monthly itemized sales records
    ├── master/                     # Consolidated master Parquet tables
    ├── processed/                  # Feature matrices & cluster-annotated datasets
    ├── models/                     # Serialized Pipelines, K-Means & PCA models (.joblib)
    ├── figures/                    # Saved plots & diagnostic charts (.png)
    └── reports/                    # Generated profile CSV summaries (.csv)
