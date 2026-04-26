# 📦 Supply Chain Anomaly Detection System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Live-red)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange)

**Live Demo:** [Click here to view the live dashboard](https://supply-chain-anomaly-detection.streamlit.app/) 

## 📌 Project Overview
Supply chains are complex networks involving millions of moving parts. When shipments are severely delayed, unauthorized discounts are applied, or system glitches wipe out profits, it costs companies billions. 

This project is a **Digital AI Auditor** designed to automatically ingest supply chain data and flag bizarre anomalies. By utilizing unsupervised Machine Learning, it helps managers detect and investigate "hidden" problems before they cause massive financial loss.

## 🚀 Key Features
* **Multi-Model AI Engine:** Utilizes Isolation Forest, Local Outlier Factor (LOF), and One-Class SVM to benchmark anomaly detection.
* **Interactive Control Tower:** A sleek, dark-themed UI built with Streamlit.
* **6 Visual Analytics Dashboards:** Powered by Plotly, including financial impact scatter plots, anomaly timelines, and regional risk mapping.
* **Lightning Fast Processing:** Features an optimized sampling engine to instantly analyze large-scale datasets.
* **Downloadable Reports:** Managers can investigate the top 100 anomalous rows and export the filtered data to CSV.

## 🧠 Machine Learning Architecture
Instead of relying on basic rules (like "flag shipping > 10 days"), this system learns the mathematical "normal" shape of the data across 5 key features:
1. `Days for shipping (real)`
2. `Order Item Profit Ratio`
3. `Order Item Discount Rate`
4. `Order Item Total`
5. `Order Item Quantity`

The AI categorizes flagged anomalies into actionable buckets: **Shipping Delays, Profit Loss, High Discounts, or Data Mismatches.**

## 💻 How to Run Locally
If you want to run this application on your own machine:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yaduraj-ds/Supply-Chain-Anomaly-Detection.git](https://github.com/yaduraj-ds/Supply-Chain-Anomaly-Detection.git)
   cd Supply-Chain-Anomaly-Detection

2. **Install the required libraries:**
   ```bash
   pip install -r requirements.txt

3. **Launch the app:**
   ```bash
   streamlit run app.py

4. Upload the dataset: Drop your CSV file into the Data Ingestion Node and watch the AI run!
