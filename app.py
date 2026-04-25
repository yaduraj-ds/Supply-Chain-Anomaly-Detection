import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# Setup web page layout
st.set_page_config(page_title="Supply Chain AI", layout="wide")
st.title("📦 Supply Chain Anomaly Dashboard")

# --- Project Overview Cards ---
st.markdown("<br>", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

with c1:
    with st.container(border=True):
        st.markdown("#### 📦 What is a Supply Chain?")
        st.write("A supply chain is the entire journey a product takes to get from the factory to your front door. It involves manufacturing, warehouses, transportation, and delivery. When it works perfectly, you get your items on time and at the intended price.")

with c2:
    with st.container(border=True):
        st.markdown("#### ⚠️ The Hidden Problems")
        st.write("Because this journey involves millions of moving parts, things often break. Shipments get severely delayed, massive unauthorized discounts are applied, or system glitches wipe out profits. These hidden errors cost companies billions of dollars every year.")

with c3:
    with st.container(border=True):
        st.markdown("#### 🤖 What Our AI System Does")
        st.write("Our platform acts as a smart digital auditor. It uses Machine Learning to instantly scan thousands of supply chain records. It automatically hunts down and flags bizarre anomalies so managers can fix the problems before the company loses money.")

st.markdown("<br>", unsafe_allow_html=True)

# Load trained AI models
@st.cache_resource
def load_models():
    return (joblib.load('models/scaler.joblib'), joblib.load('models/iso_forest.joblib'),
            joblib.load('models/lof.joblib'), joblib.load('models/svm.joblib'))

scaler, iso_forest, lof, svm = load_models()

# --- Sleek Data Ingestion Zone ---
st.markdown("<br>", unsafe_allow_html=True) # Adds a little breathing room

with st.container(border=True):
    st.markdown("### 📥 Data Ingestion Node")
    st.markdown("Upload your raw supply chain records to begin the AI benchmarking process.")
    
    uploaded_file = st.file_uploader(
        "Drag and drop your CSV file here", 
        type=['csv'],
        help="Ensure your dataset contains the standard supply chain columns."
    )
st.markdown("<br>", unsafe_allow_html=True)

if uploaded_file:
    # Read the uploaded CSV
    df = pd.read_csv(uploaded_file, encoding='latin1')
    
    # Select features and clean data
    features = ['Days for shipping (real)', 'Order Item Profit Ratio', 
                'Order Item Discount Rate', 'Order Item Total', 'Order Item Quantity']
    df_clean = df.dropna(subset=features).copy()
    
    # Scale data and run models
    scaled_data = scaler.transform(df_clean[features])
    
    # Predict anomalies (-1 = Anomaly, 1 = Normal)
    df_clean['ISOLATION_FOREST'] = ['Anomaly' if x == -1 else 'Normal' for x in iso_forest.predict(scaled_data)]
    df_clean['LOF'] = ['Anomaly' if x == -1 else 'Normal' for x in lof.predict(scaled_data)]
    df_clean['SVM'] = ['Anomaly' if x == -1 else 'Normal' for x in svm.predict(scaled_data)]
    
    # Assign anomaly categories based on logic
    def get_anomaly_type(row):
        if 'Anomaly' in [row['ISOLATION_FOREST'], row['LOF'], row['SVM']]:
            if row['Days for shipping (real)'] > 6: return 'Shipping Delay'
            if row['Order Item Profit Ratio'] < 0: return 'Profit Loss'
            if row['Order Item Discount Rate'] > 0.20: return 'High Discount'
            return 'Data Mismatch'
        return 'Normal'

    df_clean['ANOMALY_TYPE'] = df_clean.apply(get_anomaly_type, axis=1)
    
    # Filter to only show anomalies for charts
    anomalies_only = df_clean[df_clean['ANOMALY_TYPE'] != 'Normal']

    # --- Dashboard KPIs ---
    st.divider()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Shipments", len(df_clean))
    col2.metric("Total Anomalies", len(anomalies_only))
    
    # Calculate estimated loss
    loss = anomalies_only[anomalies_only['Order Item Profit Ratio'] < 0]['Order Item Total'].sum()
    col3.metric("Estimated Loss", f"${loss:,.2f}")
    
    # Calculate average delay
    avg_delay = anomalies_only[anomalies_only['ANOMALY_TYPE'] == 'Shipping Delay']['Days for shipping (real)'].mean()
    col4.metric("Avg Delay (Days)", f"{avg_delay:.1f}" if pd.notna(avg_delay) else "0")

    # # --- 6 Interactive Charts ---
    # st.subheader("📊 Visual Analytics")
    # r1c1, r1c2 = st.columns(2)
    # r2c1, r2c2 = st.columns(2)
    # r3c1, r3c2 = st.columns(2)

    # # Build the 6 charts
    # r1c1.plotly_chart(px.pie(anomalies_only, names='ANOMALY_TYPE', title="1. Anomaly Breakdown", hole=0.4), use_container_width=True)
    # r1c2.plotly_chart(px.histogram(anomalies_only, x='Order Region', title="2. Risk by Region"), use_container_width=True)
    # r2c1.plotly_chart(px.scatter(df_clean, x='Order Item Total', y='Order Item Profit Ratio', color='ANOMALY_TYPE', title="3. Financial Impact"), use_container_width=True)
    # r2c2.plotly_chart(px.histogram(anomalies_only, x='order date (DateOrders)', title="4. Anomalies Over Time"), use_container_width=True)
    # r3c1.plotly_chart(px.histogram(anomalies_only, x='Shipping Mode', color='ANOMALY_TYPE', title="5. Shipping Mode Risk"), use_container_width=True)
    
    # # Top 10 categories
    # top_cats = anomalies_only['Category Name'].value_counts().head(10).reset_index()
    # top_cats.columns = ['Category', 'Count']
    # r3c2.plotly_chart(px.bar(top_cats, x='Count', y='Category', orientation='h', title="6. Top Risky Categories"), use_container_width=True)

    # --- 6 Interactive Charts ---
    st.subheader("📊 Visual Analytics")
    r1c1, r1c2 = st.columns(2)
    r2c1, r2c2 = st.columns(2)
    r3c1, r3c2 = st.columns(2)

    # Custom Cyberpunk Color Palette
    color_map = {
        'Normal': '#2A2E39',          # Muted Dark Grey
        'Shipping Delay': '#FF2B2B',  # Neon Red
        'Profit Loss': '#00FFAA',     # Neon Mint
        'High Discount': '#FFD700',   # Cyber Gold
        'Data Mismatch': '#FF00FF'    # Neon Magenta
    }

    # 1. Anomaly Breakdown (Donut)
    fig1 = px.pie(anomalies_only, names='ANOMALY_TYPE', title="1. Anomaly Breakdown", hole=0.4, 
                  color='ANOMALY_TYPE', color_discrete_map=color_map, template='plotly_dark')
    r1c1.plotly_chart(fig1, use_container_width=True)

    # 2. Risk by Region (Bar)
    fig2 = px.histogram(anomalies_only, x='Order Region', title="2. Risk by Region", 
                        color='ANOMALY_TYPE', color_discrete_map=color_map, template='plotly_dark')
    r1c2.plotly_chart(fig2, use_container_width=True)

    # 3. Financial Impact (Scatter)
    fig3 = px.scatter(df_clean, x='Order Item Total', y='Order Item Profit Ratio', 
                      color='ANOMALY_TYPE', color_discrete_map=color_map, template='plotly_dark',
                      title="3. Financial Impact")
    r2c1.plotly_chart(fig3, use_container_width=True)

    # 4. Anomalies Over Time (Line)
    fig4 = px.histogram(anomalies_only, x='order date (DateOrders)', title="4. Anomalies Over Time", 
                        color='ANOMALY_TYPE', color_discrete_map=color_map, template='plotly_dark')
    r2c2.plotly_chart(fig4, use_container_width=True)

    # 5. Shipping Mode Risk (Stacked Bar)
    fig5 = px.histogram(anomalies_only, x='Shipping Mode', color='ANOMALY_TYPE', 
                        color_discrete_map=color_map, title="5. Shipping Mode Risk", template='plotly_dark')
    r3c1.plotly_chart(fig5, use_container_width=True)
    
    # 6. Top Risky Categories (Horizontal Bar)
    top_cats = anomalies_only['Category Name'].value_counts().head(10).reset_index()
    top_cats.columns = ['Category', 'Count']
    fig6 = px.bar(top_cats, x='Count', y='Category', orientation='h', title="6. Top Risky Categories", 
                  template='plotly_dark', color_discrete_sequence=['#FF2B2B'])
    r3c2.plotly_chart(fig6, use_container_width=True)

    # --- Clean Data Table ---
    st.divider()
    st.subheader("🔍 Filter & Investigate")
    
    # Dropdown to pick model
    selected_model = st.selectbox("Select AI Model Results:", ["ISOLATION_FOREST", "LOF", "SVM"])
    
    # Keep only the 14 columns
    final_cols = ['Order Id', 'order date (DateOrders)', 'Category Name', 'Product Name', 
                  'Order Region', 'Shipping Mode', 'Order Item Total', 'Order Item Profit Ratio', 
                  'Order Item Discount Rate', 'Days for shipping (real)', 
                  'ISOLATION_FOREST', 'LOF', 'SVM', 'ANOMALY_TYPE']
    
    # Show top 100 bad rows
    bad_rows = df_clean[df_clean[selected_model] == 'Anomaly'][final_cols]
    st.dataframe(bad_rows.head(100))

    # Download Report button
    csv_data = bad_rows.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Filtered Report", csv_data, "Anomaly_Report.csv", "text/csv")