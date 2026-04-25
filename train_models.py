import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
import joblib
import os

# Load dataset from data folder
df = pd.read_csv('data/DataCoSupplyChainDataset.csv', encoding='latin1')

# Select numeric columns for anomalies
features = ['Days for shipping (real)', 'Order Item Profit Ratio', 
            'Order Item Discount Rate', 'Order Item Total', 'Order Item Quantity']

# Drop rows with missing values
df_clean = df[features].dropna()

# Scale data to uniform variance
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_clean)

# Set 5% anomaly contamination rate
contamination_rate = 0.05

# Initialize the three ML models
iso_forest = IsolationForest(contamination=contamination_rate, random_state=42)
lof = LocalOutlierFactor(contamination=contamination_rate, novelty=True)
svm = OneClassSVM(nu=contamination_rate)

# Train models on scaled data
iso_forest.fit(scaled_data)
lof.fit(scaled_data)
svm.fit(scaled_data)

# Create models folder if missing
os.makedirs('models', exist_ok=True)

# Save trained models and scaler
joblib.dump(scaler, 'models/scaler.joblib')
joblib.dump(iso_forest, 'models/iso_forest.joblib')
joblib.dump(lof, 'models/lof.joblib')
joblib.dump(svm, 'models/svm.joblib')

print("Models trained and saved successfully!")