import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os
import json
import sqlite3
from datetime import datetime

# --- Configuration ---
RAW_DATA_PATH = 'data/online_retail.csv'
SEGMENTED_DATA_PATH = 'data/rfm_segmented.csv'
MODEL_DIR = 'backend'
LOG_FILE = 'training_log.json'
DB_PATH = 'data/prism.db'
SCHEMA_PATH = 'backend/schema.sql'

def print_header(title):
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def log_results(log_data):
    with open(LOG_FILE, 'w') as f:
        json.dump(log_data, f, indent=4)
    print(f"\n[SUCCESS] Training log saved to {LOG_FILE}")

def symmetric_log1p(x):
    """Symmetric log transform to handle positive and negative net spend skewness."""
    return np.sign(x) * np.log1p(np.abs(x))

def save_to_sqlite(raw_df, rfm_df):
    print("\n[DATABASE] Saving data to SQLite database...")
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    
    with open(SCHEMA_PATH, 'r') as f:
        schema_sql = f.read()
    conn.executescript(schema_sql)
    
    print("  - Writing transactions table...")
    conn.execute("DELETE FROM transactions")
    tx_df = raw_df[['InvoiceNo', 'StockCode', 'Description', 'Quantity', 'InvoiceDate', 'UnitPrice', 'CustomerID', 'Country']].copy()
    tx_df['InvoiceDate'] = tx_df['InvoiceDate'].dt.strftime('%Y-%m-%d %H:%M:%S')
    tx_df.to_sql('transactions', conn, if_exists='append', index=False)
    
    print("  - Writing customers table...")
    conn.execute("DELETE FROM customers")
    
    cust_df = rfm_df[['CustomerID', 'Recency', 'Frequency', 'Monetary', 'ReturnRate', 'Cluster', 'Persona', 'ReturnProbability', 'ReturnRisk']].copy()
    cust_df.to_sql('customers', conn, if_exists='append', index=False)
    
    conn.commit()
    conn.close()
    print("[SUCCESS] Data saved to SQLite database successfully.")

def step_1_run_segmentation(df):
    """
    Groups customer transactions, calculates return rates, and applies DBSCAN clustering.
    Trains a KNN classifier on top of DBSCAN to handle out-of-sample predictions.
    """
    print_header("Step 1: Running Customer Segmentation (DBSCAN)")

    print("  - Separating purchases and returns...")
    purchases = df[df['Quantity'] > 0].copy()
    returns = df[df['Quantity'] < 0].copy()

    print("  - Aggregating customer purchase profiles...")
    customer_purchases = purchases.groupby('CustomerID').agg({
        'InvoiceNo': 'nunique',
        'TotalPrice': 'sum',
        'Quantity': 'sum'
    }).rename(columns={'InvoiceNo': 'Frequency', 'TotalPrice': 'Monetary', 'Quantity': 'PurchaseQty'})

    print("  - Aggregating customer return profiles...")
    customer_returns = returns.groupby('CustomerID').agg({
        'TotalPrice': 'sum',
        'Quantity': 'sum'
    }).rename(columns={'TotalPrice': 'ReturnMonetary', 'Quantity': 'ReturnQty'})

    # Build base customer metrics sheet
    customer_metrics = pd.DataFrame(index=df['CustomerID'].unique())
    customer_metrics = customer_metrics.join(customer_purchases).join(customer_returns)
    customer_metrics.index.name = 'CustomerID'
    customer_metrics.reset_index(inplace=True)
    customer_metrics.fillna(0.0, inplace=True)

    # Convert quantities and monetary values to absolute positive numbers
    customer_metrics['ReturnQty'] = customer_metrics['ReturnQty'].abs()
    customer_metrics['ReturnMonetary'] = customer_metrics['ReturnMonetary'].abs()

    # Calculate net monetary (purchased monetary minus returned monetary)
    customer_metrics['Monetary'] = customer_metrics['Monetary'] - customer_metrics['ReturnMonetary']

    # Calculate return rate
    customer_metrics['ReturnRate'] = np.where(
        customer_metrics['PurchaseQty'] > 0,
        customer_metrics['ReturnQty'] / (customer_metrics['PurchaseQty'] + customer_metrics['ReturnQty']),
        0.0
    )
    customer_metrics['ReturnRate'] = customer_metrics['ReturnRate'].clip(upper=1.0)

    # Calculate Recency based on last purchase date
    snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    recency = purchases.groupby('CustomerID')['InvoiceDate'].max().apply(lambda x: (snapshot_date - x).days)
    customer_metrics = customer_metrics.merge(recency.rename('Recency'), on='CustomerID', how='left')
    customer_metrics['Recency'] = customer_metrics['Recency'].fillna(365.0)

    # Preprocessing log transforms
    print("  - Applying symmetric log transforms and scaling...")
    rfm_log = customer_metrics[['Recency', 'Frequency', 'Monetary', 'ReturnRate']].copy()
    rfm_log['Recency'] = np.log1p(rfm_log['Recency'])
    rfm_log['Frequency'] = np.log1p(rfm_log['Frequency'])
    rfm_log['Monetary'] = symmetric_log1p(rfm_log['Monetary'])
    # ReturnRate remains unlogged as it is already bounded [0, 1]

    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_log)

    print("  - Running DBSCAN clustering...")
    dbscan = DBSCAN(eps=0.6, min_samples=5)
    customer_metrics['Cluster'] = dbscan.fit_predict(rfm_scaled)

    # Generate dynamic persona map based on cluster metrics
    print("  - Generating dynamic persona map...")
    cluster_stats = customer_metrics.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean',
        'ReturnRate': 'mean'
    })

    persona_map = {}
    main_clusters = [c for c in cluster_stats.index if c != -1]
    
    if -1 in cluster_stats.index:
        persona_map[-1] = "Unusual Activity Outlier"

    if main_clusters:
        # Find cluster with highest average return rate
        return_sorted = cluster_stats.loc[main_clusters].sort_values(by='ReturnRate', ascending=False)
        serial_returner_cluster = return_sorted.index[0]
        persona_map[int(serial_returner_cluster)] = "Serial Returner"
        
        remaining_clusters = [c for c in main_clusters if c != serial_returner_cluster]
        if remaining_clusters:
            # Find cluster with highest net monetary spend
            monetary_sorted = cluster_stats.loc[remaining_clusters].sort_values(by='Monetary', ascending=False)
            vip_cluster = monetary_sorted.index[0]
            persona_map[int(vip_cluster)] = "VIP Buyer"
            
            low_value_clusters = [c for c in remaining_clusters if c != vip_cluster]
            for c in low_value_clusters:
                persona_map[int(c)] = "Low-Value Buyer"
        else:
            persona_map[main_clusters[0]] = "Standard Buyer"
    else:
        # Fallback if DBSCAN groups everything into noise
        persona_map[0] = "Standard Buyer"

    # Train a KNN classifier to predict DBSCAN cluster for new points (handles out-of-sample inputs)
    print("  - Training KNN classifier for out-of-sample cluster prediction...")
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(rfm_scaled, customer_metrics['Cluster'])

    # Save artifacts
    print("  - Saving segmentation artifacts...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(knn, os.path.join(MODEL_DIR, 'model.pkl'))
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))
    joblib.dump(persona_map, os.path.join(MODEL_DIR, 'persona_map.pkl'))
    customer_metrics.to_csv(SEGMENTED_DATA_PATH, index=False)
    
    print("[SUCCESS] Step 1 complete.")
    return customer_metrics, persona_map

def step_2_run_return_prediction(customer_metrics):
    """
    Trains a Logistic Regression model to predict if a customer has a return risk (ReturnRate > 15%).
    Saves the return risk model and its scaler.
    """
    print_header("Step 2: Running Return Prediction (Logistic Regression)")

    # Define target label
    customer_metrics['ReturnRiskLabel'] = np.where(customer_metrics['ReturnRate'] > 0.15, 1, 0)
    print(f"  - Return Risk defined as ReturnRate > 15%.")

    # Features (excluding ReturnRate to prevent data leakage)
    features = ['Frequency', 'Monetary']
    X = customer_metrics[features]
    y = customer_metrics['ReturnRiskLabel']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Scaling
    print("  - Scaling features...")
    churn_scaler = StandardScaler()
    X_train_scaled = churn_scaler.fit_transform(X_train)
    X_test_scaled = churn_scaler.transform(X_test)

    # Model Training
    print("  - Training Logistic Regression classifier...")
    churn_model = LogisticRegression(random_state=42, class_weight='balanced')
    churn_model.fit(X_train_scaled, y_train)

    # Evaluation
    y_pred = churn_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"  - Return Risk Model Accuracy: {accuracy:.2f}")

    # Save artifacts
    print("  - Saving return model artifacts...")
    joblib.dump(churn_model, os.path.join(MODEL_DIR, 'churn_model.pkl'))
    joblib.dump(churn_scaler, os.path.join(MODEL_DIR, 'churn_scaler.pkl'))
    
    print("[SUCCESS] Step 2 complete.")
    return churn_model, churn_scaler, accuracy

def main():
    print_header("PRISM RELOG Training Pipeline Started")
    
    print(f"  - Loading raw data from {RAW_DATA_PATH}...")
    try:
        df = pd.read_csv(RAW_DATA_PATH, encoding='ISO-8859-1')
        df = df.dropna(subset=['CustomerID'])
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
        df = df.dropna(subset=['InvoiceDate'])
        df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    except FileNotFoundError:
        print(f"[ERROR] Raw data file not found at {RAW_DATA_PATH}. Aborting.")
        return

    # Run steps
    customer_metrics, persona_map = step_1_run_segmentation(df.copy())
    churn_model, churn_scaler, model_accuracy = step_2_run_return_prediction(customer_metrics)

    # Precompute risk probabilities for database
    print("\n[PREPROCESSING] Precomputing return risk attributes for database...")
    customer_metrics['Persona'] = customer_metrics['Cluster'].map(persona_map)
    
    X_pred = customer_metrics[['Frequency', 'Monetary']]
    X_pred_scaled = churn_scaler.transform(X_pred)
    customer_metrics['ReturnProbability'] = churn_model.predict_proba(X_pred_scaled)[:, 1]
    
    pred_risk = churn_model.predict(X_pred_scaled)
    customer_metrics['ReturnRisk'] = np.where(pred_risk == 1, 'High', 'Low')

    # Save data to SQLite database
    save_to_sqlite(df, customer_metrics)

    # Save log
    training_log = {
        "last_trained_timestamp": datetime.now().isoformat(),
        "models_trained": ["DBSCAN_Clustering", "KNN_Out_Of_Sample", "LogisticRegression_ReturnRisk"],
        "return_risk_model_accuracy": round(model_accuracy, 4)
    }
    log_results(training_log)
    
    print_header("PRISM RELOG Training Pipeline Finished Successfully")

if __name__ == "__main__":
    main()
