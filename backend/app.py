from flask import Flask, render_template, jsonify, request, send_file
import pandas as pd
import joblib
import os
import sqlite3
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__, template_folder='../templates', static_folder='../static')
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Paths
MODEL_PATH = os.path.join(BASE_DIR, 'backend/model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'backend/scaler.pkl')
MAP_PATH = os.path.join(BASE_DIR, 'backend/persona_map.pkl')
CHURN_MODEL_PATH = os.path.join(BASE_DIR, 'backend/churn_model.pkl')
CHURN_SCALER_PATH = os.path.join(BASE_DIR, 'backend/churn_scaler.pkl')
DATA_PATH = os.path.join(BASE_DIR, 'data/rfm_segmented.csv')
DB_PATH = os.path.join(BASE_DIR, 'data/prism.db')

TRANSACTION_COLUMNS = {'InvoiceNo', 'StockCode', 'Description', 'Quantity', 'InvoiceDate', 'UnitPrice', 'CustomerID', 'Country'}
RFM_COLUMNS = {'CustomerID', 'Recency', 'Frequency', 'Monetary'}

# Operational Campaign Recommendations by Return Personas
CAMPAIGN_RECOMMENDATIONS = {
    "VIP Buyer": "Provide free return shipping, early VIP catalog access, and reward points to keep customer retention high.",
    "Serial Returner": "Restrict free return shipping. Apply a 10% restocking fee. Prompt dynamic sizing questionnaires during checkout.",
    "Low-Value Buyer": "Offer volume bundle discounts to increase Average Order Value; maintain standard, low-cost shipping policies.",
    "Unusual Activity Outlier": "Flag account for manual audit. Verify if purchase patterns suggest wholesale reseller behavior or automated bot testing."
}

# Persistent objects
model = None  # KNN out-of-sample classifier
scaler = None  # DBSCAN scaler
persona_map = None
churn_model = None  # Return Risk Classifier
churn_scaler = None  # Return Risk Scaler

def load_persistence():
    global model, scaler, persona_map, churn_model, churn_scaler
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
    if os.path.exists(MAP_PATH):
        persona_map = joblib.load(MAP_PATH)
    if os.path.exists(CHURN_MODEL_PATH) and os.path.exists(CHURN_SCALER_PATH):
        churn_model = joblib.load(CHURN_MODEL_PATH)
        churn_scaler = joblib.load(CHURN_SCALER_PATH)

load_persistence()

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def symmetric_log1p(x):
    """Symmetric log transform to handle positive and negative net spend values."""
    return np.sign(x) * np.log1p(np.abs(x))

def save_uploaded_data_to_db(raw_df, rfm_df, is_precomputed=False):
    """Saves raw transaction history and calculated customer profiles to SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    schema_path = os.path.join(BASE_DIR, 'backend/schema.sql')
    with open(schema_path, 'r') as f:
        schema_sql = f.read()
    conn.executescript(schema_sql)
    
    conn.execute("DELETE FROM customers")
    if not is_precomputed and raw_df is not None:
        conn.execute("DELETE FROM transactions")
        tx_df = raw_df[['InvoiceNo', 'StockCode', 'Description', 'Quantity', 'InvoiceDate', 'UnitPrice', 'CustomerID', 'Country']].copy()
        tx_df['InvoiceDate'] = pd.to_datetime(tx_df['InvoiceDate']).dt.strftime('%Y-%m-%d %H:%M:%S')
        tx_df.to_sql('transactions', conn, if_exists='append', index=False)
        
    cust_df = rfm_df[['CustomerID', 'Recency', 'Frequency', 'Monetary', 'ReturnRate', 'Cluster', 'Persona', 'ReturnProbability', 'ReturnRisk']].copy()
    cust_df.to_sql('customers', conn, if_exists='append', index=False)
    
    conn.commit()
    conn.close()

def _normalize_columns(df):
    df = df.copy()
    df.columns = df.columns.str.strip()
    return df

def _cluster_rfm(rfm, raw_df=None):
    if rfm.empty:
        raise ValueError("No valid customer rows found after cleaning the CSV.")

    # Apply DBSCAN logic
    rfm_log = rfm[['Recency', 'Frequency', 'Monetary', 'ReturnRate']].copy()
    rfm_log['Recency'] = np.log1p(rfm_log['Recency'])
    rfm_log['Frequency'] = np.log1p(rfm_log['Frequency'])
    rfm_log['Monetary'] = symmetric_log1p(rfm_log['Monetary'])

    scaler_new = StandardScaler()
    scaled_data = scaler_new.fit_transform(rfm_log)

    dbscan_new = DBSCAN(eps=0.6, min_samples=5)
    rfm['Cluster'] = dbscan_new.fit_predict(scaled_data)

    # Dynamic Persona mapping
    cluster_stats = rfm.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean',
        'ReturnRate': 'mean'
    })

    new_map = {}
    main_clusters = [c for c in cluster_stats.index if c != -1]
    
    if -1 in cluster_stats.index:
        new_map[-1] = "Unusual Activity Outlier"

    if main_clusters:
        return_sorted = cluster_stats.loc[main_clusters].sort_values(by='ReturnRate', ascending=False)
        serial_returner_cluster = return_sorted.index[0]
        new_map[int(serial_returner_cluster)] = "Serial Returner"
        
        remaining_clusters = [c for c in main_clusters if c != serial_returner_cluster]
        if remaining_clusters:
            monetary_sorted = cluster_stats.loc[remaining_clusters].sort_values(by='Monetary', ascending=False)
            vip_cluster = monetary_sorted.index[0]
            new_map[int(vip_cluster)] = "VIP Buyer"
            
            low_value_clusters = [c for c in remaining_clusters if c != vip_cluster]
            for c in low_value_clusters:
                new_map[int(c)] = "Low-Value Buyer"
        else:
            new_map[main_clusters[0]] = "Standard Buyer"
    else:
        new_map[0] = "Standard Buyer"

    rfm['Persona'] = rfm['Cluster'].map(new_map)

    # Train out-of-sample KNN classifier
    knn_new = KNeighborsClassifier(n_neighbors=3)
    knn_new.fit(scaled_data, rfm['Cluster'])

    # Save artifacts
    joblib.dump(knn_new, MODEL_PATH)
    joblib.dump(scaler_new, SCALER_PATH)
    joblib.dump(new_map, MAP_PATH)

    # Precompute return risk classifier
    if churn_model and churn_scaler:
        X_pred = rfm[['Frequency', 'Monetary']]
        X_pred_scaled = churn_scaler.transform(X_pred)
        rfm['ReturnProbability'] = churn_model.predict_proba(X_pred_scaled)[:, 1]
        pred_risk = churn_model.predict(X_pred_scaled)
        rfm['ReturnRisk'] = np.where(pred_risk == 1, 'High', 'Low')
    else:
        rfm['ReturnProbability'] = 0.0
        rfm['ReturnRisk'] = 'Low'

    if 'CustomerID' not in rfm.columns:
        rfm = rfm.reset_index()

    # Save to SQLite & CSV
    save_uploaded_data_to_db(raw_df, rfm, is_precomputed=(raw_df is None))
    rfm.to_csv(DATA_PATH, index=False)
    
    load_persistence()
    return rfm

def process_transactions(df):
    df = _normalize_columns(df)
    missing = sorted(TRANSACTION_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"Missing required transaction columns: {', '.join(missing)}")

    df = df.dropna(subset=['CustomerID']).copy()
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
    df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
    df['UnitPrice'] = pd.to_numeric(df['UnitPrice'], errors='coerce')
    df['CustomerID'] = pd.to_numeric(df['CustomerID'], errors='coerce')
    df = df.dropna(subset=['InvoiceDate', 'Quantity', 'UnitPrice', 'CustomerID'])
    df['CustomerID'] = df['CustomerID'].astype(int)
    
    # Calculate Total Price (refunds will naturally be negative)
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

    # Calculate metrics at customer level
    purchases = df[df['Quantity'] > 0].copy()
    returns = df[df['Quantity'] < 0].copy()

    customer_purchases = purchases.groupby('CustomerID').agg({
        'InvoiceNo': 'nunique',
        'TotalPrice': 'sum',
        'Quantity': 'sum'
    }).rename(columns={'InvoiceNo': 'Frequency', 'TotalPrice': 'Monetary', 'Quantity': 'PurchaseQty'})

    customer_returns = returns.groupby('CustomerID').agg({
        'TotalPrice': 'sum',
        'Quantity': 'sum'
    }).rename(columns={'TotalPrice': 'ReturnMonetary', 'Quantity': 'ReturnQty'})

    rfm = pd.DataFrame(index=df['CustomerID'].unique())
    rfm = rfm.join(customer_purchases).join(customer_returns)
    rfm.index.name = 'CustomerID'
    rfm.reset_index(inplace=True)
    rfm.fillna(0.0, inplace=True)

    rfm['ReturnQty'] = rfm['ReturnQty'].abs()
    rfm['ReturnMonetary'] = rfm['ReturnMonetary'].abs()
    rfm['Monetary'] = rfm['Monetary'] - rfm['ReturnMonetary']

    rfm['ReturnRate'] = np.where(
        rfm['PurchaseQty'] > 0,
        rfm['ReturnQty'] / (rfm['PurchaseQty'] + rfm['ReturnQty']),
        0.0
    )
    rfm['ReturnRate'] = rfm['ReturnRate'].clip(upper=1.0)

    snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    recency = purchases.groupby('CustomerID')['InvoiceDate'].max().apply(lambda x: (snapshot_date - x).days)
    rfm = rfm.merge(recency.rename('Recency'), on='CustomerID', how='left')
    rfm['Recency'] = rfm['Recency'].fillna(365.0)

    return _cluster_rfm(rfm, raw_df=df)

def process_precomputed_rfm(df):
    df = _normalize_columns(df)
    # Check for ReturnRate, otherwise default it
    if 'ReturnRate' not in df.columns:
        df['ReturnRate'] = 0.0

    rfm = df[['CustomerID', 'Recency', 'Frequency', 'Monetary', 'ReturnRate']].copy()
    for col in ['CustomerID', 'Recency', 'Frequency', 'Monetary', 'ReturnRate']:
        rfm[col] = pd.to_numeric(rfm[col], errors='coerce')
    rfm = rfm.dropna(subset=['CustomerID', 'Recency', 'Frequency', 'Monetary'])
    rfm['CustomerID'] = rfm['CustomerID'].astype(int)
    rfm = rfm.groupby('CustomerID', as_index=False).agg({
        'Recency': 'min',
        'Frequency': 'sum',
        'Monetary': 'sum',
        'ReturnRate': 'mean'
    })
    return _cluster_rfm(rfm, raw_df=None)

def process_rfm(df):
    """Accept raw transaction data or a precomputed CustomerID/RFM CSV."""
    normalized = _normalize_columns(df)
    if RFM_COLUMNS.issubset(normalized.columns):
        return process_precomputed_rfm(normalized)
    return process_transactions(normalized)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/sample_csv')
def sample_csv():
    return send_file(
        os.path.join(BASE_DIR, 'data', 'user_test.csv'),
        mimetype='text/csv',
        as_attachment=True,
        download_name='prism_sample_transactions.csv'
    )

@app.route('/upload', methods=['POST'])
def upload_data():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file and file.filename.lower().endswith('.csv'):
        try:
            df = pd.read_csv(file, encoding='ISO-8859-1')
            rfm = process_rfm(df)
            return jsonify({
                "message": "File processed successfully",
                "customers": int(rfm['CustomerID'].nunique())
            })
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return jsonify({"error": "Invalid file type"}), 400

@app.route('/segment')
def get_segments():
    if not os.path.exists(DB_PATH) or persona_map is None:
        return jsonify({"error": "empty_state"}), 200
    
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT CustomerID, Recency, Frequency, Monetary, ReturnRate, Cluster, Persona FROM customers")
        rows = cursor.fetchall()
        data = [dict(row) for row in rows]
        if not data:
            return jsonify({"error": "empty_state"}), 200
        return jsonify({
            "data": data,
            "persona_map": {str(k): v for k, v in persona_map.items()}
        })
    except sqlite3.OperationalError:
        return jsonify({"error": "empty_state"}), 200
    finally:
        conn.close()

@app.route('/metrics')
def get_metrics():
    if not os.path.exists(DB_PATH):
        return jsonify({
            "total_customers": 0,
            "total_revenue": 0,
            "avg_recency": 0,
            "avg_frequency": 0,
            "avg_monetary": 0,
            "avg_return_rate": 0,
            "restocking_cost": 0
        })
    
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT 
                COUNT(CustomerID) as total_customers,
                SUM(Monetary) as net_revenue,
                AVG(Recency) as avg_recency,
                AVG(Frequency) as avg_frequency,
                AVG(ReturnRate) as avg_return_rate
            FROM customers
        """)
        row = cursor.fetchone()
        
        # Calculate restocking overhead cost (15% processing fee on returns value)
        cursor.execute("SELECT SUM(ABS(Quantity * UnitPrice)) FROM transactions WHERE Quantity < 0")
        returned_val_row = cursor.fetchone()
        returned_value = float(returned_val_row[0] or 0.0)
        restocking_cost = returned_value * 0.15
        
        if row and row['total_customers'] > 0:
            metrics = {
                "total_customers": int(row['total_customers']),
                "total_revenue": float(row['net_revenue'] or 0),
                "avg_recency": float(row['avg_recency'] or 0),
                "avg_frequency": float(row['avg_frequency'] or 0),
                "avg_monetary": float(row['net_revenue']/row['total_customers'] if row['net_revenue'] else 0),
                "avg_return_rate": float(row['avg_return_rate'] or 0) * 100.0, # percentage
                "restocking_cost": restocking_cost
            }
        else:
            metrics = {
                "total_customers": 0,
                "total_revenue": 0,
                "avg_recency": 0,
                "avg_frequency": 0,
                "avg_monetary": 0,
                "avg_return_rate": 0,
                "restocking_cost": 0
            }
        return jsonify(metrics)
    except sqlite3.OperationalError:
        return jsonify({
            "total_customers": 0,
            "total_revenue": 0,
            "avg_recency": 0,
            "avg_frequency": 0,
            "avg_monetary": 0,
            "avg_return_rate": 0,
            "restocking_cost": 0
        })
    finally:
        conn.close()

@app.route('/predict', methods=['POST'])
def predict_persona():
    if not scaler or not model or not persona_map:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        data = request.json
        recency = float(data['recency'])
        frequency = float(data['frequency'])
        monetary = float(data['monetary'])
        return_rate = float(data.get('return_rate', 0.0)) / 100.0
        
        # Preprocess using same pipeline
        input_df = pd.DataFrame([[recency, frequency, monetary, return_rate]], columns=['Recency', 'Frequency', 'Monetary', 'ReturnRate'])
        input_log = input_df.copy()
        input_log['Recency'] = np.log1p(input_log['Recency'])
        input_log['Frequency'] = np.log1p(input_log['Frequency'])
        input_log['Monetary'] = symmetric_log1p(input_log['Monetary'])
        
        input_scaled = scaler.transform(input_log)
        
        # Predict using KNN classifier trained on DBSCAN labels
        cluster = int(model.predict(input_scaled)[0])
        persona = persona_map.get(cluster, "Unknown")
        
        return jsonify({"persona": persona})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/predict_churn', methods=['POST'])
def predict_churn():
    """Predicts next-order return risk probability using our logistic regression classifier."""
    if not churn_model or not churn_scaler:
        return jsonify({"error": "Return risk model not loaded"}), 500
    
    try:
        data = request.json
        frequency = float(data['frequency'])
        monetary = float(data['monetary'])
        
        input_df = pd.DataFrame([[frequency, monetary]], columns=['Frequency', 'Monetary'])
        input_scaled = churn_scaler.transform(input_df)
        
        prediction = int(churn_model.predict(input_scaled)[0])
        probability = float(churn_model.predict_proba(input_scaled)[0][1])
        
        return jsonify({
            "churn_prediction": prediction,
            "churn_probability": probability
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/customer_exists/<int:customer_id>')
def customer_exists(customer_id):
    """
    Checks if a customer ID exists in the SQLite database.
    """
    if not os.path.exists(DB_PATH):
        return jsonify({"exists": False, "error": "Database not initialized"}), 404
        
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT 1 FROM customers WHERE CustomerID = ?", (customer_id,))
        row = cursor.fetchone()
        return jsonify({"exists": row is not None})
    except Exception as e:
        return jsonify({"exists": False, "error": str(e)}), 500
    finally:
        conn.close()

@app.route('/customer/<int:customer_id>')
def customer_profile(customer_id):
    """
    Displays a detailed profile for a single customer by querying SQLite.
    """
    if not os.path.exists(DB_PATH):
        return "Database not initialized. Please run the training pipeline.", 404
        
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT * FROM customers WHERE CustomerID = ?", (customer_id,))
        customer_row = cursor.fetchone()
        if not customer_row:
            return "Customer not found.", 404
            
        customer_rfm = dict(customer_row)
        persona = customer_rfm.get('Persona', 'Unknown')
        return_risk = customer_rfm.get('ReturnRisk', 'Unknown')
        return_prob_percent = round(customer_rfm.get('ReturnProbability', 0) * 100, 1)
        net_monetary = customer_rfm.get('Monetary', 0.0)
        return_rate_percent = round(customer_rfm.get('ReturnRate', 0) * 100, 1)
        
        # Pull return-related recommendation
        campaign_recommendation = CAMPAIGN_RECOMMENDATIONS.get(
            persona, "Monitor return behavior and ensure standard shipping policies apply."
        )
        
        # Get transaction logs
        cursor.execute("""
            SELECT InvoiceNo, InvoiceDate, StockCode, Description, Quantity, UnitPrice, (Quantity * UnitPrice) as TotalPrice
            FROM transactions 
            WHERE CustomerID = ? 
            ORDER BY InvoiceDate DESC
        """, (customer_id,))
        transactions = [dict(r) for r in cursor.fetchall()]
        
        return render_template(
            'customer_profile.html',
            customer_id=customer_id,
            persona=persona,
            return_risk=return_risk,
            return_probability=return_prob_percent,
            net_monetary=f"{net_monetary:,.2f}",
            return_rate=return_rate_percent,
            campaign_recommendation=campaign_recommendation,
            transactions=transactions
        )
    except sqlite3.OperationalError as e:
        return f"Database error: {str(e)}", 500
    finally:
        conn.close()

if __name__ == '__main__':
    app.run(debug=True, port=5001)
