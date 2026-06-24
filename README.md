# RELOG: Reverse Logistics & Purchase Return Risk Engine

RELOG (formerly PRISM) is a production-grade machine learning application and interactive analytics dashboard designed to optimize e-commerce margins by auditing customer purchase return behaviors. 

Standard customer segmentation dashboards rely on classical RFM (Recency, Frequency, Monetary) clustering, completely overlooking return/refund transactions which drain up to 30% of online retail margins. RELOG introduces **RFM-R (Recency, Frequency, Monetary, Return Rate)** behavioral feature engineering, utilizing density-based clustering, supervised predictive wrappers, and sub-millisecond SQLite database lookups.

---

## 🚀 Key Business & Engineering Features

- **RFM-R Feature Engineering**: Ingests raw sales transaction histories (including returns/refunds indicated by negative quantities) to calculate customer-level purchase pacing, net spend values, and return ratios.
- **DBSCAN Density-Based Clustering**: Replaces simplistic K-Means with DBSCAN (`eps=0.6`, `min_samples=5`) to discover natural customer distribution groups and flag atypical checkout bots or reseller accounts as **Activity Outliers** (DBSCAN Noise label: `-1`).
- **KNN Out-Of-Sample Classifier wrapper**: Resolves the transductive nature of DBSCAN (which lacks a native `.predict()` method for new points) by training a K-Nearest Neighbors (`k=3`) classifier over DBSCAN's scaled spatial coordinate labels.
- **Return Risk Predictor (Logistic Regression)**: Trains a classifier (defined by target label `ReturnRate > 15%`) using frequency and net spend values to predict the probability of future purchase returns. Uses class-balanced weights to manage dataset imbalance.
- **High-Performance SQLite Storage Layer**: Moves away from expensive, in-memory Pandas CSV scans. Relies on an indexed SQLite database (`data/prism.db`) for profile lookups and transaction history joins under 1ms.
- **Quick Search Scanner Sidebar**: A sidebar scanner widget with input validation and asynchronous endpoints (`/api/customer_exists/<id>`) to search and route to a customer profile.
- **Printer-Friendly PDF Report Layout**: Uses pure CSS `@media print` rules to optimize the customer deep-dive view into a clean, physical paper layout (hiding UI buttons/sidebars, setting high-contrast text, and avoiding bad page breaks).

---

## 🛠 Tech Stack

- **Backend**: Python, Flask
- **Database**: SQLite (SQL query optimization with indices on `CustomerID` and `Persona`)
- **Data Engineering**: Pandas, NumPy
- **Machine Learning**: Scikit-Learn (DBSCAN, KNeighborsClassifier, LogisticRegression, StandardScaler)
- **Frontend**: HTML5 (Semantic), Vanilla CSS (Glassmorphic cards, CSS Custom Variables), JavaScript (ES6+ Fetch, Plotly.js, Lucide Icons)

---

## 📊 Customer Personas (DBSCAN Clustered)

1. **VIP Buyer** (High Net Spend, Low Return Rate)  
   *Strategy*: Reward with early catalog access and priority benefits.
2. **Serial Returner** (High Frequency, Return Rate > 40%)  
   *Strategy*: Restrict free return shipping. Charge restocking processing fees.
3. **Low-Value Buyer** (Low Frequency, Low Return Rate)  
   *Strategy*: Incentivize with volume bundling options to grow average basket sizes.
4. **Activity Outlier** (Atypical transaction pacing/volumes; DBSCAN Noise cluster `-1`)  
   *Strategy*: Hold account for manual auditing to verify bot checkouts or reseller behavior.

---

## 📂 Project Structure

```text
RELOG/
├── backend/
│   ├── app.py                  # Flask REST API, routing & SQLite DB endpoints
│   ├── schema.sql              # SQLite database schema (indexed transactions & customers)
│   ├── model.pkl               # Trained KNN out-of-sample persona classifier
│   ├── scaler.pkl              # DBSCAN StandardScaler scaler object
│   ├── persona_map.pkl         # Trained Cluster-ID-to-Persona string mappings
│   ├── churn_model.pkl         # Logistic Regression return risk classifier
│   └── churn_scaler.pkl        # Return risk StandardScaler scaler object
├── data/
│   ├── online_retail.csv       # Dataset storage (uci online retail dataset)
│   ├── rfm_segmented.csv       # Precomputed customer segments spreadsheet
│   └── prism.db                # Indexed SQLite database file (production)
├── static/
│   ├── style.css               # Premium CSS Styles & Printer @media rules
│   ├── dashboard.js            # Plotly.js render charts & overview tab manager
│   ├── scanner.js              # Sidebar scanner validation & routing
│   └── logo.png                # Brand Logo asset
├── templates/
│   ├── index.html              # Main multi-tab dashboard layout
│   └── customer_profile.html   # Customer Deep-Dive report template
├── tests/
│   └── test_app.py             # Isolation tests covering ingestion, metrics, and APIs
├── run_training_pipeline.py    # Automated ETL & ML training pipeline script
├── PROJECT_REPORT.md           # Formal project report for interviews
└── requirements.txt            # Package dependencies
```

---

## 💻 How To Run

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Machine Learning Pipeline**:
   *This loads the dataset, aggregates transaction records, trains standardizers and models, and populates the SQLite database.*
   ```bash
   python run_training_pipeline.py
   ```

3. **Start the Web Interface**:
   ```bash
   python backend/app.py
   ```

4. **Access the Dashboard**:
   Open [http://127.0.0.1:5001](http://127.0.0.1:5001) in your browser.

---

## 🧪 Testing

The codebase includes an isolated test suite running on a temporary SQLite db instance to prevent data corruption.

Run automated unit tests:
```bash
python -m unittest discover -s tests
```

---

## 📋 Interview Questions & Answers preparation

* **Q: Why use DBSCAN instead of K-Means?**
  * *A*: K-Means forces every single data point into a cluster (spherical grouping), making it highly sensitive to outliers. DBSCAN defines clusters based on spatial density (`eps` and `min_samples`), allowing it to naturally capture complex shapes and classify sparse, irregular records as outliers (noise). This is ideal for detecting bots or reseller checkout scripts.
* **Q: How does DBSCAN handle out-of-sample (new) customer data?**
  * *A*: It doesn't natively. To solve this transductive clustering limitation, we train a K-Nearest Neighbors (KNN) classifier (`k=3`) wrapper using the spatial cluster coordinates created by DBSCAN. New customer data is scaled and passed to the KNN classifier to obtain their persona segment.
* **Q: Why exclude ReturnRate from the Logistic Regression Return Risk Model features?**
  * *A*: Including `ReturnRate` directly in the training features would create a severe data leakage issue (since the target risk label itself is defined by whether `ReturnRate > 15%`). The model would learn a trivial rule. Instead, the risk model is trained strictly on `Frequency` and `Monetary` net spend features.
