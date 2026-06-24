-- RELOG Database Schema

-- Drop tables if they exist to allow schema rebuilds
DROP TABLE IF EXISTS transactions;
DROP TABLE IF EXISTS customers;

-- Table to store raw customer transactions (including returns/refunds)
CREATE TABLE IF NOT EXISTS transactions (
    InvoiceNo TEXT,
    StockCode TEXT,
    Description TEXT,
    Quantity INTEGER,
    InvoiceDate TEXT,
    UnitPrice REAL,
    CustomerID INTEGER,
    Country TEXT
);

-- Table to store processed customer metrics and model predictions
CREATE TABLE IF NOT EXISTS customers (
    CustomerID INTEGER PRIMARY KEY,
    Recency REAL,
    Frequency REAL,
    Monetary REAL,
    ReturnRate REAL,
    Cluster INTEGER,
    Persona TEXT,
    ReturnProbability REAL,
    ReturnRisk TEXT
);

-- Indexing for fast profile lookups
CREATE INDEX IF NOT EXISTS idx_transactions_customer ON transactions(CustomerID);
CREATE INDEX IF NOT EXISTS idx_customers_persona ON customers(Persona);
