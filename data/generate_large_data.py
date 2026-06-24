import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_synthetic_data(num_rows=100000):
    np.random.seed(42)
    print(f"Generating {num_rows} rows of synthetic transactions...")
    
    # Generate Customer IDs (a pool of 2000 unique customers for high repeat purchases)
    customer_ids = np.random.randint(10000, 12000, size=num_rows)
    
    # Generate Invoice Dates (over last 2 years)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)
    invoice_dates = [start_date + timedelta(seconds=np.random.randint(0, int((end_date - start_date).total_seconds()))) for _ in range(num_rows)]
    
    # Generate Stock Codes and Descriptions
    stock_codes = [f"STK{np.random.randint(100, 999)}" for _ in range(num_rows)]
    countries = np.random.choice(['United Kingdom', 'Germany', 'France', 'Spain', 'Italy', 'USA', 'India', 'Canada'], size=num_rows)
    
    # Base quantities and unit prices
    quantities = np.random.randint(1, 50, size=num_rows)
    unit_prices = np.round(np.random.uniform(0.5, 100.0, size=num_rows), 2)
    
    # Group into realistic multi-line invoices
    base_invoice_nos = [np.random.randint(536365, 551365) for _ in range(num_rows)]
    
    # Randomly make ~12% of the transactions cancellations (returns)
    is_return = np.random.random(size=num_rows) < 0.12
    
    invoice_nos = []
    final_quantities = []
    for i in range(num_rows):
        if is_return[i]:
            # Cancellation invoice prepended with 'C', quantity is negative
            invoice_nos.append(f"C{base_invoice_nos[i]}")
            final_quantities.append(-int(np.random.randint(1, max(2, quantities[i]))))
        else:
            invoice_nos.append(str(base_invoice_nos[i]))
            final_quantities.append(int(quantities[i]))

    df = pd.DataFrame({
        'InvoiceNo': invoice_nos,
        'StockCode': stock_codes,
        'Description': ['Item ' + s for s in stock_codes],
        'Quantity': final_quantities,
        'InvoiceDate': invoice_dates,
        'UnitPrice': unit_prices,
        'CustomerID': customer_ids,
        'Country': countries
    })
    
    # Sort by InvoiceDate
    df = df.sort_values(by='InvoiceDate')
    
    # Ensure directory exists
    os.makedirs('data', exist_ok=True)
    out_path = 'data/large_transactions.csv'
    df.to_csv(out_path, index=False)
    print(f"[SUCCESS] Generated {num_rows} rows of synthetic data in {out_path} (Cancellations: {sum(is_return)})")

if __name__ == "__main__":
    generate_synthetic_data()
