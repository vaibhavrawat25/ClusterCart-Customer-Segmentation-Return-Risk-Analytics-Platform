import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_synthetic_data(num_rows=5000):
    np.random.seed(42)
    
    # Generate Customer IDs
    customer_ids = np.random.randint(10000, 15000, size=num_rows)
    
    # Generate Invoice Dates (last 1 year)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    invoice_dates = [start_date + timedelta(seconds=np.random.randint(0, int((end_date - start_date).total_seconds()))) for _ in range(num_rows)]
    
    # Generate Stock Codes and Descriptions (simplified)
    stock_codes = [f"STK{np.random.randint(100, 999)}" for _ in range(num_rows)]
    countries = np.random.choice(['United Kingdom', 'Germany', 'France', 'Spain', 'Italy'], size=num_rows)
    
    # Base quantities and unit prices
    quantities = np.random.randint(1, 20, size=num_rows)
    unit_prices = np.round(np.random.uniform(1.0, 50.0, size=num_rows), 2)
    
    # Generate Invoice Numbers
    base_invoice_nos = [np.random.randint(536365, 581587) for _ in range(num_rows)]
    
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
    
    # Ensure directory exists
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/online_retail.csv', index=False)
    print(f"Generated {num_rows} rows of synthetic data in data/online_retail.csv (Cancellations: {sum(is_return)})")

if __name__ == "__main__":
    generate_synthetic_data()
