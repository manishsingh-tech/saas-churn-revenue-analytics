import pandas as pd
import numpy as np
import sqlite3

print("==================================================")
print("CHURN & REVENUE-AT-RISK: DATA LOADING & SQL PIPELINE")
print("==================================================")

# 1. Simulate Realistic Customer Churn Dataset
np.random.seed(42)
n_customers = 5000

data = {
    'Customer_ID': [f'CUST_{i:04d}' for i in range(1, n_customers + 1)],
    'Tenure_Months': np.random.randint(1, 72, size=n_customers),
    'Contract_Type': np.random.choice(['Monthly', 'Annual', 'Multi-Year'], size=n_customers, p=[0.5, 0.3, 0.2]),
    'Monthly_Charges': np.random.uniform(500, 5000, size=n_customers),
    'Total_Support_Tickets': np.random.poisson(lam=2.5, size=n_customers),
    'Login_Frequency_Days': np.random.randint(0, 30, size=n_customers),
    'Payment_Delay_Days': np.random.choice([0, 2, 5, 10, 15], size=n_customers, p=[0.7, 0.15, 0.08, 0.05, 0.02]),
    'Segment': np.random.choice(['SMB', 'Enterprise'], size=n_customers, p=[0.75, 0.25]),
}

df_raw = pd.DataFrame(data)

# Introduce realistic probability rules for Churn based on behavior
churn_probability = (
    0.05 
    + (df_raw['Payment_Delay_Days'] > 5).astype(int) * 0.35
    + (df_raw['Total_Support_Tickets'] > 4).astype(int) * 0.25
    + (df_raw['Contract_Type'] == 'Monthly').astype(int) * 0.20
    - (df_raw['Tenure_Months'] > 36).astype(int) * 0.15
)
churn_probability = np.clip(churn_probability, 0.01, 0.99)
df_raw['Churn'] = np.random.binomial(1, churn_probability)

# Introduce anomalies for cleaning step
df_raw['Monthly_Charges'] = df_raw['Monthly_Charges'].mask(df_raw.index == 15, np.nan)
df_raw['Login_Frequency_Days'] = df_raw['Login_Frequency_Days'].mask(df_raw.index == 42, np.nan)

print(f"\nRaw Dataset Shape: {df_raw.shape}")

# 2. Schema Validation Function
def validate_schema(df):
    expected_columns = [
        'Customer_ID', 'Tenure_Months', 'Contract_Type', 'Monthly_Charges',
        'Total_Support_Tickets', 'Login_Frequency_Days', 'Payment_Delay_Days', 'Segment', 'Churn'
    ]
    for col in expected_columns:
        if col not in df.columns:
            raise ValueError(f"Schema Validation Failed: Missing column '{col}'")
    print("Schema Validation Passed: All mandatory columns present.")

validate_schema(df_raw)

# 3. Data Cleaning
df_raw['Monthly_Charges'] = df_raw['Monthly_Charges'].fillna(df_raw['Monthly_Charges'].median())
df_raw['Login_Frequency_Days'] = df_raw['Login_Frequency_Days'].fillna(df_raw['Login_Frequency_Days'].median())
df_clean = df_raw.drop_duplicates()

# 4. SQL Extraction Script & Analytical Base Table (ABT) creation via SQLite
conn = sqlite3.connect("churn_analytics.db")
df_clean.to_sql("raw_customers", conn, if_exists="replace", index=False)

abt_query = """
SELECT 
    Customer_ID,
    Tenure_Months,
    Contract_Type,
    Monthly_Charges,
    Total_Support_Tickets,
    Login_Frequency_Days,
    Payment_Delay_Days,
    Segment,
    Churn,
    CASE 
        WHEN Tenure_Months <= 12 THEN 'New'
        WHEN Tenure_Months <= 36 THEN 'Mid-Tenure'
        ELSE 'Loyal'
    END AS Tenure_Cohort
FROM raw_customers;
"""

df_abt = pd.read_sql(abt_query, conn)
df_abt.to_csv("churn_analytical_base_table.csv", index=False)
print("Analytical Base Table (ABT) successfully generated and saved via SQLite SQL query.")
conn.close()