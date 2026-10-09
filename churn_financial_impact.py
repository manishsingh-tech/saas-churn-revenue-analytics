import pandas as pd
import numpy as np
import joblib

print("==================================================")
print("CHURN & REVENUE-AT-RISK: FINANCIAL IMPACT & LTV")
print("==================================================")

# 1. Load dataset with features and trained model
df = pd.read_csv("churn_features_engineered.csv")
model = joblib.load("churn_xgb_model.pkl")

feature_cols = [
    'Tenure_Months', 'Monthly_Charges', 'Total_Support_Tickets', 
    'Login_Frequency_Days', 'Payment_Delay_Days', 'Recency_Score', 
    'Frequency_Score', 'Monetary_Value', 'Support_Ticket_Ratio', 
    'Risk_Payment_Interaction', 'Is_Enterprise', 'Is_Monthly_Contract'
]

# Predict churn probabilities for all customers
df['Churn_Probability'] = model.predict_proba(df[feature_cols])[:, 1]

# 2. Estimate Customer Lifetime Value (LTV)
df['Estimated_LTV'] = df['Monthly_Charges'] * 12 / (df['Churn_Probability'] + 0.05)

# 3. Calculate Revenue-at-Risk per customer
df['Revenue_At_Risk'] = df['Estimated_LTV'] * df['Churn_Probability']

# 4. Segment Churn Analysis (Enterprise vs SMB)
print("\n--- Segment Churn Analysis ---")
segment_summary = df.groupby('Segment').agg(
    Total_Customers=('Customer_ID', 'count'),
    Average_Churn_Prob=('Churn_Probability', 'mean'),
    Total_Revenue_At_Risk=('Revenue_At_Risk', 'sum')
).reset_index()
print(segment_summary)

total_rar = df['Revenue_At_Risk'].sum()
print(f"\nTotal Portfolio Revenue-at-Risk: INR {total_rar:,.2f}")

# 5. Retention Campaign & Net Financial Impact Simulation
df['Target_For_Retention'] = np.where(df['Churn_Probability'] > 0.5, 1, 0)
retention_cost_per_user = 2000.0
total_retention_cost = df['Target_For_Retention'].sum() * retention_cost_per_user

# Assume retention intervention saves 40% of targeted at-risk revenue
saved_revenue = df.loc[df['Target_For_Retention'] == 1, 'Revenue_At_Risk'].sum() * 0.40
net_financial_impact = saved_revenue - total_retention_cost

print(f"\nTotal Retention Campaign Cost: INR {total_retention_cost:,.2f}")
print(f"Estimated Recovered Revenue: INR {saved_revenue:,.2f}")
print(f"Net Financial Impact: INR {net_financial_impact:,.2f}")

# Save financial report
df.to_csv("churn_financial_report.csv", index=False)
print("\nFinancial impact report successfully generated and saved.")
print("==================================================")