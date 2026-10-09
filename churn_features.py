import pandas as pd
import numpy as np

print("==================================================")
print("CHURN & REVENUE-AT-RISK: FEATURE ENGINEERING (RFM)")
print("==================================================")

# 1. Load the Analytical Base Table
df = pd.read_csv("churn_analytical_base_table.csv")
print(f"Loaded Base Table Shape: {df.shape}")

# 2. Engineer RFM & Behavioral Features
# Recency: Proxy using Login_Frequency_Days (lower login frequency implies higher recency gap / inactivity)
df['Recency_Score'] = df['Login_Frequency_Days']

# Frequency: Proxy using Total_Support_Tickets
df['Frequency_Score'] = df['Total_Support_Tickets']

# Monetary: Total value generated over tenure
df['Monetary_Value'] = df['Monthly_Charges'] * df['Tenure_Months']

# Additional Behavioral Ratios
df['Support_Ticket_Ratio'] = df['Total_Support_Tickets'] / (df['Tenure_Months'] + 1)
df['Risk_Payment_Interaction'] = df['Payment_Delay_Days'] * df['Monthly_Charges'] / 1000.0

# 3. Categorical Encoding for Modeling Preparation
df['Is_Enterprise'] = np.where(df['Segment'] == 'Enterprise', 1, 0)
df['Is_Monthly_Contract'] = np.where(df['Contract_Type'] == 'Monthly', 1, 0)

print("\nEngineered Features Sample:")
print(df[['Customer_ID', 'Monetary_Value', 'Support_Ticket_Ratio', 'Is_Enterprise', 'Churn']].head(5))

# Save feature-engineered dataset
df.to_csv("churn_features_engineered.csv", index=False)
print("\nFeature-engineered dataset successfully saved as 'churn_features_engineered.csv'.")
print("==================================================")