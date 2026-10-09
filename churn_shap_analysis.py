import pandas as pd
import numpy as np
import joblib
import shap

print("==================================================")
print("CHURN & REVENUE-AT-RISK: SHAP INTERPRETABILITY")
print("==================================================")

# 1. Load Dataset and Trained XGBoost Model
df = pd.read_csv("churn_features_engineered.csv")
model = joblib.load("churn_xgb_model.pkl")

feature_cols = [
    'Tenure_Months', 'Monthly_Charges', 'Total_Support_Tickets', 
    'Login_Frequency_Days', 'Payment_Delay_Days', 'Recency_Score', 
    'Frequency_Score', 'Monetary_Value', 'Support_Ticket_Ratio', 
    'Risk_Payment_Interaction', 'Is_Enterprise', 'Is_Monthly_Contract'
]

X = df[feature_cols]

# 2. Compute SHAP Values using TreeExplainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# 3. Global SHAP Summary (Identify major churn drivers across all customers)
mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
global_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Mean_Absolute_SHAP': mean_abs_shap
}).sort_values(by='Mean_Absolute_SHAP', ascending=False)

print("\nTop Global Churn Drivers:")
print(global_importance.head(5))

# Save Global SHAP summary for executive reporting and dashboards
global_importance.to_csv("churn_global_shap_importance.csv", index=False)
print("\nGlobal SHAP importance successfully saved as CSV.")
print("==================================================")