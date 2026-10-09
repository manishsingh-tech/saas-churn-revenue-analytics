import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

print("==================================================")
print("CHURN & REVENUE-AT-RISK: MODEL TRAINING & EVALUATION")
print("==================================================")

# 1. Load Feature-Engineered Dataset
df = pd.read_csv("churn_features_engineered.csv")

# 2. Define Features and Target
feature_cols = [
    'Tenure_Months', 'Monthly_Charges', 'Total_Support_Tickets', 
    'Login_Frequency_Days', 'Payment_Delay_Days', 'Recency_Score', 
    'Frequency_Score', 'Monetary_Value', 'Support_Ticket_Ratio', 
    'Risk_Payment_Interaction', 'Is_Enterprise', 'Is_Monthly_Contract'
]

X = df[feature_cols]
y = df['Churn']

# 3. Train-Test Split (80-20 stratified split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training Set Shape: {X_train.shape}")
print(f"Test Set Shape: {X_test.shape}")

# 4. Baseline Model: Logistic Regression
print("\n--- Training Baseline Model: Logistic Regression ---")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_scaled, y_train)

y_pred_lr = lr_model.predict(X_test_scaled)
y_prob_lr = lr_model.predict_proba(X_test_scaled)[:, 1]

print("Logistic Regression Performance:")
print(classification_report(y_test, y_pred_lr))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob_lr):.4f}")

# 5. Production Candidate Model: XGBoost
print("\n--- Training Production Candidate: XGBoost Classifier ---")
xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss'
)
xgb_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_test)
y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

print("XGBoost Classifier Performance:")
print(classification_report(y_test, y_pred_xgb))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob_xgb):.4f}")

# 6. Save Trained Artifacts for SHAP and Deployment
joblib.dump(xgb_model, "churn_xgb_model.pkl")
joblib.dump(scaler, "feature_scaler.pkl")
print("\nTrained XGBoost model and scaler successfully saved as artifacts.")
print("==================================================")