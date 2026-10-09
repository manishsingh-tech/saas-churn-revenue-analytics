
import streamlit as st
import pandas as pd
from pathlib import Path

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="SaaS Churn & Revenue Risk Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CONSTANTS
# =========================================================

DATA_FILE = Path(__file__).parent / "churn_financial_report.csv"
CHURN_THRESHOLD = 0.50

# These are illustrative business assumptions, not actual results.
RETENTION_COST_PER_CUSTOMER = 2000.0
ASSUMED_RECOVERY_RATE = 0.40

REQUIRED_COLUMNS = [
    "Customer_ID",
    "Tenure_Months",
    "Contract_Type",
    "Monthly_Charges",
    "Churn_Probability",
    "Revenue_At_Risk",
    "Segment",
    "Total_Support_Tickets",
]

# =========================================================
# PAGE HEADER
# =========================================================

st.title("SaaS Churn & Revenue Risk Intelligence Platform")

st.markdown(
    """
    An interactive analytics dashboard for customer churn risk assessment,
    revenue exposure analysis, and retention campaign planning.
    """
)

st.divider()

# =========================================================
# DATA LOADING AND VALIDATION
# =========================================================

@st.cache_data
def load_data(file_path, modified_time):
    return pd.read_csv(file_path)


try:
    if not DATA_FILE.exists():
        st.error(
            "The required dataset was not found. "
            "Place churn_financial_report.csv in the same folder as this app."
        )
        st.stop()

    df = load_data(str(DATA_FILE), DATA_FILE.stat().st_mtime)

except (pd.errors.ParserError, UnicodeDecodeError, OSError) as error:
    st.error(f"Unable to read the dataset: {error}")
    st.stop()

missing_columns = [
    column for column in REQUIRED_COLUMNS
    if column not in df.columns
]

if missing_columns:
    st.error(
        "The dataset is missing required columns: "
        + ", ".join(missing_columns)
    )
    st.stop()

if df.empty:
    st.warning("The dataset contains no customer records.")
    st.stop()

# Validate numeric fields
NUMERIC_COLUMNS = [
    "Tenure_Months",
    "Monthly_Charges",
    "Churn_Probability",
    "Revenue_At_Risk",
    "Total_Support_Tickets",
]

for column in NUMERIC_COLUMNS:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df = df.dropna(subset=[
    "Customer_ID",
    "Churn_Probability",
    "Revenue_At_Risk",
])

if df.empty:
    st.error("No valid customer records are available for analysis.")
    st.stop()

if not df["Churn_Probability"].between(0, 1).all():
    st.error(
        "Churn_Probability must contain values between 0 and 1."
    )
    st.stop()

# =========================================================
# BUSINESS METRICS
# =========================================================

total_customers = len(df)

total_revenue_at_risk = df["Revenue_At_Risk"].sum()

high_risk_df = df[
    df["Churn_Probability"] > CHURN_THRESHOLD
]

high_risk_customers = len(high_risk_df)

estimated_recovered_revenue = (
    high_risk_df["Revenue_At_Risk"].sum()
    * ASSUMED_RECOVERY_RATE
)

estimated_retention_cost = (
    high_risk_customers * RETENTION_COST_PER_CUSTOMER
)

estimated_net_impact = (
    estimated_recovered_revenue - estimated_retention_cost
)

# =========================================================
# SIDEBAR: EXECUTIVE SUMMARY
# =========================================================

st.sidebar.title("Executive Summary")

st.sidebar.caption("Customer portfolio and estimated financial exposure")

st.sidebar.metric(
    "Customer Records",
    f"{total_customers:,}"
)

st.sidebar.metric(
    "Total Revenue at Risk",
    f"INR {total_revenue_at_risk:,.2f}"
)

st.sidebar.metric(
    "High-Risk Customers",
    f"{high_risk_customers:,}"
)

st.sidebar.metric(
    "Estimated Net Campaign Impact",
    f"INR {estimated_net_impact:,.2f}"
)

st.sidebar.divider()

st.sidebar.subheader("Model Assumptions")

st.sidebar.write(
    f"High-risk threshold: {CHURN_THRESHOLD:.0%}"
)

st.sidebar.write(
    f"Assumed recovery rate: {ASSUMED_RECOVERY_RATE:.0%}"
)

st.sidebar.write(
    f"Retention cost per customer: "
    f"INR {RETENTION_COST_PER_CUSTOMER:,.2f}"
)

st.sidebar.caption(
    "Financial impact figures are estimates based on configurable "
    "assumptions, not measured campaign outcomes."
)

# =========================================================
# DASHBOARD NAVIGATION
# =========================================================

tab1, tab2, tab3 = st.tabs([
    "Portfolio Overview",
    "Segment Analysis",
    "Customer Risk Explorer",
])

# =========================================================
# TAB 1: PORTFOLIO OVERVIEW
# =========================================================

with tab1:
    st.subheader("Portfolio Performance and Risk Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Customers",
        f"{total_customers:,}"
    )

    col2.metric(
        "High-Risk Customers",
        f"{high_risk_customers:,}"
    )

    col3.metric(
        "Estimated Recoverable Revenue",
        f"INR {estimated_recovered_revenue:,.2f}"
    )

    st.divider()

    st.subheader("Customer Churn Risk Distribution")

    risk_distribution = pd.Series({
        "High Risk": high_risk_customers,
        "Lower Risk": total_customers - high_risk_customers,
    })

    st.bar_chart(risk_distribution)

    st.caption(
        "High risk is defined as a churn probability greater than 50%."
    )

    st.subheader("Customer Account Details")

    overview_columns = [
        "Customer_ID",
        "Tenure_Months",
        "Contract_Type",
        "Monthly_Charges",
        "Churn_Probability",
        "Revenue_At_Risk",
        "Segment",
    ]

    overview_df = df[overview_columns].copy()

    overview_df["Churn_Probability"] = (
        overview_df["Churn_Probability"] * 100
    )

    overview_df = overview_df.rename(columns={
        "Customer_ID": "Customer ID",
        "Tenure_Months": "Tenure (Months)",
        "Contract_Type": "Contract Type",
        "Monthly_Charges": "Monthly Charges (INR)",
        "Churn_Probability": "Churn Probability (%)",
        "Revenue_At_Risk": "Revenue at Risk (INR)",
        "Segment": "Customer Segment",
    })

    st.dataframe(
        overview_df.head(15),
        use_container_width=True,
        hide_index=True,
    )

# =========================================================
# TAB 2: SEGMENT ANALYSIS
# =========================================================

with tab2:
    st.subheader("Segment-Wise Churn and Financial Exposure")

    segment_summary = (
        df.groupby("Segment", dropna=False)
        .agg(
            Total_Customers=("Customer_ID", "count"),
            Average_Churn_Probability=("Churn_Probability", "mean"),
            Total_Revenue_At_Risk=("Revenue_At_Risk", "sum"),
        )
        .reset_index()
    )

    segment_summary["Average_Churn_Probability"] = (
        segment_summary["Average_Churn_Probability"] * 100
    )

    segment_summary = segment_summary.rename(columns={
        "Segment": "Customer Segment",
        "Total_Customers": "Total Customers",
        "Average_Churn_Probability": "Average Churn Probability (%)",
        "Total_Revenue_At_Risk": "Revenue at Risk (INR)",
    })

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Customers by Segment**")
        st.bar_chart(
            segment_summary.set_index("Customer Segment")[
                "Total Customers"
            ]
        )

    with col2:
        st.markdown("**Revenue at Risk by Segment**")
        st.bar_chart(
            segment_summary.set_index("Customer Segment")[
                "Revenue at Risk (INR)"
            ]
        )

    st.subheader("Segment Summary Table")

    st.dataframe(
        segment_summary,
        use_container_width=True,
        hide_index=True,
    )

# =========================================================
# TAB 3: CUSTOMER RISK EXPLORER
# =========================================================

with tab3:
    st.subheader("Individual Customer Risk Assessment")

    customer_ids = df["Customer_ID"].tolist()

    selected_customer = st.selectbox(
        "Select a customer account",
        customer_ids,
    )

    customer_rows = df[
        df["Customer_ID"] == selected_customer
    ]

    if customer_rows.empty:
        st.warning("No matching customer record was found.")

    else:
        customer = customer_rows.iloc[0]

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Tenure",
            f"{customer['Tenure_Months']:,.0f} months"
            if pd.notna(customer["Tenure_Months"])
            else "N/A"
        )

        col2.metric(
            "Monthly Charges",
            f"INR {customer['Monthly_Charges']:,.2f}"
            if pd.notna(customer["Monthly_Charges"])
            else "N/A"
        )

        col3.metric(
            "Support Tickets",
            f"{customer['Total_Support_Tickets']:,.0f}"
            if pd.notna(customer["Total_Support_Tickets"])
            else "N/A"
        )

        col4.metric(
            "Churn Probability",
            f"{customer['Churn_Probability']:.1%}"
        )

        st.divider()

        st.markdown("**Customer Profile**")

        profile_col1, profile_col2 = st.columns(2)

        with profile_col1:
            st.write(
                f"**Customer ID:** {selected_customer}"
            )
            st.write(
                f"**Contract Type:** {customer['Contract_Type']}"
            )
            st.write(
                f"**Segment:** {customer['Segment']}"
            )

        with profile_col2:
            st.write(
                f"**Revenue at Risk:** "
                f"INR {customer['Revenue_At_Risk']:,.2f}"
            )

            st.write(
                f"**Risk Threshold:** {CHURN_THRESHOLD:.0%}"
            )

        if customer["Churn_Probability"] > CHURN_THRESHOLD:
            st.error(
                f"High-Risk Account: Customer {selected_customer} "
                "exceeds the configured churn-risk threshold. "
                "Consider reviewing retention options and customer "
                "support history."
            )

        else:
            st.success(
                f"Below-Threshold Account: Customer {selected_customer} "
                "does not exceed the configured churn-risk threshold. "
                "Continue monitoring customer engagement."
            )

        st.caption(
            "Risk classifications are based on the churn probabilities "
            "provided in the dataset. They do not guarantee future outcomes."
        )

# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "SaaS Churn & Revenue Risk Intelligence | "
    "Portfolio analytics project"
)