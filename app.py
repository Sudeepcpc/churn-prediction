import streamlit as st
import numpy as np
import pandas as pd
import pickle

# ── load the saved bundle ─────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("churn_model.pkl", "rb") as f:
        return pickle.load(f)

bundle = load_model()
model = bundle["model"]
scaler = bundle["scaler"]
feature_names = bundle["feature_names"]   # the 30 columns, in order

# ── page ──────────────────────────────────────────────────────────────────────
st.title("Customer Churn Predictor")
st.write("Predicts how likely a telecom customer is to cancel. Built with scikit-learn (Logistic Regression, AUC 0.84).")

st.header("Enter Customer Details")

col1, col2 = st.columns(2)
with col1:
    tenure = st.number_input("Tenure (months as customer)", 0, 100, 12)
    MonthlyCharges = st.number_input("Monthly charges ($)", 0.0, 200.0, 70.0)
    TotalCharges = st.number_input("Total charges ($)", 0.0, 10000.0, 800.0)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
    payment = st.selectbox("Payment method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

with col2:
    senior = st.selectbox("Senior citizen?", ["No", "Yes"])
    partner = st.selectbox("Has partner?", ["No", "Yes"])
    dependents = st.selectbox("Has dependents?", ["No", "Yes"])
    paperless = st.selectbox("Paperless billing?", ["No", "Yes"])
    online_security = st.selectbox("Online security?", ["No", "Yes"])
    tech_support = st.selectbox("Tech support?", ["No", "Yes"])

if st.button("Predict Churn Risk", type="primary"):
    # ── build a row of all 30 features, starting at 0 ─────────────────────────
    # we start with every encoded column = 0, then set the ones that apply.
    # this guarantees the SAME 30 columns in the SAME order the model expects.
    row = dict.fromkeys(feature_names, 0)

    # numeric features (set directly)
    row["tenure"] = tenure
    row["MonthlyCharges"] = MonthlyCharges
    row["TotalCharges"] = TotalCharges
    row["SeniorCitizen"] = 1 if senior == "Yes" else 0

    # binary encoded columns (get_dummies made "_Yes" versions)
    if partner == "Yes":         row["Partner_Yes"] = 1
    if dependents == "Yes":      row["Dependents_Yes"] = 1
    if paperless == "Yes":       row["PaperlessBilling_Yes"] = 1

    # multi-category: only set the matching column (baseline stays all-0)
    if contract == "One year":   row["Contract_One year"] = 1
    if contract == "Two year":   row["Contract_Two year"] = 1

    if internet == "Fiber optic": row["InternetService_Fiber optic"] = 1
    if internet == "No":          row["InternetService_No"] = 1

    if online_security == "Yes":  row["OnlineSecurity_Yes"] = 1
    if tech_support == "Yes":     row["TechSupport_Yes"] = 1

    if payment == "Electronic check":            row["PaymentMethod_Electronic check"] = 1
    elif payment == "Mailed check":              row["PaymentMethod_Mailed check"] = 1
    elif payment == "Credit card (automatic)":   row["PaymentMethod_Credit card (automatic)"] = 1
    # "Bank transfer (automatic)" is the baseline → all payment cols stay 0

    # ── assemble in the exact training column order, scale, predict ───────────
    X_row = pd.DataFrame([row])[feature_names]   # enforce column order
    X_scaled = scaler.transform(X_row)
    prob = model.predict_proba(X_scaled)[0][1]   # probability of churn

    # ── show result ───────────────────────────────────────────────────────────
    st.header("Result")
    st.metric("Churn Probability", f"{prob:.1%}")

    if prob >= 0.5:
        st.error(f"HIGH RISK — {prob:.1%} chance of churning. Recommend a retention offer.")
    else:
        st.success(f"LOWER RISK — {prob:.1%} chance of churning.")

    st.caption("Tip: try Month-to-month + Fiber optic + low tenure to see risk rise.")