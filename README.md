# Customer Churn Prediction — scikit-learn

Predicting which telecom customers are likely to cancel their subscription, so the business can act before they leave. Built as a **production-style ML project** with scikit-learn and XGBoost — the companion piece to my [from-scratch loan default project](https://github.com/Sudeepcpc/loan-default).

**Live demo:** https://churn-prediction-sudeep.streamlit.app
**Author:** Sudeep C P — [github.com/Sudeepcpc](https://github.com/Sudeepcpc)

---

## The Problem

Telecom companies lose revenue every month to churn — customers cancelling their service. Retaining an existing customer is far cheaper than acquiring a new one, so the business question is: **which customers are likely to leave soon?** If a model can flag them early, the company can offer a retention deal before they cancel.

This is binary classification: each customer is scored with a probability of churning (1) or staying (0).

**Dataset:** [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (IBM sample) — 7,043 customers, 20 features, 26.5% churn rate.

---

## Results — Three-Model Comparison

All models evaluated on the same stratified 20% held-out test set (1,409 customers).

| Model | ROC-AUC | Churn Recall | Notes |
|---|---|---|---|
| **Logistic Regression** | **0.841** | **79%** | **Winner — simple, interpretable, best performer** |
| Random Forest | 0.827 | — | 200 trees, balanced class weights |
| XGBoost | 0.809 | — | Default settings, imbalance-weighted |

**The headline finding: the simplest model won.** Logistic regression beat both tree ensembles. On a small (~7K rows) dataset with largely linear relationships, added model complexity didn't earn its keep — a useful reminder that model selection should be driven by evidence, not by reaching for the fanciest tool.

At the default threshold, the winning model caught **294 of 374 churners (79% recall)** with 51% precision — a sensible trade for retention, where a false alarm costs a cheap discount offer but a missed churner costs the whole customer.

---

## What Drives Churn? (Feature Importance)

Coefficient analysis of the winning model surfaced clear, actionable drivers:

**Increases churn risk:**
- Fiber optic internet (strongest driver — premium service, high expectations, heavy competition)
- Streaming add-ons, multiple lines

**Reduces churn risk (protective):**
- **Tenure** — the longer someone has been a customer, the less likely they leave. New customers are the flight risk.
- **Long contracts** — two-year contracts strongly reduce churn versus month-to-month.
- Online security add-on

**Business recommendations:** focus retention efforts on new, month-to-month customers; incentivise longer contracts; investigate fiber-optic pricing and service quality.

---

## Technical Highlights

- **Data cleaning:** caught and fixed `TotalCharges` stored as text due to hidden blank values (new customers with tenure 0) — a classic real-world data trap. Filled via `pd.to_numeric(errors="coerce")` + zero-fill.
- **Encoding:** one-hot encoded 15 categorical columns with `pd.get_dummies(drop_first=True)`, carefully converting only boolean columns to int to preserve monetary decimals.
- **Leak-free preprocessing:** `StandardScaler` fitted on training data only; stratified split preserves the 73/27 class ratio in both sets.
- **Imbalance handling:** `class_weight="balanced"` (and `scale_pos_weight` for XGBoost) so the models take the minority churn class seriously without manual threshold hacks.
- **Deployment:** winning model + fitted scaler + feature schema pickled together; the Streamlit app rebuilds each new customer into the exact 30-column training format before scoring.

---

## Project Structure

```
churn-prediction/
├── churn_analysis.ipynb    # full pipeline: cleaning → encoding → 3 models → insights
├── app.py                  # Streamlit app (loads pickled model + scaler)
├── churn_model.pkl         # trained model bundle
├── requirements.txt
└── runtime.txt             # pins Python 3.11 for Streamlit Cloud
```

---

## Running Locally

```bash
git clone https://github.com/Sudeepcpc/churn-prediction.git
cd churn-prediction

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt

# explore the full pipeline
jupyter notebook churn_analysis.ipynb

# run the web app
streamlit run app.py
```

---

## Relationship to My Other Work

This project deliberately complements my [loan default predictor](https://github.com/Sudeepcpc/loan-default), where I implemented logistic regression, decision trees, and a random forest **from scratch in NumPy** to master the internals. Here I apply the same concepts **the production way** — scikit-learn pipelines, proper encoding, and library-driven evaluation. Together they demonstrate both depth (understanding the math) and practical tooling (shipping with industry-standard libraries).

---

## Tech Stack

Python · scikit-learn · XGBoost · pandas · NumPy · Streamlit
