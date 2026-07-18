import streamlit as st
import pandas as pd 
import numpy as np
import joblib

st.set_page_config(page_title="Customer VIP Predictor",layout="centered")
st.title("🎯 E-Commerce VIP Customer Predictor ")
st.write("Adjust the behavioral metrics below to see if a customer qualifies as a **High-Value (VIP)** buyer.")


@st.cache_resource
def load_objects():
  model=joblib.load('Customer_segmaintation/best_naive_bayse_model.pkl')
  scaler=joblib.load('Customer_segmaintation/scaler.pkl')
  return model,scaler

model,scaler=load_objects()

st.sidebar.header("📊 Customer Behavior Inputs")

recency = st.sidebar.slider(
    "Recency (Days since last purchase)", 
    min_value=1, max_value=365, value=30, step=1
)

total_qty = st.sidebar.number_input(
    "Total Quantity of Items Ordered", 
    min_value=1, max_value=50000, value=600, step=10
)

avg_qty = st.sidebar.slider(
    "Average Items Per Order Line", 
    min_value=1, max_value=100, value=12, step=1
)

avg_price = st.sidebar.slider(
    "Average Product Unit Price ($)", 
    min_value=0.5, max_value=50.0, value=3.20, step=0.1
)

# --- MODEL INFERENCE ---
# 1. Gather live inputs
raw_input = np.array([[recency, total_qty, avg_qty, avg_price]])

# 2. Scale using the training parameters
scaled_input = scaler.transform(raw_input)

# 3. Live prediction using Naive Bayes probabilities
prediction = model.predict(scaled_input)[0]
probability = model.predict_proba(scaled_input)[0][1]

# --- DISPLAY RESULTS ---
st.subheader("🔮 Prediction Results")

col1, col2 = st.columns(2)
with col1:
    if prediction == 1:
        st.success("🎉 Class Result: HIGH-VALUE VIP")
    else:
        st.error("📉 Class Result: STANDARD CUSTOMER")

with col2:
    st.metric(label="Probability of being VIP", value=f"{probability * 100:.1f}%")

st.progress(int(probability * 100))

st.markdown("---")
st.subheader("💡 Suggested Marketing Action")

if probability > 0.85:
    st.info("⭐ **VIP Strategy:** Enroll in exclusive early-access sales and assign a dedicated account tier manager.")
elif probability > 0.50:
    st.warning("📈 **Growth Strategy:** Target with up-sell bundles to turn them into solid long-term VIPs.")
else:
    st.error("🔄 **Win-Back Strategy:** Send an active discount code via email to reduce their growing Recency score.")