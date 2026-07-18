import streamlit as st
import joblib
import numpy as np

# পেজ কনফিগারেশন
st.set_page_config(page_title="Customer VIP Predictor", layout="wide")
st.title("Customer VIP Customer Predictor")

# মডেল এবং স্কেলার লোড করা
@st.cache_resource
def load_objects():
    model = joblib.load('best_naive_bayse_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_objects()

# সাইডবারে ইনপুট ক্ষেত্রগুলো
st.sidebar.header("Customer Behavior Inputs")
recency = st.sidebar.slider("Recency (Days since last purchase)", min_value=1, max_value=365, value=30, step=1)
total_qty = st.sidebar.number_input("Total Quantity of Items Ordered", min_value=1, max_value=50000, value=600, step=10)
avg_qty = st.sidebar.slider("Average Items Per Order Line", min_value=1, max_value=100, value=12, step=1)
avg_price = st.sidebar.slider("Average Product Unit Price ($)", min_value=0.0, max_value=50.0, value=3.20, step=0.1)

# প্রেডিকশন লজিক
raw_input = np.array([[recency, total_qty, avg_qty, avg_price]])
scaled_input = scaler.transform(raw_input)
prediction = model.predict(scaled_input)[0]
probability = model.predict_proba(scaled_input)[0][1]

# ফলাফল প্রদর্শন
st.subheader("Prediction Results")
col1, col2 = st.columns(2)

with col1:
    if prediction == 1:
        st.success("Class Result: HIGH-VALUE VIP")
    else:
        st.error("Class Result: STANDARD CUSTOMER")

with col2:
    st.metric(label="Probability of being VIP", value=f"{probability * 100:.1f}%")
    st.progress(int(probability * 100))
