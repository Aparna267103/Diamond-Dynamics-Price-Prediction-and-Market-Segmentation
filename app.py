import streamlit as st
import numpy as np
import pickle

# ================= LOAD MODELS =================
model = pickle.load(open("model.pkl", "rb"))
cluster = pickle.load(open("cluster.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

le_cut = pickle.load(open("le_cut.pkl", "rb"))
le_color = pickle.load(open("le_color.pkl", "rb"))
le_clarity = pickle.load(open("le_clarity.pkl", "rb"))

# ================= UI =================
st.set_page_config(page_title="Diamond Predictor", layout="centered")

st.title("💎 Diamond Price Predictor")
st.write("Enter diamond details to predict price and market segment")

# ================= INPUTS =================
carat = st.number_input("Carat (Weight)", min_value=0.1, value=0.5, step=0.1)

x = st.number_input("Length (x mm)", min_value=0.1, value=5.0)
y = st.number_input("Width (y mm)", min_value=0.1, value=5.0)
z = st.number_input("Depth (z mm)", min_value=0.1, value=3.0)

cut = st.selectbox("Cut Quality", le_cut.classes_)
color = st.selectbox("Color Grade", le_color.classes_)
clarity = st.selectbox("Clarity Grade", le_clarity.classes_)

# ================= ENCODING =================
cut_val = le_cut.transform([cut])[0]
color_val = le_color.transform([color])[0]
clarity_val = le_clarity.transform([clarity])[0]

# ================= FEATURE ENGINEERING =================
volume = x * y * z

st.info(f"Calculated Volume: {round(volume,2)}")

# IMPORTANT → must match train_model.py
features = np.array([[
    carat,
    cut_val,
    color_val,
    clarity_val,
    x,
    y,
    z,
    volume
]])

# ================= PREDICTION =================
if st.button("Predict Price"):
    price = model.predict(features)
    st.success(f"💰 Estimated Price: ₹ {int(price[0])}")

# ================= CLUSTER =================
if st.button("Predict Cluster"):
    scaled = scaler.transform(features)
    c = cluster.predict(scaled)[0]

    cluster_names = {
        0: "💎 Affordable Diamonds",
        1: "💎 Mid-range Diamonds",
        2: "💎 Premium Diamonds"
    }

    st.success(f"Market Segment: {cluster_names[c]}")