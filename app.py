import streamlit as st
import numpy as np
import pandas as pd
import pickle


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Diamond Predictor",
    layout="centered"
)


# =========================================================
# LOAD MODELS
# =========================================================

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("cluster.pkl", "rb") as f:
    cluster = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)


# =========================================================
# LOAD ENCODERS
# =========================================================

with open("le_cut.pkl", "rb") as f:
    le_cut = pickle.load(f)

with open("le_color.pkl", "rb") as f:
    le_color = pickle.load(f)

with open("le_clarity.pkl", "rb") as f:
    le_clarity = pickle.load(f)


# =========================================================
# LOAD CLUSTER NAMES
# =========================================================

with open("cluster_names.pkl", "rb") as f:
    cluster_names = pickle.load(f)


# =========================================================
# APPLICATION TITLE
# =========================================================

st.title("💎 Diamond Price Predictor")

st.write(
    "Enter diamond details to predict "
    "price and market segment."
)


# =========================================================
# INPUT SECTION
# =========================================================

st.subheader("Diamond Details")


carat = st.number_input(
    "Carat (Weight)",
    min_value=0.1,
    value=0.5,
    step=0.1
)


x = st.number_input(
    "Length (x mm)",
    min_value=0.1,
    value=5.0
)


y = st.number_input(
    "Width (y mm)",
    min_value=0.1,
    value=5.0
)


z = st.number_input(
    "Depth (z mm)",
    min_value=0.1,
    value=3.0
)


cut = st.selectbox(
    "Cut Quality",
    le_cut.classes_
)


color = st.selectbox(
    "Color Grade",
    le_color.classes_
)


clarity = st.selectbox(
    "Clarity Grade",
    le_clarity.classes_
)


# =========================================================
# ENCODING
# =========================================================

cut_val = le_cut.transform(
    [cut]
)[0]


color_val = le_color.transform(
    [color]
)[0]


clarity_val = le_clarity.transform(
    [clarity]
)[0]


# =========================================================
# FEATURE ENGINEERING
# =========================================================

volume = x * y * z


st.info(
    f"Calculated Volume: {round(volume, 2)} mm³"
)


# =========================================================
# CREATE FEATURE ARRAY
# IMPORTANT:
# Must match train_model.py feature order
# =========================================================

feature_names = ["carat", "cut", "color", "clarity", "x", "y", "z", "volume"]
features = pd.DataFrame([[
    carat,
    cut_val,
    color_val,
    clarity_val,
    x,
    y,
    z,
    volume
]], columns=feature_names)


# =========================================================
# PRICE PREDICTION
# =========================================================

st.subheader("Price Prediction")


if st.button("Predict Price"):

    price = model.predict(
        features
    )

    predicted_price = float(
        np.asarray(price).flatten()[0]
    )

    st.success(
        f"💰 Estimated Price: ₹ {predicted_price:,.0f}"
    )


# =========================================================
# MARKET SEGMENTATION
# =========================================================

st.subheader("Market Segmentation")


if st.button("Predict Cluster"):

    # Apply the SAME scaler used during training
    scaled_features = scaler.transform(
        features
    )

    # Predict cluster
    cluster_id = int(
        cluster.predict(
            scaled_features
        )[0]
    )

    # Get actual cluster name
    market_segment = cluster_names.get(
        cluster_id,
        f"Market Segment {cluster_id}"
    )

    st.success(
        f"💎 Market Segment: {market_segment}"
    )
