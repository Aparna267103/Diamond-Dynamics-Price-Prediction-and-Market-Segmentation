import numpy as np
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense


# =========================================================
# 1. LOAD DATA
# =========================================================

df = pd.read_csv("diamonds.csv")

print("Original Dataset Shape:", df.shape)


# =========================================================
# 2. DATA CLEANING
# =========================================================

# Replace invalid zero dimensions
df[['x', 'y', 'z']] = df[['x', 'y', 'z']].replace(0, np.nan)

# Remove invalid rows
df = df.dropna(subset=['x', 'y', 'z'])

print("After Cleaning:", df.shape)


# =========================================================
# 3. FEATURE ENGINEERING
# =========================================================

df["volume"] = (
    df["x"] *
    df["y"] *
    df["z"]
)

USD_TO_INR = 83.0

df["price_inr"] = (
    df["price"] *
    USD_TO_INR
)


# =========================================================
# 4. LOAD THE SAME ENCODERS USED IN TRAINING
# =========================================================

with open("le_cut.pkl", "rb") as f:
    le_cut = pickle.load(f)

with open("le_color.pkl", "rb") as f:
    le_color = pickle.load(f)

with open("le_clarity.pkl", "rb") as f:
    le_clarity = pickle.load(f)


# =========================================================
# 5. ENCODE CATEGORICAL FEATURES
# =========================================================

df["cut"] = le_cut.transform(df["cut"])

df["color"] = le_color.transform(df["color"])

df["clarity"] = le_clarity.transform(df["clarity"])


# =========================================================
# 6. FEATURES AND TARGET
# =========================================================

X = df[
    [
        "carat",
        "cut",
        "color",
        "clarity",
        "x",
        "y",
        "z",
        "volume"
    ]
]

y = df["price_inr"]


# =========================================================
# 7. TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))


# =========================================================
# 8. FEATURE SCALING
# =========================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


# =========================================================
# 9. ANN MODEL
# =========================================================

model = Sequential([

    Dense(
        64,
        activation="relu",
        input_shape=(X_train_scaled.shape[1],)
    ),

    Dense(
        32,
        activation="relu"
    ),

    Dense(
        1
    )
])


# =========================================================
# 10. COMPILE MODEL
# =========================================================

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)


# =========================================================
# 11. TRAIN MODEL
# =========================================================

history = model.fit(
    X_train_scaled,
    y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.20,
    verbose=1
)


# =========================================================
# 12. ANN PREDICTION
# =========================================================

predictions = model.predict(
    X_test_scaled,
    verbose=0
).flatten()


# =========================================================
# 13. EVALUATION
# =========================================================

mae = mean_absolute_error(
    y_test,
    predictions
)

mse = mean_squared_error(
    y_test,
    predictions
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_test,
    predictions
)


# =========================================================
# 14. PRINT RESULTS
# =========================================================

print("\n======================================")
print("ANN MODEL PERFORMANCE")
print("======================================")

print(f"MAE  : {mae:.2f}")
print(f"MSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R2   : {r2:.4f}")


# =========================================================
# 15. SAVE ANN MODEL
# =========================================================

model.save("ann_model.h5")


# =========================================================
# 16. SAVE ANN SCALER
# =========================================================

with open("ann_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)


# =========================================================
# 17. FINAL MESSAGE
# =========================================================

print("\n======================================")
print("✅ ANN TRAINING COMPLETED")
print("======================================")

print("Saved:")
print("  - ann_model.h5")
print("  - ann_scaler.pkl")
