import numpy as np
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load data
df = pd.read_csv("diamonds.csv")

df = df[(df['x'] > 0) & (df['y'] > 0) & (df['z'] > 0)]
df["volume"] = df["x"] * df["y"] * df["z"]
df["price_inr"] = df["price"] * 83

# Encoding
df['cut'] = df['cut'].astype('category').cat.codes
df['color'] = df['color'].astype('category').cat.codes
df['clarity'] = df['clarity'].astype('category').cat.codes

X = df.drop(["price", "price_inr"], axis=1)
y = df["price_inr"]

X_train, X_test, y_train, y_test = train_test_split(X, y)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ANN
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

model.fit(X_train, y_train, epochs=20, batch_size=32)

# Save
model.save("ann_model.h5")
pickle.dump(scaler, open("ann_scaler.pkl", "wb"))

print("✅ ANN Saved")