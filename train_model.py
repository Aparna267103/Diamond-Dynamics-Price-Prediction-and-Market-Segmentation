import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score
)

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.cluster import KMeans


# =========================================================
# 1. LOAD DATA
# =========================================================

df = pd.read_csv("diamonds.csv")

print("Original Dataset Shape:", df.shape)


# =========================================================
# 2. DATA CLEANING
# =========================================================

df[['x', 'y', 'z']] = df[['x', 'y', 'z']].replace(0, np.nan)

df = df.dropna(subset=['x', 'y', 'z'])

print("After Cleaning:", df.shape)


# =========================================================
# 3. FEATURE ENGINEERING
# =========================================================

# Diamond volume
df["volume"] = df["x"] * df["y"] * df["z"]

# USD to INR conversion
USD_TO_INR = 83.0

df["price_inr"] = df["price"] * USD_TO_INR


# =========================================================
# 4. ENCODING CATEGORICAL FEATURES
# =========================================================

le_cut = LabelEncoder()
le_color = LabelEncoder()
le_clarity = LabelEncoder()

df["cut"] = le_cut.fit_transform(df["cut"])
df["color"] = le_color.fit_transform(df["color"])
df["clarity"] = le_clarity.fit_transform(df["clarity"])


# =========================================================
# 5. SAVE ENCODERS
# =========================================================

with open("le_cut.pkl", "wb") as f:
    pickle.dump(le_cut, f)

with open("le_color.pkl", "wb") as f:
    pickle.dump(le_color, f)

with open("le_clarity.pkl", "wb") as f:
    pickle.dump(le_clarity, f)


# =========================================================
# 6. SELECT FEATURES AND TARGET
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
# 8. MACHINE LEARNING MODELS
# =========================================================

models = {
    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ),

    "Decision Tree": DecisionTreeRegressor(
        random_state=42
    ),

    "KNN": KNeighborsRegressor(
        n_neighbors=5
    )
}


# =========================================================
# 9. TRAIN AND EVALUATE MODELS
# =========================================================

best_model = None
best_model_name = None
best_r2 = -np.inf

results = []


for name, model in models.items():

    # Train
    model.fit(X_train, y_train)

    # Predict
    preds = model.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, preds)

    mse = mean_squared_error(y_test, preds)

    rmse = np.sqrt(mse)

    r2 = r2_score(y_test, preds)

    # Store results
    results.append({
        "Model": name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    })

    # Display
    print("\n--------------------------------------")
    print(name)
    print("--------------------------------------")
    print(f"MAE  : {mae:.2f}")
    print(f"MSE  : {mse:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R2   : {r2:.4f}")

    # Select best model based on R2
    if r2 > best_r2:
        best_r2 = r2
        best_model = model
        best_model_name = name


# =========================================================
# 10. MODEL COMPARISON TABLE
# =========================================================

results_df = pd.DataFrame(results)

print("\n\nMODEL COMPARISON")
print("==============================")
print(results_df.to_string(index=False))

print("\nBest Model:", best_model_name)
print("Best R2   :", round(best_r2, 4))


# =========================================================
# 11. SAVE BEST REGRESSION MODEL
# =========================================================

with open("model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("\nBest model saved as model.pkl")


# =========================================================
# 12. CLUSTERING DATA - Market Segmentation
# =========================================================

cluster_data = X.copy()


# =========================================================
# 13. STANDARDIZATION
# =========================================================

scaler = StandardScaler()

scaled = scaler.fit_transform(cluster_data)


# =========================================================
# 14. FIND BEST NUMBER OF CLUSTERS USING SILHOUETTE SCORE
# =========================================================

silhouette_scores = {}

print("\n\nCLUSTERING ANALYSIS")
print("==============================")

for k in range(2, 7):

    temp_kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    temp_labels = temp_kmeans.fit_predict(scaled)

    score = silhouette_score(
        scaled,
        temp_labels
    )

    silhouette_scores[k] = score

    print(
        f"K = {k} -> "
        f"Silhouette Score = {score:.4f}"
    )


# Select K with highest silhouette score
best_k = max(
    silhouette_scores,
    key=silhouette_scores.get
)

print("\nBest number of clusters:", best_k)


# =========================================================
# 15. FINAL K-MEANS MODEL
# =========================================================

kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(scaled)

df["cluster"] = clusters


# =========================================================
# 16. CLUSTER ANALYSIS
# =========================================================

cluster_summary = df.groupby("cluster").agg(
    average_carat=("carat", "mean"),
    average_price=("price_inr", "mean")
)

print("\nCLUSTER SUMMARY")
print("==============================")
print(cluster_summary)


# =========================================================
# 17. CREATE CLUSTER NAMES
# =========================================================

cluster_order = (
    cluster_summary["average_price"]
    .sort_values()
    .index
)

cluster_names = {}

labels = [
    "💎 Affordable Diamonds",
    "💎 Mid-range Diamonds",
    "💎 Premium Diamonds"
]

for i, cluster_id in enumerate(cluster_order):

    if i < len(labels):
        cluster_names[int(cluster_id)] = labels[i]

    else:
        cluster_names[int(cluster_id)] = (
            f"💎 Market Segment {i + 1}"
        )


print("\nCLUSTER NAMES")
print("==============================")

for cluster_id, name in cluster_names.items():
    print(cluster_id, "->", name)


# =========================================================
# 18. SAVE CLUSTER MODEL AND SCALER
# =========================================================

with open("cluster.pkl", "wb") as f:
    pickle.dump(kmeans, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("cluster_names.pkl", "wb") as f:
    pickle.dump(cluster_names, f)


# =========================================================
# 19. FINAL MESSAGE
# =========================================================

print("\n======================================")
print("✅ TRAINING COMPLETED SUCCESSFULLY")
print("======================================")
print("Best Model      :", best_model_name)
print("Best R2         :", round(best_r2, 4))
print("Best Clusters   :", best_k)
print("Saved Files     :")
print("  - model.pkl")
print("  - le_cut.pkl")
print("  - le_color.pkl")
print("  - le_clarity.pkl")
print("  - cluster.pkl")
print("  - scaler.pkl")
print("  - cluster_names.pkl")
