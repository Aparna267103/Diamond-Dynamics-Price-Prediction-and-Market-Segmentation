import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load
df = pd.read_csv("diamonds.csv")

# Cleaning
df = df[(df['x'] > 0) & (df['y'] > 0) & (df['z'] > 0)]

# Feature Engineering
df["volume"] = df["x"] * df["y"] * df["z"]
df["price_inr"] = df["price"] * 83

# Encoding
le_cut, le_color, le_clarity = LabelEncoder(), LabelEncoder(), LabelEncoder()

df['cut'] = le_cut.fit_transform(df['cut'])
df['color'] = le_color.fit_transform(df['color'])
df['clarity'] = le_clarity.fit_transform(df['clarity'])

# Save encoders
pickle.dump(le_cut, open("le_cut.pkl", "wb"))
pickle.dump(le_color, open("le_color.pkl", "wb"))
pickle.dump(le_clarity, open("le_clarity.pkl", "wb"))

# Split
X = df[['carat','cut','color','clarity','x','y','z','volume']]
y = df["price_inr"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Models
models = {
    "Linear": LinearRegression(),
    "RandomForest": RandomForestRegressor(),
    "DecisionTree": DecisionTreeRegressor(),
    "KNN": KNeighborsRegressor()
}

best_model = None
best_r2 = -1

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print(f"{name} -> MAE:{mae}, RMSE:{rmse}, R2:{r2}")

    if r2 > best_r2:
        best_r2 = r2
        best_model = model

# Save best model
pickle.dump(best_model, open("model.pkl", "wb"))

# ================= CLUSTERING =================
cluster_data = X.copy()

scaler = StandardScaler()
scaled = scaler.fit_transform(cluster_data)

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(scaled)

sil_score = silhouette_score(scaled, clusters)
print("Silhouette Score:", sil_score)

pickle.dump(kmeans, open("cluster.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("✅ Training Completed")