# 💎 Diamond Dynamics: Price Prediction and Market Segmentation

> **Machine Learning | Regression | ANN | K-Means Clustering | PCA | Streamlit**

Diamond Dynamics is a **Machine Learning and Data Analytics project** focused on predicting diamond prices and identifying meaningful market segments based on diamond characteristics.

The project uses diamond attributes such as **carat, cut, color, clarity, and dimensions** to build regression models for price prediction and clustering models for market segmentation. An interactive **Streamlit web application** allows users to enter diamond details and receive both a predicted price and market segment.

---

## 📌 Project Overview

The diamond market depends on several quality and physical attributes when determining prices. Accurately predicting diamond prices can support **pricing strategies, inventory management, product recommendations, and customer targeting**.

This project addresses two major problems:

1. 💰 **Diamond Price Prediction** — Predict the price of a diamond using Machine Learning regression algorithms and an Artificial Neural Network.
2. 🎯 **Market Segmentation** — Group diamonds into meaningful clusters based on their physical and qualitative characteristics.

The final solution includes a trained ML pipeline and an interactive Streamlit application. 

---

## 🎯 Objectives

* Predict diamond prices using multiple regression algorithms.
* Build an **Artificial Neural Network (ANN)** for price prediction.
* Segment diamonds into meaningful market groups using clustering.
* Identify the optimal number of clusters using methods such as the **Elbow Method** and **Silhouette Score**.
* Apply feature engineering and feature selection to improve model performance.
* Use **PCA** for dimensionality reduction and cluster visualization.
* Develop an interactive **Streamlit application** for real-time predictions. 

---

## 📊 Dataset

The project uses a **Diamond Dataset** containing:

* **53,940 rows**
* **10 features**

### Dataset Features

| Feature   | Description                             |
| --------- | --------------------------------------- |
| `carat`   | Weight of the diamond in carats         |
| `cut`     | Quality of the diamond cut              |
| `color`   | Diamond color grade from D to J         |
| `clarity` | Measurement of inclusions and blemishes |
| `depth`   | Total depth percentage                  |
| `table`   | Width of the diamond's top facet        |
| `price`   | Diamond price in USD                    |
| `x`       | Diamond length in mm                    |
| `y`       | Diamond width in mm                     |
| `z`       | Diamond depth/height in mm              |

The original project specification indicates that the price is converted from **USD to INR** for the application. 

---

## 🔄 Machine Learning Workflow

```text
                ┌─────────────────────┐
                │   Diamond Dataset   │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Data Preprocessing   │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │        EDA          │
                │ Visualization       │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Feature Engineering │
                │ & Feature Selection │
                └──────────┬──────────┘
                           │
                ┌──────────┴──────────┐
                ▼                     ▼
       ┌─────────────────┐   ┌─────────────────┐
       │ Price Prediction│   │    Clustering   │
       │   Regression    │   │   Segmentation  │
       └────────┬────────┘   └────────┬────────┘
                │                     │
                ▼                     ▼
       ┌─────────────────┐   ┌─────────────────┐
       │ Best ML / ANN   │   │ K-Means + PCA   │
       │     Model       │   │                 │
       └────────┬────────┘   └────────┬────────┘
                │                     │
                └──────────┬──────────┘
                           ▼
                ┌─────────────────────┐
                │   Streamlit App     │
                └─────────────────────┘
```

---

## 🧹 Data Preprocessing

The preprocessing stage includes:

* Handling missing values.
* Detecting invalid or zero values in `x`, `y`, and `z`.
* Imputing or treating invalid values where necessary.
* Removing irrelevant columns when appropriate.
* Detecting and handling numerical outliers.
* Checking skewness in numerical variables.
* Applying transformations such as:

  * Log transformation
  * Square-root transformation
  * Box-Cox transformation

Outlier detection can use **IQR** or **Z-Score** methods, supported by visual inspection through boxplots. 

---

## 📈 Exploratory Data Analysis

The project performs EDA to understand relationships between diamond attributes and price.

### Visualizations

* Price distribution
* Carat distribution
* Dimension distributions
* Count plots for `cut`, `color`, and `clarity`
* Price vs. carat
* Price variation by cut
* Price variation by color
* Price variation by clarity
* Correlation heatmap
* Scatterplot matrix
* Pairwise relationships
* Average price by category

These analyses help identify important patterns and relationships before model training. 

---

## ⚙️ Feature Engineering

Additional features can be derived from the existing diamond attributes.

### Engineered Features

```text
Volume = x × y × z

Price per Carat = price / carat

Dimension Ratio = (x + y) / (2 × z)
```

Diamonds can also be categorized by carat:

| Category | Carat Range |
| -------- | ----------- |
| Light    | `< 0.5`     |
| Medium   | `0.5 – 1.5` |
| Heavy    | `> 1.5`     |

The project also specifies converting the original USD price into INR using a fixed or dynamic conversion rate. 

---

## 🔍 Feature Selection

Feature selection can be performed using techniques such as:

* Correlation Matrix Analysis
* Feature Importance
* Recursive Feature Elimination (RFE)
* Variance Inflation Factor (VIF)

The selected approach depends on the model and feature relationships. 

---

# 💰 Price Prediction

The regression component predicts the price of a diamond from its characteristics.

### Models

The project includes multiple regression algorithms, such as:

* Linear Regression
* Decision Tree
* Random Forest
* XGBoost
* K-Nearest Neighbors (KNN)
* Artificial Neural Network (ANN)

The dataset is divided into training and testing sets, with an **80:20 or 70:30 split**. 

### Evaluation Metrics

Models are evaluated using:

* **MAE** — Mean Absolute Error
* **MSE** — Mean Squared Error
* **RMSE** — Root Mean Squared Error
* **R² Score** — Coefficient of Determination

The best-performing model is saved as a `.pkl` file for deployment. 

---

# 🎯 Market Segmentation

The clustering component groups diamonds into market segments based on relevant physical and qualitative characteristics.

### Clustering Approach

The project primarily uses:

**K-Means Clustering**

Other possible approaches include:

* DBSCAN
* Hierarchical Clustering

The optimal number of clusters can be identified using:

* Elbow Method
* Silhouette Score

Categorical variables such as `cut`, `color`, and `clarity` are encoded before clustering. 

> **Note:** Price is excluded from the clustering model so that market segments are based on diamond characteristics rather than directly on price.

---

## 📊 PCA Visualization

**Principal Component Analysis (PCA)** can be applied to reduce the feature space to two or three principal components.

This makes it possible to visualize the resulting clusters using 2D or 3D scatter plots. 

---

# 🏷️ Cluster Naming

After clustering, each cluster can be analyzed based on characteristics such as:

* Average carat
* Average price
* Cut distribution
* Other diamond characteristics

Example segment names include:

| Segment                        | Description                            |
| ------------------------------ | -------------------------------------- |
| 💎 Premium Heavy Diamonds      | Large, expensive, premium-grade stones |
| 💠 Affordable Small Diamonds   | Small, budget-friendly stones          |
| 🔷 Mid-range Balanced Diamonds | Balanced size and cost                 |

These names are examples from the project specification and should be adjusted according to the actual cluster characteristics obtained from the trained model. 

---

# 🖥️ Streamlit Application

The project includes an interactive **Streamlit web application**.

## 💰 Price Prediction Module

Users can provide:

* Carat
* Cut
* Color
* Clarity
* X dimension
* Y dimension
* Z dimension

The application then predicts the diamond price in **INR**. 

---

## 🎯 Market Segment Prediction

The same diamond attributes can be used to identify the diamond's market segment.

The application displays:

```text
Cluster Number
        ↓
Cluster Name
        ↓
Market Segment
```

Example:

```text
Cluster: 2
Segment: Premium Heavy Diamonds
```

The cluster name is mapped from the characteristics of the trained clusters. 

---

# 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* TensorFlow / Keras
* XGBoost
* Matplotlib
* Seaborn
* Streamlit
* Pickle
* PCA
* K-Means Clustering

### Key Concepts

```text
Machine Learning
Regression
Artificial Neural Networks
Feature Engineering
Feature Selection
EDA
Outlier Detection
Skewness Treatment
K-Means Clustering
PCA
Model Evaluation
Streamlit Deployment
```

These technologies and concepts are aligned with the technical tags in the project specification. 

---

# 📁 Suggested Project Structure

```text
Diamond-Dynamics/
│
├── 📂 data/
│   └── diamonds.csv
│
├── 📂 notebooks/
│   └── diamond_dynamics.ipynb
│
├── 📂 models/
│   ├── regression_model.pkl
│   ├── clustering_model.pkl
│   └── preprocessing.pkl
│
├── 📂 app/
│   └── app.py
│
├── 📂 images/
│   ├── eda.png
│   ├── correlation.png
│   ├── clusters.png
│   └── streamlit_app.png
│
├── requirements.txt
├── README.md
└── .gitignore
```

> Update the filenames and folders to match your actual GitHub repository structure.

---

# 🚀 How to Run the Project

## 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/Diamond-Dynamics.git
cd Diamond-Dynamics
```

## 2. Create a Virtual Environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### macOS / Linux

```bash
source venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your browser.

---

# 📦 Model Deployment

The trained regression and clustering models can be serialized using Pickle:

```text
regression_model.pkl
clustering_model.pkl
```

These models are loaded by the Streamlit application to generate predictions without retraining the models every time the application runs. The project specification explicitly includes saving the best regression and clustering models as `.pkl` files for Streamlit use. 

---

# 🌍 Real-World Applications

Diamond Dynamics can support several real-world use cases:

### 💰 Dynamic Pricing

Retailers can estimate appropriate diamond prices based on quality and physical characteristics.

### 📦 Inventory Management

Diamonds can be grouped into meaningful categories for easier inventory organization.

### 🛍️ Product Recommendations

Diamond profiles can be used to build recommendation systems for customers.

### 🎯 Personalized Marketing

Market segments can support targeted marketing strategies and customer personalization. 

---

# 📊 Project Deliverables

The project deliverables include:

* Python/Jupyter Notebook
* Data preprocessing pipeline
* EDA visualizations
* Feature engineering
* Feature selection
* Multiple regression models
* ANN regression model
* Regression evaluation metrics
* K-Means clustering model
* Cluster evaluation
* PCA visualization
* Saved `.pkl` models
* Streamlit web application
* Interactive price prediction
* Interactive market segmentation



---

# 📈 Future Enhancements

Possible improvements include:

* Deploying the Streamlit application online.
* Adding more advanced regression models.
* Improving ANN architecture and hyperparameter tuning.
* Adding automated model comparison.
* Adding interactive cluster visualizations.
* Adding customer-oriented diamond recommendations.
* Adding dynamic currency conversion.
* Implementing model monitoring and retraining pipelines.

---

# 👨‍💻 Skills Demonstrated

Through this project, the following skills are demonstrated:

* Data Cleaning & Preprocessing
* Exploratory Data Analysis
* Data Visualization
* Feature Engineering
* Feature Selection
* Outlier Detection
* Skewness Handling
* Regression
* Artificial Neural Networks
* K-Means Clustering
* PCA
* Model Evaluation
* Streamlit Application Development
* Machine Learning Model Deployment



---

# ⭐ Project Highlights

```text
💎 53,940 Diamond Records
💰 Price Prediction
🤖 Multiple ML Regression Models
🧠 Artificial Neural Network
🎯 K-Means Market Segmentation
📊 PCA Visualization
📈 Comprehensive EDA
⚙️ Feature Engineering
🖥️ Interactive Streamlit UI
📦 Pickle Model Deployment
```

---

## 📌 Conclusion

**Diamond Dynamics** combines supervised and unsupervised Machine Learning to create a practical analytics solution for the diamond market.

The project demonstrates how diamond characteristics can be transformed into actionable insights through **price prediction, market segmentation, data visualization, and interactive deployment**.

> 💎 **Predict the Price. Understand the Segment. Discover the Dynamics.**
