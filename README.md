# 🌲 EcoType: Forest Cover Type Prediction using Machine Learning

An end-to-end Machine Learning project that predicts the **Forest Cover Type** based on cartographic and environmental features such as elevation, slope, soil type, hillshade values, hydrology distance, and wilderness area information.

The project demonstrates the complete Machine Learning workflow from **data preprocessing and feature engineering** to **model training, evaluation, hyperparameter tuning, and deployment using Streamlit**.

---

## 📌 Project Overview

Forest cover classification plays a vital role in environmental monitoring, forest conservation, wildfire risk assessment, and land resource management.

This project builds multiple classification models to accurately predict the forest cover type using geographical and environmental attributes.

---

## 🎯 Objectives

- Predict forest cover type using supervised Machine Learning
- Perform comprehensive Exploratory Data Analysis (EDA)
- Clean and preprocess real-world environmental data
- Handle class imbalance using SMOTE
- Compare multiple Machine Learning algorithms
- Optimize the best-performing model
- Deploy the model with Streamlit

---

## 🌍 Real-World Applications

- 🌳 Forest Resource Management
- 🔥 Wildfire Risk Assessment
- 🛰️ Land Cover Mapping
- 🌱 Environmental Monitoring
- 📊 Geospatial Data Analysis

---

# 📊 Dataset

**Dataset:** Forest Cover Type Dataset

- Rows: **145,891**
- Columns: **13**
- Target Variable: **Cover_Type**
- Number of Classes: **7**

### Dataset Source

UCI Machine Learning Repository

https://archive.ics.uci.edu/dataset/31/covertype

---

# 📁 Features

- Elevation
- Aspect
- Slope
- Horizontal Distance to Hydrology
- Vertical Distance to Hydrology
- Horizontal Distance to Roadways
- Hillshade (9 AM)
- Hillshade (Noon)
- Hillshade (3 PM)
- Horizontal Distance to Fire Points
- Wilderness Area
- Soil Type

Target:

**Cover_Type**

---

# ⚙️ Project Workflow

```
Data Collection
        ↓
Data Cleaning
        ↓
Exploratory Data Analysis
        ↓
Feature Engineering
        ↓
Class Balancing (SMOTE)
        ↓
Model Training
        ↓
Hyperparameter Tuning
        ↓
Model Evaluation
        ↓
Model Saving
        ↓
Streamlit Deployment
```

---

# 🧹 Data Preprocessing

✔ Missing Value Handling

✔ Duplicate Removal

✔ Outlier Detection (IQR)

✔ Skewness Treatment (log1p)

✔ Feature Engineering

- Hydrology_Distance
- Hillshade_Mean

✔ Label Encoding

✔ Feature Scaling (if required)

---

# 📈 Exploratory Data Analysis

The project includes:

- Missing Value Analysis
- Duplicate Analysis
- Histograms
- Boxplots
- Correlation Heatmap
- Scatterplots
- Pairplots
- Class Distribution
- Feature Importance
- Distribution Analysis

---

# ⚖️ Handling Class Imbalance

The training dataset was balanced using

**SMOTE (Synthetic Minority Oversampling Technique)**

This significantly improved model performance across minority classes.

---

# 🤖 Machine Learning Models

The following models were trained and compared:

- Random Forest Classifier
- Decision Tree Classifier
- Logistic Regression
- K-Nearest Neighbors (KNN)
- XGBoost Classifier

---

# 📊 Model Evaluation

Evaluation Metrics:

- Accuracy Score
- Confusion Matrix
- Classification Report
- Precision
- Recall
- F1 Score

---

# 🔧 Hyperparameter Tuning

The best-performing model was optimized using

**GridSearchCV**

to improve prediction accuracy and generalization.

---

# 💾 Saved Files

```
forest_model.pkl
label_encoder.pkl
```

---

# 🌐 Streamlit Web Application

The Streamlit application allows users to

- Enter environmental feature values
- Predict forest cover type instantly
- Display prediction results interactively

---

# 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Imbalanced-learn (SMOTE)
- Streamlit
- Joblib

---

# 📂 Project Structure

```
EcoType_Project/

│── app.py
│── train.py
│── EDA.py
│── forest_model.pkl
│── label_encoder.pkl
│── requirements.txt
│── README.md
```

---

# 🚀 Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/EcoType_Project.git

cd EcoType_Project
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Download Dataset

Download the dataset from

https://archive.ics.uci.edu/dataset/31/covertype

Place the dataset inside the project folder.

---

## Run EDA

```bash
python EDA.py
```

---

## Train Model

```bash
python train.py
```

---

## Run Streamlit App

```bash
streamlit run app.py
```

---

# 📈 Future Improvements

- Deploy on Streamlit Community Cloud
- Deploy using Render
- Add Feature Selection Techniques
- Integrate Geospatial Visualization
- Compare with Deep Learning Models
- Improve User Interface

---

# 📚 Skills Demonstrated

- Data Cleaning
- Exploratory Data Analysis
- Feature Engineering
- Classification Algorithms
- Model Evaluation
- Hyperparameter Tuning
- Class Imbalance Handling
- Streamlit Deployment
- Machine Learning Pipeline Development

---

# 👩‍💻 Author

**Aparna V**

Aspiring Data Analyst | Python | SQL | Power BI | Machine Learning

---

## ⭐ If you found this project useful, don't forget to Star the repository!
