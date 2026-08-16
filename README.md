# Aerial Object Classification & Detection

A deep learning and computer vision project for **classifying aerial images as Bird or Drone**, with an optional **YOLOv8-based object detection** component for locating and labeling objects in real-world aerial scenes.

The project is designed for applications such as **aerial surveillance, wildlife monitoring, airport safety, and security & defense**.

## 📌 Project Overview

The primary goal is to build an AI-based solution capable of distinguishing between **birds and drones** from aerial images.

The project includes:

* Custom CNN-based image classification
* Transfer learning using pretrained deep learning models
* Data preprocessing and augmentation
* Model training and evaluation
* Optional YOLOv8 object detection
* Streamlit-based deployment
* Model comparison and performance analysis

## 🎯 Objectives

1. Classify aerial images into two categories:

   * 🐦 Bird
   * 🚁 Drone
2. Develop a Custom CNN classification model.
3. Apply transfer learning using pretrained architectures.
4. Compare model performance using appropriate evaluation metrics.
5. Optionally detect and localize birds and drones using YOLOv8.
6. Deploy the final solution through an interactive Streamlit application.

## 💼 Real-World Applications

### 🦅 Wildlife Protection

Detect birds near wind farms or airports to help prevent accidents.

### 🛡️ Security & Defense Surveillance

Identify drones entering restricted airspace and support timely alerts.

### ✈️ Airport Bird-Strike Prevention

Monitor runway areas for bird activity.

### 🌱 Environmental Research

Track bird populations using aerial imagery while reducing misclassification.

## 🧠 Technologies & Skills

* Python
* Deep Learning
* Computer Vision
* Image Classification
* Object Detection
* Convolutional Neural Networks (CNN)
* TensorFlow / Keras or PyTorch
* Transfer Learning
* Data Preprocessing
* Data Augmentation
* YOLOv8
* Model Evaluation
* Streamlit

## 📂 Dataset

### Classification Dataset

**Source:** `classification_dataset`

**Task:** Binary image classification — Bird vs Drone

**Data type:** RGB images
**Format:** `.jpg`

| Split      |  Bird | Drone | Total |
| ---------- | ----: | ----: | ----: |
| Train      | 1,414 | 1,248 | 2,662 |
| Validation |   217 |   225 |   442 |
| Test       |   121 |    94 |   215 |

**Total images:** 3,319

### Object Detection Dataset

**Source:** `object_detection_Dataset`

The object detection dataset contains **3,319 images** with corresponding YOLOv8-format `.txt` annotations.

Each annotation follows:

```text
<class_id> <x_center> <y_center> <width> <height>
```

Dataset split:

* Train: 2,662 images
* Validation: 442 images
* Test: 215 images

## 🔄 Project Workflow

```text
Dataset
   │
   ▼
Data Understanding
   │
   ▼
Data Preprocessing
   │
   ▼
Data Augmentation
   │
   ├───────────────┐
   ▼               ▼
Custom CNN    Transfer Learning
   │               │
   └───────┬───────┘
           ▼
      Model Training
           │
           ▼
      Model Evaluation
           │
           ▼
     Model Comparison
           │
           ▼
    Best Model Selection
           │
           ▼
   Streamlit Deployment
           │
           ▼
 Optional YOLOv8 Detection
```

The documented workflow covers dataset inspection, preprocessing, augmentation, model building, training, evaluation, model comparison, and deployment.

## 🧹 Data Preprocessing

For classification:

* Inspect the dataset structure.
* Check the number of images in each class.
* Identify potential class imbalance.
* Visualize sample images.
* Resize images to **224 × 224**.
* Normalize pixel values to the range **[0, 1]**.

For transfer learning:

* TensorFlow models use their model-specific `preprocess_input`.
* PyTorch pretrained models use ImageNet normalization according to the model's training configuration.

## 🔀 Data Augmentation

The project applies image transformations such as:

* Rotation
* Flipping
* Zoom
* Brightness adjustment
* Cropping

These transformations are intended to improve the model's ability to generalize to different aerial image conditions.

## 🏗️ Model Architecture

### Custom CNN

The Custom CNN includes:

* Convolutional layers
* Pooling layers
* Batch normalization
* Dropout
* Dense output layer

### Transfer Learning

The project can leverage pretrained architectures such as:

* ResNet50
* MobileNet
* EfficientNetB0

These models can be fine-tuned for the Bird vs Drone classification task.

## 🏋️ Model Training

Both the Custom CNN and transfer-learning models are trained and compared.

Training includes:

* EarlyStopping
* ModelCheckpoint
* Accuracy tracking
* Precision
* Recall
* F1-score

## 📊 Model Evaluation

Model performance is evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix
* Classification report
* Training/validation accuracy graphs
* Training/validation loss graphs

The models are compared based on **accuracy, training time, and generalization performance**. The best-performing model is selected for deployment.

> **Note:** The project document specifies the evaluation methodology but does not provide final trained-model performance results. Therefore, actual accuracy or other metric values should be added here after model training.

## 🚁 Optional YOLOv8 Object Detection

The project optionally extends classification into object detection using **YOLOv8**.

### Workflow

1. Install YOLOv8.
2. Prepare the YOLOv8-format dataset.
3. Create the `data.yaml` configuration.
4. Train the YOLOv8 model.
5. Validate the model.
6. Run inference on test or new images.

The detection model can provide bounding boxes around detected birds and drones.

## 🌐 Streamlit Deployment

The final model can be deployed through an interactive Streamlit application.

The application should allow users to:

1. Upload an image.
2. Run the trained classification model.
3. Display the predicted class.
4. Display the prediction confidence score.
5. Optionally display YOLOv8 detection results with bounding boxes.

Example:

```text
Upload Image
     │
     ▼
AI Model
     │
     ├── Bird → Confidence: XX%
     │
     └── Drone → Confidence: XX%

Optional:
YOLOv8 → Bounding Boxes + Labels
```

## 📁 Suggested Repository Structure

```text
Aerial-Object-Classification-Detection/
│
├── README.md
├── requirements.txt
│
├── data/
│   ├── classification_dataset/
│   │   ├── train/
│   │   │   ├── bird/
│   │   │   └── drone/
│   │   ├── valid/
│   │   │   ├── bird/
│   │   │   └── drone/
│   │   └── test/
│   │       ├── bird/
│   │       └── drone/
│   │
│   └── object_detection_Dataset/
│
├── notebooks/
│   ├── data_preprocessing.ipynb
│   ├── custom_cnn.ipynb
│   ├── transfer_learning.ipynb
│   └── yolov8_detection.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── models/
│   ├── custom_cnn/
│   ├── transfer_learning/
│   └── yolov8/
│
├── app/
│   └── app.py
│
└── results/
    ├── confusion_matrix/
    ├── training_plots/
    └── model_comparison/
```

*This repository structure is a suggested organization for implementing the deliverables described in the project document; it is not a directory structure specified in the source document.*

## 📦 Project Deliverables

The expected deliverables include:

* Trained Custom CNN model
* Trained transfer-learning model
* YOLOv8 model *(optional)*
* Streamlit classification/detection application
* Preprocessing, training, and evaluation scripts/notebooks
* Model comparison report
* GitHub repository with documentation
* Well-structured and commented code

## 🏷️ Technical Tags

`Computer Vision` · `Deep Learning` · `Image Classification` · `Object Detection` · `CNN` · `YOLOv8` · `Transfer Learning` · `Data Augmentation` · `Model Evaluation` · `Streamlit` · `Aerial Surveillance AI`

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone <your-github-repository-url>
cd Aerial-Object-Classification-Detection
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare the Dataset

Place the classification dataset and, if implementing object detection, the YOLOv8 dataset in the appropriate data directories.

### 5. Train the Model

Run the appropriate training notebook or script for:

* Custom CNN
* Transfer Learning
* YOLOv8 *(optional)*

### 6. Evaluate the Model

Generate:

* Classification report
* Confusion matrix
* Accuracy/loss plots
* Model comparison results

### 7. Run the Streamlit Application

```bash
streamlit run app/app.py
```

> The exact commands and filenames depend on the implementation of the repository and are not specified in the project document.

## 📈 Results

### Classification Results

| Model          | Accuracy | Precision | Recall | F1-Score | Training Time |
| -------------- | -------: | --------: | -----: | -------: | ------------: |
| Custom CNN     |      TBD |       TBD |    TBD |      TBD |           TBD |
| ResNet50       |      TBD |       TBD |    TBD |      TBD |           TBD |
| MobileNet      |      TBD |       TBD |    TBD |      TBD |           TBD |
| EfficientNetB0 |      TBD |       TBD |    TBD |      TBD |           TBD |

### Object Detection Results

| Model  | mAP | Precision | Recall |
| ------ | --: | --------: | -----: |
| YOLOv8 | TBD |       TBD |    TBD |

**Replace the `TBD` values with the actual results obtained during training and evaluation.**

## 🔮 Future Scope

Potential extensions based on the project's classification/detection direction include:

* Real-time aerial video analysis
* Automated drone alerts
* Improved object localization
* Deployment on edge devices
* Integration with surveillance systems
* Expansion to additional aerial object classes

These are possible extensions rather than documented project requirements.

## ⏱️ Timeline

The project document specifies a completion timeline of **10 days from the date of assignment**.

## 👥 Project Credits

**Created by:** Nilofer Mubeen
**Verified by:** Shadiya Nehlath
**Approved by:** Harmain

## 📄 Project Reference

This README is based on the provided project specification for **Aerial Object Classification & Detection**.

---

⭐ If this project is useful, consider giving the repository a star!
