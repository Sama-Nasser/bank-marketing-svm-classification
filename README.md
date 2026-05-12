# Bank Marketing Campaign Prediction — SVM Pipeline

## Overview
This project builds a complete Machine Learning pipeline to predict whether a bank client will subscribe to a term deposit based on marketing campaign data.

The solution is built using a Support Vector Machine (SVM) classification pipeline with advanced preprocessing, dimensionality reduction, imbalance handling, and model evaluation techniques.

The dataset used is the **Bank Marketing Dataset** from the UCI Machine Learning Repository.

---

## Business Problem
Banks spend significant resources running marketing campaigns through phone calls and customer outreach.

The goal of this project is to help marketing teams:

- Identify clients more likely to subscribe to a term deposit
- Reduce unnecessary campaign costs
- Improve conversion rates
- Support data-driven marketing decisions
- Increase campaign efficiency and ROI

---

## Dataset Information
- Source: UCI Bank Marketing Dataset
- Records: 41,188
- Features: 20 input features + target variable
- Target: `y`
  - `yes` → client subscribed
  - `no` → client did not subscribe

---

## Project Workflow

### 1. Data Preparation
- Train/Test split using stratification
- Missing value handling
- Data cleaning
- Outlier inspection

### 2. Exploratory Data Analysis (EDA)
- Target distribution analysis
- Numerical feature visualization
- Correlation analysis
- Business insights extraction

### 3. Feature Engineering & Preprocessing
- StandardScaler for numerical features
- OrdinalEncoder for categorical features
- PCA for dimensionality reduction

### 4. Imbalanced Data Handling
Because the dataset is highly imbalanced, the project applies:

- SMOTE oversampling
- `class_weight='balanced'`

This improves the model’s ability to detect positive subscribers.

### 5. Model Training
Model used:

- Linear Support Vector Machine (LinearSVC)
- Probability calibration using CalibratedClassifierCV

### 6. Model Evaluation
Evaluation metrics include:

- ROC-AUC
- Precision & Recall
- F1-score
- PR Curve
- Confusion Matrix

---

## Final Model Performance

| Metric | Score |
|---|---|
| Test ROC-AUC | 0.9357 |
| Recall (Yes Class) | 0.8804 |
| F1-Score (Yes Class) | 0.5876 |
| PCA Components | 12 |

### Key Result
The model successfully identifies approximately **88% of potential subscribers**, making it useful for marketing prioritization and customer targeting.

---

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Matplotlib
- Seaborn
- Jupyter Notebook
- Streamlit (Deployment)

---

## Project Structure

 
