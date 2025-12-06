# Customer Churn Prediction

An end-to-end machine learning project for predicting customer churn in a telecommunications company. This project implements a complete data science pipeline from data cleaning to model evaluation and visualization.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Code Structure](#code-structure)
- [Future Improvements](#future-improvements)


## <a name="overview"></a>🎯 Overview

This project predicts customer churn using machine learning algorithms. The pipeline includes:
- **Data Cleaning**: Handling missing values and data type conversions
- **Feature Engineering**: Automatic column type identification, one-hot encoding, and feature scaling
- **Model Training**: Comparison of three different algorithms
- **Evaluation**: Comprehensive metrics including precision, recall, F1-score, and confusion matrices
- **Visualization**: Feature importance plots and correlation matrices

## <a name="features"></a>✨ Features

- **Automatic Data Processing**: Intelligently identifies categorical vs numerical columns
- **Robust Data Cleaning**: Handles missing values and data type inconsistencies
- **Multiple Model Comparison**: Trains and evaluates Logistic Regression, Random Forest, and XGBoost
- **Comprehensive Evaluation**: Uses classification reports and confusion matrices (not just accuracy)
- **Visual Analytics**: Generates feature importance and correlation visualizations
- **Modular Design**: Clean, well-documented, and maintainable code structure

## <a name="dataset"></a>📊 Dataset

The dataset (`WA_Fn-UseC_-Telco-Customer-Churn.csv`) contains information about 7,043 customers with the following characteristics:

- **21 features** including:
  - Demographics: gender, SeniorCitizen, Partner, Dependents
  - Services: PhoneService, MultipleLines, InternetService, OnlineSecurity, etc.
  - Account information: tenure, Contract, PaperlessBilling, PaymentMethod
  - Charges: MonthlyCharges, TotalCharges
- **Target variable**: Churn (Yes/No)
- **Class distribution**: 
  - No Churn: 5,174 (73.5%)
  - Churn: 1,869 (26.5%)

## <a name="installation"></a>🚀 Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Setup

1. Clone or download this repository

2. Install required packages:
```bash
pip install -r requirements.txt
```

The required packages are:
- pandas >= 1.5.0
- numpy >= 1.23.0
- scikit-learn >= 1.2.0
- matplotlib >= 3.6.0
- seaborn >= 0.12.0
- xgboost >= 1.7.0

## <a name="usage"></a>💻 Usage

1. Ensure the dataset file `WA_Fn-UseC_-Telco-Customer-Churn.csv` is in the project directory

2. Run the main script:
```bash
python customer_churn_prediction.py
```

3. The script will:
   - Load and clean the data
   - Perform feature engineering
   - Train three machine learning models
   - Display evaluation metrics
   - Generate visualization files:
     - `feature_importance.png`
     - `correlation_matrix.png`

## <a name="project-structure"></a>📁 Project Structure

```
Customer Churn Prediction/
│
├── customer_churn_prediction.py    # Main script with complete pipeline
├── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Dataset
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
└── Generated Files (after running):
    ├── feature_importance.png      # Feature importance visualization
    └── correlation_matrix.png      # Correlation matrix of numerical features
```

## <a name="methodology"></a>🔬 Methodology

### 1. Data Cleaning
- **TotalCharges Conversion**: Converts object type to numeric (handles empty strings)
- **Missing Values**: Fills NaN values in TotalCharges with median
- **Feature Removal**: Drops customerID (not useful for prediction)
- **Target Encoding**: Converts Churn from Yes/No to binary (1/0)

### 2. Feature Engineering
- **Automatic Column Identification**: 
  - Categorical: Object types and low-cardinality integers (≤10 unique values)
  - Numerical: High-cardinality numeric features
- **One-Hot Encoding**: Applied to categorical variables with `drop_first=True` to avoid multicollinearity
- **Feature Scaling**: StandardScaler applied to numerical features (tenure, MonthlyCharges, TotalCharges)

### 3. Model Training
- **Train-Test Split**: 80% training, 20% testing (stratified to maintain class distribution)
- **Models Evaluated**:
  1. **Logistic Regression**: Baseline linear model
  2. **Random Forest**: Ensemble tree-based model (100 estimators)
  3. **XGBoost**: Gradient boosting classifier

### 4. Evaluation Metrics
Since the dataset is imbalanced, the project uses comprehensive metrics:
- **Classification Report**: Precision, Recall, F1-score for each class
- **Confusion Matrix**: True/False Positives and Negatives
- **Accuracy**: Overall classification accuracy
- **F1-Score**: Harmonic mean of precision and recall (for churn class)
- <img width="2609" height="2367" alt="correlation_matrix" src="https://github.com/user-attachments/assets/16fc3a2d-c3e8-4a93-958c-a18707bf77d8" />

-<img width="2969" height="2367" alt="feature_importance" src="https://github.com/user-attachments/assets/fdde17f1-ba4d-436e-9860-2190231c9a16" />

## <a name="results"></a>📈 Results

### Model Performance Summary

| Model | Accuracy | F1-Score (Churn) |
|-------|----------|------------------|
| **Logistic Regression** | **80.55%** | **0.6029** |
| Random Forest | 78.64% | 0.5501 |
| XGBoost | 78.21% | 0.5621 |

### Key Findings

- **Best Model**: Logistic Regression performed best overall with 80.55% accuracy and 0.6029 F1-score
- **Class Imbalance**: The dataset shows class imbalance (26.5% churn rate), which is why F1-score is more informative than accuracy alone
- **Feature Engineering**: Created 30 features from 19 original features after one-hot encoding
- **Important Features**: The feature importance plot reveals which factors most influence churn prediction

### Model Details

**Logistic Regression**:
- Precision (Churn): 0.66
- Recall (Churn): 0.56
- Confusion Matrix: 927 TN, 108 FP, 166 FN, 208 TP

**Random Forest**:
- Precision (Churn): 0.62
- Recall (Churn): 0.49
- Confusion Matrix: 924 TN, 111 FP, 190 FN, 184 TP

**XGBoost**:
- Precision (Churn): 0.60
- Recall (Churn): 0.53
- Confusion Matrix: 905 TN, 130 FP, 177 FN, 197 TP

## <a name="technologies-used"></a>🛠 Technologies Used

- **Python 3.7+**: Programming language
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **scikit-learn**: Machine learning algorithms and utilities
- **XGBoost**: Gradient boosting framework
- **matplotlib**: Plotting and visualization
- **seaborn**: Statistical data visualization

## <a name="code-structure"></a>📝 Code Structure

The script is organized into modular functions:

- `load_and_clean_data()`: Data loading and cleaning
- `identify_column_types()`: Automatic column type detection
- `engineer_features()`: Feature encoding and scaling
- `train_and_evaluate_models()`: Model training and evaluation
- `plot_feature_importance()`: Feature importance visualization
- `plot_correlation_matrix()`: Correlation matrix visualization
- `main()`: Main execution pipeline

## <a name="future-improvements"></a>🔮 Future Improvements

Potential enhancements for this project:
- Hyperparameter tuning using GridSearchCV or RandomizedSearchCV
- Handling class imbalance with SMOTE or class weights
- Feature selection to reduce dimensionality
- Cross-validation for more robust evaluation
- Model deployment using Flask/FastAPI
- Real-time prediction API



