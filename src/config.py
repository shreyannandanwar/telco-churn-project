"""
Configuration file for the Telco Churn project.
Centralizes all paths, constants, and hyperparameters.
"""

import os
from pathlib import Path

# Project root directory
ROOT_DIR = Path(__file__).parent.parent

# Data paths
DATA_DIR = ROOT_DIR / 'data'
RAW_DATA_PATH = DATA_DIR / 'Telco-Customer-Churn.csv'
PROCESSED_DATA_PATH = DATA_DIR / 'processed_data.csv'

# Model paths
MODELS_DIR = ROOT_DIR / 'models'
MODEL_PATH = MODELS_DIR / 'xgboost_churn_model.pkl'
PREPROCESSOR_PATH = MODELS_DIR / 'preprocessor.pkl'
ENCODER_PATH = MODELS_DIR / 'label_encoders.pkl'

# Reports path
REPORTS_DIR = ROOT_DIR / 'reports'
FIGURES_DIR = REPORTS_DIR / 'figures'

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODELS_DIR, FIGURES_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Data URL
DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv"

# Model hyperparameters (will be tuned)
XGB_PARAMS = {
    'n_estimators': 300,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': 3,  # Accounts for class imbalance
    'random_state': 42,
    'eval_metric': 'logloss',
    'use_label_encoder': False
}

# Test size and random seed
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Target column
TARGET_COLUMN = 'Churn'

# Numeric features
NUMERIC_FEATURES = [
    'tenure', 'MonthlyCharges', 'TotalCharges'
]

# Categorical features (will be one-hot encoded)
CATEGORICAL_FEATURES = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod'
]

# Features to drop (not useful for prediction)
DROP_FEATURES = ['customerID']

# Engineered features (will be created)
ENGINEERED_FEATURES = [
    'avg_monthly_spend',
    'tenure_group',
    'monthly_charge_per_tenure',
    'has_phone_service',
    'has_internet',
    'services_count'
]