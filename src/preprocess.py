"""
Phase 4: Preprocessing Pipeline
Creates a robust preprocessing pipeline using ColumnTransformer to prevent data leakage.
"""

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
from pathlib import Path

from src.config import (
    TARGET_COLUMN, NUMERIC_FEATURES, CATEGORICAL_FEATURES,
    DROP_FEATURES, TEST_SIZE, RANDOM_STATE, PREPROCESSOR_PATH,
    ENGINEERED_FEATURES
)

def create_preprocessor():
    """
    Create a preprocessing pipeline using ColumnTransformer.
    
    Returns:
        ColumnTransformer: The preprocessor object
    """
    print("\n🔧 Building preprocessing pipeline...")
    
    # Numeric pipeline: Impute missing values -> Scale
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical pipeline: Impute missing -> One-hot encode
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine into ColumnTransformer
    preprocessor = ColumnTransformer([
        ('numeric', numeric_pipeline, NUMERIC_FEATURES),
        ('categorical', categorical_pipeline, CATEGORICAL_FEATURES)
    ], remainder='drop')  # Drop any columns not specified
    
    print("✅ Preprocessor created successfully!")
    print(f"  - Numeric features: {len(NUMERIC_FEATURES)}")
    print(f"  - Categorical features: {len(CATEGORICAL_FEATURES)}")
    
    return preprocessor

def prepare_data(df, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """
    Split data into train and test sets.
    
    Args:
        df (pd.DataFrame): Input dataframe with features and target
        test_size (float): Proportion of test set
        random_state (int): Random seed
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    print("\n📊 Splitting data into train/test sets...")
    
    # Separate features and target
    X = df.drop(columns=[TARGET_COLUMN] + DROP_FEATURES)
    y = df[TARGET_COLUMN].apply(lambda x: 1 if x == 'Yes' else 0)  # Binary encoding
    
    # Split with stratification (maintain class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"✅ Data split complete!")
    print(f"  - Training set: {X_train.shape[0]} samples")
    print(f"  - Test set: {X_test.shape[0]} samples")
    print(f"  - Training churn rate: {y_train.mean():.2%}")
    print(f"  - Test churn rate: {y_test.mean():.2%}")
    
    return X_train, X_test, y_train, y_test

def create_smote_pipeline(preprocessor):
    """
    Create a pipeline with preprocessing and SMOTE (for training).
    
    Args:
        preprocessor (ColumnTransformer): The preprocessor
        
    Returns:
        Pipeline: Full pipeline with preprocessing + SMOTE
    """
    print("\n🔧 Creating SMOTE pipeline...")
    
    # Pipeline order: Preprocess -> SMOTE -> Classifier (will be added later)
    pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=RANDOM_STATE, sampling_strategy='auto')),
        ('classifier', None)  # Will be filled during training
    ])
    
    print("✅ SMOTE pipeline created!")
    print("  - SMOTE will be applied ONLY to training data during cross-validation")
    
    return pipeline

def save_preprocessor(preprocessor, path=PREPROCESSOR_PATH):
    """
    Save the preprocessor for later use.
    
    Args:
        preprocessor (ColumnTransformer): The preprocessor to save
        path (Path): Path to save the preprocessor
    """
    joblib.dump(preprocessor, path)
    print(f"✅ Preprocessor saved to {path}")

def load_preprocessor(path=PREPROCESSOR_PATH):
    """
    Load a saved preprocessor.
    
    Args:
        path (Path): Path to the saved preprocessor
        
    Returns:
        ColumnTransformer: The loaded preprocessor
    """
    return joblib.load(path)

def apply_preprocessing(X_train, X_test, preprocessor=None, fit_preprocessor=True):
    """
    Apply preprocessing to train and test sets.
    
    Args:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Test features
        preprocessor (ColumnTransformer): Preprocessor (if None, creates new)
        fit_preprocessor (bool): Whether to fit the preprocessor
        
    Returns:
        tuple: (X_train_processed, X_test_processed, preprocessor)
    """
    if preprocessor is None:
        preprocessor = create_preprocessor()
    
    if fit_preprocessor:
        # Fit on training data only
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        print("✅ Preprocessor fitted on training data and applied to test data")
    else:
        # Use existing preprocessor
        X_train_processed = preprocessor.transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        print("✅ Preprocessor applied to both datasets")
    
    return X_train_processed, X_test_processed, preprocessor

if __name__ == "__main__":
    # Test preprocessing pipeline
    from src.data_loader import load_and_inspect_data, fix_data_types
    from src.feature_engineering import engineer_features
    
    print("🚀 Testing preprocessing pipeline...")
    
    # Load and prepare data
    df = load_and_inspect_data()
    df = fix_data_types(df)
    df = engineer_features(df)
    
    # Split data
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    # Create and apply preprocessor
    preprocessor = create_preprocessor()
    X_train_proc, X_test_proc, _ = apply_preprocessing(X_train, X_test, preprocessor)
    
    print(f"\n📊 Processed data shapes:")
    print(f"  - X_train: {X_train_proc.shape}")
    print(f"  - X_test: {X_test_proc.shape}")
    
    print("\n✅ Preprocessing test completed!")