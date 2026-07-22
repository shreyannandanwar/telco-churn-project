u = "https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv"

"""
Phase 5-6: Model Training and Hyperparameter Tuning
Trains multiple models with cross-validation and hyperparameter optimization.
"""

import pandas as pd
import numpy as np
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, f1_score, recall_score, precision_score
import joblib
import warnings
warnings.filterwarnings('ignore')

from src.config import (
    XGB_PARAMS, MODEL_PATH, RANDOM_STATE, TEST_SIZE
)
from src.preprocess import create_smote_pipeline, save_preprocessor


def _cv_n_jobs():
    """Use single-process CV on Python 3.13+ to avoid loky resource_tracker issues."""
    return 1 if sys.version_info >= (3, 13) else -1

def train_logistic_regression(X_train, y_train, preprocessor):
    """
    Train a Logistic Regression model (baseline).
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training targets
        preprocessor: The preprocessor object
        
    Returns:
        tuple: (model, cv_scores)
    """
    print("\n" + "="*60)
    print("TRAINING: LOGISTIC REGRESSION")
    print("="*60)
    
    # Create pipeline
    pipeline = create_smote_pipeline(preprocessor)
    pipeline.set_params(classifier=LogisticRegression(
        random_state=RANDOM_STATE,
        class_weight='balanced',
        max_iter=1000
    ))
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring='f1',
        n_jobs=_cv_n_jobs()
    )
    
    print(f"CV F1 Scores: {cv_scores}")
    print(f"Mean CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    # Fit on full training data
    pipeline.fit(X_train, y_train)
    
    return pipeline, cv_scores

def train_random_forest(X_train, y_train, preprocessor):
    """
    Train a Random Forest model.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training targets
        preprocessor: The preprocessor object
        
    Returns:
        tuple: (model, cv_scores)
    """
    print("\n" + "="*60)
    print("TRAINING: RANDOM FOREST")
    print("="*60)
    
    # Create pipeline
    pipeline = create_smote_pipeline(preprocessor)
    pipeline.set_params(classifier=RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=_cv_n_jobs()
    ))
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring='f1',
        n_jobs=_cv_n_jobs()
    )
    
    print(f"CV F1 Scores: {cv_scores}")
    print(f"Mean CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    # Fit on full training data
    pipeline.fit(X_train, y_train)
    
    return pipeline, cv_scores

def train_xgboost(X_train, y_train, preprocessor, tune=False):
    """
    Train an XGBoost model with optional hyperparameter tuning.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training targets
        preprocessor: The preprocessor object
        tune (bool): Whether to perform hyperparameter tuning
        
    Returns:
        tuple: (model, cv_scores, best_params)
    """
    print("\n" + "="*60)
    print("TRAINING: XGBOOST")
    print("="*60)
    
    # Base XGBoost parameters
    xgb_params = XGB_PARAMS.copy()
    
    # Create pipeline
    pipeline = create_smote_pipeline(preprocessor)
    pipeline.set_params(classifier=XGBClassifier(**xgb_params))
    
    if tune:
        print("\n🔧 Performing hyperparameter tuning with RandomizedSearchCV...")
        
        # Parameter grid for Random Search
        param_grid = {
            'classifier__n_estimators': [100, 200, 300, 400],
            'classifier__max_depth': [3, 5, 7, 9],
            'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
            'classifier__subsample': [0.6, 0.7, 0.8, 0.9],
            'classifier__colsample_bytree': [0.6, 0.7, 0.8, 0.9],
            'classifier__scale_pos_weight': [1, 2, 3, 4, 5]
        }
        
        # Create random search
        random_search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_grid,
            n_iter=30,
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
            scoring='f1',
            n_jobs=_cv_n_jobs(),
            random_state=RANDOM_STATE,
            verbose=1
        )
        
        # Fit random search
        random_search.fit(X_train, y_train)
        
        # Best model and parameters
        best_pipeline = random_search.best_estimator_
        best_params = random_search.best_params_
        
        print(f"\n✅ Best parameters found:")
        for param, value in best_params.items():
            print(f"  - {param}: {value}")
        print(f"Best CV F1: {random_search.best_score_:.4f}")
        
        return best_pipeline, random_search.cv_results_, best_params
    
    else:
        # No tuning, just use default parameters with cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring='f1',
            n_jobs=_cv_n_jobs()
        )
        
        print(f"CV F1 Scores: {cv_scores}")
        print(f"Mean CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        
        # Fit on full training data
        pipeline.fit(X_train, y_train)
        
        return pipeline, cv_scores, None

def train_all_models(X_train, y_train, preprocessor, tune_xgboost=True):
    """
    Train all models and return results.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training targets
        preprocessor: The preprocessor object
        tune_xgboost (bool): Whether to tune XGBoost
        
    Returns:
        dict: Dictionary of trained models and their metrics
    """
    print("\n" + "🚀"*30)
    print("STARTING MODEL TRAINING")
    print("🚀"*30)
    
    models = {}
    
    # 1. Logistic Regression
    lr_model, lr_cv = train_logistic_regression(X_train, y_train, preprocessor)
    models['LogisticRegression'] = {
        'model': lr_model,
        'cv_scores': lr_cv,
        'cv_mean': lr_cv.mean()
    }
    
    # 2. Random Forest
    rf_model, rf_cv = train_random_forest(X_train, y_train, preprocessor)
    models['RandomForest'] = {
        'model': rf_model,
        'cv_scores': rf_cv,
        'cv_mean': rf_cv.mean()
    }
    
    # 3. XGBoost
    xgb_model, xgb_cv, xgb_params = train_xgboost(
        X_train, y_train, preprocessor, tune=tune_xgboost
    )
    models['XGBoost'] = {
        'model': xgb_model,
        'cv_scores': xgb_cv,
        'cv_mean': xgb_cv.mean() if isinstance(xgb_cv, np.ndarray) else None,
        'best_params': xgb_params
    }
    
    # Print summary
    print("\n" + "="*60)
    print("MODEL TRAINING SUMMARY")
    print("="*60)
    for name, info in models.items():
        if info['cv_mean'] is not None:
            print(f"{name:20} | CV F1: {info['cv_mean']:.4f}")
        elif isinstance(info.get('cv_scores'), dict):
            print(f"{name:20} | Best CV F1: {info['cv_scores']['mean_test_score'].max():.4f}")
        else:
            print(f"{name:20} | CV metric unavailable")
    
    return models

def select_best_model(models, metric='f1'):
    """
    Select the best model based on cross-validation score.
    
    Args:
        models (dict): Dictionary of trained models
        metric (str): Metric to use for selection
        
    Returns:
        tuple: (best_model_name, best_model_info)
    """
    best_name = None
    best_score = -np.inf
    
    for name, info in models.items():
        if info.get('cv_mean') is not None:
            score = info['cv_mean']
        else:
            # For tuned models, use best CV score
            score = info['cv_scores']['mean_test_score'].max()
        
        if score > best_score:
            best_score = score
            best_name = name
    
    print(f"\n🏆 Best Model: {best_name} (CV F1: {best_score:.4f})")
    
    return best_name, models[best_name]

def save_model(model, path=MODEL_PATH):
    """
    Save the trained model.
    
    Args:
        model: The model object to save
        path (Path): Path to save the model
    """
    joblib.dump(model, path)
    print(f"✅ Model saved to {path}")

def load_model(path=MODEL_PATH):
    """
    Load a saved model.
    
    Args:
        path (Path): Path to the saved model
        
    Returns:
        The loaded model
    """
    return joblib.load(path)

if __name__ == "__main__":
    # Test training pipeline
    from src.data_loader import load_and_inspect_data, fix_data_types
    from src.feature_engineering import engineer_features
    from src.preprocess import prepare_data, create_preprocessor
    
    print("🚀 Testing model training pipeline...")
    
    # Load and prepare data
    df = load_and_inspect_data()
    df = fix_data_types(df)
    df = engineer_features(df)
    
    # Split data
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    # Create preprocessor
    preprocessor = create_preprocessor()
    
    # Train models
    models = train_all_models(X_train, y_train, preprocessor, tune_xgboost=False)
    
    # Select best model
    best_name, best_model = select_best_model(models)
    
    print("\n✅ Training test completed!")