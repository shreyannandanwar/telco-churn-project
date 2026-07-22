"""
Phase 8: Model Interpretability with SHAP
Understand what drives churn predictions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
import warnings
warnings.filterwarnings('ignore')

from src.config import FIGURES_DIR, MODEL_PATH, PREPROCESSOR_PATH, NUMERIC_FEATURES
from src.preprocess import load_preprocessor
from src.train import load_model

def _prepare_shap_inputs(model, X_sample, feature_names, preprocessor=None):
    """
    Prepare SHAP inputs so they match the fitted model's expected feature space.

    Returns:
        tuple: (X_for_shap, feature_names_for_shap)
    """
    fitted_preprocessor = None

    if hasattr(model, 'named_steps') and 'preprocessor' in model.named_steps:
        fitted_preprocessor = model.named_steps['preprocessor']
    elif preprocessor is not None:
        fitted_preprocessor = preprocessor

    if fitted_preprocessor is None:
        return X_sample.copy(), feature_names

    X_processed = fitted_preprocessor.transform(X_sample)

    if hasattr(fitted_preprocessor, 'get_feature_names_out'):
        transformed_feature_names = fitted_preprocessor.get_feature_names_out().tolist()
    else:
        transformed_feature_names = feature_names

    X_for_shap = pd.DataFrame(
        X_processed,
        columns=transformed_feature_names,
        index=X_sample.index
    )

    return X_for_shap, transformed_feature_names


def explain_with_shap(model, X_sample, feature_names):
    """
    Use SHAP to explain model predictions.
    
    Args:
        model: Trained model
        X_sample (pd.DataFrame): Sample data to explain
        feature_names (list): Names of features
    Returns:
        tuple: (shap_values, explainer)
    """
    print("\n" + "="*60)
    print("SHAP ANALYSIS")
    print("="*60)
    
    # Get the underlying model (if pipeline, extract classifier)
    if hasattr(model, 'named_steps'):
        # Check if it's a pipeline with classifier
        if 'classifier' in model.named_steps:
            base_model = model.named_steps['classifier']
        else:
            base_model = model.named_steps.get('classifier', model)
    else:
        base_model = model
    
    # Handle XGBoost
    if 'XGB' in str(type(base_model)):
        print("Using TreeExplainer for XGBoost...")
        explainer = shap.TreeExplainer(base_model)
        shap_values = explainer.shap_values(X_sample)
    
    # Handle Random Forest
    elif 'RandomForest' in str(type(base_model)):
        print("Using TreeExplainer for Random Forest...")
        explainer = shap.TreeExplainer(base_model)
        shap_values = explainer.shap_values(X_sample)
    
    # Handle Linear models
    else:
        print("Using LinearExplainer...")
        explainer = shap.LinearExplainer(base_model, X_sample)
        shap_values = explainer.shap_values(X_sample)
    
    print(f"✅ SHAP values computed for {X_sample.shape[0]} samples")
    
    return shap_values, explainer


def _positive_class_shap_values(shap_values):
    """
    Normalize SHAP outputs to a 2D matrix (n_samples, n_features) for class 1.
    """
    if isinstance(shap_values, list):
        return shap_values[1] if len(shap_values) > 1 else shap_values[0]

    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        # Common format in newer SHAP for tree classifiers: (samples, features, classes)
        return shap_values[:, :, 1] if shap_values.shape[2] > 1 else shap_values[:, :, 0]

    return shap_values

def plot_shap_summary(shap_values, X_sample, feature_names):
    """
    Plot SHAP summary (beeswarm) plot.
    
    Args:
        shap_values: SHAP values
        X_sample (pd.DataFrame): Sample data
        feature_names (list): Names of features
    """
    print("\n📊 Creating SHAP Summary Plot...")
    
    plt.figure(figsize=(12, 8))
    
    shap_values_to_plot = _positive_class_shap_values(shap_values)
    
    # Create summary plot
    shap.summary_plot(
        shap_values_to_plot, 
        X_sample, 
        feature_names=feature_names,
        show=False,
        max_display=15
    )
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'shap_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ SHAP summary plot saved to {FIGURES_DIR / 'shap_summary.png'}")

def plot_shap_bar(shap_values, X_sample, feature_names):
    """
    Plot SHAP bar plot (global feature importance).
    
    Args:
        shap_values: SHAP values
        X_sample (pd.DataFrame): Sample data
        feature_names (list): Names of features
    """
    print("\n📊 Creating SHAP Bar Plot...")
    
    plt.figure(figsize=(10, 8))
    
    shap_values_to_plot = _positive_class_shap_values(shap_values)
    
    # Create bar plot
    shap.summary_plot(
        shap_values_to_plot, 
        X_sample, 
        feature_names=feature_names,
        plot_type='bar',
        show=False,
        max_display=15
    )
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'shap_bar.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ SHAP bar plot saved to {FIGURES_DIR / 'shap_bar.png'}")

def plot_shap_waterfall(shap_values, X_sample, feature_names, index=0):
    """
    Plot SHAP waterfall plot for a single prediction.
    
    Args:
        shap_values: SHAP values
        X_sample (pd.DataFrame): Sample data
        feature_names (list): Names of features
        index (int): Index of sample to explain
    """
    print(f"\n📊 Creating SHAP Waterfall Plot for sample {index}...")
    
    shap_matrix = _positive_class_shap_values(shap_values)
    shap_values_to_plot = np.asarray(shap_matrix[index])

    # Defensive fallback for SHAP outputs that still retain a class axis per sample.
    if shap_values_to_plot.ndim == 2:
        shap_values_to_plot = (
            shap_values_to_plot[:, 1]
            if shap_values_to_plot.shape[1] > 1
            else shap_values_to_plot[:, 0]
        )
    elif shap_values_to_plot.ndim != 1:
        shap_values_to_plot = shap_values_to_plot.reshape(-1)
    
    # Create waterfall plot
    plt.figure(figsize=(10, 8))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values_to_plot,
            base_values=0,  # Will be adjusted
            data=X_sample.iloc[index],
            feature_names=feature_names
        ),
        show=False
    )
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f'shap_waterfall_sample_{index}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ SHAP waterfall plot saved to {FIGURES_DIR / f'shap_waterfall_sample_{index}.png'}")

def extract_business_insights(shap_values, X_sample, feature_names):
    """
    Extract actionable business insights from SHAP values.
    
    Args:
        shap_values: SHAP values
        X_sample (pd.DataFrame): Sample data
        feature_names (list): Names of features
        
    Returns:
        dict: Dictionary of insights
    """
    print("\n📊 Extracting Business Insights...")
    
    shap_values_to_use = _positive_class_shap_values(shap_values)
    
    # Calculate mean absolute SHAP values
    mean_shap = np.abs(shap_values_to_use).mean(axis=0)
    
    # Create feature importance DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'mean_shap': mean_shap
    }).sort_values('mean_shap', ascending=False)
    
    print("\n🏆 Top 5 Churn Drivers:")
    for idx, row in importance_df.head(5).iterrows():
        print(f"  {row['feature']:30} | Importance: {row['mean_shap']:.4f}")
    
    # Generate actionable insights
    insights = {
        'top_features': importance_df.head(5)['feature'].tolist(),
        'feature_importance': importance_df.to_dict('records'),
        'key_drivers': []
    }
    
    # Add specific insights based on top features
    for feature in insights['top_features']:
        if 'Contract' in feature:
            insights['key_drivers'].append(
                "Month-to-month contracts significantly increase churn risk. "
                "Consider offering incentives for longer-term contracts."
            )
        elif 'tenure' in feature.lower():
            insights['key_drivers'].append(
                "Low tenure (especially <12 months) is a strong churn predictor. "
                "Implement onboarding programs for new customers."
            )
        elif 'MonthlyCharges' in feature:
            insights['key_drivers'].append(
                "High monthly charges correlate with churn, especially for new customers. "
                "Consider tiered pricing or loyalty discounts."
            )
        elif 'InternetService' in feature:
            insights['key_drivers'].append(
                "Fiber optic customers have higher churn than DSL. "
                "Investigate service quality and competitive positioning."
            )
        elif 'services_count' in feature.lower():
            insights['key_drivers'].append(
                "Customers with fewer services are more likely to churn. "
                "Promote service bundling and add-on services."
            )
    
    return insights

def run_shap_analysis(model, X_sample, feature_names, preprocessor=None):
    """
    Run complete SHAP analysis pipeline.
    
    Args:
        model: Trained model
        X_sample (pd.DataFrame): Sample data
        feature_names (list): Names of features
        preprocessor: The preprocessor (optional)
        
    Returns:
        dict: SHAP analysis results
    """
    print("\n" + "🚀"*30)
    print("STARTING SHAP ANALYSIS")
    print("🚀"*30)

    X_for_shap, shap_feature_names = _prepare_shap_inputs(
        model,
        X_sample,
        feature_names,
        preprocessor
    )
    
    # Compute SHAP values
    shap_values, explainer = explain_with_shap(model, X_for_shap, shap_feature_names)
    
    # Create visualizations
    plot_shap_summary(shap_values, X_for_shap, shap_feature_names)
    plot_shap_bar(shap_values, X_for_shap, shap_feature_names)
    
    # Create waterfall for a few samples
    for idx in [0, 1, 2] if len(X_for_shap) > 2 else range(len(X_for_shap)):
        plot_shap_waterfall(shap_values, X_for_shap, shap_feature_names, idx)
    
    # Extract insights
    insights = extract_business_insights(shap_values, X_for_shap, shap_feature_names)
    
    # Print actionable recommendations
    print("\n" + "="*60)
    print("📋 ACTIONABLE BUSINESS RECOMMENDATIONS")
    print("="*60)
    for idx, insight in enumerate(insights['key_drivers'], 1):
        print(f"{idx}. {insight}")
    
    return {
        'shap_values': shap_values,
        'explainer': explainer,
        'insights': insights,
        'feature_names': shap_feature_names
    }

if __name__ == "__main__":
    # Test SHAP analysis
    from src.data_loader import load_and_inspect_data, fix_data_types
    from src.feature_engineering import engineer_features
    from src.preprocess import prepare_data, create_preprocessor, load_preprocessor
    
    print("🚀 Testing SHAP analysis...")
    
    # Load data
    df = load_and_inspect_data()
    df = fix_data_types(df)
    df = engineer_features(df)
    
    # Split data
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    # Load model and preprocessor
    try:
        model = load_model()
        preprocessor = load_preprocessor()
        
        # Take a sample for SHAP (first 100 test samples)
        X_sample = X_test.iloc[:100]
        
        # Get feature names after preprocessing
        # For demo, use original feature names plus engineered
        feature_names = X_test.columns.tolist()
        
        # Run SHAP analysis
        results = run_shap_analysis(model, X_sample, feature_names, preprocessor)
        
        print("\n✅ SHAP analysis completed!")
    except:
        print("❌ No saved model found. Run training first!")