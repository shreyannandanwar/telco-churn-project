"""
Phase 7: Model Evaluation and Threshold Tuning
Comprehensive evaluation with business-focused metrics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    precision_recall_curve
)
import joblib

from src.config import FIGURES_DIR, MODEL_PATH, PREPROCESSOR_PATH
from src.preprocess import load_preprocessor
from src.train import load_model

def evaluate_model(model, X_test, y_test, threshold=0.5):
    """
    Evaluate a trained model on test data.
    
    Args:
        model: Trained model
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test targets
        threshold (float): Decision threshold
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    # Get predictions and probabilities
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    print(f"\n📊 Performance Metrics (Threshold: {threshold}):")
    print("-" * 40)
    for metric, value in metrics.items():
        print(f"  {metric.replace('_', ' ').title():15}: {value:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n📊 Confusion Matrix:")
    print(f"  True Negatives : {cm[0,0]}")
    print(f"  False Positives: {cm[0,1]}")
    print(f"  False Negatives: {cm[1,0]}")
    print(f"  True Positives : {cm[1,1]}")
    
    # Classification Report
    print(f"\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Non-Churn', 'Churn']))
    
    # Cost analysis (business metrics)
    cost_per_fp = 10  # Cost of false positive (wasted retention offer)
    cost_per_fn = 100  # Cost of false negative (lost customer)
    
    total_cost = (cm[0,1] * cost_per_fp) + (cm[1,0] * cost_per_fn)
    metrics['total_cost'] = total_cost
    metrics['cost_per_customer'] = total_cost / len(y_test)
    
    print(f"\n💰 Business Cost Analysis:")
    print(f"  Cost per False Positive : ${cost_per_fp}")
    print(f"  Cost per False Negative : ${cost_per_fn}")
    print(f"  Total Cost              : ${total_cost:,.2f}")
    print(f"  Cost per Customer       : ${metrics['cost_per_customer']:.2f}")
    
    return metrics, y_pred_proba, y_pred

def tune_threshold(model, X_test, y_test):
    """
    Find optimal threshold to maximize F1 score or minimize cost.
    
    Args:
        model: Trained model
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test targets
        
    Returns:
        dict: Threshold tuning results
    """
    print("\n" + "="*60)
    print("THRESHOLD TUNING")
    print("="*60)
    
    # Get probabilities
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Try different thresholds
    thresholds = np.arange(0.1, 0.9, 0.05)
    results = []
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # Custom business cost
        cm = confusion_matrix(y_test, y_pred)
        cost = (cm[0,1] * 10) + (cm[1,0] * 100)  # FP: $10, FN: $100
        
        results.append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'cost': cost
        })
    
    results_df = pd.DataFrame(results)
    
    # Find optimal thresholds
    best_f1_idx = results_df['f1_score'].idxmax()
    best_cost_idx = results_df['cost'].idxmin()
    
    print(f"\n📊 Optimal Threshold for F1 Score:")
    print(f"  Threshold : {results_df.loc[best_f1_idx, 'threshold']:.2f}")
    print(f"  F1 Score  : {results_df.loc[best_f1_idx, 'f1_score']:.4f}")
    print(f"  Precision : {results_df.loc[best_f1_idx, 'precision']:.4f}")
    print(f"  Recall    : {results_df.loc[best_f1_idx, 'recall']:.4f}")
    
    print(f"\n💰 Optimal Threshold for Business Cost:")
    print(f"  Threshold : {results_df.loc[best_cost_idx, 'threshold']:.2f}")
    print(f"  Cost      : ${results_df.loc[best_cost_idx, 'cost']:,.2f}")
    print(f"  F1 Score  : {results_df.loc[best_cost_idx, 'f1_score']:.4f}")
    
    # Plot threshold tuning
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot 1: Metrics vs Threshold
    ax1 = axes[0]
    ax1.plot(results_df['threshold'], results_df['precision'], label='Precision', marker='o')
    ax1.plot(results_df['threshold'], results_df['recall'], label='Recall', marker='s')
    ax1.plot(results_df['threshold'], results_df['f1_score'], label='F1 Score', marker='^', linewidth=2)
    ax1.axvline(results_df.loc[best_f1_idx, 'threshold'], color='red', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Score')
    ax1.set_title('Precision, Recall, and F1 vs Threshold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Cost vs Threshold
    ax2 = axes[1]
    ax2.plot(results_df['threshold'], results_df['cost'], marker='o', color='green')
    ax2.axvline(results_df.loc[best_cost_idx, 'threshold'], color='red', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Threshold')
    ax2.set_ylabel('Total Cost ($)')
    ax2.set_title('Business Cost vs Threshold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'threshold_tuning.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Threshold tuning plots saved to {FIGURES_DIR / 'threshold_tuning.png'}")
    
    return {
        'best_f1_threshold': results_df.loc[best_f1_idx, 'threshold'],
        'best_cost_threshold': results_df.loc[best_cost_idx, 'threshold'],
        'results_df': results_df
    }

def plot_roc_curve(model, X_test, y_test):
    """
    Plot ROC curve with AUC score.
    
    Args:
        model: Trained model
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test targets
    """
    print("\n📊 Plotting ROC Curve...")
    
    # Get probabilities
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(FIGURES_DIR / 'roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ ROC curve saved to {FIGURES_DIR / 'roc_curve.png'}")

def plot_confusion_matrix(model, X_test, y_test, threshold=0.5):
    """
    Plot confusion matrix with labels.
    
    Args:
        model: Trained model
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test targets
        threshold (float): Decision threshold
    """
    print("\n📊 Plotting Confusion Matrix...")
    
    # Get predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Churn', 'Churn'],
                yticklabels=['Non-Churn', 'Churn'])
    plt.title(f'Confusion Matrix (Threshold = {threshold:.2f})')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Confusion matrix saved to {FIGURES_DIR / 'confusion_matrix.png'}")

def full_evaluation(model, X_test, y_test, preprocessor=None, save_results=True):
    """
    Run complete evaluation pipeline.
    
    Args:
        model: Trained model
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test targets
        preprocessor: The preprocessor (optional)
        save_results (bool): Whether to save results
        
    Returns:
        dict: Complete evaluation results
    """
    print("\n" + "🚀"*30)
    print("STARTING COMPLETE EVALUATION")
    print("🚀"*30)
    
    results = {}
    
    # 1. Tune threshold
    threshold_results = tune_threshold(model, X_test, y_test)
    results['threshold_results'] = threshold_results
    
    # 2. Evaluate with best F1 threshold
    best_f1_threshold = threshold_results['best_f1_threshold']
    metrics_f1, _, _ = evaluate_model(model, X_test, y_test, best_f1_threshold)
    results['metrics_f1_threshold'] = metrics_f1
    
    # 3. Evaluate with best cost threshold
    best_cost_threshold = threshold_results['best_cost_threshold']
    metrics_cost, _, _ = evaluate_model(model, X_test, y_test, best_cost_threshold)
    results['metrics_cost_threshold'] = metrics_cost
    
    # 4. Plot ROC curve
    plot_roc_curve(model, X_test, y_test)
    
    # 5. Plot confusion matrix (with best F1 threshold)
    plot_confusion_matrix(model, X_test, y_test, best_f1_threshold)
    
    # Summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Optimal Threshold (F1): {best_f1_threshold:.3f}")
    print(f"F1 Score: {metrics_f1['f1_score']:.4f}")
    print(f"Precision: {metrics_f1['precision']:.4f}")
    print(f"Recall: {metrics_f1['recall']:.4f}")
    print(f"ROC-AUC: {metrics_f1['roc_auc']:.4f}")
    print(f"\nOptimal Threshold (Cost): {best_cost_threshold:.3f}")
    print(f"Total Cost: ${metrics_cost['total_cost']:,.2f}")
    
    return results

if __name__ == "__main__":
    # Test evaluation pipeline
    from src.data_loader import load_and_inspect_data, fix_data_types
    from src.feature_engineering import engineer_features
    from src.preprocess import prepare_data, create_preprocessor
    
    print("🚀 Testing evaluation pipeline...")
    
    # Load and prepare data
    df = load_and_inspect_data()
    df = fix_data_types(df)
    df = engineer_features(df)
    
    # Split data
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    # Load model (assuming it exists)
    try:
        model = load_model()
        preprocessor = load_preprocessor()
        
        # Run evaluation
        results = full_evaluation(model, X_test, y_test, preprocessor)
        
        print("\n✅ Evaluation test completed!")
    except:
        print("❌ No saved model found. Run training first!")