"""
Main Pipeline Script
Orchestrates the entire project from data loading to evaluation.
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.data_loader import load_and_inspect_data, fix_data_types, save_processed_data
from src.feature_engineering import engineer_features
from src.preprocess import prepare_data, create_preprocessor, save_preprocessor
from src.train import train_all_models, select_best_model, save_model
from src.evaluate import full_evaluation
from src.interpret import run_shap_analysis
from src.predict import ChurnPredictor, predict_single_customer

def run_full_pipeline(tune_xgboost=True, run_shap=True):
    """
    Run the complete data science pipeline.
    
    Args:
        tune_xgboost (bool): Whether to tune XGBoost
        run_shap (bool): Whether to run SHAP analysis
    """
    start_time = time.time()
    
    print("\n" + "="*80)
    print("🚀 TELCO CUSTOMER CHURN PREDICTION - FULL PIPELINE")
    print("="*80)
    
    # Phase 1: Data Loading
    print("\n📊 PHASE 1: Data Loading")
    print("-" * 40)
    df = load_and_inspect_data()
    df = fix_data_types(df)
    save_processed_data(df)
    
    # Phase 2: EDA (Optional - run separately if needed)
    # from src.eda import run_eda
    # insights = run_eda(df)
    
    # Phase 3: Feature Engineering
    print("\n🔧 PHASE 3: Feature Engineering")
    print("-" * 40)
    df = engineer_features(df)
    
    # Phase 4: Data Preparation
    print("\n📊 PHASE 4: Data Preparation")
    print("-" * 40)
    X_train, X_test, y_train, y_test = prepare_data(df)
    preprocessor = create_preprocessor()
    
    # Phase 5-6: Model Training
    print("\n🤖 PHASE 5-6: Model Training")
    print("-" * 40)
    models = train_all_models(X_train, y_train, preprocessor, tune_xgboost)
    best_name, best_model = select_best_model(models)
    save_model(best_model['model'])
    # Save fitted preprocessor for compatibility with non-pipeline prediction flows.
    if hasattr(best_model['model'], 'named_steps') and 'preprocessor' in best_model['model'].named_steps:
        save_preprocessor(best_model['model'].named_steps['preprocessor'])
    
    # Phase 7: Model Evaluation
    print("\n📊 PHASE 7: Model Evaluation")
    print("-" * 40)
    eval_results = full_evaluation(best_model['model'], X_test, y_test, preprocessor)
    
    # Phase 8: Model Interpretability
    if run_shap:
        print("\n🔍 PHASE 8: Model Interpretability")
        print("-" * 40)
        # Take a sample for SHAP
        X_sample = X_test.iloc[:100]
        feature_names = X_test.columns.tolist()
        
        shap_results = run_shap_analysis(
            best_model['model'], 
            X_sample, 
            feature_names, 
            preprocessor
        )
    
    # Phase 9: Business Recommendations
    print("\n📋 PHASE 9: Business Recommendations")
    print("-" * 40)
    print("Based on the analysis, here are key recommendations:")
    print("""
    1. 💰 Retention Strategy:
       - Target month-to-month customers with loyalty discounts
       - Offer 2-year contracts with premium service bundles
    
    2. 🆕 New Customer Onboarding:
       - Implement personalized onboarding for first 6 months
       - Proactive support calls for high-value new customers
    
    3. 📈 Service Bundling:
       - Promote service bundling (Internet + Phone + TV)
       - Offer discounts for multiple services
    
    4. 🔍 Early Warning System:
       - Flag customers with tenure < 6 months and MonthlyCharges > $70
       - Monitor fiber optic customers for service quality issues
    
    5. 💡 Actionable Insights:
       - Use churn probability scores for targeted interventions
       - A/B test retention offers on high-risk segments
    """)
    
    # Phase 10: Model Deployment
    print("\n🚀 PHASE 10: Model Deployment")
    print("-" * 40)
    print("Model is ready for deployment!")
    print(f"  - Model saved at: models/xgboost_churn_model.pkl")
    print(f"  - Preprocessor saved at: models/preprocessor.pkl")
    print("\n  To make predictions, use:")
    print("    from src.predict import ChurnPredictor")
    print("    predictor = ChurnPredictor()")
    print("    prediction, probability = predictor.predict(customer_data)")
    
    # Example prediction
    print("\n📊 Example Prediction:")
    predict_single_customer()
    
    # Summary
    total_time = time.time() - start_time
    print("\n" + "="*80)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"⏱️ Total execution time: {total_time/60:.2f} minutes")
    print(f"🏆 Best Model: {best_name}")
    print(f"📈 F1 Score on Test Set: {eval_results['metrics_f1_threshold']['f1_score']:.4f}")
    print(f"🎯 ROC-AUC Score: {eval_results['metrics_f1_threshold']['roc_auc']:.4f}")
    print("📁 All results saved in 'reports/' directory")

if __name__ == "__main__":
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description='Run Churn Prediction Pipeline')
    parser.add_argument('--no-tune', action='store_true', help='Skip hyperparameter tuning')
    parser.add_argument('--no-shap', action='store_true', help='Skip SHAP analysis')
    args = parser.parse_args()
    
    # Run pipeline
    run_full_pipeline(
        tune_xgboost=not args.no_tune,
        run_shap=not args.no_shap
    )
    