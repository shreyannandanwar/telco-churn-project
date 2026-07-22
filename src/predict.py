"""
Phase 10: Prediction Script
Loads the trained model and makes predictions on new data.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import warnings

from src.config import MODEL_PATH, PREPROCESSOR_PATH, NUMERIC_FEATURES, CATEGORICAL_FEATURES
from src.preprocess import load_preprocessor
from src.train import load_model
from src.feature_engineering import engineer_features

class ChurnPredictor:
    """
    A class for making churn predictions with a trained model.
    """
    
    def __init__(self, model_path=MODEL_PATH, preprocessor_path=PREPROCESSOR_PATH):
        """
        Initialize the predictor with a trained model and preprocessor.
        
        Args:
            model_path (Path): Path to saved model
            preprocessor_path (Path): Path to saved preprocessor
        """
        self.model = load_model(model_path)
        self.preprocessor = None

        # If the saved model is a full pipeline, we can predict directly on raw features.
        if hasattr(self.model, 'named_steps') and 'preprocessor' in self.model.named_steps:
            print("✅ ChurnPredictor initialized with fitted model pipeline")
        else:
            try:
                self.preprocessor = load_preprocessor(preprocessor_path)
                print("✅ ChurnPredictor initialized with model and preprocessor")
            except Exception:
                warnings.warn(
                    "Saved model is not a full pipeline and preprocessor could not be loaded. "
                    "Prediction will fail until a fitted preprocessor is available."
                )
    
    def predict(self, customer_data, threshold=0.5):
        """
        Make churn predictions for customer data.
        
        Args:
            customer_data (pd.DataFrame): Customer data
            threshold (float): Decision threshold
            
        Returns:
            tuple: (predictions, probabilities)
        """
        # Ensure data is in DataFrame format
        if not isinstance(customer_data, pd.DataFrame):
            customer_data = pd.DataFrame([customer_data])
        
        # Apply feature engineering
        customer_data = engineer_features(customer_data)
        
        # Predict directly if model is a fitted pipeline; otherwise transform manually.
        if hasattr(self.model, 'named_steps') and 'preprocessor' in self.model.named_steps:
            probabilities = self.model.predict_proba(customer_data)[:, 1]
        else:
            if self.preprocessor is None:
                raise ValueError(
                    "Preprocessor is not available for this model. "
                    "Retrain and save a fitted preprocessor, or save a full prediction pipeline."
                )
            X_processed = self.preprocessor.transform(customer_data)
            probabilities = self.model.predict_proba(X_processed)[:, 1]
        
        # Get predictions
        predictions = (probabilities >= threshold).astype(int)
        
        return predictions, probabilities
    
    def predict_batch(self, customer_data, threshold=0.5):
        """
        Make churn predictions for multiple customers.
        
        Args:
            customer_data (pd.DataFrame): Customer data
            threshold (float): Decision threshold
            
        Returns:
            pd.DataFrame: Predictions with probabilities
        """
        predictions, probabilities = self.predict(customer_data, threshold)
        
        results = customer_data.copy()
        results['churn_prediction'] = predictions
        results['churn_probability'] = probabilities
        results['churn_risk'] = results['churn_probability'].apply(
            lambda x: 'High' if x > 0.7 else ('Medium' if x > 0.3 else 'Low')
        )
        
        return results
    
    def save_predictions(self, customer_data, output_path, threshold=0.5):
        """
        Save predictions to a CSV file.
        
        Args:
            customer_data (pd.DataFrame): Customer data
            output_path (Path): Path to save predictions
            threshold (float): Decision threshold
        """
        results = self.predict_batch(customer_data, threshold)
        results.to_csv(output_path, index=False)
        print(f"✅ Predictions saved to {output_path}")

def predict_single_customer(example_customer=None):
    """
    Example function to predict churn for a single customer.
    
    Args:
        example_customer (dict): Customer data
        
    Returns:
        tuple: (prediction, probability)
    """
    if example_customer is None:
        # Example customer (high risk: month-to-month, high charges, low tenure)
        example_customer = {
            'gender': 'Female',
            'SeniorCitizen': '0',
            'Partner': 'No',
            'Dependents': 'No',
            'tenure': 2,
            'PhoneService': 'Yes',
            'MultipleLines': 'No',
            'InternetService': 'Fiber optic',
            'OnlineSecurity': 'No',
            'OnlineBackup': 'No',
            'DeviceProtection': 'No',
            'TechSupport': 'No',
            'StreamingTV': 'No',
            'StreamingMovies': 'No',
            'Contract': 'Month-to-month',
            'PaperlessBilling': 'Yes',
            'PaymentMethod': 'Electronic check',
            'MonthlyCharges': 85.00,
            'TotalCharges': 170.00
        }
    
    # Create predictor
    predictor = ChurnPredictor()
    
    # Make prediction
    prediction, probability = predictor.predict(example_customer)
    
    print("\n" + "="*60)
    print("CUSTOMER CHURN PREDICTION")
    print("="*60)
    print(f"Churn Probability: {probability[0]:.2%}")
    print(f"Prediction: {'CHURN' if prediction[0] == 1 else 'NO CHURN'}")
    print(f"Risk Level: {'High' if probability[0] > 0.7 else 'Medium' if probability[0] > 0.3 else 'Low'}")
    
    return prediction, probability

if __name__ == "__main__":
    # Test prediction
    print("🚀 Testing prediction pipeline...")
    
    # Test single customer
    predict_single_customer()
    
    # Test batch prediction
    from src.data_loader import load_and_inspect_data
    
    print("\n🚀 Testing batch prediction...")
    try:
        df = load_and_inspect_data()
        df = df.iloc[:10]  # First 10 customers
        
        predictor = ChurnPredictor()
        results = predictor.predict_batch(df)
        
        print(f"\n📊 Batch Prediction Results:")
        print(results[['gender', 'tenure', 'Contract', 'MonthlyCharges', 
                      'churn_probability', 'churn_risk']].to_string())
    except:
        print("❌ Error in batch prediction")
    
    print("\n✅ Prediction test completed!")