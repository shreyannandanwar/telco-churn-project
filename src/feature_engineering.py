"""
Phase 3: Feature Engineering
Creates new features based on domain knowledge and EDA insights.
"""

import pandas as pd
import numpy as np
from src.config import ENGINEERED_FEATURES

def engineer_features(df):
    """
    Create new features from existing data.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with new engineered features
    """
    print("\n🔧 Engineering new features...")
    
    # Make a copy to avoid modifying original
    df_engineered = df.copy()
    
    # 1. Average monthly spend (TotalCharges / tenure)
    # Handle division by zero (new customers have tenure=0)
    df_engineered['avg_monthly_spend'] = df_engineered['TotalCharges'] / (df_engineered['tenure'] + 1)
    print("  ✅ Created 'avg_monthly_spend'")
    
    # 2. Tenure groups (categorical binning)
    bins = [0, 6, 12, 24, 72]
    labels = ['0-6 months', '6-12 months', '12-24 months', '24+ months']
    df_engineered['tenure_group'] = pd.cut(df_engineered['tenure'], 
                                           bins=bins, labels=labels, right=False)
    print("  ✅ Created 'tenure_group'")
    
    # 3. Monthly charge per tenure (interaction feature)
    # Captures "new + expensive" customers who are high risk
    df_engineered['monthly_charge_per_tenure'] = df_engineered['MonthlyCharges'] / (df_engineered['tenure'] + 1)
    print("  ✅ Created 'monthly_charge_per_tenure'")
    
    # 4. Has phone service (binary flag)
    df_engineered['has_phone_service'] = (df_engineered['PhoneService'] == 'Yes').astype(int)
    print("  ✅ Created 'has_phone_service'")
    
    # 5. Has internet service (binary flag)
    df_engineered['has_internet'] = (df_engineered['InternetService'] != 'No').astype(int)
    print("  ✅ Created 'has_internet'")
    
    # 6. Number of services subscribed to
    # Count services like OnlineSecurity, OnlineBackup, DeviceProtection, etc.
    service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                   'TechSupport', 'StreamingTV', 'StreamingMovies']
    df_engineered['services_count'] = df_engineered[service_cols].apply(
        lambda row: (row == 'Yes').sum(), axis=1
    )
    print("  ✅ Created 'services_count'")
    
    # 7. Customer lifetime value (CLV) proxy
    # Higher tenure + higher charges = higher value customers
    df_engineered['customer_value'] = df_engineered['TotalCharges'] * (1 + df_engineered['MonthlyCharges'] / 100)
    print("  ✅ Created 'customer_value'")
    
    # 8. Churn risk score (composite feature based on EDA insights)
    # High risk if: Month-to-month contract + high monthly charges + low tenure
    df_engineered['churn_risk_score'] = 0
    df_engineered.loc[df_engineered['Contract'] == 'Month-to-month', 'churn_risk_score'] += 3
    df_engineered.loc[df_engineered['MonthlyCharges'] > df_engineered['MonthlyCharges'].median(), 'churn_risk_score'] += 2
    df_engineered.loc[df_engineered['tenure'] < 12, 'churn_risk_score'] += 2
    print("  ✅ Created 'churn_risk_score'")
    
    print(f"✅ Feature engineering complete! Added {len(ENGINEERED_FEATURES)} new features")
    
    return df_engineered

if __name__ == "__main__":
    # Test feature engineering
    from src.data_loader import load_and_inspect_data, fix_data_types
    
    print("🚀 Testing feature engineering...")
    df = load_and_inspect_data()
    df = fix_data_types(df)
    df_engineered = engineer_features(df)
    
    print(f"\n📊 New features added:")
    for feature in ['avg_monthly_spend', 'tenure_group', 'monthly_charge_per_tenure',
                   'has_phone_service', 'has_internet', 'services_count',
                   'customer_value', 'churn_risk_score']:
        print(f"  - {feature}: {df_engineered[feature].dtype}")
    
    print("\n✅ Feature engineering test completed!")