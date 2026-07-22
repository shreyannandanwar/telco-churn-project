"""
Phase 1: Data Loading and Initial Inspection
Handles downloading, loading, and basic validation of the dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import urllib.request
import sys
from src.config import RAW_DATA_PATH, DATA_URL, PROCESSED_DATA_PATH

def download_data(force=False):
    """
    Check if data exists locally. If not, download it.
    """
    if RAW_DATA_PATH.exists():
        print(f"✅ Data found at {RAW_DATA_PATH}")
        return
    
    print(f"📥 Data not found. Downloading from {DATA_URL}...")
    try:
        urllib.request.urlretrieve(DATA_URL, RAW_DATA_PATH)
        print(f"✅ Data downloaded successfully to {RAW_DATA_PATH}")
    except Exception as e:
        print(f"❌ Error downloading data: {e}")
        print("Please place your CSV file manually in the data/ folder")
        sys.exit(1)

def load_and_inspect_data():
    """
    Load the dataset and perform initial inspection.
    
    Returns:
        pd.DataFrame: The loaded dataset
    """
    # Download if not exists
    download_data()
    
    # Load the data
    print("\n📊 Loading dataset...")
    df = pd.read_csv(RAW_DATA_PATH)
    
    print(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Initial inspection
    print("\n" + "="*60)
    print("INITIAL DATA INSPECTION")
    print("="*60)
    
    # First 5 rows
    print("\n📋 First 5 rows:")
    print(df.head())
    
    # Data types
    print("\n📋 Data types:")
    print(df.dtypes)
    
    # Missing values
    print("\n📋 Missing values:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("✅ No missing values found!")
    
    # Summary statistics
    print("\n📋 Summary statistics for numerical features:")
    print(df.describe())
    
    # Check target distribution
    if 'Churn' in df.columns:
        print("\n📋 Target distribution:")
        churn_counts = df['Churn'].value_counts()
        churn_percent = df['Churn'].value_counts(normalize=True) * 100
        print(pd.DataFrame({
            'Count': churn_counts,
            'Percentage': churn_percent
        }))
    
    # Data quality checks
    print("\n🔍 Data Quality Checks:")
    
    # Check for duplicate rows
    duplicates = df.duplicated().sum()
    print(f"  - Duplicate rows: {duplicates}")
    
    # Check for constant columns
    constant_cols = [col for col in df.columns if df[col].nunique() == 1]
    if constant_cols:
        print(f"  - Constant columns: {constant_cols}")
    
    # Check for high cardinality categorical
    high_card = [col for col in df.select_dtypes(include=['object']).columns 
                 if df[col].nunique() > 20]
    if high_card:
        print(f"  - High cardinality categorical columns: {high_card}")
    
    return df

def fix_data_types(df):
    """
    Fix data types, especially TotalCharges which should be numeric.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with corrected types
    """
    print("\n🔧 Fixing data types...")
    
    # Convert TotalCharges to numeric (coerce errors to NaN)
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        print(f"  - Converted TotalCharges to numeric. Found {df['TotalCharges'].isnull().sum()} null values")
    
    # Convert SeniorCitizen to categorical
    if 'SeniorCitizen' in df.columns:
        df['SeniorCitizen'] = df['SeniorCitizen'].astype('object')
    
    # Ensure all object columns are strings
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('str')
    
    return df

def save_processed_data(df):
    """
    Save the processed data for later use.
    
    Args:
        df (pd.DataFrame): Processed dataframe
    """
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"✅ Processed data saved to {PROCESSED_DATA_PATH}")

if __name__ == "__main__":
    # Test the data loader
    print("🚀 Testing Data Loader...")
    df = load_and_inspect_data()
    df = fix_data_types(df)
    save_processed_data(df)
    print("\n✅ Data loader test completed successfully!")