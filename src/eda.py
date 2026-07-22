"""
Phase 2: Exploratory Data Analysis (EDA)
Generates visualizations and insights about the dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from src.config import FIGURES_DIR
from src.data_loader import load_and_inspect_data, fix_data_types

# Set up plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def create_categorical_plots(df):
    """
    Create distribution plots for categorical features.
    
    Args:
        df (pd.DataFrame): Input dataframe
    """
    print("\n📊 Creating categorical feature plots...")
    
    # Select categorical features (excluding target and high-cardinality)
    cat_cols = df.select_dtypes(include=['object']).columns
    cat_cols = [col for col in cat_cols if col not in ['customerID', 'Churn']]
    
    # Create subplots
    n_cols = 3
    n_rows = (len(cat_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten()
    
    for idx, col in enumerate(cat_cols):
        if idx < len(axes):
            # Countplot with hue for churn
            ax = axes[idx]
            sns.countplot(data=df, x=col, hue='Churn', ax=ax)
            ax.set_title(f'{col} Distribution by Churn', fontsize=12, fontweight='bold')
            ax.set_xlabel('')
            ax.tick_params(axis='x', rotation=45)
            ax.legend(title='Churn', loc='upper right')
    
    # Hide unused subplots
    for idx in range(len(cat_cols), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'categorical_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Categorical plots saved to {FIGURES_DIR / 'categorical_analysis.png'}")

def create_numerical_plots(df):
    """
    Create distribution plots for numerical features.
    
    Args:
        df (pd.DataFrame): Input dataframe
    """
    print("\n📊 Creating numerical feature plots...")
    
    # Select numerical features
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    # Create subplots
    n_cols = 2
    n_rows = (len(num_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten()
    
    for idx, col in enumerate(num_cols):
        if idx < len(axes):
            ax = axes[idx]
            
            # Histogram with KDE
            sns.histplot(data=df, x=col, kde=True, ax=ax)
            ax.set_title(f'{col} Distribution', fontsize=12, fontweight='bold')
            
            # Add vertical lines for mean and median
            ax.axvline(df[col].mean(), color='red', linestyle='--', label='Mean')
            ax.axvline(df[col].median(), color='green', linestyle='--', label='Median')
            ax.legend()
    
    # Hide unused subplots
    for idx in range(len(num_cols), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'numerical_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Numerical plots saved to {FIGURES_DIR / 'numerical_distributions.png'}")

def create_churn_comparison_plots(df):
    """
    Create comparison plots between churners and non-churners.
    
    Args:
        df (pd.DataFrame): Input dataframe
    """
    print("\n📊 Creating churn comparison plots...")
    
    # Numerical features comparison
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for idx, col in enumerate(num_cols[:4]):
        if idx < len(axes):
            ax = axes[idx]
            
            # Box plots comparing churn vs non-churn
            sns.boxplot(data=df, x='Churn', y=col, ax=ax)
            ax.set_title(f'{col} by Churn Status', fontsize=12, fontweight='bold')
            
            # Add statistical annotation
            churn_median = df[df['Churn'] == 'Yes'][col].median()
            non_churn_median = df[df['Churn'] == 'No'][col].median()
            ax.text(0.5, 0.95, f'Churn median: {churn_median:.2f}\nNon-Churn median: {non_churn_median:.2f}',
                   transform=ax.transAxes, ha='center', va='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'churn_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Churn comparison plots saved to {FIGURES_DIR / 'churn_comparison.png'}")

def create_correlation_heatmap(df):
    """
    Create correlation heatmap for numerical features.
    
    Args:
        df (pd.DataFrame): Input dataframe
    """
    print("\n📊 Creating correlation heatmap...")
    
    # Select numerical features and encode target
    num_df = df.select_dtypes(include=['int64', 'float64']).copy()
    num_df['Churn_encoded'] = (df['Churn'] == 'Yes').astype(int)
    
    # Calculate correlations
    correlation_matrix = num_df.corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                square=True, linewidths=0.5)
    plt.title('Correlation Heatmap', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Correlation heatmap saved to {FIGURES_DIR / 'correlation_heatmap.png'}")

def generate_eda_insights(df):
    """
    Generate key insights from the EDA process.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        dict: Dictionary of key insights
    """
    insights = {}
    
    # Churn rate
    churn_rate = (df['Churn'] == 'Yes').mean() * 100
    insights['churn_rate'] = f"{churn_rate:.2f}%"
    
    # Contract type analysis
    contract_churn = df.groupby('Contract')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100)
    insights['month_to_month_churn'] = f"{contract_churn.get('Month-to-month', 0):.2f}%"
    insights['one_year_churn'] = f"{contract_churn.get('One year', 0):.2f}%"
    insights['two_year_churn'] = f"{contract_churn.get('Two year', 0):.2f}%"
    
    # Tenure analysis
    avg_tenure_churn = df[df['Churn'] == 'Yes']['tenure'].mean()
    avg_tenure_non_churn = df[df['Churn'] == 'No']['tenure'].mean()
    insights['avg_tenure_churn'] = f"{avg_tenure_churn:.1f} months"
    insights['avg_tenure_non_churn'] = f"{avg_tenure_non_churn:.1f} months"
    
    # Monthly charges analysis
    avg_monthly_churn = df[df['Churn'] == 'Yes']['MonthlyCharges'].mean()
    avg_monthly_non_churn = df[df['Churn'] == 'No']['MonthlyCharges'].mean()
    insights['avg_monthly_churn'] = f"${avg_monthly_churn:.2f}"
    insights['avg_monthly_non_churn'] = f"${avg_monthly_non_churn:.2f}"
    
    # Internet service analysis
    if 'InternetService' in df.columns:
        internet_churn = df.groupby('InternetService')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100)
        insights['fiber_optic_churn'] = f"{internet_churn.get('Fiber optic', 0):.2f}%"
        insights['dsl_churn'] = f"{internet_churn.get('DSL', 0):.2f}%"
        insights['no_internet_churn'] = f"{internet_churn.get('No', 0):.2f}%"
    
    return insights

def run_eda(df):
    """
    Run the complete EDA pipeline.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        dict: EDA insights
    """
    print("\n" + "="*60)
    print("EXPLORATORY DATA ANALYSIS (EDA)")
    print("="*60)
    
    # Create all visualizations
    create_categorical_plots(df)
    create_numerical_plots(df)
    create_churn_comparison_plots(df)
    create_correlation_heatmap(df)
    
    # Generate insights
    insights = generate_eda_insights(df)
    
    print("\n📊 Key EDA Insights:")
    print("-" * 40)
    for key, value in insights.items():
        print(f"  {key.replace('_', ' ').title()}: {value}")
    
    return insights

if __name__ == "__main__":
    # Test the EDA pipeline
    print("🚀 Running EDA pipeline...")
    df = load_and_inspect_data()
    df = fix_data_types(df)
    insights = run_eda(df)
    print("\n✅ EDA pipeline completed successfully!")