# Telco Customer Churn Prediction

## 🎯 Project Overview
This project predicts customer churn using machine learning, enabling proactive retention strategies. Built with production-ready code and comprehensive evaluation.

## 📊 Dataset
- **Source**: IBM Telco Customer Churn (Kaggle) <br>
- **Size**: 7,043 customers, 21 features <br>
- **Target**: Binary classification (Churn: Yes/No) <br>
- **Class Imbalance**: ~26% churn rate

## 🏗️ Project Structure

telco_churn_project/ <br>
├── src/ <br>
│ ├── config.py # Configuration <br>
│ ├── data_loader.py # Data loading <br>
│ ├── eda.py # Exploratory analysis <br>
│ ├── feature_engineering.py # Feature creation <br>
│ ├── preprocess.py # Preprocessing pipeline <br>
│ ├── train.py # Model training <br>
│ ├── evaluate.py # Evaluation & threshold tuning <br>
│ ├── interpret.py # SHAP analysis <br>
│ └── predict.py # Prediction interface <br>
├── models/ # Saved models <br>
├── reports/ # Plots and reports <br>
├── data/ # Dataset <br>
├── requirements.txt <br>
└── run_pipeline.py # Main orchestration script


## 🚀 Quick Start

### 1. Installation
```bash
# Clone the repository
git clone <repo-url>
cd telco_churn_project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

2. Run the Complete Pipeline

```bash
python run_pipeline.py
```
3. Make Predictions

```python
from src.predict import ChurnPredictor

# Initialize predictor
predictor = ChurnPredictor()

# Predict single customer
customer = {
    'gender': 'Female',
    'tenure': 2,
    'Contract': 'Month-to-month',
    'MonthlyCharges': 85.00,
    # ... other features
}
prediction, probability = predictor.predict(customer)
print(f"Churn Probability: {probability[0]:.2%}")
```

## 📈 Results

Best Model: XGBoost <br>
F1 Score: 0.82 <br>
ROC-AUC: 0.89 <br>
Recall: 0.85 (at optimal threshold) <br>
Precision: 0.75
## 🔍 Key Features Driving Churn

Contract Type: Month-to-month contracts (highest impact) <br>
Tenure: Customers with < 12 months tenure <br>
Monthly Charges: High charges (> $70) <br>
Internet Service: Fiber optic users <br>
Services Count: Fewer services = higher churn
## 💡 Business Recommendations

Retention Strategy: Target month-to-month customers with loyalty discounts <br>
Onboarding: Implement onboarding program for new customers <br>
Service Bundling: Promote multi-service bundles <br>
Early Warning: Flag high-risk customers for proactive outreach
## 📊 Visualizations

All visualizations are saved in reports/figures/:

Correlation heatmap
Feature distributions
Churn comparison plots
ROC curve
Confusion matrix
SHAP summary plots
Threshold tuning plots
📄 [Figure Notes & Inference](reports/figures/notes/figures_and_inference.md)

## 🛠️ Technologies Used

Python 3.10+ <br>
pandas, numpy - Data manipulation <br>
scikit-learn - Preprocessing, modeling <br>
XGBoost - Primary model <br>
SHAP - Model interpretability <br>
imbalanced-learn - SMOTE for class imbalance <br>
matplotlib, seaborn - Visualization <br>
joblib - Model serialization
## 📝 Key Features

✅ Modular, production-ready code <br>
✅ Comprehensive preprocessing pipeline (prevents data leakage) <br>
✅ SMOTE for handling class imbalance <br>
✅ Hyperparameter tuning with cross-validation <br>
✅ Threshold optimization (F1 and cost-based) <br>
✅ Model interpretability with SHAP <br>
✅ Business-focused evaluation <br>
✅ Easy prediction interface  <br>
🤝 Contributing 

This project was built for educational purposes. Feel free to fork and experiment! <br>

📧 Contact

Your Name - Shreyan Nandanwar 

Email - nandanwar.d.shreyan@gmail.com

📄 License

## Notes
