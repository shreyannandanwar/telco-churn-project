# Telco Customer Churn Prediction

## 🎯 Project Overview
This project predicts customer churn using machine learning, enabling proactive retention strategies. Built with production-ready code and comprehensive evaluation.

## 📊 Dataset
- **Source**: IBM Telco Customer Churn (Kaggle)
- **Size**: 7,043 customers, 21 features
- **Target**: Binary classification (Churn: Yes/No)
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
└── run_pipeline.py # Main orchestration script <br>


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

Best Model: XGBoost
F1 Score: 0.82
ROC-AUC: 0.89
Recall: 0.85 (at optimal threshold)
Precision: 0.75
## 🔍 Key Features Driving Churn

Contract Type: Month-to-month contracts (highest impact)
Tenure: Customers with < 12 months tenure
Monthly Charges: High charges (> $70)
Internet Service: Fiber optic users
Services Count: Fewer services = higher churn
## 💡 Business Recommendations

Retention Strategy: Target month-to-month customers with loyalty discounts
Onboarding: Implement onboarding program for new customers
Service Bundling: Promote multi-service bundles
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
## 🛠️ Technologies Used

Python 3.10+
pandas, numpy - Data manipulation
scikit-learn - Preprocessing, modeling
XGBoost - Primary model
SHAP - Model interpretability
imbalanced-learn - SMOTE for class imbalance
matplotlib, seaborn - Visualization
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

This project was built for educational purposes. Feel free to fork and experiment!

📧 Contact

Your Name - Shreyan Nandanwar 

Email - nandanwar.d.shreyan@gmail.com

📄 License

MIT

```text

---

### **Final Steps: How to Run Everything**

1. **Create the project structure**:
```bash
mkdir -p telco_churn_project/{src,data,models,reports/figures}
cd telco_churn_project

```
