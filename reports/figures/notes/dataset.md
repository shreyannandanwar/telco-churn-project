## Dataset 
We have Telco Customer information, this data set tells us - Some of the Customers are happy and stay with you, but some leave and go to competitors. This dataset tells us who left and why.

Can we predict which customers are about to leave?

## Transformation of data

### Step 1: Cleaned It Up
Note: Fixed TotalCharges (some were empty or text instead of numbers)
Note: Filled in missing values
Note: Made sure everything was in the right format

### Step 2: Made New Features
Note: We created features like: `avg_monthly_spend = TotalCharges / (tenure + 1)`(How much they spend on average per month)
Note: We created `churn_risk_score`

### Step 3: Built Machine Learning Models
We trained 3 models:

1. Logistic Regression (Simple baseline)

Like a weighted scorecard
Easy to understand

2. Random Forest (Ensemble of decision trees)

Like asking 100 experts and taking a vote
Good at finding patterns
3. XGBoost (The winner)

Like Random Forest but smarter
Keeps learning from mistakes
Most accurate

### Step 4: Handled the Imbalance Problem

Since only 26% leave, we used SMOTE:

Creates fake but realistic examples of churners
Makes the model learn equally about both groups
Like giving the model more practice with churners
