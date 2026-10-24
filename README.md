Customer Churn Prediction

Predicts whether a telecom customer will churn (leave the service) using machine learning on the Telco Customer Churn dataset.

Overview
Goal: identify customers likely to churn so retention efforts can target them.
Type: binary classification (Churn: Yes / No)
Notebook: CustomerChurnPrediction.ipynb (built in Google Colab)
Dataset
Name: Telco Customer Churn (IBM sample data), file WA_Fn-UseC_-Telco-Customer-Churn.csv
Size: 7,043 customers, 21 columns
Target balance: 5,174 No (73.5%) / 1,869 Yes (26.5%), so the classes are imbalanced
Feature groups:
Demographics: gender, SeniorCitizen, Partner, Dependents
Services: PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies
Account: tenure, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges
The dataset is not included in this repo. Download it from Kaggle and place it in the project folder.
Workflow
Data cleaning
Dropped customerID
Fixed 11 blank TotalCharges values (replaced with 0) and converted the column to float
EDA
Histograms (with mean and median lines) and boxplots for tenure, MonthlyCharges, TotalCharges
Correlation heatmap and count plots for categorical features
Preprocessing
Label-encoded categorical columns (encoders saved to encoder.pkl)
80/20 train-test split (random_state=42)
SMOTE applied to the training set only (4,138 / 4,138 after balancing)
Model selection: 5-fold cross-validation on the SMOTE training data
Model	CV Accuracy
Decision Tree	0.78
Random Forest	0.84
XGBoost	0.83
Final model: Random Forest (max_depth=16), saved to customer_churn_prediction_model.pkl
Extra comparison: logistic regression with no regularization, L2, L1 and Elastic Net, evaluated by ROC-AUC with coefficient analysis
Results (test set, 1,409 customers)

Random Forest (final model)

Metric	Value
Accuracy	0.77
Precision (churn)	0.54
Recall (churn)	0.79
F1 (churn)	0.64

Confusion matrix: [[788, 248], [78, 295]]

Logistic regression variants

Model	Accuracy	Recall	F1	ROC-AUC
No regularization	0.786	0.697	0.633	0.852
L2 (Ridge)	0.784	0.689	0.628	0.852
L1 (Lasso)	0.784	0.689	0.628	0.852
Elastic Net	0.784	0.689	0.628	0.852
Tech stack
Python, Pandas, NumPy
Matplotlib, Seaborn
Scikit-learn, XGBoost, imbalanced-learn (SMOTE)
Getting started
bash
git clone https://github.com/<username>/<repo>.git
cd <repo>
pip install numpy pandas matplotlib seaborn scikit-learn xgboost imbalanced-learn jupyter
jupyter notebook CustomerChurnPrediction.ipynb
Update the CSV path in the first data-loading cell (the notebook uses /content/... from Colab).
Making a prediction
python
import pickle
import pandas as pd

with open("customer_churn_prediction_model.pkl", "rb") as f:
    model_data = pickle.load(f)
model = model_data["Model"]

with open("encoder.pkl", "rb") as f:
    encoders = pickle.load(f)

customer = pd.DataFrame([{
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
    "tenure": 1, "PhoneService": "No", "MultipleLines": "No phone service",
    "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "Yes",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 29.85, "TotalCharges": 29.85,
}])

for col, enc in encoders.items():
    if col in customer.columns:
        customer[col] = enc.transform(customer[col])

pred = model.predict(customer)[0]
prob = model.predict_proba(customer)
print("Churn" if pred == 1 else "No churn", prob)
Project structure
.
├── CustomerChurnPrediction.ipynb
├── customer_churn_prediction_model.pkl
├── encoder.pkl
├── WA_Fn-UseC_-Telco-Customer-Churn.csv   (download separately)
└── README.md
Key observations
Month-to-month contracts, short tenure and fiber-optic internet are associated with higher churn in the logistic regression coefficients.
Longer contracts and longer tenure reduce churn risk.
The model favors recall on churners, so it catches most at-risk customers at the cost of some false alarms.
Future improvements
Hyperparameter tuning (GridSearch / Optuna) for Random Forest and XGBoost
Threshold tuning for the precision / recall trade-off
Deploy as a Streamlit or Flask app
Author

Alok
