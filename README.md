# MLFlow Studio — End-to-End Machine Learning Workflow Platform

MLFlow Studio is an educational full-stack project for exploring a complete ML lifecycle using a credit-card fraud detection dataset.

## Features
- Upload a CSV dataset or generate a reproducible demo dataset
- Dataset overview, data types, summary statistics, missing values, and class balance
- Configurable missing-value imputation and preprocessing
- Feature engineering (time-derived features when `Time` exists; log transform of `Amount` when available)
- Train/test split with stratification when appropriate
- Classification models: Logistic Regression, Decision Tree, XGBoost, Random Forest, SVM, and Naive Bayes
- Regression models: Linear Regression, Multiple Linear Regression, SVR, Random Forest Regressor, and Gradient Boosting Regressor
- Classification evaluation starts with a confusion matrix, followed by accuracy, precision, recall, F1-score, specificity, ROC-AUC, PR-AUC, log loss, Matthews correlation coefficient, and balanced accuracy.
- Regression metrics: MAE, MSE, RMSE, R², adjusted R², MAPE, MSLE, RMSLE, median absolute error, and explained variance
- MSLE and RMSLE are unavailable when targets or predictions are negative; adjusted R² is unavailable when the test set is too small for the feature count.
- Confusion matrix and feature importance when supported
- Randomized hyperparameter tuning
- Predict from manually entered feature values
- Download a trained model artifact

## Tech stack
- Frontend: React, Vite, Recharts
- Backend: FastAPI, Pandas, NumPy, scikit-learn, Joblib

## Project structure
```text
MLFlow-Studio/
├── backend/
│   ├── main.py
│   ├── ml_pipeline.py
│   ├── requirements.txt
│   └── uploads/             # created automatically
├── frontend/
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   └── src/
│       ├── main.jsx
│       ├── App.jsx
│       └── styles.css
└── README.md
```

## Run locally

### 1. Backend
Python 3.10–3.12 recommended.

```bash
cd backend
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

API: http://127.0.0.1:8000  
Swagger docs: http://127.0.0.1:8000/docs

### 2. Frontend
Open a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the local URL printed by Vite (usually http://localhost:5173).

## Dataset
Download the Kaggle **Credit Card Fraud Detection** dataset and upload `creditcard.csv` from the app. It commonly contains `Time`, `Amount`, `V1`–`V28`, and target column `Class` (0 = normal, 1 = fraud). The dataset is not bundled because it is large and subject to its source's terms.

You can also click **Load demo dataset** in the app. It generates synthetic, imbalanced sample data for demonstrating the workflow. Synthetic demo results are not evidence of real-world fraud detection performance.

## Workflow
1. Upload / load demo data
2. Select the target column (`Class` for the Kaggle dataset)
3. Inspect the dataset and missing values
4. Configure imputation and run preprocessing
5. Engineer features
6. Train and compare candidate models
7. Evaluate metrics and tune the selected model
8. Enter feature values and make predictions

## Important notes
- This is an educational prototype, not a production fraud detection system.
- For fraud data, accuracy alone can be misleading because the target is highly imbalanced. Review precision, recall, PR-AUC, and the confusion matrix.
- The default test split is stratified for classification when the class distribution allows it.
- The backend reuses engineered features and the training split during hyperparameter tuning to avoid redundant preprocessing.
- The frontend production build separates charting and icon dependencies into cacheable vendor chunks.
- Never use this prototype as the sole basis for blocking financial transactions. Real deployment requires privacy controls, drift monitoring, threshold calibration, explainability, security testing, and human oversight.
