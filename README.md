# MLFlow Studio

MLFlow Studio is an end-to-end machine learning workbench for exploring tabular CSV datasets, preparing features, comparing classification or regression models, tuning a selected model, and generating predictions through a React interface.

> **Note:** This project is named MLFlow Studio, but it does not currently integrate with the MLflow tracking server or client package. Trained pipelines are saved locally with Joblib.

## Features

- Upload a CSV file or generate a reproducible synthetic fraud-like demo dataset
- Inspect dataset shape, column types, preview rows, summary statistics, missing values, duplicate rows, and target class balance
- Remove duplicate rows and rows with missing target values
- Configure median or mean imputation for numeric features
- Impute categorical features with their most frequent value
- Standardize numeric features and one-hot encode categorical features
- Generate domain features for datasets containing `Time` or `Amount`
- Automatically select features with L1 Logistic Regression or Lasso
- Split data into training and test sets with optional class stratification
- Train and compare multiple classification or regression models
- Evaluate models with task-specific metrics
- Tune supported models with randomized 3-fold cross-validation
- Enter feature values manually and generate predictions
- Download the best pipeline as a `.joblib` artifact

## Supported models

### Classification

- Logistic Regression
- Decision Tree
- XGBoost
- Random Forest
- Support Vector Machine
- Gaussian Naive Bayes

### Regression

- Linear Regression
- Multiple Linear Regression
- Support Vector Regression
- Random Forest Regressor
- Gradient Boosting Regressor

## Evaluation metrics

### Classification

- Accuracy
- Precision
- Recall
- F1 score
- Specificity
- Balanced accuracy
- ROC-AUC
- PR-AUC
- Log loss
- Matthews correlation coefficient
- Confusion matrix
- Classification report

### Regression

- MAE
- MSE
- RMSE
- R-squared
- Adjusted R-squared
- MAPE
- MSLE
- RMSLE
- Median absolute error
- Explained variance

Adjusted R-squared is omitted when the test set is too small for the feature count. MSLE and RMSLE are omitted when the target or prediction contains negative values.

## Feature engineering

When present, the pipeline generates these optional features:

| Input column | Generated feature | Formula |
|---|---|---|
| `Time` | `Time_hour` | `(Time / 3600) mod 24` |
| `Time` | `Time_day` | `Time / 86400` |
| `Amount` | `Amount_log1p` | `log(1 + max(Amount, 0))` |

The original `Time` and `Amount` columns are retained.

During preprocessing, the pipeline:

1. Imputes missing numeric values using median or mean
2. Adds an indicator for missing numeric values
3. Standardizes numeric features with `StandardScaler`
4. Imputes categorical values with the most frequent value
5. One-hot encodes categorical features
6. Applies L1 Logistic Regression or Lasso feature selection when there are more than five engineered features

Preprocessing is fitted inside the scikit-learn pipeline on training data only, preventing test data leakage.

## Tech stack

- **Frontend:** React 18, Vite, Recharts, Lucide React
- **Backend:** Python, FastAPI, Pandas, NumPy
- **Machine learning:** scikit-learn, XGBoost, Joblib

## Project structure

```text
MLFlow-Studio/
├── backend/
│   ├── main.py                 # FastAPI routes
│   ├── ml_pipeline.py          # Dataset, pipeline, training, and prediction logic
│   ├── requirements.txt        # Python dependencies
│   ├── generate_demo_csv.py    # Optional demo CSV generator
│   └── uploads/
│       └── best_model.joblib   # Created after training
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   └── styles.css
│   ├── package.json
│   └── vite.config.js
└── README.md
```

## API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/health` | Check API and dataset status |
| `POST` | `/api/dataset/demo` | Load the synthetic demo dataset |
| `POST` | `/api/dataset/upload` | Upload a CSV dataset |
| `GET` | `/api/dataset/summary` | Inspect the loaded dataset |
| `POST` | `/api/preprocess` | Configure and preprocess the dataset |
| `GET` | `/api/features/preview` | Preview derived features |
| `POST` | `/api/train` | Train and compare candidate models |
| `POST` | `/api/tune` | Tune a selected model |
| `POST` | `/api/predict` | Generate a prediction |
| `GET` | `/api/model/download` | Download the saved pipeline |

## Run locally

Python 3.10–3.12 is recommended.

### 1. Backend

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m uvicorn main:app --reload
```

For Command Prompt on Windows, activate the environment with:

```bat
.venv\Scripts\activate.bat
```

On macOS or Linux:

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m uvicorn main:app --reload
```

API: http://127.0.0.1:8000  
Swagger documentation: http://127.0.0.1:8000/docs

### 2. Frontend

Open a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the local URL printed by Vite, usually http://localhost:5173.

### 3. Generate a demo CSV

From the backend directory:

```bash
python generate_demo_csv.py
```

This creates `demo_credit_fraud.csv` with 2,500 rows and missing values. It is intended for demonstration only.

## Dataset

The project supports standard tabular CSV files containing at least two columns and at least 20 rows with a non-missing target value.

The backend:

- Limits datasets to 250,000 rows
- Limits uploaded CSV files to 80 MB
- Removes duplicate rows only when duplicate removal is enabled
- Removes rows with a missing target
- Rejects classification datasets with fewer than two target classes

You can download the Kaggle **Credit Card Fraud Detection** dataset and upload `creditcard.csv`. It commonly contains `Time`, `Amount`, `V1`-`V28`, and target column `Class` (`0` for normal, `1` for fraud). The dataset is not bundled because it is large and subject to Kaggle's licensing and redistribution terms.

## Workflow

1. Upload a CSV or load the demo dataset
2. Select a target column
3. Choose classification or regression
4. Configure preprocessing and run it
5. Preview the generated features
6. Train and compare candidate models
7. Tune a selected model
8. Review evaluation results
9. Enter values or download the trained pipeline

## Important notes

- This is an educational prototype, not a production fraud detection system
- Accuracy alone is insufficient for imbalanced data; review precision, recall, PR-AUC, specificity, and the confusion matrix
- Categorical features are supported through one-hot encoding
- Free-text, image, audio, graph, and raw time-series datasets require specialized processing not implemented here
- The project keeps trained state in backend memory and saves the latest pipeline to `backend/uploads/best_model.joblib`
- Real deployment requires reproducible environment controls, monitoring, calibration, explainability, security testing, and human oversight
