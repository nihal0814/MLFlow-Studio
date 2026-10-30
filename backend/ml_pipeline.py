from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import joblib
import numpy as np
import pandas as pd
from math import prod

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, log_loss,
    matthews_corrcoef,
    classification_report, explained_variance_score, mean_absolute_error,
    mean_absolute_percentage_error, mean_squared_error, mean_squared_log_error,
    median_absolute_error, r2_score
)
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor,
)
from sklearn.base import clone
from xgboost import XGBClassifier

UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
MODEL_PATH = UPLOAD_DIR / "best_model.joblib"
STATE: dict[str, Any] = {
    "raw_df": None,
    "processed_df": None,
    "target": None,
    "task": "classification",
    "models": {},
    "results": [],
    "best_model_name": None,
    "best_pipeline": None,
    "feature_columns": [],
    "last_metrics": None,
    "imputation": "median",
    "engineered_df": None,
    "split_indices": None,
    "feature_selection": None,
}

def json_safe(value):
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)): return bool(value)
    if pd.isna(value): return None
    return value

def dataframe_preview(df: pd.DataFrame, limit=12):
    safe = df.head(limit).copy()
    safe = safe.replace({np.nan: None})
    return {"columns": list(safe.columns), "rows": safe.to_dict(orient="records")}

def make_demo_dataset(n=2500, seed=42):
    """Create an intentionally synthetic, imbalanced fraud-like dataset for UI testing."""
    rng = np.random.default_rng(seed)
    fraud = rng.random(n) < 0.035
    data = {}
    data["Time"] = np.sort(rng.uniform(0, 172800, n))
    data["Amount"] = np.where(fraud, rng.lognormal(3.6, 1.0, n), rng.lognormal(3.1, 1.0, n))
    for i in range(1, 9):
        normal = rng.normal(0, 1, n)
        signal = rng.normal(0, 1, n) + fraud * rng.choice([-2.2, 2.2])
        data[f"V{i}"] = np.where(fraud, signal, normal)
    for i in range(9, 15):
        data[f"V{i}"] = rng.normal(0, 1.0, n) + fraud * rng.normal(0.7, 0.6, n)
    # A few missing values to demonstrate imputation.
    df = pd.DataFrame(data)
    missing_mask = rng.random(df.shape) < 0.002
    df = df.mask(missing_mask)
    df["Class"] = fraud.astype(int)
    return df

def set_dataset(df: pd.DataFrame, filename="uploaded.csv"):
    if df.empty:
        raise ValueError("The uploaded dataset is empty.")
    if len(df.columns) < 2:
        raise ValueError("Dataset must contain at least two columns.")
    if len(df) > 250_000:
        raise ValueError("For this educational prototype, datasets are limited to 250,000 rows.")
    STATE.update({
        "raw_df": df.copy(),
        "processed_df": None,
        "target": None,
        "task": "classification",
        "models": {},
        "results": [],
        "best_model_name": None,
        "best_pipeline": None,
        "feature_columns": [],
        "last_metrics": None,
        "engineered_df": None,
        "split_indices": None,
        "feature_selection": None,
    })
    if MODEL_PATH.exists():
        MODEL_PATH.unlink()
    return dataset_summary(df)

def dataset_summary(df=None):
    df = df if df is not None else STATE["raw_df"]
    if df is None:
        return {"loaded": False}
    missing = df.isna().sum()
    numeric = df.select_dtypes(include=np.number)
    target = STATE.get("target")
    class_counts = {}
    if target and target in df.columns and df[target].nunique(dropna=True) <= 30:
        class_counts = {str(k): int(v) for k, v in df[target].value_counts(dropna=False).items()}
    return {
        "loaded": True,
        "rows": int(df.shape[0]),
        "columns_count": int(df.shape[1]),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing": {c: int(v) for c, v in missing.items()},
        "missing_total": int(missing.sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_summary": json.loads(numeric.describe().replace({np.nan: None}).to_json()),
        "preview": dataframe_preview(df),
        "target": target,
        "class_counts": class_counts,
    }

def preprocess_dataset(target: str, task: str, imputation: str = "median", drop_duplicates: bool = True):
    df = STATE["raw_df"]
    if df is None:
        raise ValueError("Upload a dataset or load the demo dataset first.")
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' does not exist.")
    if df[target].isna().all():
        raise ValueError("The target column contains only missing values.")
    if task not in {"classification", "regression"}:
        raise ValueError("Task must be classification or regression.")
    work = df.copy()
    before = len(work)
    if drop_duplicates:
        work = work.drop_duplicates()
    work = work.loc[work[target].notna()].copy()
    if len(work) < 20:
        raise ValueError("At least 20 rows with a non-missing target are required.")
    # Keep target as provided; impute feature columns inside the model pipeline to avoid leakage.
    STATE["target"] = target
    STATE["task"] = task
    STATE["imputation"] = imputation if imputation in {"median", "mean"} else "median"
    STATE["processed_df"] = work
    STATE["models"] = {}
    STATE["results"] = []
    STATE["best_model_name"] = None
    STATE["best_pipeline"] = None
    STATE["last_metrics"] = None
    STATE["engineered_df"] = None
    STATE["split_indices"] = None
    STATE["feature_selection"] = None
    return {
        "message": "Dataset prepared. Feature imputation/encoding/scaling will be fitted on training data only.",
        "rows_before": int(before),
        "rows_after": int(len(work)),
        "duplicates_removed": int(before - len(df.drop_duplicates())) if drop_duplicates else 0,
        "rows_without_target_removed": int(df[target].isna().sum()),
        "target": target,
        "task": task,
        "imputation": STATE["imputation"],
        "feature_count": int(work.shape[1] - 1),
        "class_counts": {str(k): int(v) for k, v in work[target].value_counts().items()} if task == "classification" else {},
    }

def engineer_features(df: pd.DataFrame):
    out = df.copy()
    if "Time" in out.columns and pd.api.types.is_numeric_dtype(out["Time"]):
        # Feature generation is simple and domain-inspired; original Time is retained.
        out["Time_hour"] = (out["Time"].fillna(0) / 3600.0) % 24
        out["Time_day"] = (out["Time"].fillna(0) / 86400.0).astype(float)
    if "Amount" in out.columns and pd.api.types.is_numeric_dtype(out["Amount"]):
        out["Amount_log1p"] = np.log1p(out["Amount"].clip(lower=0))
    return out

def feature_engineering_preview(limit=12):
    df = STATE.get("processed_df")
    target = STATE.get("target")
    if df is None or not target:
        raise ValueError("Run preprocessing first.")
    source = df.drop(columns=[target])
    engineered = engineer_features(source)
    original_columns = list(source.columns)
    generated_columns = [column for column in engineered.columns if column not in original_columns]
    return {
        "original_columns": original_columns,
        "generated_columns": generated_columns,
        "columns": list(engineered.columns),
        "preview": dataframe_preview(engineered, limit),
    }

def _build_preprocessor(X: pd.DataFrame):
    numeric_cols = list(X.select_dtypes(include=np.number).columns)
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    transformers = []
    if numeric_cols:
        numeric_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy=STATE.get("imputation", "median"), add_indicator=True)),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("numeric", numeric_pipe, numeric_cols))
    if categorical_cols:
        categorical_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append(("categorical", categorical_pipe, categorical_cols))
    return ColumnTransformer(transformers=transformers, remainder="drop")

def _model_candidates(task):
    if task == "classification":
        return {
            "Logistic Regression": LogisticRegression(max_iter=1500, class_weight="balanced"),
            "Decision Tree": DecisionTreeClassifier(
                class_weight="balanced", random_state=42, min_samples_leaf=2
            ),
            "XGBoost": XGBClassifier(
                n_estimators=180, max_depth=6, learning_rate=0.1,
                subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
                random_state=42, n_jobs=-1,
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=180, class_weight="balanced_subsample",
                random_state=42, n_jobs=-1, min_samples_leaf=2
            ),
            "SVM": SVC(
                probability=True, class_weight="balanced",
                random_state=42, max_iter=5000,
            ),
            "Naive Bayes": GaussianNB(),
        }
    return {
        "Linear Regression": LinearRegression(),
        "Multiple Linear Regression": LinearRegression(),
        "SVR": SVR(),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=180, random_state=42, n_jobs=-1),
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            n_estimators=180, learning_rate=0.05, max_depth=3, random_state=42
        ),
    }

def _build_feature_selector(task: str, feature_count: int):
    if feature_count <= 5:
        return None
    if task == "classification":
        estimator = LogisticRegression(
            penalty="l1", solver="liblinear", C=0.1,
            class_weight="balanced", max_iter=1500, random_state=42,
        )
        method = "L1 logistic"
    else:
        estimator = Lasso(alpha=0.01, max_iter=5000, random_state=42)
        method = "Lasso regression"
    return SelectFromModel(estimator, threshold="median", importance_getter="auto"), method

def _build_model_pipeline(X: pd.DataFrame, task: str):
    selector_info = _build_feature_selector(task, X.shape[1])
    selector = selector_info[0] if selector_info else "passthrough"
    return Pipeline([
        ("preprocess", _build_preprocessor(X)),
        ("feature_selection", selector),
    ]), selector_info[1] if selector_info else None

def _score_classifier(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return (scores - np.min(scores)) / (np.ptp(scores) + 1e-12)
    return None

def _classification_probabilities(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    if not hasattr(model, "decision_function"):
        return None
    decision = np.asarray(model.decision_function(X))
    if decision.ndim == 1:
        positive = 1 / (1 + np.exp(-np.clip(decision, -50, 50)))
        return np.column_stack([1 - positive, positive])
    shifted = decision - np.max(decision, axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    return probabilities / probabilities.sum(axis=1, keepdims=True)

def _classification_metrics(y_true, predictions, model, X, classes):
    matrix = confusion_matrix(y_true, predictions, labels=classes)
    average = "binary" if len(classes) == 2 else "weighted"
    if len(classes) == 2:
        true_negatives, false_positives, false_negatives, true_positives = matrix.ravel()
        specificity = (
            float(true_negatives / (true_negatives + false_positives))
            if true_negatives + false_positives else None
        )
    else:
        specificity_values = []
        for index in range(len(classes)):
            true_negatives = matrix.sum() - matrix[index, :].sum() - matrix[:, index].sum() + matrix[index, index]
            false_positives = matrix[:, index].sum() - matrix[index, index]
            specificity_values.append(
                true_negatives / (true_negatives + false_positives)
                if true_negatives + false_positives else 0
            )
        specificity = float(np.mean(specificity_values))
    probabilities = _classification_probabilities(model, X)
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, average=average, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, average=average, zero_division=0)),
        "specificity": specificity,
        "log_loss": float(log_loss(y_true, probabilities, labels=classes)) if probabilities is not None else None,
        "mcc": float(matthews_corrcoef(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "confusion_matrix": matrix.tolist(),
    }

def _regression_metrics(y_true, predictions, feature_count):
    mse_value = float(mean_squared_error(y_true, predictions))
    r2_value = float(r2_score(y_true, predictions))
    sample_count = len(y_true)
    adjusted_r2 = (
        float(1 - (1 - r2_value) * (sample_count - 1) / (sample_count - feature_count - 1))
        if sample_count > feature_count + 1
        else None
    )
    log_metrics = {"msle": None, "rmsle": None}
    if np.all(np.asarray(y_true) >= 0) and np.all(np.asarray(predictions) >= 0):
        msle_value = float(mean_squared_log_error(y_true, predictions))
        log_metrics = {"msle": msle_value, "rmsle": float(np.sqrt(msle_value))}
    return {
        "mae": float(mean_absolute_error(y_true, predictions)),
        "mse": mse_value,
        "rmse": float(np.sqrt(mse_value)),
        "r2": r2_value,
        "adjusted_r2": adjusted_r2,
        "mape": float(mean_absolute_percentage_error(y_true, predictions)),
        **log_metrics,
        "median_absolute_error": float(median_absolute_error(y_true, predictions)),
        "explained_variance": float(explained_variance_score(y_true, predictions)),
    }

def train_models(test_size=0.2, random_state=42):
    df = STATE["processed_df"]
    target = STATE["target"]
    task = STATE["task"]
    if df is None or not target:
        raise ValueError("Run preprocessing first.")
    X = engineer_features(df.drop(columns=[target]))
    y = df[target]
    if X.shape[1] == 0:
        raise ValueError("No usable feature columns remain.")
    # For classification, require at least two classes.
    if task == "classification" and y.nunique() < 2:
        raise ValueError("Classification requires at least two target classes.")
    stratify = y if task == "classification" and y.value_counts().min() >= 2 else None
    train_indices, test_indices = train_test_split(
        np.arange(len(X)), test_size=test_size, random_state=random_state, stratify=stratify
    )
    X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
    y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
    STATE["engineered_df"] = X
    STATE["split_indices"] = (train_indices, test_indices)
    results, pipelines = [], {}
    for name, estimator in _model_candidates(task).items():
        pipeline, selection_method = _build_model_pipeline(X_train, task)
        pipeline.steps.append(("model", estimator))
        try:
            pipeline.fit(X_train, y_train)
            preds = pipeline.predict(X_test)
            row = {"model": name}
            selector = pipeline.named_steps["feature_selection"]
            row["feature_selection"] = selection_method
            row["selected_feature_count"] = (
                int(selector.get_support().sum())
                if selector != "passthrough" else int(pipeline.named_steps["preprocess"].transform(X_train).shape[1])
            )
            if task == "classification":
                classes = pipeline.named_steps["model"].classes_
                row.update(_classification_metrics(y_test, preds, pipeline, X_test, classes))
                if y.nunique() == 2:
                    try:
                        scores = _score_classifier(pipeline, X_test)
                        row["roc_auc"] = float(roc_auc_score(y_test, scores)) if scores is not None else None
                        row["pr_auc"] = float(average_precision_score(y_test, scores)) if scores is not None else None
                    except Exception:
                        row["roc_auc"], row["pr_auc"] = None, None
                row["classes"] = [json_safe(v) for v in classes]
                row["classification_report"] = classification_report(y_test, preds, output_dict=True, zero_division=0)
            else:
                row.update({
                    **_regression_metrics(y_test, preds, X_train.shape[1]),
                    "y_test": [float(v) for v in y_test.tolist()],
                    "y_pred": [float(v) for v in preds.tolist()],
                })
            results.append(row)
            pipelines[name] = pipeline
        except Exception as e:
            results.append({"model": name, "error": str(e)})
    if not pipelines:
        raise ValueError("All candidate models failed. Check the dataset's target and feature types.")
    metric = "pr_auc" if task == "classification" and any(r.get("pr_auc") is not None for r in results) else ("f1" if task == "classification" else "r2")
    valid = [r for r in results if r["model"] in pipelines and r.get(metric) is not None]
    best = max(valid, key=lambda r: r[metric])["model"] if valid else next(iter(pipelines))
    STATE["models"] = pipelines
    STATE["results"] = results
    STATE["best_model_name"] = best
    STATE["best_pipeline"] = pipelines[best]
    STATE["feature_columns"] = list(X.columns)
    best_selector = STATE["best_pipeline"].named_steps["feature_selection"]
    STATE["feature_selection"] = {
        "method": next((r.get("feature_selection") for r in results if r["model"] == best), None),
        "input_feature_count": int(X.shape[1]),
        "transformed_feature_count": int(STATE["best_pipeline"].named_steps["preprocess"].transform(X_train).shape[1]),
        "selected_feature_count": int(best_selector.get_support().sum())
        if best_selector != "passthrough"
        else int(STATE["best_pipeline"].named_steps["preprocess"].transform(X_train).shape[1]),
    }
    # Save reproducible metadata and fitted pipeline.
    joblib.dump({
        "pipeline": STATE["best_pipeline"],
        "target": target,
        "task": task,
        "feature_columns": list(X.columns),
        "model_name": best,
        "feature_selection": STATE["feature_selection"],
    }, MODEL_PATH)
    STATE["last_metrics"] = {"best_model": best, "selection_metric": metric}
    return {
        "results": results,
        "best_model": best,
        "selection_metric": metric,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "feature_count": int(X.shape[1]),
        "feature_selection": STATE["feature_selection"],
        "target": target,
        "task": task,
    }

def tune_model(model_name: str, n_iter: int = 8):
    df, target, task = STATE["processed_df"], STATE["target"], STATE["task"]
    if df is None:
        raise ValueError("Run preprocessing first.")
    if model_name not in STATE["models"]:
        raise ValueError("Train models first and select a model from the comparison results.")
    X = STATE.get("engineered_df")
    split_indices = STATE.get("split_indices")
    if X is None or split_indices is None:
        raise ValueError("Train models first to prepare the tuning data.")
    y = df[target]
    train_indices, test_indices = split_indices
    X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
    y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
    estimator = clone(STATE["models"][model_name].named_steps["model"])
    if model_name == "Logistic Regression":
        params = {"model__C": [0.01, 0.1, 1, 5, 10], "model__solver": ["lbfgs"]}
    elif model_name == "Decision Tree":
        params = {"model__max_depth": [None, 5, 10, 20], "model__min_samples_leaf": [1, 2, 4]}
    elif model_name == "XGBoost":
        params = {
            "model__n_estimators": [100, 180, 250],
            "model__max_depth": [3, 5, 7],
            "model__learning_rate": [0.03, 0.1, 0.2],
        }
    elif model_name == "SVM":
        params = {"model__C": [0.01, 0.1, 1, 5, 10], "model__gamma": ["scale", "auto"]}
    elif model_name == "Naive Bayes":
        params = {"model__var_smoothing": np.logspace(-11, -7, 8)}
    elif model_name == "Random Forest":
        params = {"model__n_estimators": [100, 180, 250], "model__max_depth": [None, 8, 14, 22], "model__min_samples_leaf": [1, 2, 4]}
    elif model_name == "Random Forest Regressor":
        params = {"model__n_estimators": [100, 180, 250], "model__max_depth": [None, 8, 14, 22], "model__min_samples_leaf": [1, 2, 4]}
    elif model_name == "SVR":
        params = {
            "model__C": [0.1, 1, 10, 50],
            "model__epsilon": [0.01, 0.1, 0.2],
            "model__kernel": ["rbf", "linear"],
        }
    elif model_name == "Gradient Boosting Regressor":
        params = {
            "model__n_estimators": [100, 180, 250],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__max_depth": [2, 3, 5],
        }
    else:
        params = {}
    scoring = "average_precision" if task == "classification" and y.nunique() == 2 else ("f1_weighted" if task == "classification" else "r2")
    pipe, selection_method = _build_model_pipeline(X_train, task)
    pipe.steps.append(("model", estimator))
    search_space_size = max(1, prod(len(values) for values in params.values()))
    search = RandomizedSearchCV(pipe, params, n_iter=min(n_iter, search_space_size), cv=3,
                                scoring=scoring, random_state=42, n_jobs=-1, error_score="raise")
    search.fit(X_train, y_train)
    tuned = search.best_estimator_
    pred = tuned.predict(X_test)
    metrics = {}
    if task == "classification":
        classes = tuned.named_steps["model"].classes_
        metrics = _classification_metrics(y_test, pred, tuned, X_test, classes)
        if y.nunique() == 2:
            scores = _score_classifier(tuned, X_test)
            metrics["roc_auc"] = float(roc_auc_score(y_test, scores)) if scores is not None else None
            metrics["pr_auc"] = float(average_precision_score(y_test, scores)) if scores is not None else None
        metrics["classes"] = [json_safe(v) for v in classes]
    else:
        metrics = {
            **_regression_metrics(y_test, pred, X_train.shape[1]),
            "y_test": [float(v) for v in y_test.tolist()],
            "y_pred": [float(v) for v in pred.tolist()],
        }
    STATE["best_model_name"] = model_name + " (tuned)"
    STATE["best_pipeline"] = tuned
    STATE["feature_columns"] = list(X.columns)
    selector = tuned.named_steps["feature_selection"]
    STATE["feature_selection"] = {
        "method": selection_method,
        "input_feature_count": int(X.shape[1]),
        "transformed_feature_count": int(tuned.named_steps["preprocess"].transform(X_train).shape[1]),
        "selected_feature_count": int(selector.get_support().sum())
        if selector != "passthrough"
        else int(tuned.named_steps["preprocess"].transform(X_train).shape[1]),
    }
    joblib.dump({"pipeline": tuned, "target": target, "task": task,
                 "feature_columns": list(X.columns), "model_name": STATE["best_model_name"],
                 "feature_selection": STATE["feature_selection"]}, MODEL_PATH)
    return {
        "model": model_name,
        "best_params": search.best_params_,
        "cv_best_score": float(search.best_score_),
        "scoring": scoring,
        "metrics": metrics,
        "best_model": STATE["best_model_name"],
        "feature_selection": STATE["feature_selection"],
    }

def predict_one(values: dict):
    pipeline = STATE.get("best_pipeline")
    if pipeline is None:
        raise ValueError("Train or tune a model before making predictions.")
    features = STATE.get("feature_columns", [])
    row = {feature: values.get(feature, None) for feature in features}
    X = engineer_features(pd.DataFrame([row]))
    # Keep the exact trained feature order after creating derived fields.
    X = X.reindex(columns=features)
    prediction = pipeline.predict(X)[0]
    result = {"prediction": json_safe(prediction), "model": STATE.get("best_model_name")}
    try:
        scores = _score_classifier(pipeline, X)
        if scores is not None:
            result["positive_class_probability"] = float(scores[0])
            result["probability_note"] = "Score is a model score rescaled to 0–1 for Linear SVM; it is not a calibrated probability." if "SVM" in str(STATE.get("best_model_name")) else "Estimated probability for class 1."
    except Exception:
        pass
    return result

def model_artifact_path():
    if not MODEL_PATH.exists():
        raise ValueError("No trained model artifact exists yet.")
    return MODEL_PATH
