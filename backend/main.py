from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import pandas as pd
import io
import traceback

from ml_pipeline import (
    STATE, set_dataset, dataset_summary, make_demo_dataset,
    preprocess_dataset, feature_engineering_preview, train_models, tune_model, predict_one,
    model_artifact_path
)

app = FastAPI(
    title="MLFlow Studio API",
    description="An educational end-to-end machine learning workflow platform.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def handle_error(exc):
    return HTTPException(status_code=400, detail=str(exc))

class PreprocessRequest(BaseModel):
    target: str
    task: str = "classification"
    imputation: str = "median"
    drop_duplicates: bool = True

class TrainRequest(BaseModel):
    test_size: float = Field(default=0.2, ge=0.1, le=0.4)
    random_state: int = 42

class TuneRequest(BaseModel):
    model: str
    n_iter: int = Field(default=8, ge=1, le=20)

class PredictRequest(BaseModel):
    values: dict

@app.get("/")
def root():
    return {"name": "MLFlow Studio API", "status": "running", "docs": "/docs"}

@app.get("/api/health")
def health():
    return {"status": "ok", "dataset_loaded": STATE["raw_df"] is not None}

@app.post("/api/dataset/demo")
def load_demo():
    try:
        df = make_demo_dataset()
        summary = set_dataset(df, "synthetic_demo.csv")
        return {"message": "Synthetic demo dataset loaded.", "summary": summary,
                "warning": "This is synthetic data for demonstrating the workflow, not a real-world benchmark."}
    except Exception as e:
        raise handle_error(e)

@app.post("/api/dataset/upload")
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file.")
    try:
        contents = await file.read()
        if len(contents) > 80 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Maximum upload size is 80 MB.")
        df = pd.read_csv(io.BytesIO(contents), low_memory=False)
        summary = set_dataset(df, file.filename)
        return {"message": f"Loaded {file.filename}", "summary": summary}
    except HTTPException:
        raise
    except Exception as e:
        raise handle_error(e)

@app.get("/api/dataset/summary")
def get_summary():
    if STATE["raw_df"] is None:
        return {"loaded": False}
    return dataset_summary()

@app.post("/api/preprocess")
def preprocess(request: PreprocessRequest):
    try:
        result = preprocess_dataset(request.target, request.task, request.imputation, request.drop_duplicates)
        return {"result": result, "summary": dataset_summary(STATE["processed_df"])}
    except Exception as e:
        raise handle_error(e)

@app.get("/api/features/preview")
def features_preview():
    try:
        return feature_engineering_preview()
    except Exception as e:
        raise handle_error(e)

@app.post("/api/train")
def train(request: TrainRequest):
    try:
        return train_models(request.test_size, request.random_state)
    except Exception as e:
        raise handle_error(e)

@app.post("/api/tune")
def tune(request: TuneRequest):
    try:
        return tune_model(request.model, request.n_iter)
    except Exception as e:
        raise handle_error(e)

@app.post("/api/predict")
def predict(request: PredictRequest):
    try:
        return predict_one(request.values)
    except Exception as e:
        raise handle_error(e)

@app.get("/api/model/download")
def download_model():
    try:
        path = model_artifact_path()
        return FileResponse(path, filename="mlflow_studio_best_model.joblib", media_type="application/octet-stream")
    except Exception as e:
        raise handle_error(e)
