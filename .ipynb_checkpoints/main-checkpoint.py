"""
Production MLOps FastAPI Application - Evidently 0.6+ API
Uses common_functions for all ML operations
"""

import os
import pandas as pd
import numpy as np
from typing import List
import joblib
import mlflow
import mlflow.sklearn
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import subprocess

# Import common functions
from common_functions import (
    check_and_pull_data,
    load_and_split_data,
    poison_labels,
    poison_features,
    train_model_with_grid_search,
    calculate_class_metrics,
    calculate_fairness_metrics,
    generate_shap_plots,
    generate_confusion_matrix,
    generate_drift_report,
    generate_drift_report_with_tests,
    log_predictions_to_file,
    log_metrics_to_mlflow,
    log_artifacts_to_mlflow,
    PARAM_GRID
)

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MLFLOW_TRACKING_URI = "http://34.10.224.106:5000/"
PRODUCTION_EXPERIMENT = "fastapi-production"
REGISTERED_MODEL_NAME = "iris-tree-classifier"

SAVE_PATH = "artifacts"
MODEL_ARTIFACT_DIR = os.path.join(SAVE_PATH, "model")
MODEL_FILE_PATH_PKL = os.path.join(MODEL_ARTIFACT_DIR, "model.pkl")
PREDICTIONS_LOG_PATH = os.path.join(SAVE_PATH, "predictions_log.csv")

os.makedirs(SAVE_PATH, exist_ok=True)

# Initialize MLflow
MLFLOW_STATUS = "FAIL: Check URI"
try:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    if mlflow.tracking.get_tracking_uri() == MLFLOW_TRACKING_URI:
        MLFLOW_STATUS = "OK"
except:
    pass

# FastAPI app
app = FastAPI(
    title="MLOps Production API - Iris Classifier",
    description="Complete MLOps API with Evidently 0.6+ for drift detection",
    version="2.2.0"
)

# --- PYDANTIC MODELS ---

class IrisSample(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class PredictionResponse(BaseModel):
    prediction: str
    model_version: str

class PoisonConfig(BaseModel):
    poison_type: str = "label"
    levels: List[int] = [0, 5, 10, 25, 50]

# --- API ENDPOINTS ---

@app.get("/", tags=["System"])
def root():
    return {
        "service": "MLOps Production API - Iris Classifier",
        "version": "2.2.0",
        "evidently_version": "0.6+",
        "status": "running",
        "mlflow_status": MLFLOW_STATUS,
        "production_experiment": PRODUCTION_EXPERIMENT
    }

@app.get("/health", tags=["System"])
def health_check():
    model_present = os.path.exists(MODEL_FILE_PATH_PKL)
    data_present = os.path.exists("./data/iris.csv")
    
    return {
        "service": "MLOps Production API",
        "system_status": "OK" if model_present and data_present else "DEGRADED",
        "file_checks": {
            "iris_csv": "OK" if data_present else "MISSING",
            "model_file": "OK" if model_present else "MISSING",
            "artifacts_dir": "OK" if os.path.exists(SAVE_PATH) else "MISSING"
        },
        "dependencies": {
            "mlflow_uri": MLFLOW_TRACKING_URI,
            "mlflow_status": MLFLOW_STATUS,
            "evidently_api": "0.6+"
        }
    }

@app.post("/data/pull", tags=["Data Management"])
def pull_data():
    try:
        result = check_and_pull_data()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DVC pull failed: {str(e)}")

@app.post("/train/baseline", tags=["Model Training"])
def train_baseline():
    """Trains baseline model without poisoning."""
    try:
        X_train, X_test, y_train, y_test, _, _ = load_and_split_data()
        
        mlflow.set_experiment(PRODUCTION_EXPERIMENT)
        
        with mlflow.start_run(run_name="Baseline_DecisionTree") as run:
            mlflow.set_tag("experiment_type", "baseline")
            mlflow.set_tag("poison_percent", "0%")
            
            best_model, metrics = train_model_with_grid_search(X_train, y_train, X_test, y_test)
            
            log_metrics_to_mlflow(metrics)
            mlflow.log_params(metrics["best_params"])
            
            return {
                "status": "success",
                "model_type": "baseline",
                "test_accuracy": metrics["test_accuracy"],
                "cv_score": metrics["cv_score"],
                "run_id": run.info.run_id
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.post("/train/poisoning", tags=["Model Training"])
def train_poisoning(config: PoisonConfig):
    """Runs data poisoning experiments."""
    try:
        if config.poison_type not in ["label", "feature"]:
            raise ValueError("poison_type must be 'label' or 'feature'")
        
        X_train, X_test, y_train, y_test, _, _ = load_and_split_data()
        mlflow.set_experiment(PRODUCTION_EXPERIMENT)
        
        results = []
        for percent in config.levels:
            run_name = f"DecisionTree_{config.poison_type.capitalize()}Poison_{percent}pct"
            
            with mlflow.start_run(run_name=run_name) as run:
                mlflow.set_tag("poison_type", config.poison_type)
                mlflow.set_tag("poison_percent_tag", f"{percent}%")
                mlflow.log_metric("poison_percent", percent)
                
                if config.poison_type == "label":
                    y_train_used = poison_labels(y_train, percent)
                    X_train_used = X_train
                else:
                    X_train_used = poison_features(X_train, percent)
                    y_train_used = y_train
                
                best_model, metrics = train_model_with_grid_search(
                    X_train_used, y_train_used, X_test, y_test
                )
                
                log_metrics_to_mlflow(metrics)
                mlflow.log_params(metrics["best_params"])
                
                results.append({
                    "poison_percent": percent,
                    "test_accuracy": metrics["test_accuracy"],
                    "cv_score": metrics["cv_score"],
                    "run_id": run.info.run_id
                })
        
        return {
            "status": "success",
            "poison_type": config.poison_type,
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Poisoning experiment failed: {str(e)}")

@app.post("/train/fairness", tags=["Model Training"])
def train_fairness():
    """Trains model with fairness analysis and SHAP for all targets."""
    try:
        X_train, X_test, y_train, y_test, _, _ = load_and_split_data(add_location=True)
        
        mlflow.set_experiment(PRODUCTION_EXPERIMENT)
        
        with mlflow.start_run(run_name="Fairness_SHAP_AllTargets") as run:
            mlflow.set_tag("experiment_type", "fairness_shap")
            
            # Train model
            best_model, train_metrics = train_model_with_grid_search(
                X_train, y_train, X_test, y_test
            )
            log_metrics_to_mlflow(train_metrics)
            
            # Predictions
            y_pred = best_model.predict(X_test)
            species_list = sorted(y_train.unique())
            
            # Class-specific metrics
            class_metrics = calculate_class_metrics(y_test, y_pred, species_list)
            for species, metrics in class_metrics.items():
                for metric_name, value in metrics.items():
                    mlflow.log_metric(f"{species}_{metric_name}", value)
            
            # Fairness metrics
            fairness_metrics = calculate_fairness_metrics(y_test, y_pred, X_test['location'])
            mlflow.log_metric("fairness_acc_diff", fairness_metrics["accuracy_difference"])
            
            # SHAP plots
            shap_paths = generate_shap_plots(best_model, X_test, species_list, SAVE_PATH, "prod_")
            log_artifacts_to_mlflow(shap_paths)
            
            # Confusion matrix
            cm_path = generate_confusion_matrix(y_test, y_pred, species_list, SAVE_PATH, "prod_confusion_matrix.png")
            mlflow.log_artifact(cm_path)
            
            return {
                "status": "success",
                "test_accuracy": train_metrics["test_accuracy"],
                "class_metrics": class_metrics,
                "fairness_acc_diff": fairness_metrics["accuracy_difference"],
                "shap_plots_generated": len(shap_paths),
                "run_id": run.info.run_id
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fairness training failed: {str(e)}")

@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(sample: IrisSample):
    """Makes prediction and logs for drift monitoring."""
    try:
        if not os.path.exists(MODEL_FILE_PATH_PKL):
            raise HTTPException(status_code=503, detail="Model not available. Train model first.")
        
        loaded_model = joblib.load(MODEL_FILE_PATH_PKL)
        input_data = pd.DataFrame([sample.model_dump()])
        input_data = input_data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
        
        prediction = loaded_model.predict(input_data)[0]
        
        # Log for drift monitoring
        log_predictions_to_file(sample.model_dump(), prediction, "v1.0", PREDICTIONS_LOG_PATH)
        
        return PredictionResponse(prediction=prediction, model_version="v1.0")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/monitoring/drift", tags=["Monitoring"])
def view_drift_report():
    """Generates and serves Evidently data drift report (0.6+ API)."""
    try:
        _, _, _, _, train, test = load_and_split_data()
        
        # Use predictions log if available, otherwise use test set
        if os.path.exists(PREDICTIONS_LOG_PATH) and os.path.getsize(PREDICTIONS_LOG_PATH) > 100:
            current_df = pd.read_csv(PREDICTIONS_LOG_PATH)
            feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
            current_data = current_df[feature_cols + ['prediction']].rename(columns={'prediction': 'species'})
        else:
            current_data = test.copy()
        
        result = generate_drift_report(train, current_data, SAVE_PATH, "drift_report.html")
        
        return FileResponse(
            result["report_path"],
            media_type="text/html",
            filename="drift_report.html"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Drift report failed: {str(e)}")

@app.get("/monitoring/drift/json", tags=["Monitoring"])
def drift_report_json():
    """Returns drift analysis as JSON."""
    try:
        _, _, _, _, train, test = load_and_split_data()
        
        if os.path.exists(PREDICTIONS_LOG_PATH) and os.path.getsize(PREDICTIONS_LOG_PATH) > 100:
            current_df = pd.read_csv(PREDICTIONS_LOG_PATH)
            feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
            current_data = current_df[feature_cols + ['prediction']].rename(columns={'prediction': 'species'})
        else:
            current_data = test.copy()
        
        result = generate_drift_report(train, current_data, SAVE_PATH, "drift_report_temp.html")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Drift analysis failed: {str(e)}")

@app.post("/monitoring/drift/tests", tags=["Monitoring"])
def run_drift_tests():
    """Runs Evidently drift report with test conditions (CI/CD style)."""
    try:
        _, _, _, _, train, test = load_and_split_data()
        result = generate_drift_report_with_tests(train, test, SAVE_PATH, "drift_tests.html")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Drift tests failed: {str(e)}")

@app.post("/test/run_data_tests", tags=["Testing"])
def run_data_tests():
    """Executes pytest on test_data.py."""
    test_file = "test_data.py"
    
    if not os.path.exists(test_file):
        raise HTTPException(status_code=404, detail=f"Test file {test_file} not found")
    
    try:
        result = subprocess.run(
            ["pytest", "-s", "--tb=line", test_file],
            capture_output=True,
            text=True,
            check=False
        )
        
        return {
            "status": "completed",
            "exit_code": result.returncode,
            "test_output": f"Exit Code: {result.returncode}\n\n{result.stdout}\n\n{result.stderr}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Test execution failed: {str(e)}")


