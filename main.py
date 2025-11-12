import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from typing import List
import joblib
from sklearn import metrics
import mlflow
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import subprocess 
from tabulate import tabulate # Used in the new fetch API

# --- 1. CONFIGURATION AND INITIALIZATION ---

# Get the path of the current script (main.py)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# FastAPI app initialization
app = FastAPI(
    title="Iris Decision Tree Classifier API",
    description="An API to train and serve an Iris classification model using MLflow."
)

# MLflow Configuration
MLFLOW_TRACKING_URI = "http://34.71.75.36:5000/"
REGISTERED_MODEL_NAME = "iris-tree-classifier"
DATA_PATH = os.path.join(SCRIPT_DIR, "data", "iris.csv") 
SAVE_PATH = "artifacts"  # Matches the path in your fetch script
MODEL_ARTIFACT_DIR = os.path.join(SAVE_PATH, "model")
# Common file names expected after MLflow download
MODEL_FILE_PATH_PKL = os.path.join(MODEL_ARTIFACT_DIR, "model.pkl") 
MODEL_FILE_PATH_CORE = os.path.join(MODEL_ARTIFACT_DIR, "model")    

# Initialize MLflow
MLFLOW_STATUS = "FAIL: Check URI"
try:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.sklearn.autolog(
        max_tuning_runs=15,
        registered_model_name=REGISTERED_MODEL_NAME
    )
    if mlflow.tracking.get_tracking_uri() == MLFLOW_TRACKING_URI:
        MLFLOW_STATUS = "OK"
except Exception:
    pass

# --- 2. DATA AND PARAMETER SETUP ---

PARAM_GRID = {
    'criterion': ['gini', 'entropy', 'log_loss'],
    'max_depth': [None, 5, 10, 20, 30],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1],
    'class_weight':[None]
}

# --- 3. DVC HELPER FUNCTION ---

def check_and_pull_data() -> dict:
    """Checks for the data file, and if not found, runs 'dvc pull'."""
    
    data_present_before = os.path.exists(DATA_PATH)
    
    if not data_present_before:
        print(f"Data file not found at {DATA_PATH}. Attempting to run 'dvc pull'...")
        
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        
        subprocess.run(
            ['dvc', 'pull'],
            check=True,  
            capture_output=True,
            text=True
        )
        print("DVC Pull successful.")
        
        data_present_after = os.path.exists(DATA_PATH)
        
        if not data_present_after:
            raise FileNotFoundError(
                f"DVC pull completed, but data file {DATA_PATH} is still missing. Training cannot proceed."
            )
        
        return {
            "status": "success",
            "message": "Data was missing, DVC pull completed successfully.",
            "data_was_pulled": True,
            "data_path": DATA_PATH
        }
            
    else:
        print(f"Data file found at {DATA_PATH}. Skipping DVC pull.")
        return {
            "status": "success",
            "message": "Data file was already present. DVC pull skipped.",
            "data_was_pulled": False,
            "data_path": DATA_PATH
        }


# --- 4. MODEL TRAINING FUNCTION ---

def train_and_tune_model() -> dict:
    """Loads data (pulling with DVC if necessary), trains the model, and logs to MLflow."""
    
    pull_result = check_and_pull_data()

    data = pd.read_csv(DATA_PATH)
    train, test = train_test_split(data, test_size=0.2, stratify=data['species'], random_state=55)

    X_train = train[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y_train = train.species
    X_test = test[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y_test = test.species

    with mlflow.start_run(run_name="DecisionTree Classifier Hyperparameter Tuning"):
        
        model = DecisionTreeClassifier(random_state=1)
        grid_search = GridSearchCV(
            model,
            PARAM_GRID,
            cv=5,
            scoring="accuracy",
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        
        best_score_cv = grid_search.best_score_
        best_params = grid_search.best_params_
        
        best_model = grid_search.best_estimator_
        test_score = best_model.score(X_test, y_test)
        
        mlflow.log_metric("final_test_accuracy", test_score)
        mlflow.log_params(best_params)
        
        return {
            "status": "success",
            "message": "Model training and hyperparameter tuning complete.",
            "best_cv_score": round(best_score_cv, 4),
            "test_accuracy": round(test_score, 3),
            "best_parameters": best_params,
            "data_pull_info": pull_result
        }

# --- 5. PYDANTIC SCHEMAS ---

class IrisSample(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class PredictionResponse(BaseModel):
    prediction: str

# --- 6. API ROUTES (ENDPOINTS) ---

@app.get("/", tags=["System"])
def root_status():
    """Simple status check at the root."""
    return {"status": "ok"}

@app.get("/health", tags=["System"])
def health_check():
    """Detailed health check of file presence and external dependencies."""
    
    # Check for model file presence
    model_present = os.path.exists(MODEL_FILE_PATH_PKL) or os.path.exists(MODEL_FILE_PATH_CORE)
    model_status = "OK" if model_present else "FAIL: Model file missing in artifacts/model/"

    health_data = {
        "service": "Iris Decision Tree Classifier API",
        "system_status": "OK",
        "file_checks": {
            "main_script": "OK" if os.path.exists(os.path.join(SCRIPT_DIR, "main.py")) else "FAIL: Not Found",
            "data_dir": "OK" if os.path.exists(os.path.dirname(DATA_PATH)) else "FAIL: Not Found",
            "iris_csv": "OK" if os.path.exists(DATA_PATH) else "FAIL: Not Found (Requires DVC pull)",
            "dvc_config": "OK" if os.path.exists(os.path.join(SCRIPT_DIR, "dvc.yaml")) else "WARN: dvc.yaml Missing",
            "local_model_file": model_status
        },
        "dependency_checks": {
            "mlflow_tracking_uri": MLFLOW_TRACKING_URI,
            "mlflow_connection": MLFLOW_STATUS
        }
    }
    
    # Update system_status if any critical check fails
    for key, value in health_data['file_checks'].items():
        if value.startswith("FAIL"):
            health_data['system_status'] = "DEGRADED"
            break
            
    if health_data['dependency_checks']['mlflow_connection'].startswith("FAIL"):
        health_data['system_status'] = "DEGRADED"

    return health_data

@app.post("/data/pull", tags=["Data Management"])
def pull_data_endpoint():
    """Forces a 'dvc pull' to update the local data repository."""
    try:
        result = check_and_pull_data()
        return result
    except subprocess.CalledProcessError as e:
        return HTTPException(
            status_code=500, 
            detail=f"DVC pull failed. Check DVC remote and configuration. Stderr: {e.stderr}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Data pull failed: {e.__class__.__name__}: {e}"
        )

@app.post("/model/fetch_and_predict", tags=["Inference/Deployment"])
def fetch_and_predict_local():
    """
    Fetches the latest registered model from MLflow using the provided script logic, 
    saves it locally, loads the local data, and runs prediction/evaluation on the test set.
    """
    try:
        client = mlflow.tracking.MlflowClient()
        
        # 1. Fetch latest model version details
        versions = client.search_model_versions(
            filter_string=f"name='{REGISTERED_MODEL_NAME}'",
            order_by=["version_number DESC"],
            max_results=1
        )
        if not versions:
            raise RuntimeError(f"No registered versions found for model: {REGISTERED_MODEL_NAME}")
            
        latest_version = versions[0]
        run_id = latest_version.run_id
        
        # 2. Download Artifacts (strictly using your code logic)
        os.makedirs(SAVE_PATH, exist_ok=True) # Ensure artifacts directory exists
        
        mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path="model",
            dst_path=SAVE_PATH
        )

        mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path="training_confusion_matrix.png",
            dst_path=SAVE_PATH
        )

        # 3. Load the local model and test data
        
        # Determine the correct model file path
        if os.path.exists(MODEL_FILE_PATH_PKL):
            model_path_to_load = MODEL_FILE_PATH_PKL
        elif os.path.exists(MODEL_FILE_PATH_CORE):
            model_path_to_load = MODEL_FILE_PATH_CORE
        else:
            raise FileNotFoundError(f"Downloaded model file not found in {MODEL_ARTIFACT_DIR}")

        # MLflow's autologging models are often saved as 'model' or 'model.pkl' 
        # in the 'model' subdirectory. We use joblib.load for scikit-learn models.
        # However, for robustness with autologging, we use mlflow.sklearn.load_model 
        # pointed to the local folder, as it handles the model signature and environment.
        local_model = mlflow.sklearn.load_model(MODEL_ARTIFACT_DIR)
        
        # Load the full dataset (data presence check is separate, but pull before predicting if necessary)
        check_and_pull_data()
        data = pd.read_csv(DATA_PATH)
        train, test = train_test_split(data, test_size=0.2, stratify=data['species'], random_state=55)

        X_test = test[['sepal_length','sepal_width','petal_length','petal_width']]
        y_test = test.species
        
        # 4. Predict and Evaluate
        y_pred = local_model.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, y_pred)
        
        # 5. Format Metrics (strictly using your code logic for the table)
        run = client.get_run(run_id)
        model_metrics = run.data.metrics
        
        header = ["Metric", "Value"]
        table_data = []
        for key, val in model_metrics.items():
            val_str = f"{val:.4f}" if isinstance(val, float) else str(val)
            table_data.append([key, val_str])

        metrics_table_string = tabulate(
            table_data, 
            headers=header, 
            tablefmt="github"
        )
        
        # Create a simple report file (using your file writing logic)
        with open(os.path.join(SAVE_PATH, "metrics.md"), "w") as f:
            f.write("# Metrics Table\n\n")
            f.write(f"{metrics_table_string}\n\n")
            f.write("# Confusion Matrix")
            f.write("![](./training_confusion_matrix.png)")
            
        return {
            "status": "success",
            "message": "Model fetched, loaded, and predicted on test set.",
            "model_version": latest_version.version,
            "test_accuracy_from_local_model": round(accuracy, 4),
            "metrics_report_location": f"{SAVE_PATH}/metrics.md",
            "artifact_location": MODEL_ARTIFACT_DIR
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Model fetch/predict failed: {e.__class__.__name__}: {e}"
        )

# (The old /train and /predict routes remain for comparison/completeness)

@app.post("/train", tags=["Model Training"])
def train_model_endpoint():
    """Triggers the training, which includes an attempt to run 'dvc pull' if data is missing."""
    try:
        results = train_and_tune_model()
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {e.__class__.__name__}: {e}")

@app.post("/predict", response_model=PredictionResponse, tags=["Inference/Deployment"])
def predict(sample: IrisSample):
    """
    (OLD ROUTE) Makes a prediction using the latest production model loaded directly 
    from the MLflow Model Registry URI.
    """
    try:
        model_uri = "artifacts/model/model.pkl"
        
        loaded_model = joblib.load(model_uri)
        
        
    except Exception:
        raise HTTPException(
            status_code=503,
            detail="Model not available. Check MLflow connection/registry stage."
        )

    input_data = pd.DataFrame([sample.model_dump()])
    
    input_data = input_data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]

    try:
        prediction = loaded_model.predict(input_data)[0]
        return PredictionResponse(prediction=prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e.__class__.__name__}")

@app.get("/test/run_data_tests", tags=["Testing"])
@app.post("/test/run_data_tests", tags=["Testing"])
def run_data_tests():
    """
    Executes 'pytest test_data.py' and returns the console output.
    The test file 'test_data.py' must be in the same directory as this script.
    """
    test_file = "test_data.py"
    
    # 1. Check if the test file exists
    if not os.path.exists(test_file):
        raise HTTPException(
            status_code=404, 
            detail=f"Test file not found: {test_file}. Ensure it is in the current directory."
        )

    # 2. Execute Pytest using subprocess
    try:
        # We use ['pytest', test_file] to avoid shell=True, which is safer.
        # -s flag is used to show print statements and prevent pytest from capturing all stdout.
        # --tb=line shows traceback just one line per error/failure for cleaner output.
        command = ["pytest", "-s", "--tb=line", test_file]
        
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,  # Capture output as string (decoded)
            check=False # Do NOT raise an exception on non-zero exit code (i.e., test failure)
        )
        
        # 3. Format Output
        # Pytest prints success/failure summary to stdout and errors/failures to stderr.
        output = f"--- Pytest Command: {' '.join(command)} ---\n"
        output += f"--- Exit Code: {result.returncode} (0=Success, >0=Failures) ---\n\n"
        output += "--- STDOUT (Summary/Details) ---\n"
        output += result.stdout
        
        if result.stderr:
            output += "\n\n--- STDERR (Errors/Tracebacks) ---\n"
            output += result.stderr

        return {
            "status": "completed",
            "exit_code": result.returncode,
            "test_output": output,
            "message": "Pytest execution finished."
        }
        
    except FileNotFoundError:
        # This catches if the 'pytest' command itself is not found in the environment's PATH
        raise HTTPException(
            status_code=500, 
            detail="Pytest command not found. Ensure 'pytest' is installed and available in the environment."
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"An unexpected error occurred during test execution: {e.__class__.__name__}: {e}"
        )