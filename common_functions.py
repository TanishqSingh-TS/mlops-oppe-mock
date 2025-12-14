"""
Common MLOps Functions - Evidently 0.7+ (Working Pattern from test_experiment.py)
Shared utilities for training, evaluation, drift detection, and visualization
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import mlflow
import mlflow.sklearn
import shap
from fairlearn.metrics import MetricFrame, selection_rate
import seaborn as sns
from typing import Dict, List, Tuple
import subprocess

# ✅ Evidently imports - Same as your working code
from evidently import Report
from evidently.presets import DataDriftPreset

# --- CONFIGURATION ---
DATA_PATH = "./data/iris.csv"
PARAM_GRID = {
    'criterion': ['gini', 'entropy', 'log_loss'],
    'max_depth': [None, 5, 10, 20, 30],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1],
    'class_weight': [None]
}

PARAM_GRID_FAST = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5]
}

# --- DATA MANAGEMENT ---

def check_and_pull_data() -> Dict:
    """Checks for data file and runs DVC pull if needed."""
    data_present = os.path.exists(DATA_PATH)
    
    if not data_present:
        print(f"Data not found at {DATA_PATH}. Running 'dvc pull'...")
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        
        try:
            subprocess.run(['dvc', 'pull'], check=True, capture_output=True, text=True)
            print("✓ DVC pull successful")
        except subprocess.CalledProcessError as e:
            raise FileNotFoundError(f"DVC pull failed: {e.stderr}")
        
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(f"Data file {DATA_PATH} still missing after DVC pull")
        
        return {"status": "success", "message": "Data pulled via DVC", "data_was_pulled": True}
    
    return {"status": "success", "message": "Data already present", "data_was_pulled": False}

def load_and_split_data(add_location: bool = False) -> Tuple:
    """Loads iris data and performs train-test split."""
    check_and_pull_data()
    data = pd.read_csv(DATA_PATH)
    
    if add_location:
        np.random.seed(42)
        data['location'] = np.random.randint(0, 2, size=len(data))
    
    if add_location:
        feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'location']
    else:
        feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    
    train, test = train_test_split(data, test_size=0.2, stratify=data['species'], random_state=55)
    
    X_train = train[feature_cols]
    y_train = train['species']
    X_test = test[feature_cols]
    y_test = test['species']
    
    return X_train, X_test, y_train, y_test, train, test

# --- POISONING FUNCTIONS ---

def poison_labels(y_train: pd.Series, poison_percent: float) -> pd.Series:
    """Flips labels for specified percentage of training data."""
    if poison_percent == 0:
        print("Training on 0% poisoned (clean) data")
        return y_train.copy()
    
    y_poisoned = y_train.copy()
    num_to_poison = int(len(y_poisoned) * (poison_percent / 100.0))
    classes = y_train.unique()
    poison_indices = np.random.choice(y_poisoned.index, num_to_poison, replace=False)
    
    poisoned_count = 0
    for idx in poison_indices:
        original = y_poisoned.loc[idx]
        possible = [c for c in classes if c != original]
        if possible:
            y_poisoned.loc[idx] = np.random.choice(possible)
            poisoned_count += 1
    
    print(f"✓ Poisoned {poisoned_count} labels ({poison_percent}%)")
    return y_poisoned

def poison_features(X_train: pd.DataFrame, poison_percent: float) -> pd.DataFrame:
    """Adds extreme outlier noise to features."""
    if poison_percent == 0:
        print("Training on 0% poisoned (clean) features")
        return X_train.copy()
    
    X_poisoned = X_train.copy()
    num_to_poison = int(len(X_poisoned) * (poison_percent / 100.0))
    poison_indices = np.random.choice(X_poisoned.index, num_to_poison, replace=False)
    features_to_poison = ['sepal_length', 'sepal_width']
    
    for idx in poison_indices:
        for feature in features_to_poison:
            noise_factor = np.random.uniform(5, 10)
            X_poisoned.loc[idx, feature] *= noise_factor
    
    print(f"✓ Poisoned features for {len(poison_indices)} samples ({poison_percent}%)")
    return X_poisoned

# --- TRAINING FUNCTIONS ---

def train_model_with_grid_search(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    param_grid: Dict = None,
    cv: int = 5
) -> Tuple[DecisionTreeClassifier, Dict]:
    """Trains model with GridSearchCV and returns best model + metrics."""
    if param_grid is None:
        param_grid = PARAM_GRID
    
    model = DecisionTreeClassifier(random_state=1)
    grid_search = GridSearchCV(model, param_grid, cv=cv, scoring="accuracy", n_jobs=-1, verbose=0)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    cv_score = grid_search.best_score_
    test_score = best_model.score(X_test, y_test)
    
    metrics = {
        "cv_score": round(cv_score, 4),
        "test_accuracy": round(test_score, 4),
        "best_params": grid_search.best_params_
    }
    
    return best_model, metrics

# --- METRICS CALCULATION ---

def calculate_class_metrics(y_test: pd.Series, y_pred: np.ndarray, species_list: List[str]) -> Dict:
    """Calculates precision, recall, F1 for each species class."""
    class_metrics = {}
    
    for species in species_list:
        precision = precision_score(y_test, y_pred, labels=[species], average=None, zero_division=0)[0]
        recall = recall_score(y_test, y_pred, labels=[species], average=None, zero_division=0)[0]
        f1 = f1_score(y_test, y_pred, labels=[species], average=None, zero_division=0)[0]
        
        class_metrics[species] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4)
        }
    
    return class_metrics

def calculate_fairness_metrics(
    y_test: pd.Series,
    y_pred: np.ndarray,
    sensitive_features: pd.Series
) -> Dict:
    """Calculates fairness metrics using Fairlearn."""
    metrics_dict = {
        'accuracy': accuracy_score,
        'selection_rate': selection_rate
    }
    
    grouped = MetricFrame(
        metrics=metrics_dict,
        y_true=y_test,
        y_pred=y_pred,
        sensitive_features=sensitive_features
    )
    
    acc_diff = grouped.difference(method='between_groups')['accuracy']
    
    return {
        "accuracy_difference": round(acc_diff, 4),
        "by_group": grouped.by_group.to_dict()
    }

# --- SHAP VISUALIZATION ---

def generate_shap_plots(
    model: DecisionTreeClassifier,
    X_test: pd.DataFrame,
    species_list: List[str],
    artifacts_dir: str,
    prefix: str = ""
) -> List[str]:
    """Generates SHAP plots (waterfall, beeswarm, bar) for all species."""
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)
    
    artifact_paths = []
    
    for idx, species in enumerate(species_list):
        # Waterfall plot
        plt.figure()
        shap.plots.waterfall(shap_values[0, :, idx], show=False)
        plt.title(f"SHAP Waterfall - {species}")
        path = os.path.join(artifacts_dir, f"{prefix}shap_waterfall_{species}.png")
        plt.savefig(path, bbox_inches='tight', dpi=150)
        plt.close()
        artifact_paths.append(path)
        
        # Beeswarm plot
        plt.figure(figsize=(10, 6))
        shap.plots.beeswarm(shap_values[..., idx], show=False)
        plt.title(f"SHAP Beeswarm - {species}")
        path = os.path.join(artifacts_dir, f"{prefix}shap_beeswarm_{species}.png")
        plt.savefig(path, bbox_inches='tight', dpi=150)
        plt.close()
        artifact_paths.append(path)
        
        # Bar plot
        plt.figure()
        shap.plots.bar(shap_values[..., idx], show=False)
        plt.title(f"SHAP Bar - {species}")
        path = os.path.join(artifacts_dir, f"{prefix}shap_bar_{species}.png")
        plt.savefig(path, bbox_inches='tight', dpi=150)
        plt.close()
        artifact_paths.append(path)
    
    print(f"✓ Generated {len(artifact_paths)} SHAP plots")
    return artifact_paths

def generate_confusion_matrix(
    y_test: pd.Series,
    y_pred: np.ndarray,
    species_list: List[str],
    artifacts_dir: str,
    filename: str = "confusion_matrix.png"
) -> str:
    """Generates and saves confusion matrix heatmap."""
    cm = confusion_matrix(y_test, y_pred, labels=species_list)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=species_list, yticklabels=species_list)
    plt.title('Confusion Matrix - All Species')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    path = os.path.join(artifacts_dir, filename)
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"✓ Generated confusion matrix: {filename}")
    return path

# --- EVIDENTLY DATA DRIFT (USING YOUR WORKING PATTERN) ---

def generate_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    artifacts_dir: str,
    filename: str = "drift_report.html"
) -> Dict:
    """
    Generates Evidently data drift report using the exact pattern from test_experiment.py
    that's already working in your code.
    """
    print("\n--- Generating Evidently Data Drift Report ---")
    
    # Prepare data - only numerical features
    numerical_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    available_cols = [col for col in numerical_cols if col in reference_df.columns and col in current_df.columns]
    
    reference_data = reference_df[available_cols].copy()
    current_data = current_df[available_cols].copy()
    
    print(f"Reference data shape: {reference_data.shape}")
    print(f"Current data shape: {current_data.shape}")
    
    try:
        # ✅ Generate Report using latest API (EXACT PATTERN from your code)
        report = Report([
            DataDriftPreset()
        ],
        include_tests=True)
        
        # ✅ Run returns evaluation object
        my_eval = report.run(current_data, reference_data)
        
        # ✅ Save HTML using returned object
        report_path = os.path.join(artifacts_dir, filename)
        my_eval.save_html(report_path)
        
        print(f"✓ Drift report saved: {report_path}")
        
        # ✅ Optional: Get drift results as dict or JSON
        drift_dict = None
        dataset_drift = False
        drift_share = 0.0
        num_drifted_columns = 0
        drifted_features = []
        
        try:
            drift_dict = my_eval.dict()
            print("✓ Drift detection completed successfully")
            
            # Parse drift metrics
            metrics = drift_dict.get('metrics', [])
            for metric in metrics:
                metric_name = metric.get('metric', '')
                result = metric.get('result', {})
                
                if 'DatasetDrift' in metric_name or 'drift' in metric_name.lower():
                    dataset_drift = result.get('dataset_drift', False)
                    drift_share = result.get('drift_share', 0.0)
                    num_drifted_columns = result.get('number_of_drifted_columns', 0)
                    
                    # Extract drifted column names
                    if 'drift_by_columns' in result:
                        for col, col_result in result['drift_by_columns'].items():
                            if col_result.get('drift_detected', False):
                                drifted_features.append(col)
                    
                    break
            
            print(f"✓ Dataset drift: {dataset_drift}")
            print(f"✓ Drift share: {drift_share:.2%}")
            print(f"✓ Drifted columns: {num_drifted_columns}/{len(available_cols)}")
            if drifted_features:
                print(f"✓ Drifted features: {', '.join(drifted_features)}")
                
        except Exception as e:
            print(f"Warning: Could not extract drift results: {e}")
        
        return {
            "status": "success",
            "report_path": report_path,
            "reference_size": len(reference_data),
            "current_size": len(current_data),
            "dataset_drift": dataset_drift,
            "drift_share": round(drift_share, 4),
            "num_drifted_columns": num_drifted_columns,
            "total_columns": len(available_cols),
            "drifted_features": drifted_features,
            "drift_dict": drift_dict
        }
        
    except Exception as e:
        print(f"⚠️ Error generating drift report: {e}")
        return {
            "status": "error",
            "error_message": str(e),
            "report_path": None,
            "dataset_drift": False,
            "drift_share": 0.0
        }


def generate_drift_report_with_predictions(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    y_reference: pd.Series,
    y_current: pd.Series,
    y_pred_reference: np.ndarray,
    y_pred_current: np.ndarray,
    artifacts_dir: str,
    filename: str = "drift_report_with_predictions.html"
) -> Dict:
    """
    Enhanced drift report including target and prediction columns.
    Follows the exact pattern from test_experiment.py Experiment 4.
    """
    print("\n--- Generating Enhanced Drift Report with Predictions ---")
    
    # Prepare DataFrames with target and predictions (like your experiment)
    reference_data = reference_df.copy()
    reference_data['target'] = y_reference.values
    reference_data['prediction'] = y_pred_reference

    current_data = current_df.copy()
    current_data['target'] = y_current.values
    current_data['prediction'] = y_pred_current
    
    try:
        # Generate Report using latest API
        report = Report([
            DataDriftPreset()
        ],
        include_tests=True)
        
        # Run returns evaluation object
        my_eval = report.run(current_data, reference_data)
        
        # Save HTML using returned object
        report_path = os.path.join(artifacts_dir, filename)
        my_eval.save_html(report_path)
        
        print(f"✓ Enhanced drift report saved: {report_path}")
        
        # Get drift results
        try:
            drift_dict = my_eval.dict()
            print("✓ Drift detection with predictions completed successfully")
        except Exception as e:
            print(f"Warning: Could not extract drift results: {e}")
            drift_dict = None
        
        return {
            "status": "success",
            "report_path": report_path,
            "reference_size": len(reference_data),
            "current_size": len(current_data),
            "drift_dict": drift_dict
        }
        
    except Exception as e:
        print(f"⚠️ Error generating enhanced drift report: {e}")
        return {
            "status": "error",
            "error_message": str(e),
            "report_path": None
        }


def create_drift_summary_for_cml(drift_result: Dict) -> str:
    """
    Creates a markdown summary of drift results for CML reporting.
    """
    if drift_result.get('status') == 'error':
        return f"""
### ⚠️ Data Drift Analysis - Error

**Status:** Failed to generate drift report

**Error:** {drift_result.get('error_message', 'Unknown error')}
"""
    
    drift_status = "🚨 **DRIFT DETECTED**" if drift_result.get('dataset_drift', False) else "✅ **NO SIGNIFICANT DRIFT**"
    drift_share = drift_result.get('drift_share', 0)
    num_drifted = drift_result.get('num_drifted_columns', 0)
    total_cols = drift_result.get('total_columns', 0)
    drifted_features = drift_result.get('drifted_features', [])
    
    summary = f"""
### 📊 Data Drift Analysis Results

**Status:** {drift_status}

**Drift Share:** {drift_share:.2%} of features show drift

**Drifted Features:** {num_drifted}/{total_cols} columns
"""
    
    if drifted_features:
        features_str = ', '.join([f'`{f}`' for f in drifted_features])
        summary += f"\n**Features with drift:** {features_str}\n"
    
    summary += f"""
**Dataset Sizes:**
- Reference: {drift_result.get('reference_size', 0)} samples
- Current: {drift_result.get('current_size', 0)} samples

📄 **Full interactive HTML report available in artifacts**
"""
    
    return summary


# --- LOGGING HELPERS ---

def log_predictions_to_file(
    sample_data: Dict,
    prediction: str,
    model_version: str,
    log_path: str
):
    """Logs predictions for drift monitoring."""
    from datetime import datetime
    
    try:
        timestamp = datetime.now().isoformat()
        log_entry = {
            'timestamp': timestamp,
            'model_version': model_version,
            'prediction': prediction,
            **sample_data
        }
        
        if os.path.exists(log_path) and os.path.getsize(log_path) > 0:
            log_df = pd.read_csv(log_path)
            log_df = pd.concat([log_df, pd.DataFrame([log_entry])], ignore_index=True)
        else:
            log_df = pd.DataFrame([log_entry])
        
        log_df.to_csv(log_path, index=False)
    except Exception as e:
        print(f"Warning: Failed to log prediction: {e}")

# --- MLFLOW HELPERS ---

def log_metrics_to_mlflow(metrics: Dict, prefix: str = ""):
    """Logs dictionary of metrics to MLflow."""
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            mlflow.log_metric(f"{prefix}{key}" if prefix else key, value)

def log_artifacts_to_mlflow(artifact_paths: List[str]):
    """Logs multiple artifact files to MLflow."""
    for path in artifact_paths:
        if os.path.exists(path):
            mlflow.log_artifact(path)
