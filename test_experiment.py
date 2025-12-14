"""
CI Experiment Tests - Evidently 0.6+ API
Runs all experiments once for CI/CD pipeline
"""

import os
import mlflow
import mlflow.sklearn
from common_functions import (
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
    log_metrics_to_mlflow,
    log_artifacts_to_mlflow,
    PARAM_GRID_FAST
)
from sklearn.metrics import accuracy_score

# Configuration
CI_EXPERIMENT = "ci-experiment"
ARTIFACTS_DIR = "ci_artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

MLFLOW_TRACKING_URI = "http://34.10.224.106:5000/"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def test_baseline_training():
    """Test: Baseline model training."""
    print("\n=== TEST: Baseline Training ===")
    
    X_train, X_test, y_train, y_test, _, _ = load_and_split_data()
    mlflow.set_experiment(CI_EXPERIMENT)
    
    with mlflow.start_run(run_name="CI_Baseline"):
        mlflow.set_tag("test_type", "baseline")
        
        best_model, metrics = train_model_with_grid_search(
            X_train, y_train, X_test, y_test, PARAM_GRID_FAST, cv=3
        )
        
        log_metrics_to_mlflow(metrics)
        
        print(f"✓ Baseline Test Accuracy: {metrics['test_accuracy']}")
        assert metrics['test_accuracy'] > 0.8, "Baseline accuracy too low"
    
    return {"status": "passed", "accuracy": metrics['test_accuracy']}

def test_label_poisoning():
    """Test: Label poisoning experiments."""
    print("\n=== TEST: Label Poisoning ===")
    
    X_train, X_test, y_train, y_test, _, _ = load_and_split_data()
    mlflow.set_experiment(CI_EXPERIMENT)
    
    poison_levels = [0, 10, 25, 50, 75]
    results = []
    
    for percent in poison_levels:
        with mlflow.start_run(run_name=f"CI_LabelPoison_{percent}pct"):
            mlflow.set_tag("poison_type", "label")
            mlflow.log_metric("poison_percent", percent)
            
            y_train_poisoned = poison_labels(y_train, percent)
            best_model, metrics = train_model_with_grid_search(
                X_train, y_train_poisoned, X_test, y_test, PARAM_GRID_FAST, cv=3
            )
            
            log_metrics_to_mlflow(metrics)
            results.append({"poison_%": percent, "accuracy": metrics['test_accuracy']})
            print(f"✓ Label Poison {percent}%: Accuracy = {metrics['test_accuracy']}")
    
    return {"status": "passed", "results": results}

def test_feature_poisoning():
    """Test: Feature poisoning experiment."""
    print("\n=== TEST: Feature Poisoning ===")
    
    X_train, X_test, y_train, y_test, _, _ = load_and_split_data()
    mlflow.set_experiment(CI_EXPERIMENT)
    
    with mlflow.start_run(run_name="CI_FeaturePoison_25pct"):
        mlflow.set_tag("poison_type", "feature")
        mlflow.log_metric("poison_percent", 25)
        
        X_train_poisoned = poison_features(X_train, 25)
        best_model, metrics = train_model_with_grid_search(
            X_train_poisoned, y_train, X_test, y_test, PARAM_GRID_FAST, cv=3
        )
        
        log_metrics_to_mlflow(metrics)
        print(f"✓ Feature Poison 25%: Accuracy = {metrics['test_accuracy']}")
    
    return {"status": "passed", "accuracy": metrics['test_accuracy']}

def test_fairness_and_shap():
    """Test: Fairness metrics and SHAP for all species."""
    print("\n=== TEST: Fairness & SHAP for All Targets ===")
    
    X_train, X_test, y_train, y_test, _, _ = load_and_split_data(add_location=True)
    mlflow.set_experiment(CI_EXPERIMENT)
    
    with mlflow.start_run(run_name="CI_Fairness_SHAP_AllTargets"):
        mlflow.set_tag("test_type", "fairness_shap")
        
        best_model, train_metrics = train_model_with_grid_search(
            X_train, y_train, X_test, y_test, PARAM_GRID_FAST, cv=3
        )
        log_metrics_to_mlflow(train_metrics)
        
        y_pred = best_model.predict(X_test)
        species_list = sorted(y_train.unique())
        
        # Class metrics
        class_metrics = calculate_class_metrics(y_test, y_pred, species_list)
        for species, metrics in class_metrics.items():
            for metric_name, value in metrics.items():
                mlflow.log_metric(f"{species}_{metric_name}", value)
            print(f"✓ {species}: P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={metrics['f1']:.3f}")
        
        # Fairness
        fairness_metrics = calculate_fairness_metrics(y_test, y_pred, X_test['location'])
        mlflow.log_metric("fairness_acc_diff", fairness_metrics["accuracy_difference"])
        print(f"✓ Fairness Accuracy Difference: {fairness_metrics['accuracy_difference']:.4f}")
        
        # SHAP plots
        shap_paths = generate_shap_plots(best_model, X_test, species_list, ARTIFACTS_DIR, "ci_")
        log_artifacts_to_mlflow(shap_paths)
        
        # Confusion matrix
        cm_path = generate_confusion_matrix(y_test, y_pred, species_list, ARTIFACTS_DIR, "ci_confusion_matrix.png")
        mlflow.log_artifact(cm_path)
        
        print(f"✓ Generated SHAP plots for all {len(species_list)} species")
    
    return {
        "status": "passed",
        "accuracy": train_metrics['test_accuracy'],
        "fairness_diff": fairness_metrics["accuracy_difference"]
    }

def test_data_drift():
    """Test: Evidently data drift detection using 0.6+ API."""
    print("\n=== TEST: Data Drift Detection (Evidently 0.6+ API) ===")
    
    _, _, _, _, train, test = load_and_split_data()
    mlflow.set_experiment(CI_EXPERIMENT)
    
    # Generate drift report
    drift_result = generate_drift_report(train, test, ARTIFACTS_DIR, "ci_drift_report.html")
    
    # Generate drift report with tests
    test_result = generate_drift_report_with_tests(train, test, ARTIFACTS_DIR, "ci_drift_tests.html")
    
    with mlflow.start_run(run_name="CI_DataDrift"):
        mlflow.log_artifact(drift_result["report_path"])
        mlflow.log_artifact(test_result["report_path"])
        mlflow.set_tag("test_type", "drift_detection")
        mlflow.set_tag("evidently_version", "0.6+")
        mlflow.log_metric("dataset_drift_detected", 1 if drift_result["drift_detected"] else 0)
        mlflow.log_metric("drift_share", drift_result["drift_share"])
        
        print(f"✓ Drift Report: {drift_result['report_path']}")
        print(f"✓ Dataset Drift Detected: {drift_result['drift_detected']}")
        print(f"✓ Drift Share: {drift_result['drift_share']:.4f}")
    
    return {
        "status": "passed",
        "drift_detected": drift_result["drift_detected"],
        "drift_share": drift_result["drift_share"]
    }

def run_all_tests():
    """Runs all CI tests sequentially."""
    print("\n" + "="*60)
    print("STARTING CI EXPERIMENT TESTS (Evidently 0.6+)")
    print("="*60)
    
    results = {}
    
    try:
        results['baseline'] = test_baseline_training()
        results['label_poisoning'] = test_label_poisoning()
        results['feature_poisoning'] = test_feature_poisoning()
        results['fairness_shap'] = test_fairness_and_shap()
        results['data_drift'] = test_data_drift()
        
        print("\n" + "="*60)
        print("✓ ALL CI TESTS PASSED")
        print("="*60)
        print(f"\nResults stored in MLflow experiment: '{CI_EXPERIMENT}'")
        print(f"Artifacts saved in: '{ARTIFACTS_DIR}/'")
        
        return {"status": "all_passed", "results": results}
    
    except Exception as e:
        print(f"\n✗ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"status": "failed", "error": str(e), "results": results}

if __name__ == "__main__":
    run_all_tests()
