import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support, 
    confusion_matrix
)
import shap
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from fairlearn.metrics import MetricFrame, selection_rate

# CORRECTED Evidently imports
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently import ColumnMapping

def log_confusion_matrix(y_true, y_pred, class_names, save_path="confusion_matrix.png"):
    """
    Generate and save confusion matrix plot
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Confusion matrix saved to {save_path}")
    return save_path


def log_shap_plots(model, X_test, feature_names, class_names, save_dir="shap_plots"):
    """
    Generate SHAP plots for model explainability
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    saved_plots = []
    
    # For multiclass, shap_values is a list (one array per class)
    for idx, class_name in enumerate(class_names):
        class_shap_values = shap_values[idx] if isinstance(shap_values, list) else shap_values
        
        # Waterfall plot (single prediction)
        try:
            waterfall_path = os.path.join(save_dir, f"shap_waterfall_{class_name}.png")
            shap.plots.waterfall(
                shap.Explanation(
                    values=class_shap_values[0],
                    base_values=explainer.expected_value[idx] if isinstance(explainer.expected_value, list) else explainer.expected_value,
                    data=X_test[0],
                    feature_names=feature_names
                ),
                show=False
            )
            plt.savefig(waterfall_path, dpi=300, bbox_inches='tight')
            plt.close()
            saved_plots.append(waterfall_path)
            print(f"✓ SHAP waterfall plot saved for {class_name}")
        except Exception as e:
            print(f"⚠ Failed to create waterfall plot for {class_name}: {e}")
        
        # Beeswarm plot (global feature importance)
        try:
            beeswarm_path = os.path.join(save_dir, f"shap_beeswarm_{class_name}.png")
            shap.plots.beeswarm(
                shap.Explanation(
                    values=class_shap_values,
                    base_values=explainer.expected_value[idx] if isinstance(explainer.expected_value, list) else explainer.expected_value,
                    data=X_test,
                    feature_names=feature_names
                ),
                show=False
            )
            plt.savefig(beeswarm_path, dpi=300, bbox_inches='tight')
            plt.close()
            saved_plots.append(beeswarm_path)
            print(f"✓ SHAP beeswarm plot saved for {class_name}")
        except Exception as e:
            print(f"⚠ Failed to create beeswarm plot for {class_name}: {e}")
        
        # Bar plot (mean absolute SHAP values)
        try:
            bar_path = os.path.join(save_dir, f"shap_bar_{class_name}.png")
            shap.plots.bar(
                shap.Explanation(
                    values=class_shap_values,
                    base_values=explainer.expected_value[idx] if isinstance(explainer.expected_value, list) else explainer.expected_value,
                    data=X_test,
                    feature_names=feature_names
                ),
                show=False
            )
            plt.savefig(bar_path, dpi=300, bbox_inches='tight')
            plt.close()
            saved_plots.append(bar_path)
            print(f"✓ SHAP bar plot saved for {class_name}")
        except Exception as e:
            print(f"⚠ Failed to create bar plot for {class_name}: {e}")
    
    return saved_plots


def compute_data_drift(reference_data, current_data, feature_columns, save_path="drift_report.html"):
    """
    Compute data drift between reference and current datasets using Evidently
    """
    # Prepare column mapping
    column_mapping = ColumnMapping()
    column_mapping.numerical_features = feature_columns
    
    # Create drift report
    drift_report = Report(metrics=[
        DataDriftPreset()
    ])
    
    # Run the report
    drift_report.run(
        reference_data=reference_data[feature_columns],
        current_data=current_data[feature_columns],
        column_mapping=column_mapping
    )
    
    # Save report
    drift_report.save_html(save_path)
    print(f"✓ Data drift report saved to {save_path}")
    
    return save_path


def compute_fairness_metrics(y_true, y_pred, sensitive_features):
    """
    Compute fairness metrics using Fairlearn
    """
    # Define metrics
    metric_frame = MetricFrame(
        metrics=accuracy_score,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features
    )
    
    # Get accuracy by group
    by_group = metric_frame.by_group
    
    # Calculate fairness metrics
    fairness_diff = by_group.max() - by_group.min()
    
    print(f"\n=== Fairness Analysis ===")
    print(f"Accuracy by group:\n{by_group}")
    print(f"Fairness Accuracy Difference: {fairness_diff:.4f}")
    
    return {
        "accuracy_by_group": by_group.to_dict(),
        "fairness_difference": float(fairness_diff)
    }


def log_metrics_to_mlflow(y_true, y_pred, class_names):
    """
    Log classification metrics to MLflow
    """
    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    mlflow.log_metric("accuracy", accuracy)
    print(f"Overall Accuracy: {accuracy:.4f}")
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=range(len(class_names))
    )
    
    for idx, class_name in enumerate(class_names):
        mlflow.log_metric(f"precision_{class_name}", precision[idx])
        mlflow.log_metric(f"recall_{class_name}", recall[idx])
        mlflow.log_metric(f"f1_{class_name}", f1[idx])
        
        print(f"{class_name}: P={precision[idx]:.4f}, R={recall[idx]:.4f}, F1={f1[idx]:.4f}")
    
    # Macro and weighted averages
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro'
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )
    
    mlflow.log_metric("precision_macro", precision_macro)
    mlflow.log_metric("recall_macro", recall_macro)
    mlflow.log_metric("f1_macro", f1_macro)
    mlflow.log_metric("precision_weighted", precision_weighted)
    mlflow.log_metric("recall_weighted", recall_weighted)
    mlflow.log_metric("f1_weighted", f1_weighted)
    
    return {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro
    }


def save_model_artifact(model, artifact_path="model.pkl"):
    """
    Save model as pickle file
    """
    with open(artifact_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"✓ Model saved to {artifact_path}")
    return artifact_path
