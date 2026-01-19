"""
Pipeline nodes for Iris model training with MLflow tracking.
"""
import os
import json
import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Any


def split_features_and_target() -> Tuple[pd.DataFrame, pd.Series, list]:
    """Load Iris dataset and split into features and target."""
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name='target')
    class_names = iris.target_names.tolist()
    return X, y, class_names


def train_test_split_node(
    X: pd.DataFrame,
    y: pd.Series,
    parameters: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into train and test sets."""
    test_size = parameters.get('test_size', 0.2)
    random_state = parameters.get('random_state', 42)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def train_single_model(
    model,
    model_name: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series
) -> Dict[str, Any]:
    """Train, evaluate, and log a single model with MLflow."""
    with mlflow.start_run(run_name=model_name, nested=True):
        # Train the model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

        # Log parameters
        mlflow.log_params(model.get_params())
        
        # Log tags
        mlflow.set_tag("version", "v1.0.0")
        mlflow.set_tag("model_type", model_name)

        # Calculate and log metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")
        precision = precision_score(y_test, y_pred, average="macro")
        recall = recall_score(y_test, y_pred, average="macro")
        
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_macro", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)

        if y_proba is not None:
            roc_auc = roc_auc_score(y_test, y_proba, multi_class="ovr")
            mlflow.log_metric("roc_auc", roc_auc)

        # Log confusion matrix artifact
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix - {model_name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        cm_path = f"confusion_matrix_{model_name}.png"
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path, "confusion_matrix")
        os.remove(cm_path)
        plt.close()

        # Log classification report artifact
        report = classification_report(y_test, y_pred)
        report_path = f"classification_report_{model_name}.txt"
        with open(report_path, "w") as f:
            f.write(report)
        mlflow.log_artifact(report_path, "classification_report")
        os.remove(report_path)

        # Log model
        mlflow.sklearn.log_model(model, "model")

        run_id = mlflow.active_run().info.run_id
        
        return {
            "model_name": model_name,
            "f1_score": f1,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "run_id": run_id
        }


def train_knn(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    parameters: Dict[str, Any]
) -> Dict[str, Any]:
    """Train KNN model."""
    model = KNeighborsClassifier(**parameters)
    return train_single_model(model, "KNN", X_train, X_test, y_train, y_test)


def train_logreg(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    parameters: Dict[str, Any]
) -> Dict[str, Any]:
    """Train Logistic Regression model."""
    model = LogisticRegression(**parameters)
    return train_single_model(model, "LogisticRegression", X_train, X_test, y_train, y_test)


def train_rf(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    parameters: Dict[str, Any]
) -> Dict[str, Any]:
    """Train Random Forest model."""
    model = RandomForestClassifier(**parameters)
    return train_single_model(model, "RandomForest", X_train, X_test, y_train, y_test)


def train_svm(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    parameters: Dict[str, Any]
) -> Dict[str, Any]:
    """Train SVM model."""
    model = SVC(**parameters)
    return train_single_model(model, "SVM", X_train, X_test, y_train, y_test)


def select_best_model(
    knn_result: Dict[str, Any],
    logreg_result: Dict[str, Any],
    rf_result: Dict[str, Any],
    svm_result: Dict[str, Any]
) -> Tuple[str, str]:
    """Select the best model based on F1 score."""
    results = [knn_result, logreg_result, rf_result, svm_result]
    best_result = max(results, key=lambda x: x['f1_score'])
    
    print(f"\n{'='*50}")
    print(f"Best model: {best_result['model_name']}")
    print(f"F1 Score: {best_result['f1_score']:.4f}")
    print(f"Accuracy: {best_result['accuracy']:.4f}")
    print(f"{'='*50}\n")
    
    return best_result['model_name'], best_result['run_id']


def save_best_model_locally(best_run_id: str) -> None:
    """Save the best model to the app directory."""
    best_model_uri = f"runs:/{best_run_id}/model"
    best_model = mlflow.sklearn.load_model(best_model_uri)
    
    # Save to parent directory's app folder
    output_path = "../app/model.joblib"
    joblib.dump(best_model, output_path)
    print(f"Model saved to {output_path}")


def save_model_metadata(
    best_model_name: str,
    best_run_id: str
) -> str:
    """Save model metadata to JSON file."""
    client = mlflow.tracking.MlflowClient()
    run_details = client.get_run(best_run_id)
    metrics = run_details.data.metrics
    
    model_meta = {
        "best_model": best_model_name,
        "metrics": {
            "accuracy": metrics.get("accuracy"),
            "f1_macro": metrics.get("f1_macro")
        },
        "mlflow_run_id": best_run_id,
        "version": "v1.0.0"
    }

    output_path = "../app/model_meta.json"
    with open(output_path, "w") as f:
        json.dump(model_meta, f, indent=4)
    
    print(f"Model metadata saved to {output_path}")
    
    # Register the best model
    model_name = "IrisModel"
    best_model_uri = f"runs:/{best_run_id}/model"
    mlflow.register_model(best_model_uri, model_name)
    
    return output_path
