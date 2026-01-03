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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Set MLflow experiment
mlflow.set_experiment("iris-model-zoo")

def train_and_evaluate(model, model_name, X_train, X_test, y_train, y_test):
    """Train, evaluate, and log a model."""
    with mlflow.start_run(run_name=model_name):
        # Train the model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

        # Log parameters
        mlflow.log_params(model.get_params())

        # Log metrics
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

        # Log artifacts
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        cm_path = f"confusion_matrix_{model_name}.png"
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path, "confusion_matrix")
        os.remove(cm_path)

        # Classification Report
        report = classification_report(y_test, y_pred)
        report_path = f"classification_report_{model_name}.txt"
        with open(report_path, "w") as f:
            f.write(report)
        mlflow.log_artifact(report_path, "classification_report")
        os.remove(report_path)

        # Log model
        mlflow.sklearn.log_model(model, "model")

        return f1, mlflow.active_run().info.run_id

def main():
    # Load data
    iris = load_iris()
    X = iris.data
    y = iris.target
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Models to train
    models = {
        "RandomForest": RandomForestClassifier(random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=200),
        "SVM": SVC(probability=True, random_state=42),
        "KNN": KNeighborsClassifier()
    }

    best_f1 = -1
    best_model_name = ""
    best_model_run_id = ""

    for name, model in models.items():
        print(f"Training {name}...")
        f1, run_id = train_and_evaluate(model, name, X_train, X_test, y_train, y_test)
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_model_run_id = run_id

    print(f"Best model: {best_model_name} with F1-score: {best_f1}")

    # Save the best model locally
    best_model_uri = f"runs:/{best_model_run_id}/model"
    best_model = mlflow.sklearn.load_model(best_model_uri)
    joblib.dump(best_model, "app/model.joblib")

    # Save model metadata
    client = mlflow.tracking.MlflowClient()
    run_details = client.get_run(best_model_run_id)
    metrics = run_details.data.metrics
    
    model_meta = {
        "best_model": best_model_name,
        "metrics": {
            "accuracy": metrics.get("accuracy"),
            "f1_macro": metrics.get("f1_macro")
        },
        "mlflow_run_id": best_model_run_id,
        "version": "v1.0.0"
    }

    with open("app/model_meta.json", "w") as f:
        json.dump(model_meta, f, indent=4)

    # Register the best model
    model_name = "IrisModel"
    mlflow.register_model(best_model_uri, model_name)


if __name__ == "__main__":
    main()
