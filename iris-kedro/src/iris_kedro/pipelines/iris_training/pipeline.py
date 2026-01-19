"""
Iris training pipeline definition.
"""
from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    split_features_and_target,
    train_test_split_node,
    train_knn,
    train_logreg,
    train_rf,
    train_svm,
    select_best_model,
    save_best_model_locally,
    save_model_metadata
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create the iris training pipeline."""
    return pipeline(
        [
            # Load and split data
            node(
                func=split_features_and_target,
                inputs=None,
                outputs=["X", "y", "class_names"],
                name="split_features_and_target",
            ),
            node(
                func=train_test_split_node,
                inputs=["X", "y", "params:train_test_split"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="train_test_split_node",
            ),
            # Train individual models
            node(
                func=train_knn,
                inputs=["X_train", "X_test", "y_train", "y_test", "params:knn_model"],
                outputs="knn_result",
                name="train_knn",
            ),
            node(
                func=train_logreg,
                inputs=["X_train", "X_test", "y_train", "y_test", "params:logreg_model"],
                outputs="logreg_result",
                name="train_logreg",
            ),
            node(
                func=train_rf,
                inputs=["X_train", "X_test", "y_train", "y_test", "params:rf_model"],
                outputs="rf_result",
                name="train_rf",
            ),
            node(
                func=train_svm,
                inputs=["X_train", "X_test", "y_train", "y_test", "params:svm_model"],
                outputs="svm_result",
                name="train_svm",
            ),
            # Select best model
            node(
                func=select_best_model,
                inputs=["knn_result", "logreg_result", "rf_result", "svm_result"],
                outputs=["best_model_name", "best_run_id"],
                name="select_best_model",
            ),
            # Save best model and metadata
            node(
                func=save_best_model_locally,
                inputs="best_run_id",
                outputs=None,
                name="save_best_model_locally",
            ),
            node(
                func=save_model_metadata,
                inputs=["best_model_name", "best_run_id"],
                outputs="model_metadata_path",
                name="save_model_metadata",
            ),
        ]
    )
