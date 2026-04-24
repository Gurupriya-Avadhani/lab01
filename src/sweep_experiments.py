import mlflow
import time
import logging
import pickle
import os
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from typing import Tuple, Dict, List
from src.mlflow_tracking import (
    initialize_mlflow_experiment,
    log_model_parameters,
    log_model_metrics,
    log_model_artifact,
    log_run_tags
)

logger = logging.getLogger(__name__)


def train_and_evaluate_knn(
    k_value: int,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame
) -> Dict[str, float]:
    start_time = time.time()

    model = NearestNeighbors(
        n_neighbors=min(k_value, len(X_train)),
        metric="cosine",
        algorithm="brute"
    )
    model.fit(X_train)

    distances, indices = model.kneighbors(X_test)

    predictions = []
    for neighbor_indices in indices:
        neighbor_ratings = y_train.iloc[neighbor_indices]["rating"].values
        avg_rating = np.mean(neighbor_ratings) if len(neighbor_ratings) > 0 else 3.0
        predictions.append(avg_rating)

    predictions = np.array(predictions)
    actual = y_test["rating"].values

    rmse = float(np.sqrt(np.mean((predictions - actual) ** 2)))
    mae = float(np.mean(np.abs(predictions - actual)))
    coverage = float(np.sum(predictions >= 0.5) / len(predictions))
    training_time = float(time.time() - start_time)

    return {
        "rmse": rmse,
        "mae": mae,
        "coverage": coverage,
        "training_time": training_time,
        "model": model
    }


def run_parameter_sweep(
    k_values: List[int],
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    experiment_name: str = "movielens_knn_sweep"
) -> Dict[str, Dict]:
    initialize_mlflow_experiment(experiment_name)

    os.makedirs("models", exist_ok=True)
    results: Dict[int, Dict] = {}

    for k in k_values:
        try:
            with mlflow.start_run(run_name=f"k_{k}") as run:
                metrics = train_and_evaluate_knn(
                    k, X_train, X_test, y_train, y_test
                )

                log_model_parameters(k_value=k)
                log_model_metrics(
                    rmse=metrics["rmse"],
                    mae=metrics["mae"],
                    coverage=metrics["coverage"],
                    training_time_seconds=metrics["training_time"]
                )

                model_path = f"models/knn_k{k}.pkl"
                with open(model_path, "wb") as f:
                    pickle.dump(metrics["model"], f)

                log_model_artifact(model_path)

                run_id = run.info.run_id
                log_run_tags(run_id, {
                    "dataset": "movielens_100k",
                    "k_value": str(k),
                    "status": "completed"
                })

                results[k] = {
                    "rmse": metrics["rmse"],
                    "mae": metrics["mae"],
                    "coverage": metrics["coverage"],
                    "training_time": metrics["training_time"],
                    "run_id": run_id,
                    "model_path": model_path
                }

        except Exception as e:
            logger.error(f"K={k} failed: {e}")
            if mlflow.active_run() is not None:
                mlflow.set_tag("status", "failed")
                mlflow.end_run()

    return results


def identify_best_run(results: Dict, metric: str = "rmse") -> Tuple[int, Dict]:
    if not results:
        raise ValueError("No results available")

    if metric == "rmse":
        best_k = min(results, key=lambda k: results[k]["rmse"])
    elif metric == "mae":
        best_k = min(results, key=lambda k: results[k]["mae"])
    elif metric == "coverage":
        best_k = max(results, key=lambda k: results[k]["coverage"])
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return best_k, results[best_k]