import mlflow
import logging
from typing import Dict

logger = logging.getLogger(__name__)


def initialize_mlflow_experiment(
    experiment_name: str,
    tracking_uri: str = "http://localhost:5001"
) -> str:
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        exp_id = mlflow.create_experiment(experiment_name)
        logger.info(f"Created experiment: {experiment_name} (ID: {exp_id})")
    else:
        exp_id = experiment.experiment_id
        logger.info(f"Using existing experiment: {experiment_name} (ID: {exp_id})")

    mlflow.set_experiment(experiment_name)
    return exp_id


def _ensure_active_run():
    if mlflow.active_run() is None:
        mlflow.start_run()


def log_model_parameters(
    k_value: int,
    test_size: float = 0.2,
    random_state: int = 42
) -> None:
    _ensure_active_run()
    mlflow.log_param("k_neighbors", k_value)
    mlflow.log_param("test_size", test_size)
    mlflow.log_param("random_state", random_state)
    mlflow.log_param("algorithm", "k_neighbors")
    logger.info(f"Logged parameters: k={k_value}, test_size={test_size}")


def log_model_metrics(
    rmse: float,
    mae: float,
    coverage: float,
    training_time_seconds: float
) -> None:
    _ensure_active_run()
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("coverage", coverage)
    mlflow.log_metric("training_time_seconds", training_time_seconds)
    logger.info(f"Logged metrics: RMSE={rmse:.3f}, MAE={mae:.3f}, Coverage={coverage:.1%}")


def log_model_artifact(
    model_path: str,
    artifact_name: str = "model"
) -> None:
    _ensure_active_run()
    mlflow.log_artifact(model_path, artifact_path=artifact_name)
    logger.info(f"Logged artifact: {model_path}")


def get_mlflow_client():
    return mlflow.tracking.MlflowClient()


def log_run_tags(run_id: str, tags: Dict[str, str]) -> None:
    client = get_mlflow_client()
    for key, value in tags.items():
        client.set_tag(run_id, key, str(value))
    logger.info(f"Tagged run {run_id}: {tags}")