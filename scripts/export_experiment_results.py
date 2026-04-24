import mlflow
import pandas as pd

mlflow.set_tracking_uri("http://localhost:5001")

experiment = mlflow.get_experiment_by_name("movielens_knn_sweep")
experiment_id = experiment.experiment_id

runs = mlflow.search_runs(experiment_ids=[experiment_id])

comparison_data = []

for _, run in runs.iterrows():
    comparison_data.append({
        "run_id": run["run_id"],
        "k_value": run.get("params.k_neighbors"),
        "rmse": run.get("metrics.rmse"),
        "mae": run.get("metrics.mae"),
        "coverage": run.get("metrics.coverage"),
        "training_time": run.get("metrics.training_time_seconds"),
        "status": run.get("status")
    })

df = pd.DataFrame(comparison_data)
df_sorted = df.sort_values("rmse")

df_sorted.to_csv("evaluations/experiment_comparison.csv", index=False)

print(df_sorted.to_string(index=False))

print(f"Best K: {df_sorted.iloc[0]['k_value']} with RMSE: {df_sorted.iloc[0]['rmse']:.3f}")