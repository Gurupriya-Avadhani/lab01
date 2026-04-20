import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging
from typing import List, Dict, Tuple
from src.train import KNNRecommendationModel

logger = logging.getLogger(__name__)


def tune_k_parameter(
    features,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    k_values: List[int] = None
) -> Tuple[int, List[Dict]]:

    if k_values is None:
        k_values = [3, 5, 10, 15, 20]

    logger.info("=" * 70)
    logger.info("HYPERPARAMETER TUNING: K-NN K Parameter")
    logger.info("=" * 70)
    logger.info(f"Tuning K values: {k_values}")
    logger.info(f"Validation set size: {len(val_df)}")

    results = []

    for k in k_values:
        logger.info(f"\n--- Testing K={k} ---")

        model = KNNRecommendationModel(k=k, use_similarity_weights=False)
        model.fit(features, train_df)

        y_true = val_df['rating'].values
        y_pred = model.predict_batch(val_df)

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)

        logger.info(f"  RMSE: {rmse:.4f}")
        logger.info(f"  MAE:  {mae:.4f}")

        results.append({
            'k': k,
            'rmse': rmse,
            'mae': mae,
            'model': model
        })

    best_result = min(results, key=lambda x: x['rmse'])
    best_k = best_result['k']

    logger.info("=" * 70)
    logger.info(f"✓ Best K={best_k} (RMSE={best_result['rmse']:.4f})")
    logger.info("=" * 70)

    return best_k, [
        {'k': r['k'], 'rmse': r['rmse'], 'mae': r['mae']}
        for r in results
    ]


def plot_tuning_results(results: List[Dict]) -> None:

    logger.info("\nK     RMSE        MAE")
    logger.info("-" * 30)

    for r in results:
        rmse_bar = "█" * int(r['rmse'] * 10)
        mae_bar = "█" * int(r['mae'] * 10)
        logger.info(f"{r['k']:2}    {r['rmse']:.4f} {mae_bar}")