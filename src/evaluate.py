# import json
# import joblib
# import pandas as pd
# import numpy as np
# from sklearn.metrics import mean_squared_error, mean_absolute_error
# import logging
# from typing import Dict

# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)


# # ============================================================================
# # LOADERS
# # ============================================================================

# def load_model(model_path: str):
#     try:
#         model = joblib.load(model_path)
#         logger.info(f"✓ Loaded model from {model_path}")
#         return model
#     except Exception as e:
#         logger.error(f"Failed to load model: {e}")
#         raise


# def load_metadata(metadata_path: str) -> Dict:
#     try:
#         with open(metadata_path, "r") as f:
#             metadata = json.load(f)
#         logger.info(f"✓ Loaded metadata from {metadata_path}")
#         return metadata
#     except Exception as e:
#         logger.error(f"Failed to load metadata: {e}")
#         raise


# # ============================================================================
# # RATING EVALUATION
# # ============================================================================

# def evaluate_rating_prediction(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
#     if len(y_true) == 0:
#         raise ValueError("Empty input arrays")

#     if len(y_true) != len(y_pred):
#         raise ValueError("y_true and y_pred must have same length")

#     y_true = np.array(y_true)
#     y_pred = np.array(y_pred)

#     rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
#     mae = float(mean_absolute_error(y_true, y_pred))

#     errors = np.abs(y_true - y_pred)

#     percentiles = np.percentile(errors, [25, 50, 75, 90, 95]) if len(errors) > 0 else [0]*5

#     logger.info("\n" + "=" * 60)
#     logger.info("RATING PREDICTION EVALUATION")
#     logger.info("=" * 60)
#     logger.info(f"Samples: {len(y_true)}")
#     logger.info(f"RMSE: {rmse:.4f}")
#     logger.info(f"MAE:  {mae:.4f}")

#     return {
#         "rmse": rmse,
#         "mae": mae,
#         "median_error": float(np.median(errors)),
#         "mean_error": float(np.mean(errors)),
#         "std_error": float(np.std(errors)),
#         "percentiles": {
#             "p25": float(percentiles[0]),
#             "p50": float(percentiles[1]),
#             "p75": float(percentiles[2]),
#             "p90": float(percentiles[3]),
#             "p95": float(percentiles[4]),
#         },
#         "n_samples": int(len(y_true))
#     }


# # ============================================================================
# # COVERAGE
# # ============================================================================

# def compute_coverage(model, test_df: pd.DataFrame, min_prediction: float = 0.5) -> Dict:
#     all_movies = test_df["movie_id"].unique()

#     recommended_movies = set()

#     # safety check
#     if not hasattr(model, "predict_rating"):
#         raise AttributeError("Model must implement predict_rating(user_id, movie_id)")

#     for movie_id in all_movies:
#         movie_data = test_df[test_df["movie_id"] == movie_id]

#         for user_id in movie_data["user_id"].unique():
#             try:
#                 pred = model.predict_rating(int(user_id), int(movie_id))
#                 if pred >= min_prediction:
#                     recommended_movies.add(int(movie_id))
#                     break
#             except Exception:
#                 continue

#     total_movies = len(all_movies)

#     coverage = len(recommended_movies) / total_movies if total_movies > 0 else 0.0

#     logger.info("\n" + "=" * 60)
#     logger.info("COVERAGE ANALYSIS")
#     logger.info("=" * 60)
#     logger.info(f"Coverage: {coverage:.2%}")

#     return {
#         "coverage_ratio": float(coverage),
#         "n_recommended": int(len(recommended_movies)),
#         "n_total": int(total_movies),
#         "threshold": float(min_prediction)
#     }


# # ============================================================================
# # SPARSITY
# # ============================================================================

# def analyze_sparsity(ratings_df: pd.DataFrame, n_movies: int) -> Dict:
#     n_users = ratings_df["user_id"].nunique()
#     n_ratings = len(ratings_df)

#     max_possible = n_users * n_movies
#     density = n_ratings / max_possible if max_possible > 0 else 0
#     sparsity = 1 - density

#     logger.info("\n" + "=" * 60)
#     logger.info("SPARSITY ANALYSIS")
#     logger.info("=" * 60)
#     logger.info(f"Density:  {density:.4%}")
#     logger.info(f"Sparsity: {sparsity:.4%}")

#     return {
#         "n_users": int(n_users),
#         "n_movies": int(n_movies),
#         "n_ratings": int(n_ratings),
#         "density": float(density),
#         "sparsity": float(sparsity)
#     }


# # ============================================================================
# # BASELINES
# # ============================================================================

# def compute_baseline_metrics(y_true: np.ndarray) -> Dict:
#     y_true = np.array(y_true)

#     mean_val = float(np.mean(y_true))
#     median_val = float(np.median(y_true))

#     mean_pred = np.full_like(y_true, mean_val, dtype=float)
#     median_pred = np.full_like(y_true, median_val, dtype=float)
#     constant_pred = np.full_like(y_true, 3.0, dtype=float)

#     mean_rmse = float(np.sqrt(mean_squared_error(y_true, mean_pred)))
#     median_rmse = float(np.sqrt(mean_squared_error(y_true, median_pred)))
#     constant_rmse = float(np.sqrt(mean_squared_error(y_true, constant_pred)))

#     best = min(mean_rmse, median_rmse, constant_rmse)

#     return {
#         "mean_rmse": mean_rmse,
#         "median_rmse": median_rmse,
#         "constant_rmse": constant_rmse,
#         "best_baseline": best
#     }


# # ============================================================================
# # ERROR DISTRIBUTION
# # ============================================================================

# def analyze_error_distribution(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
#     errors = np.abs(np.array(y_true) - np.array(y_pred))

#     bins = [(0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 10)]

#     distribution = {}

#     for low, high in bins:
#         count = np.sum((errors >= low) & (errors < high))
#         pct = count / len(errors) * 100 if len(errors) > 0 else 0

#         distribution[f"{low}-{high}"] = {
#             "count": int(count),
#             "percent": float(pct)
#         }

#     return {
#         "mean_error": float(np.mean(errors)),
#         "std_error": float(np.std(errors)),
#         "distribution": distribution
#     }


import json
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging
from typing import Dict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model(model_path: str):
    model = joblib.load(model_path)
    logger.info(f"Loaded model from {model_path}")
    return model


def load_metadata(metadata_path: str) -> Dict:
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    logger.info(f"Loaded metadata from {metadata_path}")
    return metadata


def evaluate_rating_prediction(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    if len(y_true) == 0 or len(y_true) != len(y_pred):
        raise ValueError("Invalid input arrays")

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))

    errors = np.abs(y_true - y_pred)
    percentiles = np.percentile(errors, [25, 50, 75, 90, 95]) if len(errors) > 0 else [0]*5

    return {
        "rmse": rmse,
        "mae": mae,
        "median_error": float(np.median(errors)),
        "mean_error": float(np.mean(errors)),
        "std_error": float(np.std(errors)),
        "percentiles": {
            "p25": float(percentiles[0]),
            "p50": float(percentiles[1]),
            "p75": float(percentiles[2]),
            "p90": float(percentiles[3]),
            "p95": float(percentiles[4]),
        },
        "n_samples": int(len(y_true))
    }


def compute_coverage(model, test_df: pd.DataFrame, min_prediction: float = 0.5) -> Dict:
    all_movies = test_df["movie_id"].unique()
    recommended_movies = set()

    for movie_id in all_movies:
        movie_data = test_df[test_df["movie_id"] == movie_id]
        for user_id in movie_data["user_id"].unique():
            try:
                pred = model.predict_rating(int(user_id), int(movie_id))
                if pred >= min_prediction:
                    recommended_movies.add(int(movie_id))
                    break
            except Exception:
                continue

    total_movies = len(all_movies)
    coverage = len(recommended_movies) / total_movies if total_movies > 0 else 0.0

    return {
        "coverage_ratio": float(coverage),
        "n_recommended": int(len(recommended_movies)),
        "n_total": int(total_movies),
        "threshold": float(min_prediction)
    }


def analyze_sparsity(ratings_df: pd.DataFrame, n_movies: int) -> Dict:
    n_users = ratings_df["user_id"].nunique()
    n_ratings = len(ratings_df)

    max_possible = n_users * n_movies
    density = n_ratings / max_possible if max_possible > 0 else 0
    sparsity = 1 - density

    return {
        "n_users": int(n_users),
        "n_movies": int(n_movies),
        "n_ratings": int(n_ratings),
        "density": float(density),
        "sparsity": float(sparsity)
    }


def compute_baseline_metrics(y_true: np.ndarray) -> Dict:
    y_true = np.array(y_true)

    mean_val = float(np.mean(y_true))
    median_val = float(np.median(y_true))

    mean_pred = np.full_like(y_true, mean_val, dtype=float)
    median_pred = np.full_like(y_true, median_val, dtype=float)
    constant_pred = np.full_like(y_true, 3.0, dtype=float)

    mean_rmse = float(np.sqrt(mean_squared_error(y_true, mean_pred)))
    median_rmse = float(np.sqrt(mean_squared_error(y_true, median_pred)))
    constant_rmse = float(np.sqrt(mean_squared_error(y_true, constant_pred)))

    best = min(mean_rmse, median_rmse, constant_rmse)

    return {
        "mean_rmse": mean_rmse,
        "median_rmse": median_rmse,
        "constant_rmse": constant_rmse,
        "best_baseline": best
    }


def analyze_error_distribution(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    errors = np.abs(np.array(y_true) - np.array(y_pred))

    bins = [(0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 10)]
    distribution = {}

    for low, high in bins:
        count = np.sum((errors >= low) & (errors < high))
        pct = count / len(errors) * 100 if len(errors) > 0 else 0
        distribution[f"{low}-{high}"] = {
            "count": int(count),
            "percent": float(pct)
        }

    return {
        "mean_error": float(np.mean(errors)),
        "std_error": float(np.std(errors)),
        "distribution": distribution
    }


def analyze_by_user_engagement(y_true: np.ndarray, y_pred: np.ndarray, test_df: pd.DataFrame) -> Dict:
    df = test_df.copy()
    df["error"] = np.abs(y_true - y_pred)

    user_counts = df["user_id"].value_counts()
    df["engagement"] = df["user_id"].map(user_counts)

    segments = {
        "low": df[df["engagement"] <= 5],
        "medium": df[(df["engagement"] > 5) & (df["engagement"] <= 20)],
        "high": df[df["engagement"] > 20],
    }

    results = {}

    for name, seg in segments.items():
        if len(seg) > 0:
            results[name] = {
                "n_samples": int(len(seg)),
                "mae": float(seg["error"].mean()),
                "mean_engagement": float(seg["engagement"].mean())
            }

    return results