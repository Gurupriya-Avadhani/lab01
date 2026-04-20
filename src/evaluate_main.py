# import argparse
# import json
# import pandas as pd
# import logging
# from pathlib import Path

# from src.evaluate import (
#     load_model,
#     load_metadata,
#     evaluate_rating_prediction,
#     compute_coverage,
#     analyze_sparsity,
#     analyze_error_distribution,
#     compute_baseline_metrics,
#     analyze_by_user_engagement
# )

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s"
# )

# logger = logging.getLogger(__name__)


# def main(args):
#     logger.info("=" * 70)
#     logger.info("MODEL EVALUATION PIPELINE")
#     logger.info("=" * 70)

#     try:
#         model = load_model(args.model_path)
#         metadata = load_metadata(args.metadata_path)

#         logger.info(f"Model loaded (K={metadata.get('hyperparameters', {}).get('k')})")

#         test_df = pd.read_csv(args.test_path)
#         ratings_df = pd.read_csv(args.ratings_path)

#         logger.info(f"Test samples: {len(test_df)}")
#         logger.info(f"Total ratings: {len(ratings_df)}")

#         y_true = test_df["rating"].to_numpy()

#         if hasattr(model, "predict_batch"):
#             y_pred = model.predict_batch(test_df)
#         else:
#             y_pred = [
#                 model.predict_rating(int(u), int(m))
#                 for u, m in zip(test_df["user_id"], test_df["movie_id"])
#             ]

#         y_pred = pd.Series(y_pred).fillna(0).to_numpy()

#         rating_metrics = evaluate_rating_prediction(y_true, y_pred)
#         coverage_metrics = compute_coverage(model, test_df)
#         sparsity_metrics = analyze_sparsity(ratings_df, args.n_movies)
#         error_metrics = analyze_error_distribution(y_true, y_pred)
#         baseline_metrics = compute_baseline_metrics(y_true)
#         segment_metrics = analyze_by_user_engagement(y_true, y_pred, test_df)

#         report = {
#             "metadata": metadata,
#             "test_set": {
#                 "n_samples": int(len(test_df))
#             },
#             "rating_prediction": rating_metrics,
#             "coverage": coverage_metrics,
#             "sparsity": sparsity_metrics,
#             "error_distribution": error_metrics,
#             "baselines": baseline_metrics,
#             "by_engagement": segment_metrics
#         }

#         Path(args.eval_dir).mkdir(parents=True, exist_ok=True)

#         output_path = Path(args.eval_dir) / "evaluation_report.json"

#         with open(output_path, "w") as f:
#             json.dump(report, f, indent=2)

#         logger.info(f"Saved report to {output_path}")

#         logger.info("=" * 70)
#         logger.info("SUMMARY")
#         logger.info("=" * 70)

#         logger.info(f"RMSE: {rating_metrics['rmse']:.4f}")
#         logger.info(f"MAE: {rating_metrics['mae']:.4f}")
#         logger.info(f"Coverage: {coverage_metrics['coverage_ratio']:.2%}")
#         logger.info(f"Sparsity: {sparsity_metrics['sparsity']:.2%}")
#         logger.info(f"Baseline RMSE: {baseline_metrics['best_baseline']:.4f}")

#         if rating_metrics["rmse"] < baseline_metrics["best_baseline"]:
#             logger.info("Model beats baseline")
#         else:
#             logger.warning("Model does NOT beat baseline")

#         return report

#     except Exception as e:
#         logger.error(f"Evaluation failed: {e}")
#         raise


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()

#     parser.add_argument("--model_path", default="models/model.pkl")
#     parser.add_argument("--metadata_path", default="models/metadata.json")
#     parser.add_argument("--test_path", default="data/processed/ratings_clean.csv")
#     parser.add_argument("--ratings_path", default="data/processed/ratings_clean.csv")
#     parser.add_argument("--n_movies", type=int, default=100)
#     parser.add_argument("--eval_dir", default="evaluations")

#     args = parser.parse_args()
#     main(args)


import argparse
import json
import pandas as pd
import logging
from pathlib import Path

from src.evaluate import (
    load_model,
    load_metadata,
    evaluate_rating_prediction,
    compute_coverage,
    analyze_sparsity,
    analyze_error_distribution,
    compute_baseline_metrics,
    analyze_by_user_engagement
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def main(args):
    try:
        model = load_model(args.model_path)
        metadata = load_metadata(args.metadata_path)

        test_df = pd.read_csv(args.test_path)
        ratings_df = pd.read_csv(args.ratings_path)

        y_true = test_df["rating"].to_numpy()

        if hasattr(model, "predict_batch"):
            y_pred = model.predict_batch(test_df)
        else:
            y_pred = [
                model.predict_rating(int(u), int(m))
                for u, m in zip(test_df["user_id"], test_df["movie_id"])
            ]

        y_pred = pd.Series(y_pred).fillna(0).to_numpy()

        rating_metrics = evaluate_rating_prediction(y_true, y_pred)
        coverage_metrics = compute_coverage(model, test_df)
        sparsity_metrics = analyze_sparsity(ratings_df, args.n_movies)
        error_metrics = analyze_error_distribution(y_true, y_pred)
        baseline_metrics = compute_baseline_metrics(y_true)
        segment_metrics = analyze_by_user_engagement(y_true, y_pred, test_df)

        report = {
            "metadata": metadata,
            "test_set": {"n_samples": int(len(test_df))},
            "rating_prediction": rating_metrics,
            "coverage": coverage_metrics,
            "sparsity": sparsity_metrics,
            "error_distribution": error_metrics,
            "baselines": baseline_metrics,
            "by_engagement": segment_metrics
        }

        Path(args.eval_dir).mkdir(parents=True, exist_ok=True)
        output_path = Path(args.eval_dir) / "evaluation_report.json"

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"RMSE: {rating_metrics['rmse']:.4f}")
        logger.info(f"MAE: {rating_metrics['mae']:.4f}")
        logger.info(f"Coverage: {coverage_metrics['coverage_ratio']:.2%}")
        logger.info(f"Sparsity: {sparsity_metrics['sparsity']:.2%}")
        logger.info(f"Baseline RMSE: {baseline_metrics['best_baseline']:.4f}")

        if rating_metrics["rmse"] < baseline_metrics["best_baseline"]:
            logger.info("Model beats baseline")
        else:
            logger.warning("Model does NOT beat baseline")

        return report

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="models/model.pkl")
    parser.add_argument("--metadata_path", default="models/metadata.json")
    parser.add_argument("--test_path", default="data/processed/ratings_clean.csv")
    parser.add_argument("--ratings_path", default="data/processed/ratings_clean.csv")
    parser.add_argument("--n_movies", type=int, default=100)
    parser.add_argument("--eval_dir", default="evaluations")

    args = parser.parse_args()
    main(args)