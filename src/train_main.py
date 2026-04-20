import argparse
import json
import joblib
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from src.train import KNNRecommendationModel
from src.tune_hyperparameters import tune_k_parameter
from sklearn.metrics import mean_squared_error, mean_absolute_error

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main(args):
    logger.info("=" * 70)
    logger.info("MODEL TRAINING PIPELINE: K-NN Recommendation Model")
    logger.info("=" * 70)

    try:
        logger.info(f"\nStep 1: Loading feature store from {args.features_path}")
        features = joblib.load(args.features_path)
        logger.info(f"  Users: {len(features.user_ids)}")
        logger.info(f"  Movies: {len(features.movie_ids)}")

        logger.info(f"\nStep 2: Loading ratings from {args.ratings_path}")
        ratings_df = pd.read_csv(args.ratings_path)
        logger.info(f"  Loaded {len(ratings_df)} ratings")

        logger.info("\nStep 3: Splitting into train/validation (80/20)")
        n = len(ratings_df)
        np.random.seed(42)
        train_idx = np.random.choice(n, int(0.8 * n), replace=False)
        val_idx = np.setdiff1d(np.arange(n), train_idx)

        train_df = ratings_df.iloc[train_idx].reset_index(drop=True)
        val_df = ratings_df.iloc[val_idx].reset_index(drop=True)

        logger.info(f"  Train set: {len(train_df)} samples")
        logger.info(f"  Val set: {len(val_df)} samples")

        if args.tune:
            logger.info("\nStep 4: Tuning K parameter")
            best_k, tuning_results = tune_k_parameter(
                features,
                train_df,
                val_df,
                k_values=args.k_values
            )
            k_to_use = best_k

            tuning_log = {
                "tuning_results": tuning_results,
                "best_k": int(best_k)
            }

            Path(args.model_dir).mkdir(parents=True, exist_ok=True)
            with open(f"{args.model_dir}/tuning_results.json", "w") as f:
                json.dump(tuning_log, f, indent=2)

        else:
            logger.info(f"\nStep 4: Using K={args.k}")
            k_to_use = args.k

        logger.info(f"\nStep 5: Training final model (K={k_to_use})")

        model = KNNRecommendationModel(
            k=int(k_to_use),
            use_similarity_weights=args.use_weights
        )

        model.fit(features, train_df)

        logger.info("\nStep 6: Evaluating on validation set")

        y_true = val_df["rating"].values
        y_pred = model.predict_batch(val_df)

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)

        logger.info(f"  RMSE: {rmse:.4f}")
        logger.info(f"  MAE: {mae:.4f}")

        logger.info("\nStep 7: Saving model and metadata")

        Path(args.model_dir).mkdir(parents=True, exist_ok=True)

        model_path = f"{args.model_dir}/model.pkl"
        model.save(model_path)

        metadata = {
            "model_type": "KNNRecommendation",
            "hyperparameters": {
                "k": int(k_to_use),
                "use_similarity_weights": args.use_weights,
                "default_rating": model.default_rating
            },
            "training": {
                "n_train_samples": int(len(train_df)),
                "n_val_samples": int(len(val_df)),
                "split_ratio": 0.8
            },
            "evaluation": {
                "rmse": float(rmse),
                "mae": float(mae)
            },
            "features_path": args.features_path,
            "ratings_path": args.ratings_path
        }

        with open(f"{args.model_dir}/metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info("\n" + "=" * 70)
        logger.info("✓ TRAINING COMPLETE")
        logger.info("=" * 70)

        return model, metadata

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_path", default="models/rating_features.pkl")
    parser.add_argument("--ratings_path", default="data/processed/ratings_clean.csv")
    parser.add_argument("--model_dir", default="models")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--k_values", type=int, nargs="+", default=[3, 5, 10, 15, 20])
    parser.add_argument("--use_weights", action="store_true")

    args = parser.parse_args()
    main(args)



# import argparse
# import json
# import joblib
# import pandas as pd
# import numpy as np
# import logging
# from pathlib import Path
# from sklearn.metrics import mean_squared_error, mean_absolute_error
# from src.train import KNNRecommendationModel
# from src.tune_hyperparameters import tune_k_parameter

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s"
# )

# logger = logging.getLogger(__name__)


# def main(args):
#     logger.info("=" * 70)
#     logger.info("MODEL TRAINING PIPELINE: K-NN Recommendation Model")
#     logger.info("=" * 70)

#     try:
#         features = joblib.load(args.features_path)

#         logger.info(f"\nStep 1: Loaded features")
#         logger.info(f"Users: {len(features.user_ids)}")
#         logger.info(f"Movies: {len(features.movie_ids)}")

#         ratings_df = pd.read_csv(args.ratings_path)

#         logger.info(f"\nStep 2: Loaded ratings: {len(ratings_df)}")

#         n = len(ratings_df)
#         np.random.seed(42)

#         train_idx = np.random.choice(n, int(0.8 * n), replace=False)
#         val_idx = np.setdiff1d(np.arange(n), train_idx)

#         train_df = ratings_df.iloc[train_idx].reset_index(drop=True)
#         val_df = ratings_df.iloc[val_idx].reset_index(drop=True)

#         logger.info(f"\nTrain size: {len(train_df)}")
#         logger.info(f"Val size: {len(val_df)}")

#         if args.tune:
#             best_k, tuning_results = tune_k_parameter(
#                 features,
#                 train_df,
#                 val_df,
#                 k_values=args.k_values
#             )
#             k_to_use = best_k

#             Path(args.model_dir).mkdir(parents=True, exist_ok=True)

#             with open(f"{args.model_dir}/tuning_results.json", "w") as f:
#                 json.dump(
#                     {"best_k": int(best_k), "results": tuning_results},
#                     f,
#                     indent=2
#                 )
#         else:
#             k_to_use = args.k

#         logger.info(f"\nTraining final model with K={k_to_use}")

#         model = KNNRecommendationModel(
#             k=int(k_to_use),
#             use_similarity_weights=args.use_weights
#         )

#         model.fit(features, train_df)

#         y_true = val_df["rating"].values
#         y_pred = model.predict_batch(val_df)

#         y_pred = np.clip(y_pred, 0.5, 5.0)

#         rmse = np.sqrt(mean_squared_error(y_true, y_pred))
#         mae = mean_absolute_error(y_true, y_pred)

#         logger.info(f"RMSE: {rmse:.4f}")
#         logger.info(f"MAE: {mae:.4f}")

#         Path(args.model_dir).mkdir(parents=True, exist_ok=True)

#         model.save(f"{args.model_dir}/model.pkl")

#         metadata = {
#             "model_type": "KNNRecommendation",
#             "k": int(k_to_use),
#             "use_similarity_weights": args.use_weights,
#             "rmse": float(rmse),
#             "mae": float(mae),
#             "train_size": int(len(train_df)),
#             "val_size": int(len(val_df)),
#             "features_path": args.features_path,
#             "ratings_path": args.ratings_path
#         }

#         with open(f"{args.model_dir}/metadata.json", "w") as f:
#             json.dump(metadata, f, indent=2)

#         logger.info("\nTRAINING COMPLETE")
#         logger.info("=" * 70)

#         return model, metadata

#     except Exception as e:
#         logger.error(str(e))
#         raise


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()

#     parser.add_argument("--features_path", default="models/rating_features.pkl")
#     parser.add_argument("--ratings_path", default="data/processed/ratings_clean.csv")
#     parser.add_argument("--model_dir", default="models")
#     parser.add_argument("--k", type=int, default=5)
#     parser.add_argument("--tune", action="store_true")
#     parser.add_argument("--k_values", type=int, nargs="+", default=[3, 5, 10, 15, 20])
#     parser.add_argument("--use_weights", action="store_true")

#     args = parser.parse_args()
#     main(args)