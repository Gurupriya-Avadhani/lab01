import pandas as pd
import sys
import logging
from pathlib import Path
from src.features import RatingFeatures

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def validate_ratings_data(ratings_df: pd.DataFrame) -> bool:

    required_cols = {'user_id', 'movie_id', 'rating', 'timestamp'}

    if not required_cols.issubset(ratings_df.columns):
        raise ValueError(f"Missing columns. Expected {required_cols}")

    if ratings_df.empty:
        raise ValueError("Ratings DataFrame is empty")

    invalid_ratings = ratings_df[
        (ratings_df['rating'] < 0.5) | (ratings_df['rating'] > 5.5)
    ]

    if not invalid_ratings.empty:
        logger.warning(f"Found {len(invalid_ratings)} ratings outside [0.5, 5.5] range")

    if ratings_df.isnull().any().any():
        missing_cols = ratings_df.columns[ratings_df.isnull().any()].tolist()
        raise ValueError(f"Found NaN values in columns: {missing_cols}")

    logger.info(f"✓ Validation passed: {len(ratings_df)} ratings from "
                f"{ratings_df['user_id'].nunique()} users on "
                f"{ratings_df['movie_id'].nunique()} movies")

    return True

def prepare_features(ratings_path: str, output_dir: str = 'models') -> RatingFeatures:

    logger.info("=" * 70)
    logger.info("FEATURE ENGINEERING PIPELINE: User Similarity Matrix")
    logger.info("=" * 70)

    try:
        logger.info(f"\nStep 1: Loading ratings from {ratings_path}")
        ratings_df = pd.read_csv(ratings_path)
        logger.info(f"  Loaded {len(ratings_df)} ratings")

        logger.info(f"\nStep 2: Validating data structure")
        validate_ratings_data(ratings_df)

        logger.info(f"\nStep 3: Computing user similarity features")
        features = RatingFeatures()
        features.fit(ratings_df)

        logger.info(f"\nStep 4: Computing feature statistics")
        stats = features.get_movie_rating_stats()

        logger.info(f"  Movie rating stats:")
        logger.info(f"    Mean:  {stats['mean_rating']:.2f}")
        logger.info(f"    Std:   {stats['std_rating']:.2f}")
        logger.info(f"    Range: [{stats['min_rating']:.1f}, {stats['max_rating']:.1f}]")
        logger.info(f"    Total: {stats['n_rated']} rated (user, movie) pairs")

        logger.info(f"\nStep 5: Saving feature store")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        features.save(f'{output_dir}/rating_features.pkl')

        logger.info(f"\nStep 6: Testing feature lookups")

        for test_user_id in features.user_ids[:3]:
            similar = features.get_similar_users(user_id=int(test_user_id), n=3)

            if similar:
                sim_str = ", ".join([f"User {u} ({s:.3f})" for u, s in similar])
                logger.info(f"  User {test_user_id} similar to: {sim_str}")

        logger.info("\n" + "=" * 70)
        logger.info("✓ FEATURE ENGINEERING COMPLETE")
        logger.info("=" * 70)

        return features

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == '__main__':

    ratings_path = sys.argv[1] if len(sys.argv) > 1 else 'data/ratings_clean.csv'
    output_dir = sys.argv[2] if len(sys.argv) > 2 else 'models'

    try:
        prepare_features(ratings_path, output_dir)
    except Exception as e:
        logger.critical(f"Fatal error: {str(e)}")
        sys.exit(1)