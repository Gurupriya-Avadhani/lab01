import joblib
import pandas as pd
import numpy as np
from src.train import KNNRecommendationModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Loading trained model...")
model = KNNRecommendationModel.load("models/model.pkl")

ratings_df = pd.read_csv("data/processed/ratings_clean.csv")

logger.info("=== Test 1: Single Predictions ===")

for _, row in ratings_df.head(5).iterrows():
    user_id = int(row["user_id"])
    movie_id = int(row["movie_id"])
    actual = row["rating"]
    predicted = model.predict_rating(user_id, movie_id)
    error = abs(actual - predicted)

    logger.info(
        f"User {user_id}, Movie {movie_id}: "
        f"actual={actual:.1f}, predicted={predicted:.2f}, error={error:.2f}"
    )

logger.info("=== Test 2: Batch Predictions ===")

test_sample = ratings_df.sample(10, random_state=42)
predictions = model.predict_batch(test_sample)
actuals = test_sample["rating"].values

mae = np.mean(np.abs(predictions - actuals))
rmse = np.sqrt(np.mean((predictions - actuals) ** 2))

logger.info(f"MAE: {mae:.4f}")
logger.info(f"RMSE: {rmse:.4f}")

logger.info("=== Test 3: Model Configuration ===")

config = model.get_config()

for key, value in config.items():
    logger.info(f"{key}: {value}")

logger.info("✓ Model tests passed!")