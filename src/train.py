import argparse
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging
from typing import Tuple, Optional
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


class KNNRecommendationModel:
    def __init__(self, k: int = 5, use_similarity_weights: bool = False):
        self.k = k
        self.use_similarity_weights = use_similarity_weights
        self.features = None
        self.ratings_df = None
        self.default_rating = 3.0
        self.fitted = False

    def fit(self, features, ratings_df: pd.DataFrame) -> 'KNNRecommendationModel':
        if not hasattr(features, 'fitted') or not features.fitted:
            raise ValueError("features must be fitted RatingFeatures object")

        if ratings_df.empty:
            raise ValueError("ratings_df cannot be empty")

        required_cols = {'user_id', 'movie_id', 'rating'}
        if not required_cols.issubset(ratings_df.columns):
            missing = required_cols - set(ratings_df.columns)
            raise ValueError(f"Missing columns: {missing}")

        self.features = features
        self.ratings_df = ratings_df
        self.fitted = True

        logger.info("✓ Fitted KNN model")
        logger.info(f"  K: {self.k}")
        logger.info(f"  Similarity weights: {self.use_similarity_weights}")
        logger.info(f"  Training samples: {len(ratings_df)}")

        return self

    def predict_rating(self, user_id: int, movie_id: int) -> float:
        if not self.fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        try:
            similar_users = self.features.get_similar_users(user_id, n=self.k)

            if not similar_users:
                return self.default_rating

            similar_user_ids = [u for u, s in similar_users]
            similarities = np.array([s for u, s in similar_users])

            ratings_for_movie = self.ratings_df[
                (self.ratings_df['user_id'].isin(similar_user_ids))
                & (self.ratings_df['movie_id'] == movie_id)
            ]

            if ratings_for_movie.empty:
                return self.default_rating

            ratings_list = []
            similarity_list = []

            for neighbor_id, sim_score in similar_users:
                neighbor_rating = ratings_for_movie[
                    ratings_for_movie['user_id'] == neighbor_id
                ]['rating'].values

                if len(neighbor_rating) > 0:
                    ratings_list.append(neighbor_rating[0])
                    similarity_list.append(sim_score)

            if len(ratings_list) == 0:
                return self.default_rating

            if self.use_similarity_weights:
                similarity_list = np.array(similarity_list)
                weights = similarity_list / np.sum(similarity_list)
                prediction = np.sum(np.array(ratings_list) * weights)
            else:
                prediction = np.mean(ratings_list)

            prediction = np.clip(prediction, 0.5, 5.0)

            return float(prediction)

        except Exception as e:
            logger.error(f"Prediction error for user {user_id}, movie {movie_id}: {e}")
            return self.default_rating

    def predict_batch(self, test_df: pd.DataFrame) -> np.ndarray:
        predictions = np.zeros(len(test_df))

        for i, (_, row) in enumerate(test_df.iterrows()):
            predictions[i] = self.predict_rating(
                int(row['user_id']),
                int(row['movie_id'])
            )

        return predictions

    def get_config(self) -> dict:
        return {
            'model_type': 'KNNRecommendation',
            'k': self.k,
            'use_similarity_weights': self.use_similarity_weights,
            'default_rating': self.default_rating
        }

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        joblib.dump(self, filepath)
        logger.info(f"✓ Saved model: {filepath}")

    @staticmethod
    def load(filepath: str) -> 'KNNRecommendationModel':
        model = joblib.load(filepath)
        logger.info(f"✓ Loaded model: {filepath}")
        return model