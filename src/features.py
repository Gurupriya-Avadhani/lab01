import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import logging
from typing import Tuple, List, Dict
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

class RatingFeatures:

    def __init__(self):
        self.ratings_matrix = None
        self.similarity_matrix = None
        self.user_ids = None
        self.movie_ids = None
        self.fitted = False

    def fit(self, ratings_df: pd.DataFrame) -> 'RatingFeatures':

        if ratings_df.empty:
            raise ValueError("ratings_df cannot be empty")

        required_cols = {'user_id', 'movie_id', 'rating'}
        if not required_cols.issubset(ratings_df.columns):
            missing = required_cols - set(ratings_df.columns)
            raise ValueError(f"Missing required columns: {missing}")

        self.ratings_matrix = ratings_df.pivot_table(
            index='user_id',
            columns='movie_id',
            values='rating',
            fill_value=0.0
        )

        self.user_ids = self.ratings_matrix.index.values
        self.movie_ids = self.ratings_matrix.columns.values

        self.similarity_matrix = cosine_similarity(self.ratings_matrix.values)

        np.fill_diagonal(self.similarity_matrix, 0)

        self.fitted = True

        return self

    def get_similar_users(self, user_id: int, n: int = 5) -> List[Tuple[int, float]]:

        if not self.fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        user_indices = np.where(self.user_ids == user_id)[0]

        if len(user_indices) == 0:
            return []

        user_idx = user_indices[0]

        similarities = self.similarity_matrix[user_idx]

        top_indices = np.argpartition(similarities, -n)[-n:]
        top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]

        valid_indices = [i for i in top_indices if similarities[i] > 0]

        return [(self.user_ids[i], similarities[i]) for i in valid_indices]

    def get_user_ratings_vector(self, user_id: int) -> np.ndarray:

        if user_id not in self.user_ids:
            raise ValueError(f"User {user_id} not in training data")

        user_idx = np.where(self.user_ids == user_id)[0][0]

        return self.ratings_matrix.iloc[user_idx].values

    def get_movie_rating_stats(self) -> Dict[str, float]:

        rated_values = self.ratings_matrix[self.ratings_matrix > 0].values.flatten()

        return {
            'mean_rating': float(np.mean(rated_values)),
            'std_rating': float(np.std(rated_values)),
            'min_rating': float(np.min(rated_values)),
            'max_rating': float(np.max(rated_values)),
            'n_rated': int(len(rated_values))
        }

    def save(self, filepath: str) -> None:

        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        joblib.dump(self, filepath)

    @staticmethod
    def load(filepath: str) -> 'RatingFeatures':

        features = joblib.load(filepath)

        if not isinstance(features, RatingFeatures):
            raise TypeError("Loaded object is not RatingFeatures")

        return features