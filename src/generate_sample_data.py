#!/usr/bin/env python3
"""
Generate synthetic MovieLens-like ratings for development.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def generate_synthetic_ratings(
    n_ratings: int = 2000,
    n_users: int = 189,
    n_movies: int = 100,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic MovieLens-like ratings.

    Args:
        n_ratings: Number of ratings to generate
        n_users: Number of unique users
        n_movies: Number of unique movies
        random_seed: For reproducibility

    Returns:
        DataFrame with columns: user_id, movie_id, rating, timestamp
    """

    np.random.seed(random_seed)

    # Generate random user-movie pairs
    user_ids = np.random.randint(1, n_users + 1, size=n_ratings)
    movie_ids = np.random.randint(1, n_movies + 1, size=n_ratings)

    # Ratings: normal distribution, clipped to [1, 5] and rounded to 0.5
    ratings = np.random.normal(loc=3.5, scale=1.0, size=n_ratings)
    ratings = np.clip(ratings, 1.0, 5.0)
    ratings = np.round(ratings * 2) / 2  # enforce 0.5 increments

    # Timestamps: Unix time from 1995 to 2005
    start_timestamp = int(pd.Timestamp("1995-01-01").timestamp())
    end_timestamp = int(pd.Timestamp("2005-01-01").timestamp())
    timestamps = np.random.randint(start_timestamp, end_timestamp, size=n_ratings)

    df = pd.DataFrame({
        "user_id": user_ids,
        "movie_id": movie_ids,
        "rating": ratings,
        "timestamp": timestamps,
    })

    # Remove duplicate user-movie pairs (keep latest timestamp)
    df = df.sort_values("timestamp")
    df = df.drop_duplicates(subset=["user_id", "movie_id"], keep="last")

    # Reset index
    df = df.reset_index(drop=True)

    return df


def main():
    """Generate and save synthetic data."""

    print("Generating synthetic MovieLens ratings...")

    df = generate_synthetic_ratings(
        n_ratings=2000,
        n_users=189,
        n_movies=100
    )

    # Create directory
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "ratings.csv"

    # Save (IMPORTANT: use comma, not tab for consistency with pandas defaults)
    df.to_csv(output_path, index=False)

    # Stats
    n_users_unique = df["user_id"].nunique()
    n_movies_unique = df["movie_id"].nunique()
    sparsity = 1 - len(df) / (n_users_unique * n_movies_unique)

    print(f"✓ Generated {len(df)} ratings")
    print(f"✓ Unique users: {n_users_unique}")
    print(f"✓ Unique movies: {n_movies_unique}")
    print(f"✓ Sparsity: {sparsity * 100:.1f}%")
    print(f"✓ Saved to {output_path}")

    print("\nFirst 5 rows:")
    print(df.head())


if __name__ == "__main__":
    main()