#!/usr/bin/env python3
"""
Process MovieLens ratings: validate, deduplicate, and save clean CSV.
"""

import pandas as pd
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def validate_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """Validate MovieLens schema and rating ranges."""
    required = {'user_id', 'movie_id', 'rating', 'timestamp'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Remove invalid ratings
    initial = len(df)
    df = df[(df['rating'] >= 1.0) & (df['rating'] <= 5.0)]
    invalid = initial - len(df)
    if invalid > 0:
        logger.warning(f"Removed {invalid} invalid ratings")

    return df


def process_ratings():
    """Load, validate, deduplicate, and save clean ratings."""
    raw_path = Path('data/raw/ratings.csv')
    processed_dir = Path('data/processed')
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_path = processed_dir / 'ratings_clean.csv'

    logger.info(f"Loading raw ratings from {raw_path}...")
    df = pd.read_csv(raw_path)
    logger.info(f"Loaded {len(df)} ratings")

    # Validate ratings
    df = validate_ratings(df)

    # Remove duplicates based on (user_id, movie_id)
    before = len(df)
    df = df.drop_duplicates(subset=['user_id', 'movie_id'])
    removed = before - len(df)
    if removed > 0:
        logger.info(f"Removed {removed} duplicate ratings")

    # Save cleaned CSV
    df.to_csv(output_path, index=False)
    logger.info(f"✓ Saved {len(df)} clean ratings to {output_path}")
    logger.info(f"  Users: {df['user_id'].nunique()}")
    logger.info(f"  Movies: {df['movie_id'].nunique()}")


if __name__ == '__main__':
    process_ratings()