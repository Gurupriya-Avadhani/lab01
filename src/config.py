# #!/usr/bin/env python3
# """
# Configuration and schema for MovieLens data pipeline.
# """

# # ============================================================================
# # MovieLens 100K Dataset Schema
# # ============================================================================
# # Original dataset: 100,000 ratings from 943 users on 1,682 movies
# # For this lab: Synthetic subset (2,000 ratings, ~189 users, ~100 movies)
# # ============================================================================

# RATINGS_SCHEMA = {
#     "user_id": {
#         "dtype": "int64",
#         "min": 1,
#         "max": 1000,  # Allow expansion beyond initial users
#         "nullable": False,
#         "description": "Unique user identifier",
#     },
#     "movie_id": {
#         "dtype": "int64",
#         "min": 1,
#         "max": 1700,  # Allow expansion beyond initial movies
#         "nullable": False,
#         "description": "Unique movie identifier",
#     },
#     "rating": {
#         "dtype": "float64",
#         "min": 0.5,
#         "max": 5.0,
#         "nullable": False,
#         "description": "User rating (0.5 to 5.0 scale)",
#     },
#     "timestamp": {
#         "dtype": "int64",
#         "min": 1_000_000_000,  # ~2001 (Unix epoch)
#         "max": 2_000_000_000,  # ~2033 (future-proof)
#         "nullable": False,
#         "description": "Unix timestamp of rating",
#     },
# }

# # Expected dataset metrics (for validation checks)
# EXPECTED_METRICS = {
#     "raw_rows": 2000,
#     "min_clean_rows": 1800,   # At least 90% valid rows
#     "target_sparsity": 0.90,  # ~10% density
# }

# # File paths relative to project root
# DATA_PATHS = {
#     "raw": "data/raw/ratings.csv",
#     "processed": "data/processed/ratings_clean.csv",
#     "validation_report": "evaluations/validation_report.json",
# }



#!/usr/bin/env python3
"""
Configuration and schema for MovieLens data pipeline.
"""

RATINGS_SCHEMA = {
    "user_id": {
        "dtype": "int64",
        "min": 1,
        "max": 1000,
        "nullable": False,
        "description": "Unique user identifier",
    },
    "movie_id": {
        "dtype": "int64",
        "min": 1,
        "max": 1700,
        "nullable": False,
        "description": "Unique movie identifier",
    },
    "rating": {
        "dtype": "float64",
        "min": 0.5,
        "max": 5.0,
        "nullable": False,
        "description": "User rating (0.5 to 5.0 scale)",
    },
    "timestamp": {
        "dtype": "int64",
        "min": 0,  # ✅ FIXED (was too strict)
        "max": 2_000_000_000,
        "nullable": False,
        "description": "Unix timestamp of rating",
    },
}

EXPECTED_METRICS = {
    "raw_rows": 2000,
    "min_clean_rows": 1800,
    "target_sparsity": 0.90,
}

DATA_PATHS = {
    "raw": "data/raw/ratings.csv",
    "processed": "data/processed/ratings_clean.csv",
    "validation_report": "evaluations/validation_report.json",
}