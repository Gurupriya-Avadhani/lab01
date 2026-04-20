#!/usr/bin/env python3
"""
Data ingestion and validation for MovieLens ratings.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))


import pandas as pd
import json
import logging
from pathlib import Path
from typing import Tuple, Dict, Optional

from src.config import RATINGS_SCHEMA, DATA_PATHS

# -----------------------------------------------------------------------------
# Logging setup
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Schema Validator
# -----------------------------------------------------------------------------
class RatingsSchemaValidator:
    """Validate ratings data against schema."""

    def __init__(self, schema: Dict = RATINGS_SCHEMA):
        self.schema = schema
        self.validation_report: Dict = {}

    def validate_columns(self, df: pd.DataFrame) -> bool:
        required = set(self.schema.keys())
        present = set(df.columns)

        if required != present:
            missing = required - present
            extra = present - required
            logger.error(f"Column mismatch. Missing: {missing}, Extra: {extra}")
            return False

        logger.info(f"✓ Columns validated: {list(df.columns)}")
        return True

    def validate_datatypes(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        errors = 0

        for col, rules in self.schema.items():
            target_dtype = rules["dtype"]

            try:
                # Convert safely
                df[col] = pd.to_numeric(df[col], errors="coerce")

                # Count conversion errors BEFORE casting
                nulls = df[col].isnull().sum()
                if nulls > 0:
                    logger.warning(f"{col}: {nulls} conversion errors")
                    errors += nulls

                # Now cast
                df[col] = df[col].astype(target_dtype)

                logger.info(f"✓ {col}: {target_dtype}")

            except Exception as e:
                logger.error(f"Failed converting {col}: {e}")
                return df, -1

        return df, errors

    def validate_ranges(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        rows_before = len(df)

        for col, rules in self.schema.items():
            min_val = rules["min"]
            max_val = rules["max"]

            mask = df[col].between(min_val, max_val)
            invalid = (~mask).sum()

            if invalid > 0:
                logger.warning(f"{col}: {invalid} out-of-range values")

            df = df[mask]

        removed = rows_before - len(df)
        logger.info(f"✓ Range validation removed {removed} rows")
        return df, removed

    def validate_nulls(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        rows_before = len(df)

        for col, rules in self.schema.items():
            if not rules["nullable"]:
                nulls = df[col].isnull().sum()
                if nulls > 0:
                    logger.warning(f"{col}: {nulls} null values removed")
                    df = df.dropna(subset=[col])

        removed = rows_before - len(df)
        return df, removed

    def validate(self, df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Dict]:
        logger.info(f"Starting validation on {len(df)} rows")

        if not self.validate_columns(df):
            return None, {"error": "Invalid columns"}

        df, dtype_errors = self.validate_datatypes(df)
        if dtype_errors < 0:
            return None, {"error": "Datatype conversion failed"}

        df, range_removed = self.validate_ranges(df)
        df, null_removed = self.validate_nulls(df)

        self.validation_report = {
            "total_errors": dtype_errors + range_removed + null_removed,
            "dtype_errors": dtype_errors,
            "range_violations": range_removed,
            "null_violations": null_removed,
            "rows_retained": len(df),
        }

        logger.info(f"✓ Validation complete: {len(df)} rows valid")
        return df, self.validation_report


# -----------------------------------------------------------------------------
# Loader
# -----------------------------------------------------------------------------
class RatingsLoader:
    """Load, validate, and save ratings."""

    def __init__(self, filepath: str = DATA_PATHS["raw"]):
        self.filepath = filepath
        self.validator = RatingsSchemaValidator()
        self.raw_df: Optional[pd.DataFrame] = None
        self.clean_df: Optional[pd.DataFrame] = None

    def load(self) -> pd.DataFrame:
        path = Path(self.filepath)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        logger.info(f"Loading from {path}")

        # Auto-detect separator (comma or tab)
        self.raw_df = pd.read_csv(path, sep=None, engine="python")

        logger.info(f"✓ Loaded {len(self.raw_df)} rows")
        return self.raw_df

    def deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        rows_before = len(df)

        df = df.sort_values("timestamp")
        df = df.drop_duplicates(
            subset=["user_id", "movie_id"],
            keep="last"
        )

        removed = rows_before - len(df)
        if removed > 0:
            logger.info(f"✓ Removed {removed} duplicates")

        return df

    def validate_and_clean(self) -> Tuple[pd.DataFrame, Dict]:
        if self.raw_df is None:
            raise ValueError("Call load() first")

        df = self.deduplicate(self.raw_df)
        df, report = self.validator.validate(df)

        self.clean_df = df
        return df, report

    def save(self, output_path: str = DATA_PATHS["processed"]):
        if self.clean_df is None:
            raise ValueError("No data to save")

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        self.clean_df.to_csv(output, index=False)
        logger.info(f"✓ Saved cleaned data → {output}")

    def save_report(self, report_path: str = DATA_PATHS["validation_report"]):
        if not self.validator.validation_report:
            logger.warning("No report to save")
            return

        path = Path(report_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(self.validator.validation_report, f, indent=2)

        logger.info(f"✓ Saved report → {path}")


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------
def main():
    logger.info("=" * 60)
    logger.info("MovieLens Data Ingestion Pipeline")
    logger.info("=" * 60)

    try:
        loader = RatingsLoader()

        loader.load()
        loader.validate_and_clean()
        loader.save()
        loader.save_report()

        logger.info("✓ Pipeline complete")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()