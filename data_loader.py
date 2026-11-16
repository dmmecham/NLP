
"""
Data loading and utilities for Amazon customer reviews.

Expected standardized output:
    text:   Review text
    rating: Numeric star rating from 1 to 5

The loader accepts several common Amazon/Kaggle column-name variants.
It does not perform text preprocessing; that is handled separately by
TextPreprocessor.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


class AmazonReviewsDataLoader:
    """Loads, validates, samples, and summarizes Amazon review data."""

    TEXT_COLUMN_ALIASES = (
        "text",
        "review_text",
        "review_body",
        "review_body_text",
        "review text",
        "review body",
    )

    RATING_COLUMN_ALIASES = (
        "rating",
        "star_rating",
        "stars",
        "star rating",
    )

    @staticmethod
    def _normalized_column_lookup(df: pd.DataFrame) -> dict:
        """Map normalized column names to their original names."""
        return {
            str(column).strip().lower(): column
            for column in df.columns
        }

    @classmethod
    def _find_column(
        cls,
        df: pd.DataFrame,
        aliases: tuple,
        description: str,
    ):
        """Find the first matching column using case-insensitive aliases."""
        lookup = cls._normalized_column_lookup(df)

        for alias in aliases:
            column = lookup.get(alias.lower())
            if column is not None:
                return column

        raise ValueError(
            f"Could not identify the {description} column. "
            f"Accepted names: {', '.join(aliases)}. "
            f"Available columns: {list(df.columns)}"
        )

    @classmethod
    def load_from_csv(
        cls,
        filepath: str,
        nrows: Optional[int] = None,
        encoding: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Load reviews from a CSV file and standardize the required columns.

        Args:
            filepath: Path to the source CSV.
            nrows: Optional maximum number of rows to read.
            encoding: Optional CSV encoding. If omitted, pandas uses its
                default encoding.

        Returns:
            DataFrame containing standardized 'text' and 'rating' columns.

        Notes:
            The loader accepts common Amazon review fields such as
            review_body and star_rating. If the source contains multiple
            product categories, filter it to the desired category before
            running the analysis.
        """
        if nrows is not None and nrows < 1:
            raise ValueError("nrows must be a positive integer or None.")

        filepath = Path(filepath)

        if not filepath.is_file():
            raise FileNotFoundError(
                f"Review CSV file not found: {filepath}"
            )

        read_options = {"nrows": nrows}
        if encoding is not None:
            read_options["encoding"] = encoding

        df = pd.read_csv(filepath, **read_options)

        if df.empty:
            raise ValueError(f"The CSV file contains no rows: {filepath}")

        text_column = cls._find_column(
            df,
            cls.TEXT_COLUMN_ALIASES,
            "review text",
        )
        rating_column = cls._find_column(
            df,
            cls.RATING_COLUMN_ALIASES,
            "star rating",
        )

        standardized = pd.DataFrame({
            "text": df[text_column],
            "rating": df[rating_column],
        })

        original_count = len(standardized)

        # Remove missing and blank review text.
        standardized["text"] = standardized["text"].fillna("").astype(str)
        standardized["text"] = standardized["text"].str.strip()
        standardized = standardized[
            standardized["text"].str.len() > 0
        ].copy()

        # Convert ratings to numeric and retain valid integer star ratings.
        standardized["rating"] = pd.to_numeric(
            standardized["rating"],
            errors="coerce",
        )

        standardized = standardized.dropna(subset=["rating"]).copy()

        valid_rating_mask = (
            standardized["rating"].between(1, 5)
            & np.isclose(
                standardized["rating"] % 1,
                0,
            )
        )
        standardized = standardized[valid_rating_mask].copy()
        standardized["rating"] = standardized["rating"].astype(int)

        standardized = standardized.reset_index(drop=True)

        if standardized.empty:
            raise ValueError(
                "No valid reviews remain after cleaning. "
                "Check the source text and rating columns."
            )

        removed_count = original_count - len(standardized)

        print(f"Loaded {len(standardized):,} valid reviews.")
        print(f"Removed {removed_count:,} invalid or empty rows.")
        print(
            "Rating distribution:\n"
            f"{standardized['rating'].value_counts().sort_index()}"
        )

        return standardized

    @staticmethod
    def sample_by_rating(
        df: pd.DataFrame,
        samples_per_rating: int = 1000,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """
        Sample up to a fixed number of reviews from each star rating.

        This creates a balanced sample when every rating has enough
        observations. If a rating has fewer observations than requested,
        all available observations for that rating are retained.

        Args:
            df: DataFrame containing 'text' and 'rating'.
            samples_per_rating: Maximum number of reviews per rating.
            random_state: Seed for reproducible sampling.

        Returns:
            Sampled and shuffled DataFrame.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame.")

        required_columns = {"text", "rating"}
        missing = required_columns.difference(df.columns)

        if missing:
            raise ValueError(
                f"DataFrame is missing required columns: {sorted(missing)}"
            )

        if samples_per_rating < 1:
            raise ValueError("samples_per_rating must be at least 1.")

        if df.empty:
            raise ValueError("Cannot sample an empty dataframe.")

        sampled_groups = []

        for rating, group in df.groupby("rating", sort=True):
            sample_count = min(len(group), samples_per_rating)

            sampled_groups.append(
                group.sample(
                    n=sample_count,
                    random_state=random_state,
                )
            )

        if not sampled_groups:
            raise ValueError("No rating groups were available for sampling.")

        sampled = pd.concat(sampled_groups, ignore_index=True)

        # Shuffle the combined sample while preserving reproducibility.
        sampled = sampled.sample(
            frac=1,
            random_state=random_state,
        ).reset_index(drop=True)

        print(f"Sampled dataset: {len(sampled):,} reviews.")
        print(
            "Rating distribution:\n"
            f"{sampled['rating'].value_counts().sort_index()}"
        )

        return sampled

    @staticmethod
    def create_demo_dataset(
        num_samples: int = 100,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """
        Create a reproducible synthetic electronics-review dataset.

        This dataset is intended only for testing the software pipeline.
        It is not real Amazon data and must not be used to support
        research conclusions.

        Args:
            num_samples: Number of synthetic reviews to generate.
            random_state: Seed for reproducible generation.

        Returns:
            DataFrame containing 'text' and 'rating'.
        """
        if num_samples < 1:
            raise ValueError("num_samples must be at least 1.")

        demo_reviews = [
            # Positive reviews
            (
                "This phone is amazing! Fast processor, beautiful display, "
                "and excellent battery life.",
                5,
            ),
            (
                "Great quality and fast shipping. Highly recommend!",
                5,
            ),
            (
                "The screen quality is incredible. Worth every penny.",
                5,
            ),
            (
                "Fast charging and reliable performance. Love it!",
                5,
            ),
            (
                "The camera takes excellent pictures and the battery "
                "lasts all day.",
                5,
            ),
            (
                "Comfortable design, responsive touchscreen, and "
                "great overall performance.",
                5,
            ),

            # Four-star reviews
            (
                "Pretty good phone. Minor issues with the camera.",
                4,
            ),
            (
                "Excellent build quality and decent performance.",
                4,
            ),
            (
                "The display looks great, although charging is a little slow.",
                4,
            ),
            (
                "Works well overall. The battery could last longer.",
                4,
            ),

            # Three-star reviews
            (
                "Good phone but battery life could be better. Overall okay.",
                3,
            ),
            (
                "Works fine but expensive. Average performance.",
                3,
            ),
            (
                "The screen is acceptable, but the speakers are disappointing.",
                3,
            ),
            (
                "Mixed experience. Good camera, but the device gets warm.",
                3,
            ),

            # Two-star reviews
            (
                "Overheats constantly during use. Not worth the price.",
                2,
            ),
            (
                "The display is too dim and has dead pixels.",
                2,
            ),
            (
                "Battery drains quickly and charging takes too long.",
                2,
            ),
            (
                "The buttons feel cheap and the device freezes frequently.",
                2,
            ),

            # One-star reviews
            (
                "Battery drains too quickly. Very disappointed.",
                1,
            ),
            (
                "Screen cracked after one week. Terrible build quality.",
                1,
            ),
            (
                "Slow processor, keeps lagging. Waste of money.",
                1,
            ),
            (
                "The device stopped working after a few days.",
                1,
            ),
            (
                "The charger broke immediately and customer support was unhelpful.",
                1,
            ),
        ]

        rng = np.random.default_rng(random_state)

        selected_indices = rng.integers(
            low=0,
            high=len(demo_reviews),
            size=num_samples,
        )

        selected_reviews = [
            demo_reviews[index] for index in selected_indices
        ]

        df = pd.DataFrame(
            selected_reviews,
            columns=["text", "rating"],
        )

        # Shuffle reproducibly.
        df = df.sample(
            frac=1,
            random_state=random_state,
        ).reset_index(drop=True)

        return df

    @staticmethod
    def save_processed_data(
        df: pd.DataFrame,
        output_path: str,
    ) -> None:
        """Save a dataframe to CSV, creating parent directories as needed."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_path, index=False, encoding="utf-8")
        print(f"Saved {len(df):,} rows to {output_path}")

    @staticmethod
    def get_review_statistics(df: pd.DataFrame) -> dict:
        """
        Calculate basic descriptive statistics for a review dataset.

        Returns:
            Dictionary containing dataset size, rating statistics,
            text length statistics, and rating distribution.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame.")

        required_columns = {"text", "rating"}
        missing = required_columns.difference(df.columns)

        if missing:
            raise ValueError(
                f"DataFrame is missing required columns: {sorted(missing)}"
            )

        if df.empty:
            raise ValueError("Cannot calculate statistics for an empty dataset.")

        text_lengths = df["text"].fillna("").astype(str).str.len()
        word_counts = (
            df["text"]
            .fillna("")
            .astype(str)
            .str.split()
            .str.len()
        )

        rating_distribution = (
            df["rating"]
            .value_counts()
            .sort_index()
            .to_dict()
        )

        return {
            "total_reviews": int(len(df)),
            "avg_rating": float(df["rating"].mean()),
            "median_rating": float(df["rating"].median()),
            "avg_text_length": float(text_lengths.mean()),
            "avg_word_count": float(word_counts.mean()),
            "rating_distribution": rating_distribution,
        }
