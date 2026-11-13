"""
Data loading and utilities for Amazon reviews dataset.
"""

import pandas as pd
from typing import Tuple, Optional
import os


class AmazonReviewsDataLoader:
    """Loads and preprocesses Amazon reviews dataset."""

    @staticmethod
    def load_from_csv(filepath: str, nrows: Optional[int] = None) -> pd.DataFrame:
        """
        Load Amazon reviews from CSV file.
        
        Expected columns: 'review_text' or 'text', 'rating' or 'star_rating'
        
        Args:
            filepath: Path to CSV file
            nrows: Maximum number of rows to load
        
        Returns:
            DataFrame with reviews
        """
        df = pd.read_csv(filepath, nrows=nrows)
        
        # Standardize column names
        if 'review_text' in df.columns:
            df['text'] = df['review_text']
        elif 'Review Text' in df.columns:
            df['text'] = df['Review Text']
        
        if 'star_rating' in df.columns:
            df['rating'] = df['star_rating']
        elif 'rating' not in df.columns and 'Rating' in df.columns:
            df['rating'] = df['Rating']
        
        # Keep only necessary columns
        required_cols = ['text', 'rating']
        df = df[required_cols].copy()
        
        # Remove rows with missing values
        df = df.dropna()
        
        # Ensure rating is numeric
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
        df = df.dropna(subset=['rating'])
        
        # Filter to valid ratings (1-5)
        df = df[(df['rating'] >= 1) & (df['rating'] <= 5)]
        
        print(f"Loaded {len(df)} reviews")
        print(f"Rating distribution:\n{df['rating'].value_counts().sort_index()}")
        
        return df

    @staticmethod
    def sample_by_rating(df: pd.DataFrame, samples_per_rating: int = 1000) -> pd.DataFrame:
        """Sample evenly from each rating to balance dataset."""
        sampled = df.groupby('rating').apply(
            lambda x: x.sample(min(len(x), samples_per_rating), random_state=42),
            include_groups=False
        ).reset_index(drop=True)
        
        print(f"Sampled dataset: {len(sampled)} reviews")
        print(f"Rating distribution:\n{sampled['rating'].value_counts().sort_index()}")
        
        return sampled

    @staticmethod
    def create_demo_dataset(num_samples: int = 100) -> pd.DataFrame:
        """
        Create a demo dataset for testing.
        
        Args:
            num_samples: Number of synthetic reviews to create
        
        Returns:
            DataFrame with demo reviews
        """
        import random
        
        demo_reviews = [
            # Positive reviews (5 stars)
            ("This phone is amazing! Fast processor, beautiful display, and excellent battery life.", 5),
            ("Great quality and fast shipping. Highly recommend!", 5),
            ("The screen quality is incredible. Worth every penny.", 5),
            ("Fast charging and reliable performance. Love it!", 5),
            
            # Negative reviews (1-2 stars)
            ("Battery drains too quickly. Very disappointed.", 1),
            ("Screen cracked after one week. Terrible build quality.", 1),
            ("Overheats constantly during use. Not worth the price.", 2),
            ("The display is too dim and has dead pixels.", 2),
            ("Slow processor, keeps lagging. Waste of money.", 1),
            
            # Mixed reviews (3-4 stars)
            ("Good phone but battery could be better. Overall okay.", 3),
            ("Works fine but expensive. Average performance.", 3),
            ("Pretty good. Minor issues with the camera.", 4),
            ("Excellent build, decent performance. Recommended.", 4),
        ]
        
        # Repeat to reach desired sample size
        reviews_extended = demo_reviews * (num_samples // len(demo_reviews) + 1)
        reviews_extended = reviews_extended[:num_samples]
        
        # Shuffle
        random.shuffle(reviews_extended)
        
        df = pd.DataFrame(reviews_extended, columns=['text', 'rating'])
        
        return df

    @staticmethod
    def save_processed_data(df: pd.DataFrame, output_path: str):
        """Save processed reviews to CSV."""
        df.to_csv(output_path, index=False)
        print(f"Saved to {output_path}")

    @staticmethod
    def get_review_statistics(df: pd.DataFrame) -> dict:
        """Get statistics about the reviews."""
        stats = {
            'total_reviews': len(df),
            'avg_rating': df['rating'].mean(),
            'median_rating': df['rating'].median(),
            'avg_text_length': df['text'].apply(len).mean(),
            'avg_word_count': df['text'].apply(lambda x: len(x.split())).mean(),
            'rating_distribution': df['rating'].value_counts().to_dict()
        }
        return stats
