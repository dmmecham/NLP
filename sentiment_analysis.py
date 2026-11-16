
"""
Sentiment and aspect synthesis module.
Maps review ratings to extracted topics and computes aspect-level sentiment.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns


class AspectSentimentAnalyzer:
    """Analyzes sentiment at aspect/topic level."""

    @staticmethod
    def compute_aspect_sentiment_distribution(
        topic_dist_df: pd.DataFrame,
        topic_columns: List[str] = None
    ) -> pd.DataFrame:
        """
        Compute sentiment distribution for each aspect/topic.

        Args:
            topic_dist_df: DataFrame with topic probabilities and ratings
            topic_columns: List of topic column names (e.g., ['topic_0', 'topic_1', ...])

        Returns:
            DataFrame with sentiment statistics per topic
        """
        if topic_columns is None:
            topic_columns = [
                col for col in topic_dist_df.columns
                if col.startswith('topic_')
            ]

        results = []

        for topic_col in topic_columns:
            topic_id = int(topic_col.split('_')[1])

            # Get average rating for documents where this topic dominates.
            # A topic is considered present if its probability is > 0.1.
            threshold = 0.1
            topic_documents = topic_dist_df[
                topic_dist_df[topic_col] > threshold
            ]

            if len(topic_documents) > 0:
                avg_rating = topic_documents['rating'].mean()
                median_rating = topic_documents['rating'].median()
                std_rating = topic_documents['rating'].std()
                count_1star = (topic_documents['rating'] == 1).sum()
                count_5star = (topic_documents['rating'] == 5).sum()

                results.append({
                    'topic_id': topic_id,
                    'avg_rating': avg_rating,
                    'median_rating': median_rating,
                    'std_rating': std_rating,
                    'count_docs': len(topic_documents),
                    'count_1star': count_1star,
                    'count_5star': count_5star,
                    'pct_1star': (
                        count_1star / len(topic_documents)
                    ) * 100,
                    'pct_5star': (
                        count_5star / len(topic_documents)
                    ) * 100,
                    'negative_sentiment_score': 1 - (avg_rating / 5.0)
                })

        if not results:
            return pd.DataFrame()

        return pd.DataFrame(results).sort_values(
            'negative_sentiment_score',
            ascending=False
        )

    @staticmethod
    def calculate_one_star_aspect_percentages(
        topic_dist_df: pd.DataFrame,
        topic_columns: List[str] = None,
        threshold: float = 0.05
    ) -> pd.DataFrame:
        """
        Calculate the percentage of all one-star reviews associated
        with each topic/aspect.

        Args:
            topic_dist_df: DataFrame containing topic assignments
                           and review ratings.
            topic_columns: Optional list of topic columns.
            threshold: Minimum topic probability required to consider
                       a review associated with a topic.

        Returns:
            DataFrame containing the number and percentage of
            one-star reviews associated with each topic.
        """

        if topic_columns is None:
            topic_columns = [
                col for col in topic_dist_df.columns
                if col.startswith('topic_')
            ]

        if 'rating' not in topic_dist_df.columns:
            raise ValueError(
                "topic_dist_df must contain a 'rating' column."
            )

        if not topic_columns:
            return pd.DataFrame(columns=[
                'topic_id',
                'count_1star_reviews',
                'total_1star_reviews',
                'pct_of_one_star_reviews'
            ])

        # Isolate one-star reviews.
        one_star_reviews = topic_dist_df[
            topic_dist_df['rating'] == 1
        ]

        # Handle datasets without one-star reviews.
        if one_star_reviews.empty:
            return pd.DataFrame(columns=[
                'topic_id',
                'count_1star_reviews',
                'total_1star_reviews',
                'pct_of_one_star_reviews'
            ])

        results = []

        for topic_col in topic_columns:
            if topic_col not in topic_dist_df.columns:
                raise ValueError(
                    f"Topic column '{topic_col}' was not found."
                )

            topic_id = int(topic_col.split('_')[1])

            # Find one-star reviews associated with this topic.
            matching_reviews = one_star_reviews[
                one_star_reviews[topic_col] > threshold
            ]

            count = len(matching_reviews)

            results.append({
                'topic_id': topic_id,
                'count_1star_reviews': count,
                'total_1star_reviews': len(one_star_reviews),
                'pct_of_one_star_reviews': (
                    count / len(one_star_reviews)
                ) * 100
            })

        results_df = pd.DataFrame(results)

        return results_df.sort_values(
            'pct_of_one_star_reviews',
            ascending=False
        ).reset_index(drop=True)

    @staticmethod
    def identify_negative_drivers(
        topic_dist_df: pd.DataFrame,
        aspect_sentiment_df: pd.DataFrame,
        topic_keywords: Dict[int, List[Tuple[str, float]]] = None,
        threshold_stars: int = 2
    ) -> pd.DataFrame:
        """
        Identify which topics/aspects most strongly correlate with negative ratings.

        Args:
            topic_dist_df: DataFrame with topic probabilities and ratings
            aspect_sentiment_df: DataFrame from compute_aspect_sentiment_distribution
            topic_keywords: Dict mapping topic_id to list of (keyword, weight) tuples
            threshold_stars: Ratings <= this value are considered negative

        Returns:
            DataFrame ranking topics by their association with negative reviews
        """
        # Filter to only negative reviews.
        negative_reviews = topic_dist_df[
            topic_dist_df['rating'] <= threshold_stars
        ]

        if len(negative_reviews) == 0:
            print("No negative reviews found")
            return pd.DataFrame()

        results = []

        for _, row in aspect_sentiment_df.iterrows():
            topic_id = int(row['topic_id'])
            topic_col = f'topic_{topic_id}'

            # Average topic prominence in negative reviews.
            avg_topic_prominence = negative_reviews[topic_col].mean()

            # Percentage of negative reviews where this topic is mentioned.
            topic_present = (
                negative_reviews[topic_col] > 0.05
            ).sum()

            pct_negative_with_topic = (
                topic_present / len(negative_reviews)
            ) * 100

            keywords = ""
            if topic_keywords and topic_id in topic_keywords:
                top_keywords = [
                    kw[0] for kw in topic_keywords[topic_id][:5]
                ]
                keywords = ", ".join(top_keywords)

            results.append({
                'topic_id': topic_id,
                'keywords': keywords,
                'avg_prominence_in_negative': avg_topic_prominence,
                'pct_negative_reviews': pct_negative_with_topic,
                'avg_rating': row['avg_rating'],
                'count_negative_mentions': topic_present,
                'impact_score': (
                    avg_topic_prominence * pct_negative_with_topic
                )
            })

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(
            'impact_score',
            ascending=False
        )

        return results_df

    @staticmethod
    def compare_topic_ratings(
        topic_dist_df: pd.DataFrame,
        topic_keywords: Dict[int, List[Tuple[str, float]]] = None
    ) -> pd.DataFrame:
        """Create comparison table of topics and their associated ratings."""
        topic_columns = [
            col for col in topic_dist_df.columns
            if col.startswith('topic_')
        ]

        comparisons = []

        for topic_col in topic_columns:
            topic_id = int(topic_col.split('_')[1])

            # Get ratings for documents with this topic.
            topic_docs = topic_dist_df[
                topic_dist_df[topic_col] > 0.1
            ]

            if len(topic_docs) > 0:
                keywords = ""
                if topic_keywords and topic_id in topic_keywords:
                    top_keywords = [
                        kw[0] for kw in topic_keywords[topic_id][:5]
                    ]
                    keywords = ", ".join(top_keywords)

                comparisons.append({
                    'topic_id': topic_id,
                    'keywords': keywords,
                    'mean_rating': topic_docs['rating'].mean(),
                    'std_rating': topic_docs['rating'].std(),
                    'min_rating': topic_docs['rating'].min(),
                    'max_rating': topic_docs['rating'].max(),
                    'doc_count': len(topic_docs)
                })

        if not comparisons:
            return pd.DataFrame()

        return pd.DataFrame(comparisons).sort_values('mean_rating')

    @staticmethod
    def visualize_topic_sentiment(
        aspect_sentiment_df: pd.DataFrame,
        title: str = "Topic Sentiment Distribution"
    ):
        """Visualize average ratings by topic."""
        fig, ax = plt.subplots(figsize=(12, 6))

        topics = aspect_sentiment_df['topic_id'].astype(str)
        ratings = aspect_sentiment_df['avg_rating']

        colors = plt.cm.RdYlGn(ratings / 5.0)
        ax.bar(topics, ratings, color=colors)
        ax.axhline(
            y=3,
            color='gray',
            linestyle='--',
            alpha=0.5,
            label='Neutral (3.0)'
        )
        ax.set_ylabel('Average Rating')
        ax.set_xlabel('Topic ID')
        ax.set_ylim([0, 5])
        ax.set_title(title)
        ax.legend()

        plt.tight_layout()
        return fig

    @staticmethod
    def visualize_negative_drivers(
        negative_drivers_df: pd.DataFrame,
        title: str = "Drivers of Negative Reviews"
    ):
        """Visualize topics most associated with negative ratings."""
        fig, ax = plt.subplots(
            figsize=(12, max(6, len(negative_drivers_df) * 0.3))
        )

        if len(negative_drivers_df) == 0:
            print("No data to visualize")
            return fig

        # Show top 15 drivers.
        top_n = min(15, len(negative_drivers_df))
        top_drivers = negative_drivers_df.head(top_n).copy()

        y_pos = np.arange(len(top_drivers))
        ax.barh(y_pos, top_drivers['impact_score'])
        ax.set_yticks(y_pos)
        ax.set_yticklabels([
            f"T{int(t)} - {k[:30]}"
            for t, k in zip(
                top_drivers['topic_id'],
                top_drivers['keywords']
            )
        ])
        ax.set_xlabel(
            'Impact Score (Prominence × Negative Review %)'
        )
        ax.set_title(title)
        ax.invert_yaxis()

        plt.tight_layout()
        return fig
