
"""
Main pipeline orchestrator.
Runs complete end-to-end analysis with both LDA and BERTopic.
"""

from typing import Dict, Tuple
from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing import TextPreprocessor
from lda_model import LDATopicModel
from bertopic_model import BERTopicModel
from sentiment_analysis import AspectSentimentAnalyzer
from data_loader import AmazonReviewsDataLoader
from evaluation import ModelEvaluator


class AspectSentimentPipeline:
    """Complete pipeline for aspect-based topic modeling and sentiment analysis."""

    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.preprocessor = TextPreprocessor()
        self.lda_model = LDATopicModel()
        self.bertopic_model = BERTopicModel()

        self.df = None
        self.lda_topics = None
        self.lda_coherence = None
        self.bertopic_topics = None
        self.bertopic_coherence = None

        self.lda_sentiment = None
        self.bertopic_sentiment = None

        self.coherence_comparison = None

    def load_data(
        self,
        filepath: str = None,
        use_demo: bool = False,
        nrows: int = None,
        sample_per_rating: int = None
    ) -> pd.DataFrame:
        """Load the review dataset."""
        print("\n" + "=" * 80)
        print("STEP 1: LOADING DATA")
        print("=" * 80)

        if use_demo:
            print("Creating demo dataset...")
            self.df = AmazonReviewsDataLoader.create_demo_dataset(
                num_samples=100
            )
        elif filepath:
            print(f"Loading reviews from {filepath}...")
            self.df = AmazonReviewsDataLoader.load_from_csv(
                filepath,
                nrows=nrows
            )

            if sample_per_rating:
                self.df = AmazonReviewsDataLoader.sample_by_rating(
                    self.df,
                    samples_per_rating=sample_per_rating
                )
        else:
            raise ValueError(
                "Either filepath or use_demo must be provided."
            )

        stats = AmazonReviewsDataLoader.get_review_statistics(self.df)

        print("\nDataset Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

        return self.df

    def preprocess_data(self) -> pd.DataFrame:
        """Preprocess reviews for both LDA and BERTopic."""
        print("\n" + "=" * 80)
        print("STEP 2: PREPROCESSING DATA")
        print("=" * 80)

        if self.df is None:
            raise ValueError(
                "Data not loaded. Call load_data() first."
            )

        print("Preprocessing reviews...")

        self.df = self.preprocessor.process_reviews_df(
            self.df,
            text_column="text"
        )

        print("Sample preprocessed review:")
        idx = 0
        print(f"  Original: {self.df.iloc[idx]['text'][:100]}...")
        print(f"  Tokens: {self.df.iloc[idx]['tokens']}")
        print(
            f"  Cleaned: "
            f"{self.df.iloc[idx]['text_cleaned'][:100]}..."
        )

        return self.df

    def compare_topic_counts(
        self,
        topic_range=range(5, 21, 2)
    ) -> pd.DataFrame:
        """
        Evaluate LDA and BERTopic over the same topic-count range.

        LDA and BERTopic are evaluated independently using U_Mass
        coherence. The best topic count for each model is selected
        from the resulting scores.

        The BERTopic model reuses its sentence embeddings across
        topic counts because the input review texts do not change.
        """
        if self.df is None:
            raise ValueError(
                "Data not preprocessed. Call preprocess_data() first."
            )

        tokens = self.df["tokens"].tolist()
        texts = self.df["text_cleaned"].tolist()

        valid_topic_counts = [
            count for count in topic_range
            if 2 <= count <= len(self.df)
        ]

        if not valid_topic_counts:
            raise ValueError(
                "The topic-count range contains no valid values "
                "for the current dataset."
            )

        print("\n" + "=" * 80)
        print("TOPIC COUNT COMPARISON")
        print("=" * 80)

        # Evaluate LDA across the selected topic counts.
        print("\nEvaluating LDA topic counts...")

        lda_scores = self.lda_model.optimize_topics(
            tokens,
            topic_range=valid_topic_counts
        )

        # Evaluate BERTopic across the same topic counts.
        print("\nEvaluating BERTopic topic counts...")

        bertopic_scores = {}

        # Generate embeddings once; train() will reuse them for
        # each subsequent topic count.
        self.bertopic_model.generate_embeddings(texts)

        for count in valid_topic_counts:
            print(f"\nEvaluating BERTopic with {count} topics...")

            self.bertopic_model.train(
                texts,
                num_topics=count
            )

            score = self.bertopic_model.calculate_coherence(
                tokens
            )

            bertopic_scores[count] = score

        # Keep only counts for which both models returned scores.
        comparison_rows = []

        for count in valid_topic_counts:
            comparison_rows.append({
                "num_topics": count,
                "lda_coherence": lda_scores.get(count, np.nan),
                "bertopic_coherence": bertopic_scores.get(
                    count,
                    np.nan
                )
            })

        comparison_df = pd.DataFrame(comparison_rows)

        comparison_df.to_csv(
            self.output_dir / "coherence_comparison.csv",
            index=False
        )

        self.coherence_comparison = comparison_df

        print("\nCoherence comparison:")
        print(comparison_df.to_string(index=False))

        # Select the best score independently for each model.
        valid_lda_scores = {
            count: score
            for count, score in lda_scores.items()
            if score is not None and np.isfinite(score)
        }

        valid_bertopic_scores = {
            count: score
            for count, score in bertopic_scores.items()
            if score is not None and np.isfinite(score)
        }

        if not valid_lda_scores:
            raise ValueError(
                "LDA did not produce any valid coherence scores."
            )

        if not valid_bertopic_scores:
            raise ValueError(
                "BERTopic did not produce any valid coherence scores. "
                "Check the vocabulary overlap used by its coherence "
                "calculation."
            )

        best_lda_count = max(
            valid_lda_scores,
            key=valid_lda_scores.get
        )

        best_bertopic_count = max(
            valid_bertopic_scores,
            key=valid_bertopic_scores.get
        )

        print(
            f"\nBest LDA topic count: {best_lda_count} "
            f"(U_Mass: {valid_lda_scores[best_lda_count]:.4f})"
        )

        print(
            f"Best BERTopic topic count: {best_bertopic_count} "
            f"(U_Mass: "
            f"{valid_bertopic_scores[best_bertopic_count]:.4f})"
        )

        return comparison_df

    def train_lda(
        self,
        num_topics: int = 10
    ) -> Tuple[LDATopicModel, pd.DataFrame]:
        """Train the final LDA model."""
        print("\n" + "=" * 80)
        print("STEP 3: TRAINING LDA MODEL (Classical Stream)")
        print("=" * 80)

        if self.df is None:
            raise ValueError(
                "Data not preprocessed. Call preprocess_data() first."
            )

        self.lda_model.train(
            self.df["tokens"].tolist(),
            num_topics=num_topics
        )

        self.lda_coherence = self.lda_model.calculate_coherence()

        self.lda_topics = self.lda_model.get_topics(num_words=10)

        print(f"\nFinal LDA coherence: {self.lda_coherence:.4f}")
        print("\nTop 3 topics from LDA:")

        for topic in self.lda_topics[:3]:
            print(f"\n  Topic {topic['topic_id']}:")

            for word, weight in topic["terms"][:5]:
                print(f"    {word}: {weight:.4f}")

        return self.lda_model, self.lda_topics

    def train_bertopic(
        self,
        num_topics: int = 10
    ) -> Tuple[BERTopicModel, Dict]:
        """Train the final BERTopic model."""
        print("\n" + "=" * 80)
        print("STEP 4: TRAINING BERTOPIC MODEL (Contextual Stream)")
        print("=" * 80)

        if self.df is None:
            raise ValueError(
                "Data not preprocessed. Call preprocess_data() first."
            )

        texts = self.df["text_cleaned"].tolist()

        self.bertopic_model.train(
            texts,
            num_topics=num_topics
        )

        # Evaluate BERTopic using the LDA-tokenized corpus.
        self.bertopic_coherence = (
            self.bertopic_model.calculate_coherence(
                self.df["tokens"].tolist()
            )
        )

        self.bertopic_topics = self.bertopic_model.get_topics(
            top_n=10
        )

        print(
            f"\nFinal BERTopic coherence: "
            f"{self.bertopic_coherence:.4f}"
        )

        print("\nTop 3 topics from BERTopic:")

        for topic_id, terms in sorted(
            self.bertopic_topics.items()
        )[:3]:
            print(f"\n  Topic {topic_id}:")

            for word, weight in terms[:5]:
                print(f"    {word}: {weight:.4f}")

        return self.bertopic_model, self.bertopic_topics

    def analyze_sentiment_lda(self) -> pd.DataFrame:
        """Analyze sentiment for LDA topics."""
        print("\n" + "=" * 80)
        print("STEP 5: SENTIMENT ANALYSIS - LDA")
        print("=" * 80)

        if self.lda_model.model is None:
            raise ValueError(
                "LDA model not trained. Call train_lda() first."
            )

        topic_dist = self.lda_model.get_review_topic_distribution(
            self.df
        )

        topic_dist.to_csv(
            self.output_dir / "lda_topic_distribution.csv",
            index=False
        )

        aspect_sentiment = (
            AspectSentimentAnalyzer.compute_aspect_sentiment_distribution(
                topic_dist
            )
        )

        aspect_sentiment.to_csv(
            self.output_dir / "lda_aspect_sentiment.csv",
            index=False
        )

        negative_drivers = (
            AspectSentimentAnalyzer.identify_negative_drivers(
                topic_dist,
                aspect_sentiment,
                topic_keywords={
                    topic["topic_id"]: topic["terms"]
                    for topic in self.lda_topics
                },
                threshold_stars=2
            )
        )

        negative_drivers.to_csv(
            self.output_dir / "lda_negative_drivers.csv",
            index=False
        )

        # Calculate the percentage of all one-star reviews
        # associated with each LDA topic.
        one_star_percentages = (
            AspectSentimentAnalyzer.calculate_one_star_aspect_percentages(
                topic_dist
            )
        )

        one_star_percentages.to_csv(
            self.output_dir / "lda_one_star_aspect_percentages.csv",
            index=False
        )

        print("\nTop 5 negative drivers (LDA):")
        print(
            negative_drivers.head()[
                ["topic_id", "keywords", "impact_score", "avg_rating"]
            ]
        )

        self.lda_sentiment = {
            "aspect_sentiment": aspect_sentiment,
            "negative_drivers": negative_drivers,
            "one_star_percentages": one_star_percentages,
            "topic_distribution": topic_dist
        }

        return aspect_sentiment

    def analyze_sentiment_bertopic(self) -> pd.DataFrame:
        """Analyze sentiment for BERTopic topics."""
        print("\n" + "=" * 80)
        print("STEP 6: SENTIMENT ANALYSIS - BERTOPIC")
        print("=" * 80)

        if self.bertopic_model.model is None:
            raise ValueError(
                "BERTopic model not trained. Call train_bertopic() first."
            )

        topic_dist = self.bertopic_model.get_review_topic_distribution(
            self.df
        )

        topic_dist.to_csv(
            self.output_dir / "bertopic_topic_distribution.csv",
            index=False
        )

        aspect_sentiment = (
            AspectSentimentAnalyzer.compute_aspect_sentiment_distribution(
                topic_dist
            )
        )

        aspect_sentiment.to_csv(
            self.output_dir / "bertopic_aspect_sentiment.csv",
            index=False
        )

        negative_drivers = (
            AspectSentimentAnalyzer.identify_negative_drivers(
                topic_dist,
                aspect_sentiment,
                topic_keywords=self.bertopic_topics,
                threshold_stars=2
            )
        )

        negative_drivers.to_csv(
            self.output_dir / "bertopic_negative_drivers.csv",
            index=False
        )

        # Calculate the percentage of all one-star reviews
        # associated with each BERTopic topic.
        one_star_percentages = (
            AspectSentimentAnalyzer.calculate_one_star_aspect_percentages(
                topic_dist
            )
        )

        one_star_percentages.to_csv(
            self.output_dir / "bertopic_one_star_aspect_percentages.csv",
            index=False
        )

        print("\nTop 5 negative drivers (BERTopic):")
        print(
            negative_drivers.head()[
                ["topic_id", "keywords", "impact_score", "avg_rating"]
            ]
        )

        self.bertopic_sentiment = {
            "aspect_sentiment": aspect_sentiment,
            "negative_drivers": negative_drivers,
            "one_star_percentages": one_star_percentages,
            "topic_distribution": topic_dist
        }

        return aspect_sentiment

    def evaluate_and_compare(self) -> Dict:
        """Compare LDA and BERTopic results."""
        print("\n" + "=" * 80)
        print("STEP 7: EVALUATION AND COMPARISON")
        print("=" * 80)

        if (
            self.lda_sentiment is None
            or self.bertopic_sentiment is None
        ):
            raise ValueError(
                "Both models must be trained and analyzed. "
                "Call the analyze_sentiment methods first."
            )

        results = {
            "lda": {
                "aspect_sentiment": self.lda_sentiment["aspect_sentiment"],
                "negative_drivers": self.lda_sentiment["negative_drivers"],
                "one_star_percentages": (
                    self.lda_sentiment["one_star_percentages"]
                ),
                "dataset_stats": (
                    AmazonReviewsDataLoader.get_review_statistics(
                        self.df
                    )
                ),
                "optimal_topics": len(self.lda_topics),
                "best_coherence": self.lda_coherence
            },
            "bertopic": {
                "aspect_sentiment": (
                    self.bertopic_sentiment["aspect_sentiment"]
                ),
                "negative_drivers": (
                    self.bertopic_sentiment["negative_drivers"]
                ),
                "one_star_percentages": (
                    self.bertopic_sentiment["one_star_percentages"]
                ),
                "num_topics": len(self.bertopic_topics),
                "best_coherence": self.bertopic_coherence
            }
        }

        # Save the one-star aspect comparison in one file.
        lda_one_star = (
            self.lda_sentiment["one_star_percentages"]
            .assign(model="LDA")
        )

        bert_one_star = (
            self.bertopic_sentiment["one_star_percentages"]
            .assign(model="BERTopic")
        )

        one_star_comparison = pd.concat(
            [lda_one_star, bert_one_star],
            ignore_index=True
        )

        one_star_comparison.to_csv(
            self.output_dir / "one_star_aspect_comparison.csv",
            index=False
        )

        report = ModelEvaluator.create_summary_report(
            results["lda"],
            results["bertopic"]
        )

        report_path = self.output_dir / "COMPARISON_REPORT.txt"

        with open(report_path, "w", encoding="utf-8") as file:
            file.write(report)

        print(report)

        return results

    def generate_visualizations(self):
        """Generate comparison visualizations."""
        print("\n" + "=" * 80)
        print("STEP 8: GENERATING VISUALIZATIONS")
        print("=" * 80)

        if (
            self.lda_sentiment is None
            or self.bertopic_sentiment is None
        ):
            raise ValueError(
                "Both models must be analyzed first."
            )

        fig = ModelEvaluator.visualize_comparison(
            self.lda_sentiment["aspect_sentiment"],
            self.bertopic_sentiment["aspect_sentiment"]
        )

        fig.savefig(
            self.output_dir / "comparison_sentiment.png",
            dpi=150,
            bbox_inches="tight"
        )

        print("Saved comparison visualization")

        fig_lda = AspectSentimentAnalyzer.visualize_negative_drivers(
            self.lda_sentiment["negative_drivers"],
            title="LDA - Drivers of Negative Reviews"
        )

        fig_lda.savefig(
            self.output_dir / "lda_negative_drivers.png",
            dpi=150,
            bbox_inches="tight"
        )

        fig_bert = AspectSentimentAnalyzer.visualize_negative_drivers(
            self.bertopic_sentiment["negative_drivers"],
            title="BERTopic - Drivers of Negative Reviews"
        )

        fig_bert.savefig(
            self.output_dir / "bertopic_negative_drivers.png",
            dpi=150,
            bbox_inches="tight"
        )

        print("Saved negative drivers visualizations")

    def run_full_pipeline(
        self,
        filepath: str = None,
        use_demo: bool = True,
        num_topics: int = 10,
        nrows: int = None,
        sample_per_rating: int = None,
        optimize_lda: bool = False
    ):
        """
        Run the complete pipeline end-to-end.

        If optimize_lda is True, evaluate both LDA and BERTopic
        over the same topic-count range and select the best count
        for each model using U_Mass coherence.
        """
        try:
            # Step 1: Load data.
            self.load_data(
                filepath=filepath,
                use_demo=use_demo,
                nrows=nrows,
                sample_per_rating=sample_per_rating
            )

            # Step 2: Preprocess data.
            self.preprocess_data()

            # Step 3: Optionally compare topic counts.
            if optimize_lda:
                comparison_df = self.compare_topic_counts()

                valid_lda = comparison_df.dropna(
                    subset=["lda_coherence"]
                )

                valid_bertopic = comparison_df.dropna(
                    subset=["bertopic_coherence"]
                )

                lda_topic_count = int(
                    valid_lda.loc[
                        valid_lda["lda_coherence"].idxmax(),
                        "num_topics"
                    ]
                )

                bertopic_topic_count = int(
                    valid_bertopic.loc[
                        valid_bertopic["bertopic_coherence"].idxmax(),
                        "num_topics"
                    ]
                )
            else:
                lda_topic_count = num_topics
                bertopic_topic_count = num_topics

            # Step 4: Train final models.
            self.train_lda(num_topics=lda_topic_count)

            self.train_bertopic(
                num_topics=bertopic_topic_count
            )

            # Step 5-6: Analyze sentiment.
            self.analyze_sentiment_lda()
            self.analyze_sentiment_bertopic()

            # Step 7: Evaluate and compare.
            self.evaluate_and_compare()

            # Step 8: Generate visualizations.
            self.generate_visualizations()

            print("\n" + "=" * 80)
            print("PIPELINE COMPLETE!")
            print(f"Results saved to: {self.output_dir}")
            print("=" * 80)

        except Exception as e:
            print(f"\nError during pipeline execution: {e}")

            import traceback
            traceback.print_exc()

            raise


if __name__ == "__main__":
    pipeline = AspectSentimentPipeline(output_dir="results")

    # Run with demo data.
    pipeline.run_full_pipeline(
        use_demo=True,
        num_topics=5,
        optimize_lda=False
    )

    # Or use a real dataset (adjust the path):
    #
    # pipeline.run_full_pipeline(
    #     filepath="path/to/amazon_reviews.csv",
    #     num_topics=10,
    #     nrows=5000,
    #     sample_per_rating=500,
    #     optimize_lda=True
    # )
