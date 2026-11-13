"""
Main pipeline orchestrator.
Runs complete end-to-end analysis with both LDA and BERTopic.
"""

import os
import sys
from typing import Dict, Tuple
import pandas as pd
import numpy as np
from pathlib import Path

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
        self.output_dir.mkdir(exist_ok=True)
        
        self.preprocessor = TextPreprocessor()
        self.lda_model = LDATopicModel()
        self.bertopic_model = BERTopicModel()
        
        self.df = None
        self.lda_topics = None
        self.lda_coherence = None
        self.bertopic_topics = None
        self.lda_sentiment = None
        self.bertopic_sentiment = None

    def load_data(self, filepath: str = None, use_demo: bool = False, 
                  nrows: int = None, sample_per_rating: int = None) -> pd.DataFrame:
        """Load and preprocess data."""
        print("\n" + "="*80)
        print("STEP 1: LOADING DATA")
        print("="*80)
        
        if use_demo:
            print("Creating demo dataset...")
            self.df = AmazonReviewsDataLoader.create_demo_dataset(num_samples=100)
        elif filepath:
            print(f"Loading reviews from {filepath}...")
            self.df = AmazonReviewsDataLoader.load_from_csv(filepath, nrows=nrows)
            if sample_per_rating:
                self.df = AmazonReviewsDataLoader.sample_by_rating(
                    self.df, 
                    samples_per_rating=sample_per_rating
                )
        else:
            raise ValueError("Either filepath or use_demo must be provided")
        
        stats = AmazonReviewsDataLoader.get_review_statistics(self.df)
        print(f"\nDataset Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        return self.df

    def preprocess_data(self) -> pd.DataFrame:
        """Preprocess reviews for both LDA and BERTopic."""
        print("\n" + "="*80)
        print("STEP 2: PREPROCESSING DATA")
        print("="*80)
        
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        print("Preprocessing reviews...")
        self.df = self.preprocessor.process_reviews_df(self.df, text_column='text')
        
        print(f"Sample preprocessed review:")
        idx = 0
        print(f"  Original: {self.df.iloc[idx]['text'][:100]}...")
        print(f"  Tokens: {self.df.iloc[idx]['tokens']}")
        print(f"  Cleaned: {self.df.iloc[idx]['text_cleaned'][:100]}...")
        
        return self.df

    def train_lda(self, num_topics: int = 10, optimize_topics: bool = False) -> Tuple[LDATopicModel, pd.DataFrame]:
        """Train LDA model."""
        print("\n" + "="*80)
        print("STEP 3: TRAINING LDA MODEL (Classical Stream)")
        print("="*80)
        
        if self.df is None:
            raise ValueError("Data not preprocessed. Call preprocess_data() first.")
        
        if optimize_topics:
            print("Optimizing topic count...")
            coherence_scores = self.lda_model.optimize_topics(
                self.df['tokens'].tolist(),
                topic_range=range(5, 21, 2)
            )
            
            # Save coherence scores
            coherence_df = pd.DataFrame(list(coherence_scores.items()), 
                                       columns=['num_topics', 'coherence_score'])
            coherence_df.to_csv(self.output_dir / 'lda_coherence_scores.csv', index=False)
            
            num_topics = max(coherence_scores, key=coherence_scores.get)
            print(f"Optimal topics selected: {num_topics}")
        
        self.lda_model.train(self.df['tokens'].tolist(), num_topics=num_topics)
        
        # Get coherence score
        self.lda_coherence = self.lda_model.calculate_coherence()
        
        # Get topics
        self.lda_topics = self.lda_model.get_topics(num_words=10)
        print(f"\nTop 3 topics from LDA:")
        for i, topic in enumerate(self.lda_topics[:3]):
            print(f"\n  Topic {topic['topic_id']}:")
            for word, weight in topic['terms'][:5]:
                print(f"    {word}: {weight:.4f}")
        
        return self.lda_model, self.lda_topics

    def train_bertopic(self, num_topics: int = 10) -> Tuple[BERTopicModel, Dict]:
        """Train BERTopic model."""
        print("\n" + "="*80)
        print("STEP 4: TRAINING BERTOPIC MODEL (Contextual Stream)")
        print("="*80)
        
        if self.df is None:
            raise ValueError("Data not preprocessed. Call preprocess_data() first.")
        
        # Generate embeddings
        self.bertopic_model.generate_embeddings(self.df['text_cleaned'].tolist())
        
        # Train model
        topics, probabilities = self.bertopic_model.train(
            self.df['text_cleaned'].tolist(),
            num_topics=num_topics
        )
        
        # Get topics
        self.bertopic_topics = self.bertopic_model.get_topics(top_n=10)
        print(f"\nTop 3 topics from BERTopic:")
        for i, (topic_id, terms) in enumerate(sorted(self.bertopic_topics.items())[:3]):
            print(f"\n  Topic {topic_id}:")
            for word, weight in terms[:5]:
                print(f"    {word}: {weight:.4f}")
        
        return self.bertopic_model, self.bertopic_topics

    def analyze_sentiment_lda(self) -> pd.DataFrame:
        """Analyze sentiment for LDA topics."""
        print("\n" + "="*80)
        print("STEP 5: SENTIMENT ANALYSIS - LDA")
        print("="*80)
        
        if self.lda_model.model is None:
            raise ValueError("LDA model not trained. Call train_lda() first.")
        
        # Get topic distribution for all reviews
        topic_dist = self.lda_model.get_review_topic_distribution(self.df)
        topic_dist.to_csv(self.output_dir / 'lda_topic_distribution.csv', index=False)
        
        # Compute aspect sentiment
        aspect_sentiment = AspectSentimentAnalyzer.compute_aspect_sentiment_distribution(topic_dist)
        aspect_sentiment.to_csv(self.output_dir / 'lda_aspect_sentiment.csv', index=False)
        
        # Identify negative drivers
        negative_drivers = AspectSentimentAnalyzer.identify_negative_drivers(
            topic_dist,
            aspect_sentiment,
            topic_keywords={t['topic_id']: t['terms'] for t in self.lda_topics},
            threshold_stars=2
        )
        negative_drivers.to_csv(self.output_dir / 'lda_negative_drivers.csv', index=False)
        
        print("\nTop 5 negative drivers (LDA):")
        print(negative_drivers.head()[['topic_id', 'keywords', 'impact_score', 'avg_rating']])
        
        self.lda_sentiment = {
            'aspect_sentiment': aspect_sentiment,
            'negative_drivers': negative_drivers,
            'topic_distribution': topic_dist
        }
        
        return aspect_sentiment

    def analyze_sentiment_bertopic(self) -> pd.DataFrame:
        """Analyze sentiment for BERTopic topics."""
        print("\n" + "="*80)
        print("STEP 6: SENTIMENT ANALYSIS - BERTOPIC")
        print("="*80)
        
        if self.bertopic_model.model is None:
            raise ValueError("BERTopic model not trained. Call train_bertopic() first.")
        
        # Get topic distribution for all reviews
        topic_dist = self.bertopic_model.get_review_topic_distribution(self.df)
        topic_dist.to_csv(self.output_dir / 'bertopic_topic_distribution.csv', index=False)
        
        # Compute aspect sentiment
        aspect_sentiment = AspectSentimentAnalyzer.compute_aspect_sentiment_distribution(topic_dist)
        aspect_sentiment.to_csv(self.output_dir / 'bertopic_aspect_sentiment.csv', index=False)
        
        # Identify negative drivers
        negative_drivers = AspectSentimentAnalyzer.identify_negative_drivers(
            topic_dist,
            aspect_sentiment,
            topic_keywords=self.bertopic_topics,
            threshold_stars=2
        )
        negative_drivers.to_csv(self.output_dir / 'bertopic_negative_drivers.csv', index=False)
        
        print("\nTop 5 negative drivers (BERTopic):")
        print(negative_drivers.head()[['topic_id', 'keywords', 'impact_score', 'avg_rating']])
        
        self.bertopic_sentiment = {
            'aspect_sentiment': aspect_sentiment,
            'negative_drivers': negative_drivers,
            'topic_distribution': topic_dist
        }
        
        return aspect_sentiment

    def evaluate_and_compare(self) -> Dict:
        """Compare LDA and BERTopic results."""
        print("\n" + "="*80)
        print("STEP 7: EVALUATION AND COMPARISON")
        print("="*80)
        
        if self.lda_sentiment is None or self.bertopic_sentiment is None:
            raise ValueError("Both models must be trained and analyzed. Call analyze_sentiment methods.")
        
        # Create results dict
        results = {
            'lda': {
                'aspect_sentiment': self.lda_sentiment['aspect_sentiment'],
                'negative_drivers': self.lda_sentiment['negative_drivers'],
                'dataset_stats': AmazonReviewsDataLoader.get_review_statistics(self.df),
                'optimal_topics': len(self.lda_topics),
                'best_coherence': self.lda_coherence,
            },
            'bertopic': {
                'aspect_sentiment': self.bertopic_sentiment['aspect_sentiment'],
                'negative_drivers': self.bertopic_sentiment['negative_drivers'],
                'num_topics': len(self.bertopic_topics),
            }
        }
        
        # Create comparison report
        report = ModelEvaluator.create_summary_report(results['lda'], results['bertopic'])
        report_path = self.output_dir / 'COMPARISON_REPORT.txt'
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(report)
        
        return results

    def generate_visualizations(self):
        """Generate comparison visualizations."""
        print("\n" + "="*80)
        print("STEP 8: GENERATING VISUALIZATIONS")
        print("="*80)
        
        if self.lda_sentiment is None or self.bertopic_sentiment is None:
            raise ValueError("Both models must be analyzed first.")
        
        # Topic sentiment comparison
        fig = ModelEvaluator.visualize_comparison(
            self.lda_sentiment['aspect_sentiment'],
            self.bertopic_sentiment['aspect_sentiment']
        )
        fig.savefig(self.output_dir / 'comparison_sentiment.png', dpi=150, bbox_inches='tight')
        print(f"Saved comparison visualization")
        
        # Negative drivers comparison
        fig_lda = AspectSentimentAnalyzer.visualize_negative_drivers(
            self.lda_sentiment['negative_drivers'],
            title="LDA - Drivers of Negative Reviews"
        )
        fig_lda.savefig(self.output_dir / 'lda_negative_drivers.png', dpi=150, bbox_inches='tight')
        
        fig_bert = AspectSentimentAnalyzer.visualize_negative_drivers(
            self.bertopic_sentiment['negative_drivers'],
            title="BERTopic - Drivers of Negative Reviews"
        )
        fig_bert.savefig(self.output_dir / 'bertopic_negative_drivers.png', dpi=150, bbox_inches='tight')
        print(f"Saved negative drivers visualizations")

    def run_full_pipeline(self, filepath: str = None, use_demo: bool = True,
                         num_topics: int = 10, nrows: int = None,
                         sample_per_rating: int = None, optimize_lda: bool = False):
        """Run complete pipeline end-to-end."""
        try:
            # Load data
            self.load_data(filepath=filepath, use_demo=use_demo, 
                          nrows=nrows, sample_per_rating=sample_per_rating)
            
            # Preprocess
            self.preprocess_data()
            
            # Train models
            self.train_lda(num_topics=num_topics, optimize_topics=optimize_lda)
            self.train_bertopic(num_topics=num_topics)
            
            # Analyze sentiment
            self.analyze_sentiment_lda()
            self.analyze_sentiment_bertopic()
            
            # Evaluate and compare
            self.evaluate_and_compare()
            
            # Visualizations
            self.generate_visualizations()
            
            print("\n" + "="*80)
            print("PIPELINE COMPLETE!")
            print(f"Results saved to: {self.output_dir}")
            print("="*80)
            
        except Exception as e:
            print(f"\nError during pipeline execution: {e}")
            import traceback
            traceback.print_exc()
            raise


if __name__ == "__main__":
    # Example usage
    pipeline = AspectSentimentPipeline(output_dir="results")
    
    # Run with demo data
    pipeline.run_full_pipeline(use_demo=True, num_topics=5, optimize_lda=False)
    
    # Or with real data (uncomment and adjust path):
    # pipeline.run_full_pipeline(
    #     filepath="path/to/amazon_reviews.csv",
    #     num_topics=10,
    #     nrows=5000,
    #     sample_per_rating=500,
    #     optimize_lda=True
    # )
