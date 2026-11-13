"""
Model evaluation and comparison utilities.
Compares LDA and BERTopic coherence, topic quality, and sentiment analysis results.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns


class ModelEvaluator:
    """Evaluates and compares topic modeling approaches."""

    @staticmethod
    def _fmt_num(val: Any, decimals: int = 4) -> str:
        """Safely format numeric values, falling back to 'N/A' for missing/NaN values."""
        if val is None or isinstance(val, bool):
            return "N/A"
        try:
            num = float(val)
            if np.isnan(num):
                return "N/A"
            return f"{num:.{decimals}f}"
        except (ValueError, TypeError):
            return "N/A"

    @staticmethod
    def compare_coherence_scores(lda_scores: Dict[int, float],
                                 bertopic_scores: Dict = None) -> pd.DataFrame:
        """
        Compare coherence scores between LDA and BERTopic.
        
        Args:
            lda_scores: Dict of {num_topics: coherence_score} for LDA
            bertopic_scores: Optional coherence scores for BERTopic
        
        Returns:
            DataFrame with comparison
        """
        results = []
        
        for num_topics, score in sorted(lda_scores.items()):
            results.append({
                'num_topics': num_topics,
                'lda_coherence': score,
                'bertopic_coherence': bertopic_scores.get(num_topics, None) if bertopic_scores else None
            })
        
        return pd.DataFrame(results)

    @staticmethod
    def evaluate_topic_semantic_grouping(lda_topics: List[Dict],
                                        bertopic_topics: Dict[int, List[Tuple[str, float]]]) -> Dict:
        """
        Qualitatively evaluate semantic grouping in topics.
        
        Looks for semantically related terms (e.g., "screen", "display", "glass" in same topic).
        
        Args:
            lda_topics: List of topic dicts from LDA
            bertopic_topics: Dict of topics from BERTopic
        
        Returns:
            Dict with evaluation metrics
        """
        semantic_groups = {
            'display': ['screen', 'display', 'glass', 'pixel', 'brightness'],
            'battery': ['battery', 'charge', 'charging', 'drain', 'power'],
            'build': ['build', 'quality', 'material', 'construction', 'durability'],
            'performance': ['fast', 'slow', 'lag', 'speed', 'processor', 'cpu'],
            'camera': ['camera', 'photo', 'image', 'lens', 'video']
        }
        
        # Exclude outlier topic -1 from BERTopic evaluation
        valid_bertopic_terms = [
            terms for topic_id, terms in bertopic_topics.items() if topic_id != -1
        ]
        
        evaluation = {
            'lda_grouping_score': ModelEvaluator._score_grouping(
                [t['terms'] for t in lda_topics],
                semantic_groups
            ),
            'bertopic_grouping_score': ModelEvaluator._score_grouping(
                valid_bertopic_terms,
                semantic_groups
            )
        }
        
        return evaluation

    @staticmethod
    def _score_grouping(topics: List[List[Tuple[str, float]]], 
                       semantic_groups: Dict[str, List[str]]) -> float:
        """Calculate semantic grouping score for topics."""
        total_score = 0
        total_groups_found = 0
        
        for topic in topics:
            words = [word for word, _ in topic]
            
            for group_name, related_words in semantic_groups.items():
                matching_words = [w for w in words if w in related_words]
                if len(matching_words) > 1:
                    total_score += len(matching_words) / len(related_words)
                    total_groups_found += 1
        
        if total_groups_found == 0:
            return 0.0
        
        return total_score / total_groups_found

    @staticmethod
    def evaluate_sentiment_consistency(lda_sentiment_df: pd.DataFrame,
                                      bertopic_sentiment_df: pd.DataFrame) -> Dict:
        """
        Compare sentiment analysis results from both models.
        
        Args:
            lda_sentiment_df: Aspect sentiment DataFrame from LDA
            bertopic_sentiment_df: Aspect sentiment DataFrame from BERTopic
        
        Returns:
            Dict with comparison metrics
        """
        evaluation = {
            'lda_avg_neg_sentiment': lda_sentiment_df['negative_sentiment_score'].mean(),
            'bertopic_avg_neg_sentiment': bertopic_sentiment_df['negative_sentiment_score'].mean(),
            'lda_topics_with_low_rating': len(lda_sentiment_df[lda_sentiment_df['avg_rating'] < 3]),
            'bertopic_topics_with_low_rating': len(bertopic_sentiment_df[bertopic_sentiment_df['avg_rating'] < 3]),
        }
        
        return evaluation

    @staticmethod
    def create_summary_report(lda_results: Dict,
                            bertopic_results: Dict,
                            output_path: str = None) -> str:
        """
        Create a comprehensive comparison report.
        
        Args:
            lda_results: Results dict from LDA pipeline
            bertopic_results: Results dict from BERTopic pipeline
            output_path: Optional path to save report
        
        Returns:
            Formatted report string
        """
        fmt = ModelEvaluator._fmt_num
        report = []
        report.append("=" * 80)
        report.append("ASPECT-BASED TOPIC MODELING AND SENTIMENT ANALYSIS - COMPARISON REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Dataset info
        report.append("DATASET STATISTICS")
        report.append("-" * 80)
        if 'dataset_stats' in lda_results and isinstance(lda_results['dataset_stats'], dict):
            stats = lda_results['dataset_stats']
            report.append(f"Total Reviews: {stats.get('total_reviews', 'N/A')}")
            report.append(f"Average Rating: {fmt(stats.get('avg_rating'), 2)}")
            report.append(f"Average Text Length: {fmt(stats.get('avg_text_length'), 0)} chars")
        report.append("")
        
        # LDA Results
        report.append("LDA (LATENT DIRICHLET ALLOCATION) RESULTS")
        report.append("-" * 80)
        if 'optimal_topics' in lda_results:
            report.append(f"Optimal Topic Count: {lda_results['optimal_topics']}")
            report.append(f"Coherence Score: {fmt(lda_results.get('best_coherence'), 4)}")
        report.append("")
        
        # BERTopic Results
        report.append("BERTOPIC RESULTS")
        report.append("-" * 80)
        if 'num_topics' in bertopic_results:
            report.append(f"Number of Topics: {bertopic_results['num_topics']}")
        report.append("")
        
        # Sentiment Analysis
        report.append("SENTIMENT ANALYSIS COMPARISON")
        report.append("-" * 80)
        lda_drivers = lda_results.get('negative_drivers')
        if isinstance(lda_drivers, pd.DataFrame) and not lda_drivers.empty:
            top_lda_driver = lda_drivers.iloc[0]
            topic_id = fmt(top_lda_driver.get('topic_id'), 0)
            report.append(f"LDA - Top Negative Driver: Topic {topic_id}")
            report.append(f"  Keywords: {top_lda_driver.get('keywords', 'N/A')}")
            report.append(f"  Impact Score: {fmt(top_lda_driver.get('impact_score'), 4)}")
        else:
            report.append("LDA - Top Negative Driver: N/A")
        report.append("")
        
        bert_drivers = bertopic_results.get('negative_drivers')
        if isinstance(bert_drivers, pd.DataFrame) and not bert_drivers.empty:
            top_bert_driver = bert_drivers.iloc[0]
            topic_id = fmt(top_bert_driver.get('topic_id'), 0)
            report.append(f"BERTopic - Top Negative Driver: Topic {topic_id}")
            report.append(f"  Keywords: {top_bert_driver.get('keywords', 'N/A')}")
            report.append(f"  Impact Score: {fmt(top_bert_driver.get('impact_score'), 4)}")
        else:
            report.append("BERTopic - Top Negative Driver: N/A")
        report.append("")
        
        # Key Findings
        report.append("KEY FINDINGS")
        report.append("-" * 80)
        report.append("1. BERTopic uses contextual embeddings for semantic understanding")
        report.append("2. LDA uses word frequency and co-occurrence patterns")
        report.append("3. Both approaches identify topics related to negative reviews")
        report.append("4. Semantic grouping quality differs between approaches")
        report.append("")
        
        report.append("=" * 80)
        
        report_text = "\n".join(report)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"Report saved to {output_path}")
            
        return report_text

    @staticmethod
    def visualize_comparison(lda_sentiment_df: pd.DataFrame,
                             bertopic_sentiment_df: pd.DataFrame,
                             title: str = "LDA vs BERTopic - Sentiment Analysis"):
        """Create a visualization comparing both models' sentiment results."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for ax, sentiment, model_name in (
            (axes[0], lda_sentiment_df, "LDA"),
            (axes[1], bertopic_sentiment_df, "BERTopic"),
        ):
            ratings = sentiment['avg_rating']
            ax.bar(sentiment['topic_id'].astype(str), ratings, color=plt.cm.RdYlGn(ratings / 5.0))
            ax.axhline(y=3, color='gray', linestyle='--', alpha=0.5)
            ax.set_ylabel('Average Rating')
            ax.set_xlabel('Topic ID')
            ax.set_ylim([0, 5])
            ax.set_title(f'{model_name} Model')

        fig.suptitle(title, fontsize=14, fontweight='bold')
        fig.tight_layout()
        return fig
