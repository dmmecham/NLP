
"""
Model evaluation and comparison utilities.

Compares LDA and BERTopic using:
- Topic coherence scores across topic counts
- Automated semantic-keyword grouping (supplementary heuristic)
- Sentiment consistency and negative-rating drivers
- One-star review aspect percentages
- Human topic-inspection worksheets with representative reviews

Important:
The semantic grouping score is an automated keyword heuristic. It is
not a substitute for manual inspection. The worksheet generator prepares
topics and representative reviews for a human evaluator to inspect.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class ModelEvaluator:
    """Evaluates and compares LDA and BERTopic models."""

    @staticmethod
    def _fmt_num(value: Any, decimals: int = 4) -> str:
        """Safely format numeric values, returning N/A for missing values."""
        if value is None or isinstance(value, bool):
            return "N/A"

        try:
            number = float(value)
            if not np.isfinite(number):
                return "N/A"
            return f"{number:.{decimals}f}"
        except (ValueError, TypeError):
            return "N/A"

    @staticmethod
    def _topic_terms_to_text(terms: Any) -> str:
        """Convert a topic's keyword representation into readable text."""
        if terms is None:
            return ""

        formatted = []
        for item in terms:
            if isinstance(item, (tuple, list)) and len(item) >= 1:
                word = str(item[0])
                weight = item[1] if len(item) > 1 else None
                if weight is not None:
                    formatted.append(f"{word} ({float(weight):.4f})")
                else:
                    formatted.append(word)
            else:
                formatted.append(str(item))

        return ", ".join(formatted)

    @staticmethod
    def compare_coherence_scores(
        lda_scores: Dict[int, float],
        bertopic_scores: Optional[Dict[int, float]] = None
    ) -> pd.DataFrame:
        """
        Compare LDA and BERTopic coherence scores at matching topic counts.

        Args:
            lda_scores: Mapping of topic count to LDA coherence.
            bertopic_scores: Mapping of topic count to BERTopic coherence.

        Returns:
            DataFrame with one row per topic count appearing in either model.

        Note:
            The scores are only directly comparable if both models use the
            same coherence metric and compatible tokenization/corpus inputs.
        """
        lda_scores = lda_scores or {}
        bertopic_scores = bertopic_scores or {}

        topic_counts = sorted(set(lda_scores) | set(bertopic_scores))

        rows = []
        for count in topic_counts:
            lda_score = lda_scores.get(count)
            bertopic_score = bertopic_scores.get(count)

            rows.append({
                "num_topics": count,
                "lda_coherence": lda_score,
                "bertopic_coherence": bertopic_score,
                "difference_bertopic_minus_lda": (
                    bertopic_score - lda_score
                    if pd.notna(bertopic_score) and pd.notna(lda_score)
                    else np.nan
                )
            })

        return pd.DataFrame(rows)

    @staticmethod
    def evaluate_topic_semantic_grouping(
        lda_topics: List[Dict],
        bertopic_topics: Dict[int, List[Tuple[str, float]]]
    ) -> Dict:
        """
        Calculate a supplementary keyword-based semantic grouping score.

        This checks whether predefined related words occur together in topics.
        It is not a human qualitative evaluation and should not be described
        as one in the project report.
        """
        semantic_groups = {
            "display": [
                "screen", "display", "glass", "pixel", "brightness"
            ],
            "battery": [
                "battery", "charge", "charging", "drain", "power"
            ],
            "build": [
                "build", "quality", "material", "construction", "durability"
            ],
            "performance": [
                "fast", "slow", "lag", "speed", "processor", "cpu"
            ],
            "camera": [
                "camera", "photo", "image", "lens", "video"
            ]
        }

        valid_bertopic_topics = [
            terms
            for topic_id, terms in (bertopic_topics or {}).items()
            if topic_id != -1
        ]

        lda_term_lists = [
            topic.get("terms", [])
            for topic in (lda_topics or [])
        ]

        return {
            "lda_grouping_score": ModelEvaluator._score_grouping(
                lda_term_lists, semantic_groups
            ),
            "bertopic_grouping_score": ModelEvaluator._score_grouping(
                valid_bertopic_topics, semantic_groups
            ),
            "interpretation": (
                "Automated exact-keyword heuristic only; manual topic "
                "inspection is still required."
            )
        }

    @staticmethod
    def _score_grouping(
        topics: List[List[Tuple[str, float]]],
        semantic_groups: Dict[str, List[str]]
    ) -> float:
        """Calculate the average match proportion for detected word groups."""
        total_score = 0.0
        groups_found = 0

        for topic in topics:
            words = set()

            for item in topic:
                if isinstance(item, (tuple, list)) and item:
                    words.add(str(item[0]).lower())
                else:
                    words.add(str(item).lower())

            for related_words in semantic_groups.values():
                matches = words.intersection(related_words)

                if len(matches) > 1:
                    total_score += len(matches) / len(related_words)
                    groups_found += 1

        if groups_found == 0:
            return 0.0

        return total_score / groups_found

    @staticmethod
    def evaluate_sentiment_consistency(
        lda_sentiment_df: pd.DataFrame,
        bertopic_sentiment_df: pd.DataFrame
    ) -> Dict:
        """Compare high-level sentiment statistics for the two models."""
        required = {"negative_sentiment_score", "avg_rating"}

        for name, frame in (
            ("LDA", lda_sentiment_df),
            ("BERTopic", bertopic_sentiment_df)
        ):
            if not isinstance(frame, pd.DataFrame):
                raise TypeError(f"{name} sentiment results must be a DataFrame.")
            missing = required - set(frame.columns)
            if missing:
                raise ValueError(
                    f"{name} sentiment results are missing columns: "
                    f"{sorted(missing)}"
                )

        return {
            "lda_avg_neg_sentiment":
                lda_sentiment_df["negative_sentiment_score"].mean(),
            "bertopic_avg_neg_sentiment":
                bertopic_sentiment_df["negative_sentiment_score"].mean(),
            "lda_topics_with_low_rating":
                int((lda_sentiment_df["avg_rating"] < 3).sum()),
            "bertopic_topics_with_low_rating":
                int((bertopic_sentiment_df["avg_rating"] < 3).sum())
        }

    @staticmethod
    def build_topic_inspection_worksheet(
        model_name: str,
        topics: Any,
        topic_distribution_df: pd.DataFrame,
        review_text_column: str = "text",
        top_n_reviews: int = 5
    ) -> pd.DataFrame:
        """
        Build a worksheet for human inspection of model topics.

        Each topic includes its keywords, prevalence, and the most strongly
        associated representative reviews. Human annotation columns are
        intentionally left blank.

        Args:
            model_name: A label such as "LDA" or "BERTopic".
            topics: LDA list of topic dictionaries or BERTopic topic dictionary.
            topic_distribution_df: Review-level topic distribution DataFrame.
            review_text_column: Column containing original review text.
            top_n_reviews: Number of representative reviews per topic.

        Returns:
            One row per representative review, with blank human annotation
            columns ready to be completed manually.
        """
        if not isinstance(topic_distribution_df, pd.DataFrame):
            raise TypeError("topic_distribution_df must be a DataFrame.")

        if review_text_column not in topic_distribution_df.columns:
            raise ValueError(
                f"Review text column '{review_text_column}' was not found. "
                "Pass a DataFrame containing the original review text."
            )

        if top_n_reviews < 1:
            raise ValueError("top_n_reviews must be at least 1.")

        # Normalize both model topic formats into {topic_id: keyword list}.
        normalized_topics = {}

        if isinstance(topics, dict):
            for topic_id, terms in topics.items():
                if topic_id == -1:
                    continue
                normalized_topics[topic_id] = terms or []

        elif isinstance(topics, list):
            for topic in topics:
                if not isinstance(topic, dict):
                    continue
                topic_id = topic.get("topic_id")
                if topic_id is not None:
                    normalized_topics[topic_id] = topic.get("terms", [])

        else:
            raise TypeError(
                "topics must be a BERTopic dictionary or an LDA topic list."
            )

        rows = []

        for topic_id, terms in sorted(
            normalized_topics.items(), key=lambda item: str(item[0])
        ):
            column = f"topic_{topic_id}"

            if column not in topic_distribution_df.columns:
                continue

            scores = pd.to_numeric(
                topic_distribution_df[column], errors="coerce"
            ).fillna(0)

            associated = topic_distribution_df.loc[scores > 0].copy()
            associated_scores = scores.loc[scores > 0]

            if associated.empty:
                continue

            # Select the most representative reviews by topic weight.
            selected_indices = associated_scores.nlargest(
                top_n_reviews
            ).index

            prevalence = float((scores > 0).mean() * 100)
            keyword_text = ModelEvaluator._topic_terms_to_text(terms)

            for rank, index in enumerate(selected_indices, start=1):
                review = associated.loc[index, review_text_column]

                rating = (
                    associated.loc[index, "rating"]
                    if "rating" in associated.columns
                    else np.nan
                )

                rows.append({
                    "model": model_name,
                    "topic_id": topic_id,
                    "keywords": keyword_text,
                    "topic_prevalence_pct": prevalence,
                    "representative_review_rank": rank,
                    "topic_probability_or_weight": float(scores.loc[index]),
                    "rating": rating,
                    "representative_review": review,
                    "human_topic_label": "",
                    "human_interpretation": "",
                    "coherence_observation": "",
                    "is_topic_coherent": ""
                })

        columns = [
            "model",
            "topic_id",
            "keywords",
            "topic_prevalence_pct",
            "representative_review_rank",
            "topic_probability_or_weight",
            "rating",
            "representative_review",
            "human_topic_label",
            "human_interpretation",
            "coherence_observation",
            "is_topic_coherent"
        ]

        return pd.DataFrame(rows, columns=columns)

    @staticmethod
    def create_summary_report(
        lda_results: Dict,
        bertopic_results: Dict,
        output_path: Optional[str] = None
    ) -> str:
        """
        Create a results-driven comparison report.

        The method accepts the result dictionaries produced by the project
        pipeline. Missing optional metrics are reported as N/A rather than
        replaced with invented findings.
        """
        fmt = ModelEvaluator._fmt_num
        report = []

        def add_section(title: str):
            report.append(title)
            report.append("-" * 80)

        def append_one_star_results(label: str, frame: Any):
            report.append(f"{label} - One-star review aspect percentages:")

            if not isinstance(frame, pd.DataFrame) or frame.empty:
                report.append("  Not available.")
                return

            required = {"topic_id", "one_star_percentage"}
            if not required.issubset(frame.columns):
                report.append(
                    "  DataFrame does not contain the expected "
                    "'topic_id' and 'one_star_percentage' columns."
                )
                return

            for _, row in frame.sort_values(
                "one_star_percentage", ascending=False
            ).iterrows():
                report.append(
                    f"  Topic {row['topic_id']}: "
                    f"{fmt(row['one_star_percentage'], 2)}%"
                )

        report.extend([
            "=" * 80,
            "ASPECT-BASED TOPIC MODELING AND SENTIMENT ANALYSIS",
            "MODEL COMPARISON REPORT",
            "=" * 80,
            ""
        ])

        # Dataset statistics
        add_section("DATASET STATISTICS")
        stats = lda_results.get("dataset_stats")

        if isinstance(stats, dict):
            report.append(
                f"Total Reviews: {stats.get('total_reviews', 'N/A')}"
            )
            report.append(
                f"Average Rating: {fmt(stats.get('avg_rating'), 2)}"
            )
            report.append(
                "Average Text Length: "
                f"{fmt(stats.get('avg_text_length'), 0)} chars"
            )
        else:
            report.append("Dataset statistics: N/A")

        report.append("")

        # Model configuration and coherence
        add_section("TOPIC MODELING RESULTS")

        lda_topic_count = lda_results.get(
            "optimal_topics", lda_results.get("num_topics")
        )
        lda_coherence = lda_results.get(
            "best_coherence", lda_results.get("lda_coherence")
        )

        bertopic_topic_count = bertopic_results.get("num_topics")
        bertopic_coherence = bertopic_results.get(
            "best_coherence", bertopic_results.get("bertopic_coherence")
        )

        report.append("LDA:")
        report.append(f"  Topic Count: {lda_topic_count or 'N/A'}")
        report.append(f"  Coherence: {fmt(lda_coherence)}")

        report.append("BERTopic:")
        report.append(f"  Topic Count: {bertopic_topic_count or 'N/A'}")
        report.append(f"  Coherence: {fmt(bertopic_coherence)}")

        if (
            lda_coherence is not None
            and bertopic_coherence is not None
            and np.isfinite(float(lda_coherence))
            and np.isfinite(float(bertopic_coherence))
        ):
            report.append(
                "  BERTopic minus LDA coherence: "
                f"{fmt(float(bertopic_coherence) - float(lda_coherence))}"
            )

        report.append("")

        # Topic-count sweep, if supplied by the pipeline
        coherence_comparison = lda_results.get(
            "coherence_comparison",
            bertopic_results.get("coherence_comparison")
        )

        add_section("COHERENCE COMPARISON ACROSS TOPIC COUNTS")

        if (
            isinstance(coherence_comparison, pd.DataFrame)
            and not coherence_comparison.empty
        ):
            expected = {
                "num_topics", "lda_coherence", "bertopic_coherence"
            }

            if expected.issubset(coherence_comparison.columns):
                report.append(
                    "Topic count | LDA coherence | BERTopic coherence"
                )
                for _, row in coherence_comparison.sort_values(
                    "num_topics"
                ).iterrows():
                    report.append(
                        f"{row['num_topics']:>11} | "
                        f"{fmt(row['lda_coherence']):>13} | "
                        f"{fmt(row['bertopic_coherence']):>18}"
                    )
            else:
                report.append(
                    "A coherence comparison was supplied, but it is missing "
                    "one or more expected columns."
                )
        else:
            report.append(
                "No topic-count comparison table was supplied."
            )

        report.append(
            "Note: Interpret cross-model coherence comparisons cautiously "
            "if the models use different tokenization or vocabulary coverage."
        )
        report.append("")

        # Automated semantic grouping
        add_section("AUTOMATED SEMANTIC GROUPING (SUPPLEMENTARY)")

        grouping = lda_results.get(
            "semantic_grouping",
            bertopic_results.get("semantic_grouping")
        )

        if isinstance(grouping, dict):
            report.append(
                f"LDA keyword grouping score: "
                f"{fmt(grouping.get('lda_grouping_score'))}"
            )
            report.append(
                f"BERTopic keyword grouping score: "
                f"{fmt(grouping.get('bertopic_grouping_score'))}"
            )
        else:
            report.append("Automated semantic grouping scores: N/A")

        report.append(
            "This is an exact-keyword heuristic, not a human qualitative "
            "evaluation. Inspect the generated topic worksheet and record "
            "human labels and observations separately."
        )
        report.append("")

        # Negative drivers
        add_section("NEGATIVE REVIEW DRIVERS")

        for label, results in (
            ("LDA", lda_results),
            ("BERTopic", bertopic_results)
        ):
            drivers = results.get("negative_drivers")
            report.append(f"{label}:")

            if isinstance(drivers, pd.DataFrame) and not drivers.empty:
                for _, driver in drivers.head(5).iterrows():
                    report.append(
                        f"  Topic {driver.get('topic_id', 'N/A')}: "
                        f"{driver.get('keywords', 'N/A')}"
                    )
                    report.append(
                        f"    Impact score: "
                        f"{fmt(driver.get('impact_score'))}"
                    )
                    if "pct_negative_reviews" in driver.index:
                        report.append(
                            "    Percentage of negative reviews: "
                            f"{fmt(driver.get('pct_negative_reviews'), 2)}%"
                        )
            else:
                report.append("  No negative-driver results available.")

        report.append("")

        # One-star aspect percentages
        add_section("ONE-STAR REVIEW ASPECT PERCENTAGES")
        append_one_star_results(
            "LDA", lda_results.get("one_star_percentages")
        )
        report.append("")
        append_one_star_results(
            "BERTopic", bertopic_results.get("one_star_percentages")
        )
        report.append(
            "Note: LDA topic memberships can overlap, so its percentages "
            "may sum to more than 100%. BERTopic assignments may be exclusive "
            "when using hard clustering."
        )
        report.append("")

        # Sentiment summary
        add_section("SENTIMENT SUMMARY")

        for label, results in (
            ("LDA", lda_results),
            ("BERTopic", bertopic_results)
        ):
            sentiment = results.get("aspect_sentiment")

            report.append(f"{label}:")
            if isinstance(sentiment, pd.DataFrame) and not sentiment.empty:
                if "avg_rating" in sentiment.columns:
                    report.append(
                        f"  Mean topic rating: "
                        f"{fmt(sentiment['avg_rating'].mean(), 2)}"
                    )
                if "negative_sentiment_score" in sentiment.columns:
                    report.append(
                        "  Mean negative sentiment score: "
                        f"{fmt(sentiment['negative_sentiment_score'].mean())}"
                    )
            else:
                report.append("  Aspect sentiment results: N/A")

        report.append("")
        report.append("=" * 80)

        report_text = "\n".join(report)

        if output_path:
            destination = Path(output_path)
            destination.parent.mkdir(parents=True, exist_ok=True)

            with destination.open("w", encoding="utf-8") as file:
                file.write(report_text)

            print(f"Report saved to {destination}")

        return report_text

    @staticmethod
    def visualize_comparison(
        lda_sentiment_df: pd.DataFrame,
        bertopic_sentiment_df: pd.DataFrame,
        title: str = "LDA vs BERTopic - Sentiment Analysis"
    ):
        """Create a side-by-side visualization of average topic ratings."""
        required = {"topic_id", "avg_rating"}

        for name, frame in (
            ("LDA", lda_sentiment_df),
            ("BERTopic", bertopic_sentiment_df)
        ):
            if not isinstance(frame, pd.DataFrame):
                raise TypeError(f"{name} sentiment results must be a DataFrame.")
            missing = required - set(frame.columns)
            if missing:
                raise ValueError(
                    f"{name} sentiment results are missing columns: "
                    f"{sorted(missing)}"
                )

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for ax, sentiment, model_name in (
            (axes[0], lda_sentiment_df, "LDA"),
            (axes[1], bertopic_sentiment_df, "BERTopic")
        ):
            ratings = pd.to_numeric(
                sentiment["avg_rating"], errors="coerce"
            ).fillna(0)

            ax.bar(
                sentiment["topic_id"].astype(str),
                ratings,
                color=plt.cm.RdYlGn(ratings / 5.0)
            )
            ax.axhline(y=3, color="gray", linestyle="--", alpha=0.5)
            ax.set_ylabel("Average Rating")
            ax.set_xlabel("Topic ID")
            ax.set_ylim([0, 5])
            ax.set_title(f"{model_name} Model")

        fig.suptitle(title, fontsize=14, fontweight="bold")
        fig.tight_layout()
        return fig
