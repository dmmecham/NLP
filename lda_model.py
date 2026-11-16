
"""
LDA topic modeling baseline implementation.

Uses Gensim to train Latent Dirichlet Allocation models on
bag-of-words representations of preprocessed Amazon reviews.
"""

from typing import List, Dict, Tuple, Optional, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from gensim import corpora, models
from gensim.models import CoherenceModel

import logging

logging.basicConfig(level=logging.WARNING)


class LDATopicModel:
    """Latent Dirichlet Allocation topic model for product reviews."""

    def __init__(self, workers: int = 4, random_state: int = 42):
        self.dictionary = None
        self.corpus = None
        self.model = None
        self.coherence_scores: Dict[int, float] = {}
        self.id2word = None

        self.workers = workers
        self.random_state = random_state

        # Keep track of which documents produced the current dictionary
        # and corpus. This prevents accidentally reusing stale data.
        self._prepared_reviews = None

    @staticmethod
    def _validate_reviews(
        tokenized_reviews: Sequence[Sequence[str]]
    ) -> List[List[str]]:
        """Validate and normalize a collection of tokenized documents."""
        if tokenized_reviews is None:
            raise ValueError("tokenized_reviews cannot be None.")

        reviews = [
            [str(token) for token in review if str(token).strip()]
            for review in tokenized_reviews
        ]

        if not reviews:
            raise ValueError("No reviews were provided.")

        if not any(reviews):
            raise ValueError(
                "All reviews are empty after tokenization. "
                "Check the preprocessing output."
            )

        return reviews

    def _prepare_if_needed(
        self,
        tokenized_reviews: Sequence[Sequence[str]]
    ) -> None:
        """
        Create the dictionary and corpus if the input documents have changed.
        """
        reviews = self._validate_reviews(tokenized_reviews)

        if (
            self._prepared_reviews is None
            or reviews != self._prepared_reviews
            or self.dictionary is None
            or self.corpus is None
        ):
            self.create_dictionary_and_corpus(reviews)

    def create_dictionary_and_corpus(
        self,
        tokenized_reviews: List[List[str]]
    ):
        """
        Create a Gensim dictionary and bag-of-words document corpus.

        Words appearing in fewer than two documents or in more than 95%
        of documents are filtered out.
        """
        reviews = self._validate_reviews(tokenized_reviews)

        self.dictionary = corpora.Dictionary(reviews)

        self.dictionary.filter_extremes(
            no_below=2,
            no_above=0.95,
            keep_n=100000
        )

        if len(self.dictionary) == 0:
            raise ValueError(
                "The LDA dictionary is empty after filtering. "
                "Try relaxing no_below/no_above or check the tokenized data."
            )

        self.corpus = [
            self.dictionary.doc2bow(review)
            for review in reviews
        ]

        self.id2word = self.dictionary
        self._prepared_reviews = reviews

        print(f"Dictionary created: {len(self.dictionary)} terms")
        print(f"Corpus created: {len(self.corpus)} documents")

        empty_documents = sum(1 for document in self.corpus if not document)
        if empty_documents:
            print(
                f"Warning: {empty_documents} documents contain no "
                "words remaining in the filtered dictionary."
            )

    def train(
        self,
        tokenized_reviews: List[List[str]],
        num_topics: int = 10,
        passes: int = 10
    ):
        """Train an LDA model using a bag-of-words corpus."""
        if not isinstance(num_topics, int) or num_topics < 2:
            raise ValueError("num_topics must be an integer of at least 2.")

        if not isinstance(passes, int) or passes < 1:
            raise ValueError("passes must be a positive integer.")

        reviews = self._validate_reviews(tokenized_reviews)
        self._prepare_if_needed(reviews)

        print(f"Training LDA with {num_topics} topics...")

        self.model = models.LdaMulticore(
            corpus=self.corpus,
            id2word=self.id2word,
            num_topics=num_topics,
            random_state=self.random_state,
            passes=passes,
            workers=self.workers,
            minimum_probability=0.0,
            per_word_topics=True
        )

        print("LDA training complete!")
        return self.model

    def calculate_coherence(self) -> float:
        """Calculate U_Mass coherence for the currently trained model."""
        if self.model is None:
            raise ValueError("Model not trained yet.")

        if self.corpus is None or self.dictionary is None:
            raise ValueError(
                "The training corpus and dictionary are unavailable. "
                "Recreate them using create_dictionary_and_corpus()."
            )

        coherence_model = CoherenceModel(
            model=self.model,
            corpus=self.corpus,
            dictionary=self.dictionary,
            coherence="u_mass"
        )

        score = coherence_model.get_coherence()

        print(f"Coherence Score (U_Mass): {score:.4f}")
        return score

    def optimize_topics(
        self,
        tokenized_reviews: List[List[str]],
        topic_range: range = range(5, 21, 2),
        passes: int = 10
    ) -> Dict[int, float]:
        """
        Find candidate topic counts by comparing U_Mass coherence.

        A higher (less negative) U_Mass score is preferred.

        Returns:
            Dictionary mapping each tested topic count to its score.
        """
        reviews = self._validate_reviews(tokenized_reviews)
        topic_counts = list(topic_range)

        if not topic_counts:
            raise ValueError("topic_range must contain at least one value.")

        if any(
            not isinstance(count, int) or count < 2
            for count in topic_counts
        ):
            raise ValueError(
                "Every topic count must be an integer of at least 2."
            )

        if len(set(topic_counts)) != len(topic_counts):
            raise ValueError("topic_range must not contain duplicate counts.")

        if any(count > len(reviews) for count in topic_counts):
            raise ValueError(
                "A requested topic count exceeds the number of reviews."
            )

        if passes < 1:
            raise ValueError("passes must be a positive integer.")

        self._prepare_if_needed(reviews)

        # Reset results so repeated optimization runs do not mix scores.
        self.coherence_scores = {}

        print(
            "Optimizing topic count in range "
            f"{min(topic_counts)} to {max(topic_counts)}..."
        )

        for num_topics in topic_counts:
            print(f"  Testing {num_topics} topics...")

            candidate_model = models.LdaMulticore(
                corpus=self.corpus,
                id2word=self.id2word,
                num_topics=num_topics,
                random_state=self.random_state,
                passes=passes,
                workers=self.workers,
                minimum_probability=0.0
            )

            coherence_model = CoherenceModel(
                model=candidate_model,
                corpus=self.corpus,
                dictionary=self.dictionary,
                coherence="u_mass"
            )

            score = coherence_model.get_coherence()
            self.coherence_scores[num_topics] = score

            print(f"    {num_topics} topics: {score:.4f}")

        best_topics = max(
            self.coherence_scores,
            key=self.coherence_scores.get
        )

        print(
            f"Optimal topic count: {best_topics} "
            f"(score: {self.coherence_scores[best_topics]:.4f})"
        )

        return self.coherence_scores

    def get_topics(self, num_words: int = 10) -> List[Dict]:
        """Return each topic's most important words and their weights."""
        if self.model is None:
            raise ValueError("Model not trained yet.")

        if num_words < 1:
            raise ValueError("num_words must be at least 1.")

        topics = []

        for topic_id, topic_terms in self.model.show_topics(
            num_topics=-1,
            num_words=num_words,
            formatted=False
        ):
            topics.append({
                "topic_id": topic_id,
                "terms": [
                    (word, float(weight))
                    for word, weight in topic_terms
                ]
            })

        return topics

    def get_document_topics(
        self,
        tokenized_text: List[str]
    ) -> List[Tuple[int, float]]:
        """Return the topic distribution for one tokenized document."""
        if self.model is None:
            raise ValueError("Model not trained yet.")

        if self.dictionary is None:
            raise ValueError("LDA dictionary is unavailable.")

        bow = self.dictionary.doc2bow(tokenized_text)

        return self.model.get_document_topics(
            bow,
            minimum_probability=0.0
        )

    def get_review_topic_distribution(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Return topic distributions for every review in a DataFrame.

        Retains review IDs, ratings, and original review text where available.
        Keeping the original text allows the evaluation module to select
        representative reviews for manual topic inspection.
        """
        if self.model is None:
            raise ValueError("Model not trained yet.")

        if "tokens" not in df.columns:
            raise ValueError(
                "Input DataFrame must contain a 'tokens' column."
            )

        results = []

        for idx, row in df.iterrows():
            doc_topics = self.get_document_topics(row["tokens"])

            doc_dict = {
                f"topic_{topic_id}": float(weight)
                for topic_id, weight in doc_topics
            }

            doc_dict["review_id"] = idx

            # Preserve either common rating-column name.
            doc_dict["rating"] = row.get(
                "rating",
                row.get("star_rating", np.nan)
            )

            # Preserve original text for the manual inspection worksheet.
            if "text" in df.columns:
                doc_dict["text"] = row["text"]
            elif "review_text" in df.columns:
                doc_dict["text"] = row["review_text"]

            results.append(doc_dict)

        result_df = pd.DataFrame(results)

        if result_df.empty:
            return result_df

        # Fill missing topic weights with zero, but do not overwrite
        # missing review text or rating values.
        topic_columns = [
            column for column in result_df.columns
            if column.startswith("topic_")
        ]

        result_df[topic_columns] = result_df[topic_columns].fillna(0)

        return result_df

    def save_model(self, path: str):
        """Save the trained Gensim model to disk."""
        if self.model is None:
            raise ValueError("No trained model to save.")

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)

        self.model.save(str(destination))
        print(f"Model saved to {destination}")

    def load_model(self, path: str):
        """
        Load a saved Gensim model.

        The model's dictionary is restored from its id2word attribute.
        The original training corpus is not automatically restored, so
        calculate_coherence() requires the corpus to be supplied again.
        """
        self.model = models.LdaMulticore.load(path)

        self.id2word = self.model.id2word
        self.dictionary = self.id2word

        # Gensim model files do not necessarily contain the original corpus.
        self.corpus = None
        self._prepared_reviews = None

        print(f"Model loaded from {path}")
        return self.model
