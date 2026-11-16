
"""
BERTopic neural topic modeling implementation.
Uses contextual embeddings to identify semantically related topics.
"""

from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from sklearn.cluster import KMeans
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
import logging

logging.basicConfig(level=logging.WARNING)


class BERTopicModel:
    """BERTopic neural topic model using transformer embeddings."""

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        language: str = "english"
    ):
        """
        Initialize BERTopic model.

        Args:
            embedding_model: Sentence transformer model to use.
            language: Language for stopword removal.
        """
        self.embedding_model_name = embedding_model
        self.language = language
        self.embeddings = None
        self.embedding_texts = None
        self.model = None
        self._sentence_model = None

    def _get_sentence_model(self):
        if self._sentence_model is None:
            device = "cuda" if torch.cuda.is_available() else None
            self._sentence_model = SentenceTransformer(
                self.embedding_model_name,
                device=device
            )
            print(f"Sentence embedding device: {self._sentence_model.device}")

        return self._sentence_model

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate sentence embeddings for texts."""
        texts = list(texts)

        print(
            f"Generating embeddings using "
            f"{self.embedding_model_name}..."
        )

        sentence_model = self._get_sentence_model()

        self.embeddings = sentence_model.encode(
            texts,
            show_progress_bar=True
        )

        self.embedding_texts = texts

        print(
            f"Generated {len(self.embeddings)} embeddings "
            f"of dimension {self.embeddings.shape[1]}"
        )

        return self.embeddings

    def train(
        self,
        texts: List[str],
        num_topics: int = 10,
        min_topic_size: int = 10
    ):
        """
        Train BERTopic model.

        Args:
            texts: List of review texts (already preprocessed).
            num_topics: Target number of topics.
            min_topic_size: Unused with KMeans, which does not
                            enforce a minimum size.

        Returns:
            Topic assignments and probabilities. KMeans does not
            produce probabilities, so the second return value is None.
        """
        texts = list(texts)

        # Generate embeddings if they are missing or belong to
        # a different set of input texts.
        if (
            self.embeddings is None
            or self.embedding_texts != texts
        ):
            self.generate_embeddings(texts)

        if num_topics > len(texts):
            raise ValueError(
                "num_topics cannot exceed the number of documents."
            )

        print(f"Training BERTopic with {num_topics} topics...")

        # Initialize clustering with KMeans for consistent topics.
        clustering_model = KMeans(
            n_clusters=num_topics,
            random_state=42,
            n_init=10
        )

        self.model = BERTopic(
            embedding_model=self._get_sentence_model(),
            language=self.language,
            hdbscan_model=clustering_model,
            verbose=True
        )

        topics, probabilities = self.model.fit_transform(
            texts,
            self.embeddings
        )

        print("BERTopic training complete!")

        return topics, probabilities

    def calculate_coherence(
        self,
        tokenized_texts: List[List[str]],
        top_n: int = 10
    ) -> float:
        """
        Calculate BERTopic coherence using U_Mass.

        BERTopic keywords are evaluated against the same
        lemmatized token corpus used by LDA.

        Keywords that do not appear in the LDA vocabulary
        are excluded from the calculation.

        Args:
            tokenized_texts: LDA-preprocessed review tokens.
            top_n: Number of top keywords per topic to evaluate.

        Returns:
            Overall U_Mass coherence score.
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        if not tokenized_texts:
            raise ValueError("No tokenized documents provided.")

        # Create the same type of vocabulary and corpus used by LDA.
        dictionary = Dictionary(tokenized_texts)
        corpus = [
            dictionary.doc2bow(text)
            for text in tokenized_texts
        ]

        if len(dictionary) == 0:
            raise ValueError(
                "The tokenized corpus has an empty vocabulary."
            )

        topic_words = []
        excluded_topics = 0

        # Extract BERTopic keywords and retain only words
        # that exist in the LDA vocabulary.
        for topic_id in sorted(set(self.model.topics_)):
            if topic_id == -1:
                continue

            keywords = self.model.get_topic(topic_id)

            if not keywords:
                excluded_topics += 1
                continue

            valid_words = [
                word
                for word, _ in keywords[:top_n]
                if word in dictionary.token2id
            ]

            # U_Mass requires meaningful topic-word overlap.
            if len(valid_words) >= 2:
                topic_words.append(valid_words)
            else:
                excluded_topics += 1

        if not topic_words:
            print(
                "Unable to calculate BERTopic coherence: "
                "no topics had enough words in the LDA vocabulary."
            )
            return np.nan

        coherence_model = CoherenceModel(
            topics=topic_words,
            corpus=corpus,
            dictionary=dictionary,
            coherence='u_mass'
        )

        score = coherence_model.get_coherence()

        print(f"BERTopic U_Mass Coherence: {score:.4f}")

        if excluded_topics > 0:
            print(
                f"Note: {excluded_topics} topics were excluded "
                "because they had insufficient vocabulary overlap."
            )

        return score

    def get_topics(
        self,
        top_n: int = 10
    ) -> Dict[int, List[Tuple[str, float]]]:
        """Get topics with top keywords."""
        if self.model is None:
            raise ValueError("Model not trained yet")

        topics_dict = {}

        for topic_id in sorted(set(self.model.topics_)):
            if topic_id != -1:
                keywords = self.model.get_topic(topic_id)

                if keywords:
                    topics_dict[topic_id] = keywords[:top_n]

        return topics_dict

    def get_topic_info(self) -> pd.DataFrame:
        """Get detailed information about all topics."""
        if self.model is None:
            raise ValueError("Model not trained yet")

        return self.model.get_topic_info()

    def get_document_topics(self, text: str) -> Dict[int, float]:
        """Get the assigned topic and its strength (1.0 for KMeans)."""
        if self.model is None:
            raise ValueError("Model not trained yet")

        topics, probabilities = self.model.transform([text])

        topic_dist = {}

        if probabilities is None:
            return {
                topic_id: 1.0
                for topic_id in topics
                if topic_id != -1
            }

        for topic_id, prob in zip(topics, probabilities):
            if topic_id != -1:
                topic_dist[topic_id] = float(prob)

        return topic_dist

    def get_review_topic_distribution(
        self,
        df: pd.DataFrame,
        text_column: str = 'text_cleaned'
    ) -> pd.DataFrame:
        """Get topic assignments for reviews; KMeans assignments are one-hot."""
        if self.model is None:
            raise ValueError("Model not trained yet")

        print("Generating topic distributions for all reviews...")

        topics, probabilities = self.model.transform(
            df[text_column].tolist()
        )

        topic_ids = sorted(set(self.model.topics_) - {-1})

        results = []

        for idx, (row, topic) in enumerate(
            zip(df.itertuples(), topics)
        ):
            if probabilities is None:
                prob = 1.0
            elif isinstance(probabilities, np.ndarray) and probabilities.ndim == 2:
                topic_col_idx = topic_ids.index(topic) if topic in topic_ids else 0
                prob = float(probabilities[idx][topic_col_idx])
            else:
                prob = float(probabilities[idx])

            doc_dict = {}

            for topic_id in topic_ids:
                if topic_id == topic:
                    doc_dict[f'topic_{topic_id}'] = prob
                else:
                    doc_dict[f'topic_{topic_id}'] = 0.0

            doc_dict['review_id'] = row.Index
            
            doc_dict['rating'] = (
                row.rating
                if hasattr(row, 'rating')
                else row.star_rating
            )

            results.append(doc_dict)

        return pd.DataFrame(results).fillna(0)

    def reduce_topics(self, nr_topics: int):
        """Reduce number of topics using merging."""
        if self.model is None:
            raise ValueError("Model not trained yet")

        print(f"Reducing to {nr_topics} topics...")

        self.model.reduce_topics(nr_topics=nr_topics)

        print("Topics reduced!")

    def save_model(self, path: str):
        """Save model to disk."""
        if self.model is None:
            raise ValueError("No model to save")

        self.model.save(path)

        print(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load model from disk."""
        self.model = BERTopic.load(path)

        print(f"Model loaded from {path}")
