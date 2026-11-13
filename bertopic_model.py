"""
BERTopic neural topic modeling implementation.
Uses contextual embeddings to identify semantically related topics.
"""

from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from sklearn.cluster import KMeans
import logging

logging.basicConfig(level=logging.WARNING)


class BERTopicModel:
    """BERTopic neural topic model using transformer embeddings."""

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2", language: str = "english"):
        """
        Initialize BERTopic model.
        
        Args:
            embedding_model: Sentence transformer model to use
            language: Language for stopword removal
        """
        self.embedding_model_name = embedding_model
        self.language = language
        self.embeddings = None
        self.model = None

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate sentence embeddings for texts."""
        print(f"Generating embeddings using {self.embedding_model_name}...")
        sentence_model = SentenceTransformer(self.embedding_model_name)
        self.embeddings = sentence_model.encode(texts, show_progress_bar=True)
        print(f"Generated {len(self.embeddings)} embeddings of dimension {self.embeddings.shape[1]}")
        return self.embeddings

    def train(self, texts: List[str], num_topics: int = 10, min_topic_size: int = 10):
        """
        Train BERTopic model.
        
        Args:
            texts: List of review texts (already preprocessed)
            num_topics: Target number of topics
            min_topic_size: Unused with KMeans, which does not enforce a minimum size

        Returns:
            Topic assignments and probabilities. KMeans does not produce probabilities,
            so the second return value is None.
        """
        if self.embeddings is None:
            self.generate_embeddings(texts)
        
        print(f"Training BERTopic with {num_topics} topics...")
        
        # Initialize clustering with KMeans for more consistent topics
        clustering_model = KMeans(
            n_clusters=num_topics,
            random_state=42,
            n_init=10
        )
        
        self.model = BERTopic(
            embedding_model=self.embedding_model_name,
            language=self.language,
            hdbscan_model=clustering_model,
            verbose=True
        )
        
        topics, probabilities = self.model.fit_transform(texts, self.embeddings)
        print("BERTopic training complete!")
        
        return topics, probabilities

    def get_topics(self, top_n: int = 10) -> Dict[int, List[Tuple[str, float]]]:
        """Get topics with top keywords."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        topics_dict = {}
        for topic_id in sorted(self.model.topics_):
            if topic_id != -1:  # Skip outlier topic
                keywords = self.model.get_topic(topic_id)
                topics_dict[topic_id] = keywords[:top_n]
        
        return topics_dict

    def get_topic_info(self) -> pd.DataFrame:
        """Get detailed information about all topics."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        return self.model.get_topic_info()

    def get_document_topics(self, text: str) -> Dict[int, float]:
        """Get the assigned topic and its strength (1.0 for a KMeans assignment)."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        topics, probabilities = self.model.transform([text])
        topic_dist = {}
        if probabilities is None:
            return {topic_id: 1.0 for topic_id in topics if topic_id != -1}

        for topic_id, prob in zip(topics, probabilities):
            if topic_id != -1:
                topic_dist[topic_id] = float(prob)
        
        return topic_dist

    def get_review_topic_distribution(self, df: pd.DataFrame, text_column: str = 'text_cleaned') -> pd.DataFrame:
        """Get topic assignments for reviews; KMeans assignments are one-hot."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        print("Generating topic distributions for all reviews...")
        topics, probabilities = self.model.transform(df[text_column].tolist())
        topic_ids = sorted(set(self.model.topics_) - {-1})

        results = []
        for idx, (row, topic) in enumerate(zip(df.itertuples(), topics)):
            prob = 1.0 if probabilities is None else float(probabilities[idx])

            doc_dict = {}
            for topic_id in topic_ids:
                if topic_id == topic:
                    doc_dict[f'topic_{topic_id}'] = prob
                else:
                    doc_dict[f'topic_{topic_id}'] = 0.0
            
            doc_dict['review_id'] = idx
            doc_dict['rating'] = row.rating if hasattr(row, 'rating') else row.star_rating
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
