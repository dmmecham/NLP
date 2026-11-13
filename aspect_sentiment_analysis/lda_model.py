"""
LDA topic modeling baseline implementation.
Uses Gensim to train Latent Dirichlet Allocation models.
"""

from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from gensim import corpora, models
from gensim.models import CoherenceModel
import logging

logging.basicConfig(level=logging.WARNING)


class LDATopicModel:
    """Latent Dirichlet Allocation topic model for product reviews."""

    def __init__(self):
        self.dictionary = None
        self.corpus = None
        self.model = None
        self.coherence_scores = {}
        self.id2word = None

    def create_dictionary_and_corpus(self, tokenized_reviews: List[List[str]]):
        """Create dictionary and document-term matrix from tokenized reviews."""
        # Create dictionary
        self.dictionary = corpora.Dictionary(tokenized_reviews)
        
        # Filter extremes: words appearing in < 2 docs or > 95% of docs
        self.dictionary.filter_extremes(no_below=2, no_above=0.95, keep_n=100000)
        
        # Create corpus (bag of words representation)
        self.corpus = [self.dictionary.doc2bow(doc) for doc in tokenized_reviews]
        self.id2word = self.dictionary
        
        print(f"Dictionary created: {len(self.dictionary)} terms")
        print(f"Corpus created: {len(self.corpus)} documents")

    def train(self, tokenized_reviews: List[List[str]], num_topics: int = 10, passes: int = 10):
        """Train LDA model."""
        if self.corpus is None:
            self.create_dictionary_and_corpus(tokenized_reviews)
        
        print(f"Training LDA with {num_topics} topics...")
        self.model = models.LdaMulticore(
            corpus=self.corpus,
            id2word=self.id2word,
            num_topics=num_topics,
            random_state=42,
            passes=passes,
            workers=4,
            minimum_probability=0.0,
            per_word_topics=True
        )
        print("LDA training complete!")
        return self.model

    def calculate_coherence(self) -> float:
        """Calculate coherence score for the trained model."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        coherence_model = CoherenceModel(
            model=self.model,
            corpus=self.corpus,
            dictionary=self.dictionary,
            coherence='u_mass'
        )
        score = coherence_model.get_coherence()
        print(f"Coherence Score: {score:.4f}")
        return score

    def optimize_topics(self, tokenized_reviews: List[List[str]], 
                       topic_range: range = range(5, 21, 2), 
                       passes: int = 10) -> Dict[int, float]:
        """Find optimal number of topics by coherence score."""
        print(f"Optimizing topic count in range {min(topic_range)} to {max(topic_range)}...")
        
        if self.corpus is None:
            self.create_dictionary_and_corpus(tokenized_reviews)
        
        for num_topics in topic_range:
            print(f"  Testing {num_topics} topics...")
            model = models.LdaMulticore(
                corpus=self.corpus,
                id2word=self.id2word,
                num_topics=num_topics,
                random_state=42,
                passes=passes,
                workers=4,
                minimum_probability=0.0
            )
            
            coherence_model = CoherenceModel(
                model=model,
                corpus=self.corpus,
                dictionary=self.dictionary,
                coherence='u_mass'
            )
            score = coherence_model.get_coherence()
            self.coherence_scores[num_topics] = score
            print(f"    {num_topics} topics: {score:.4f}")
        
        best_topics = max(self.coherence_scores, key=self.coherence_scores.get)
        print(f"Optimal topic count: {best_topics} (score: {self.coherence_scores[best_topics]:.4f})")
        return self.coherence_scores

    def get_topics(self, num_words: int = 10) -> List[Dict]:
        """Get topics with top words."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        topics = []
        for topic_id, topic_terms in self.model.show_topics(
            num_topics=-1, 
            num_words=num_words, 
            formatted=False
        ):
            topic_info = {
                'topic_id': topic_id,
                'terms': [(word, float(weight)) for word, weight in topic_terms]
            }
            topics.append(topic_info)
        
        return topics

    def get_document_topics(self, tokenized_text: List[str]) -> List[Tuple[int, float]]:
        """Get topic distribution for a single document."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        bow = self.dictionary.doc2bow(tokenized_text)
        return self.model.get_document_topics(bow)

    def get_review_topic_distribution(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get topic distribution for all reviews in dataframe."""
        results = []
        
        for idx, row in df.iterrows():
            doc_topics = self.get_document_topics(row['tokens'])
            doc_dict = {f'topic_{topic_id}': weight for topic_id, weight in doc_topics}
            doc_dict['review_id'] = idx
            doc_dict['rating'] = row.get('rating', row.get('star_rating', None))
            results.append(doc_dict)
        
        return pd.DataFrame(results).fillna(0)

    def save_model(self, path: str):
        """Save model to disk."""
        if self.model is None:
            raise ValueError("No model to save")
        self.model.save(path)
        print(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load model from disk."""
        self.model = models.LdaMulticore.load(path)
        print(f"Model loaded from {path}")
