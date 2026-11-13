"""
Text preprocessing pipeline for product reviews.
Includes tokenization, stopword removal, lemmatization for LDA.
"""

import re
from typing import List
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4', quiet=True)


class TextPreprocessor:
    """Preprocesses product reviews for topic modeling."""

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        # Add domain-specific stopwords
        self.stop_words.update(['product', 'would', 'could', 'also', 'one', 'two', 'get', 'got'])

    def clean_text(self, text: str) -> str:
        """Remove URLs, special characters, and normalize whitespace."""
        # Convert to lowercase
        text = text.lower()
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def tokenize_and_lemmatize(self, text: str) -> List[str]:
        """Tokenize and lemmatize text."""
        tokens = word_tokenize(text)
        lemmatized = [
            self.lemmatizer.lemmatize(token, pos='v')
            for token in tokens
            if token.isalpha() and len(token) > 2
        ]
        return lemmatized

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove stopwords from token list."""
        return [token for token in tokens if token not in self.stop_words]

    def preprocess_lda(self, text: str) -> List[str]:
        """Complete preprocessing pipeline for LDA (classical stream)."""
        text = self.clean_text(text)
        tokens = self.tokenize_and_lemmatize(text)
        tokens = self.remove_stopwords(tokens)
        return tokens

    def preprocess_bertopic(self, text: str) -> str:
        """Light normalization for BERTopic (contextual stream)."""
        text = self.clean_text(text)
        # Keep more structure for transformer models
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def process_reviews_df(self, df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        """Process all reviews in a dataframe."""
        df = df.copy()
        df['text_cleaned'] = df[text_column].apply(self.preprocess_bertopic)
        df['tokens'] = df[text_column].apply(self.preprocess_lda)
        return df
