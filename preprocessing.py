
"""
Text preprocessing for Amazon product reviews.

Two separate representations are produced:

- LDA: normalized, tokenized, lemmatized tokens with stopword removal.
- BERTopic: lightly normalized natural-language text that preserves
  punctuation and sentence structure for transformer embeddings.
"""

import html
import re
from typing import List

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


class TextPreprocessor:
    """Preprocess product reviews for LDA and BERTopic."""

    def __init__(self):
        self._ensure_nltk_resource(
            "corpora/stopwords",
            "stopwords"
        )
        self._ensure_nltk_resource(
            "corpora/wordnet",
            "wordnet"
        )
        self._ensure_nltk_resource(
            "corpora/omw-1.4",
            "omw-1.4"
        )
        self._ensure_nltk_resource(
            "tokenizers/punkt",
            "punkt"
        )

        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))

        # Domain-specific words that are unlikely to distinguish aspects.
        self.stop_words.update({
            "product",
            "would",
            "could",
            "also",
            "one",
            "two",
            "get",
            "got",
        })

        # Retain negation because it can change the meaning of a review.
        self.stop_words.difference_update({
            "not",
            "no",
            "nor",
            "never",
        })

    @staticmethod
    def _ensure_nltk_resource(resource_path: str, package_name: str) -> None:
        """Download an NLTK resource if it is not already installed."""
        try:
            nltk.data.find(resource_path)
        except LookupError:
            downloaded = nltk.download(package_name, quiet=True)

            if not downloaded:
                raise RuntimeError(
                    f"Could not download the NLTK resource "
                    f"'{package_name}'. Install it manually with "
                    f"nltk.download('{package_name}')."
                )

            try:
                nltk.data.find(resource_path)
            except LookupError as exc:
                raise RuntimeError(
                    f"NLTK resource '{package_name}' is still unavailable "
                    "after attempting to download it."
                ) from exc

    @staticmethod
    def _normalize_input(text) -> str:
        """Convert missing or non-string input into safe text."""
        if text is None or pd.isna(text):
            return ""

        if not isinstance(text, str):
            text = str(text)

        return html.unescape(text)

    def _remove_html_and_urls(self, text: str) -> str:
        """Remove HTML tags and URLs without removing sentence punctuation."""
        text = self._normalize_input(text)

        # Remove URLs.
        text = re.sub(
            r"(https?://\S+|www\.\S+)",
            " ",
            text,
            flags=re.IGNORECASE
        )

        # Remove HTML tags.
        text = re.sub(r"<[^>]*>", " ", text)

        # Remove non-printing control characters.
        text = re.sub(r"[\x00-\x1f\x7f]", " ", text)

        # Normalize whitespace.
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def clean_text(self, text: str) -> str:
        """
        Clean text for the LDA stream.

        This method lowercases text and removes punctuation and other
        characters that are not useful for a bag-of-words representation.
        """
        text = self._remove_html_and_urls(text).lower()

        # Expand common negative contractions before removing apostrophes.
        contractions = {
            "can't": "cannot",
            "won't": "will not",
            "shan't": "shall not",
            "ain't": "is not",
            "isn't": "is not",
            "aren't": "are not",
            "wasn't": "was not",
            "weren't": "were not",
            "don't": "do not",
            "doesn't": "does not",
            "didn't": "did not",
            "hasn't": "has not",
            "haven't": "have not",
            "hadn't": "had not",
            "couldn't": "could not",
            "wouldn't": "would not",
            "shouldn't": "should not",
            "mustn't": "must not",
            "mightn't": "might not",
        }

        for contraction, expanded in contractions.items():
            text = re.sub(
                rf"\b{re.escape(contraction)}\b",
                expanded,
                text
            )

        # Handle remaining common n't contractions.
        text = re.sub(r"n't\b", " not", text)

        # Keep letters, numbers, and whitespace.
        text = re.sub(r"[^a-z0-9\s]", " ", text)

        # Normalize whitespace again after punctuation removal.
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def tokenize_and_lemmatize(self, text: str) -> List[str]:
        """Tokenize text and lemmatize words for LDA."""
        if not text:
            return []

        # preserve_line=True avoids requiring the separate Punkt tokenizer
        # model while still using NLTK's word tokenizer.
        tokens = word_tokenize(text, preserve_line=True)

        lemmatized = []

        for token in tokens:
            if not token.isalpha() or len(token) <= 2:
                continue

            # Apply noun lemmatization first, then verb lemmatization.
            # This handles common forms such as "batteries" and "charging".
            token = self.lemmatizer.lemmatize(token, pos="n")
            token = self.lemmatizer.lemmatize(token, pos="v")

            lemmatized.append(token)

        return lemmatized

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove common and domain-specific stopwords."""
        return [
            token
            for token in tokens
            if token not in self.stop_words
        ]

    def preprocess_lda(self, text: str) -> List[str]:
        """
        Produce the classical LDA representation.

        Output is a list of normalized tokens suitable for constructing
        a Gensim dictionary and bag-of-words corpus.
        """
        cleaned_text = self.clean_text(text)
        tokens = self.tokenize_and_lemmatize(cleaned_text)
        tokens = self.remove_stopwords(tokens)

        return tokens

    def preprocess_bertopic(self, text: str) -> str:
        """
        Produce the contextual BERTopic representation.

        Only light normalization is applied. Punctuation, capitalization,
        and sentence structure are retained for transformer embeddings.
        """
        return self._remove_html_and_urls(text)

    def process_reviews_df(
        self,
        df: pd.DataFrame,
        text_column: str = "text"
    ) -> pd.DataFrame:
        """
        Create both preprocessing representations in a dataframe.

        Original columns are preserved. Two columns are added:
        - text_cleaned: normalized natural-language text for BERTopic.
        - tokens: preprocessed token lists for LDA.

        Rows with missing or empty review text are retained, but their
        generated text and token representations will be empty.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame.")

        if text_column not in df.columns:
            raise ValueError(
                f"Required text column '{text_column}' was not found. "
                f"Available columns: {list(df.columns)}"
            )

        processed_df = df.copy()

        processed_df["text_cleaned"] = (
            processed_df[text_column].apply(self.preprocess_bertopic)
        )

        processed_df["tokens"] = (
            processed_df[text_column].apply(self.preprocess_lda)
        )

        return processed_df
