import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from bertopic import BERTopic
from bertopic.backend import BaseEmbedder
from sentence_transformers import SentenceTransformer, models

from bertopic_model import BERTopicModel


TEXTS = [
    "battery power charger",
    "battery charging power",
    "battery charger power",
    "screen display glass",
    "screen glass display",
    "display screen glass",
]
EMBEDDINGS = np.array([
    [0.0, 0.0],
    [0.1, 0.0],
    [0.0, 0.1],
    [5.0, 5.0],
    [5.1, 5.0],
    [5.0, 5.1],
])


class TestEmbedder(BaseEmbedder):
    def embed(self, documents, verbose=False):
        return np.array([
            EMBEDDINGS[TEXTS.index(document)] if document in TEXTS else [1.0, 1.0]
            for document in documents
        ])


class IdentityReducer:
    def fit(self, embeddings, y=None):
        return self

    def transform(self, embeddings):
        return embeddings


class BERTopicModelTest(unittest.TestCase):
    def test_embedding_model_device_and_reuse(self):
        for cuda_available, expected_device in ((True, "cuda"), (False, None)):
            with self.subTest(cuda_available=cuda_available):
                sentence_model = Mock()
                sentence_model.device = expected_device or "cpu"
                sentence_model.encode.return_value = EMBEDDINGS
                topic_model = Mock()
                topic_model.fit_transform.return_value = ([0] * len(TEXTS), None)

                with patch(
                    "bertopic_model.torch.cuda.is_available",
                    return_value=cuda_available
                ), patch(
                    "bertopic_model.SentenceTransformer",
                    return_value=sentence_model
                ) as constructor, patch(
                    "bertopic_model.BERTopic", return_value=topic_model
                ) as bertopic:
                    model = BERTopicModel()
                    model.generate_embeddings(TEXTS)
                    model.train(TEXTS, num_topics=2)
                    model.train(TEXTS, num_topics=2)

                constructor.assert_called_once_with(
                    "all-MiniLM-L6-v2", device=expected_device
                )
                sentence_model.encode.assert_called_once_with(
                    TEXTS, show_progress_bar=True
                )
                self.assertEqual(bertopic.call_count, 2)
                for call in bertopic.call_args_list:
                    self.assertIs(call.kwargs["embedding_model"], sentence_model)

    def test_sentence_transformer_is_used_for_later_predictions(self):
        encoder = SentenceTransformer(
            modules=[models.BoW(vocab=[
                "battery", "power", "charger", "charging",
                "screen", "display", "glass"
            ])],
            device="cpu"
        )

        def create_bertopic(**kwargs):
            kwargs["umap_model"] = IdentityReducer()
            return BERTopic(**kwargs)

        with patch(
            "bertopic_model.torch.cuda.is_available", return_value=False
        ), patch(
            "bertopic_model.SentenceTransformer", return_value=encoder
        ), patch("bertopic_model.BERTopic", side_effect=create_bertopic):
            model = BERTopicModel()
            topics, _ = model.train(TEXTS, num_topics=2)
            for text, topic in zip(TEXTS, topics):
                self.assertEqual(model.get_document_topics(text), {topic: 1.0})

    def test_kmeans_training_and_topic_assignments(self):
        model = BERTopicModel()
        model.embeddings = EMBEDDINGS
        model.embedding_texts = TEXTS
        model._sentence_model = TestEmbedder()

        def create_bertopic(**kwargs):
            kwargs["embedding_model"] = TestEmbedder()
            kwargs["umap_model"] = IdentityReducer()
            return BERTopic(**kwargs)

        with patch("bertopic_model.BERTopic", side_effect=create_bertopic):
            topics, probabilities = model.train(TEXTS, num_topics=2)

        self.assertIsNone(probabilities)
        self.assertEqual(len(set(topics)), 2)
        self.assertEqual(model.get_document_topics(TEXTS[0]), {topics[0]: 1.0})

        reviews = pd.DataFrame({"text_cleaned": TEXTS, "rating": [1, 2, 3, 4, 5, 5]})
        distribution = model.get_review_topic_distribution(reviews)
        self.assertEqual(set(distribution.columns), {"topic_0", "topic_1", "review_id", "rating"})
        self.assertEqual(distribution[["topic_0", "topic_1"]].sum(axis=1).tolist(), [1.0] * len(TEXTS))
        for index, topic in enumerate(topics):
            self.assertEqual(distribution.loc[index, f"topic_{topic}"], 1.0)
        self.assertEqual(distribution["rating"].tolist(), reviews["rating"].tolist())

    def test_outliers_and_noncontiguous_topics(self):
        model = BERTopicModel()
        model.model = Mock()
        model.model.topics_ = [-1, 0, 2]
        model.model.transform.return_value = ([-1, 0, 2], None)
        reviews = pd.DataFrame({"text_cleaned": ["a", "b", "c"], "rating": [1, 2, 3]})

        distribution = model.get_review_topic_distribution(reviews)
        self.assertEqual(distribution["topic_0"].tolist(), [0.0, 1.0, 0.0])
        self.assertEqual(distribution["topic_2"].tolist(), [0.0, 0.0, 1.0])

        model.model.transform.return_value = ([-1], None)
        self.assertEqual(model.get_document_topics("a"), {})

    def test_existing_assignment_strengths_are_preserved(self):
        model = BERTopicModel()
        model.model = Mock()
        model.model.topics_ = [0, 1]
        model.model.transform.return_value = ([1], np.array([0.75]))

        self.assertEqual(model.get_document_topics("a"), {1: 0.75})
        reviews = pd.DataFrame({"text_cleaned": ["a"], "rating": [4]})
        distribution = model.get_review_topic_distribution(reviews)
        self.assertEqual(distribution.loc[0, "topic_0"], 0.0)
        self.assertEqual(distribution.loc[0, "topic_1"], 0.75)


if __name__ == "__main__":
    unittest.main()
