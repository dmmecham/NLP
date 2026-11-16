import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from evaluation import ModelEvaluator


class ModelEvaluatorTest(unittest.TestCase):
    def test_comparison_visualization_plots_both_models(self):
        lda = pd.DataFrame({"topic_id": [0, 1], "avg_rating": [2.5, 4.0]})
        bertopic = pd.DataFrame({"topic_id": [0], "avg_rating": [3.5]})

        figure = ModelEvaluator.visualize_comparison(lda, bertopic)
        try:
            self.assertEqual(len(figure.axes), 2)
            self.assertEqual([bar.get_height() for bar in figure.axes[0].patches], [2.5, 4.0])
            self.assertEqual([bar.get_height() for bar in figure.axes[1].patches], [3.5])
        finally:
            plt.close(figure)

    def test_report_handles_missing_coherence_and_empty_drivers(self):
        report = ModelEvaluator.create_summary_report(
            {"optimal_topics": 5, "dataset_stats": {}, "negative_drivers": pd.DataFrame()},
            {"num_topics": 5, "negative_drivers": pd.DataFrame()},
        )
        self.assertIn("LDA:\n  Topic Count: 5\n  Coherence: N/A", report)
        self.assertIn("BERTopic:\n  Topic Count: 5\n  Coherence: N/A", report)
        self.assertIn("LDA:\n  No negative-driver results available.", report)
        self.assertIn("BERTopic:\n  No negative-driver results available.", report)

    def test_report_includes_coherence_and_drivers(self):
        drivers = pd.DataFrame([{"topic_id": 2, "keywords": "battery", "impact_score": 0.125}])
        report = ModelEvaluator.create_summary_report(
            {"optimal_topics": 3, "best_coherence": -1.25, "negative_drivers": drivers},
            {"num_topics": 3, "negative_drivers": drivers},
        )
        self.assertIn("LDA:\n  Topic Count: 3\n  Coherence: -1.2500", report)
        self.assertIn("LDA:\n  Topic 2: battery\n    Impact score: 0.1250", report)
        self.assertIn("BERTopic:\n  Topic 2: battery\n    Impact score: 0.1250", report)


if __name__ == "__main__":
    unittest.main()
