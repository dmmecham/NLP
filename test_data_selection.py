import contextlib
import runpy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import DEFAULT, patch

import pandas as pd

from pipeline import AspectSentimentPipeline


class PipelineDataSelectionTest(unittest.TestCase):
    def run_pipeline(self, **kwargs):
        pipeline = AspectSentimentPipeline.__new__(AspectSentimentPipeline)
        pipeline.output_dir = Path("unused")
        with patch.multiple(
            pipeline,
            preprocess_data=DEFAULT,
            train_lda=DEFAULT,
            train_bertopic=DEFAULT,
            analyze_sentiment_lda=DEFAULT,
            analyze_sentiment_bertopic=DEFAULT,
            evaluate_and_compare=DEFAULT,
            generate_visualizations=DEFAULT,
        ):
            pipeline.run_full_pipeline(**kwargs)
        return pipeline.df

    def test_filepath_selects_real_reviews_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reviews.csv"
            pd.DataFrame({"review_body": ["Real review"], "star_rating": [4]}).to_csv(
                path, index=False
            )
            result = self.run_pipeline(filepath=str(path))
            self.assertEqual(result.to_dict("records"), [
                {"text": "Real review", "rating": 4}
            ])

    def test_missing_filepath_does_not_fall_back_to_demo(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(
            None
        ), contextlib.redirect_stderr(None):
            with self.assertRaises(FileNotFoundError):
                self.run_pipeline(filepath=str(Path(directory) / "missing.csv"))

    def test_tsv_reviews_are_loaded_without_csv_quote_parsing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reviews.tsv"
            path.write_text(
                'review_body\tstar_rating\nGreat "screen, sharp\t5\nOther\t2\n',
                encoding="utf-8"
            )
            result = self.run_pipeline(filepath=str(path), nrows=1)
            self.assertEqual(result.to_dict("records"), [
                {"text": 'Great "screen, sharp', "rating": 5}
            ])

    def test_default_and_explicit_demo_mode(self):
        for kwargs in ({}, {"filepath": "missing.csv", "use_demo": True}):
            with self.subTest(kwargs=kwargs):
                result = self.run_pipeline(**kwargs)
                self.assertEqual(len(result), 100)

    def test_explicit_real_mode_requires_filepath(self):
        with contextlib.redirect_stdout(None), contextlib.redirect_stderr(None):
            with self.assertRaisesRegex(ValueError, "Either filepath or use_demo"):
                self.run_pipeline(use_demo=False)

    def test_real_reviews_script_explicitly_requests_real_data(self):
        results = pd.DataFrame(columns=[
            "topic_id", "avg_rating", "negative_sentiment_score", "count_docs",
            "keywords", "impact_score",
        ])
        with patch("pipeline.AspectSentimentPipeline") as pipeline_class, patch(
            "kagglehub.dataset_download", return_value="/cache/reviews.tsv"
        ) as download, patch(
            "pandas.read_csv", return_value=results
        ), contextlib.redirect_stdout(None):
            runpy.run_path(str(Path(__file__).with_name("real_reviews.py")))

        download.assert_called_once_with(
            "cynthiarempel/amazon-us-customer-reviews-dataset",
            path="amazon_reviews_us_Electronics_v1_00.tsv",
        )
        args = pipeline_class.return_value.run_full_pipeline.call_args.kwargs
        self.assertEqual(args["filepath"], "/cache/reviews.tsv")
        self.assertIs(args["use_demo"], False)

    def test_download_errors_stop_real_reviews_script(self):
        with patch(
            "kagglehub.dataset_download", side_effect=RuntimeError("download failed")
        ), patch("pipeline.AspectSentimentPipeline") as pipeline_class, \
                contextlib.redirect_stdout(None):
            with self.assertRaisesRegex(RuntimeError, "download failed"):
                runpy.run_path(str(Path(__file__).with_name("real_reviews.py")))
        pipeline_class.assert_not_called()


if __name__ == "__main__":
    unittest.main()
