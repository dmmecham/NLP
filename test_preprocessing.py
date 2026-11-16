import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

import nltk

from preprocessing import TextPreprocessor


class NltkResourceTests(unittest.TestCase):
    def test_existing_extracted_and_zipped_resources_do_not_download(self):
        with tempfile.TemporaryDirectory() as directory:
            corpora = Path(directory) / "corpora"
            (corpora / "stopwords").mkdir(parents=True)
            corpora.chmod(0o700)
            (corpora / "stopwords").chmod(0o700)
            with ZipFile(corpora / "wordnet.zip", "w") as archive:
                archive.writestr("wordnet/data.noun", "data")
            (corpora / "wordnet.zip").chmod(0o600)

            with patch.object(nltk.data, "path", [directory]), patch.object(
                nltk, "download"
            ) as download:
                TextPreprocessor._ensure_nltk_resource("corpora/stopwords", "stopwords")
                TextPreprocessor._ensure_nltk_resource("corpora/wordnet", "wordnet")

            download.assert_not_called()

    def test_downloaded_zip_is_recognized(self):
        with tempfile.TemporaryDirectory() as directory:
            corpora = Path(directory) / "corpora"
            corpora.mkdir()
            corpora.chmod(0o700)

            def download_archive(package_name, quiet):
                with ZipFile(corpora / f"{package_name}.zip", "w") as archive:
                    archive.writestr(f"{package_name}/data.noun", "data")
                (corpora / f"{package_name}.zip").chmod(0o600)
                return True

            with patch.object(nltk.data, "path", [directory]), patch.object(
                nltk, "download", side_effect=download_archive
            ) as download:
                TextPreprocessor._ensure_nltk_resource("corpora/wordnet", "wordnet")

            download.assert_called_once_with("wordnet", quiet=True)

    def test_missing_resource_after_download_raises(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            nltk.data, "path", [directory]
        ), patch.object(nltk, "download", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "still unavailable"):
                TextPreprocessor._ensure_nltk_resource("corpora/wordnet", "wordnet")

    def test_failed_download_raises(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            nltk.data, "path", [directory]
        ), patch.object(nltk, "download", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "Could not download"):
                TextPreprocessor._ensure_nltk_resource("corpora/wordnet", "wordnet")


if __name__ == "__main__":
    unittest.main()
