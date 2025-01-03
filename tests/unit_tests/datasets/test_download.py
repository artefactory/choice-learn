"""Test the file downloading function."""

from choice_learn.datasets.base import download_from_url

def test_download():
    """Tests downloading a dummy csv file."""
    url = "https://github.com/artefactory/choice-learn/tests/data/test_data.csv"
    download_from_url(url)