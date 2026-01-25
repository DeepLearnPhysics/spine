"""Tests for config download functionality."""

import hashlib
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from spine.config.download import (
    compute_file_hash,
    download_file,
    download_from_url,
    get_cache_dir,
    url_to_filename,
)
from spine.config.loader import ConfigLoader, load_config


class TestDownloadUtilities:
    """Test suite for download utility functions."""

    def test_get_cache_dir_default(self):
        """Test default cache directory."""
        with patch.dict(os.environ, {}, clear=True):
            cache_dir = get_cache_dir()
            assert cache_dir == Path.cwd() / "weights"

    def test_get_cache_dir_env_override(self):
        """Test cache directory from environment variable."""
        with patch.dict(os.environ, {"SPINE_CACHE_DIR": "/custom/cache"}):
            cache_dir = get_cache_dir()
            assert cache_dir == Path("/custom/cache")

    def test_url_to_filename(self):
        """Test URL to filename conversion."""
        url = "https://example.com/models/my_model.ckpt"
        filename = url_to_filename(url)

        # Should have the original extension
        assert filename.endswith(".ckpt")
        # Should be deterministic (same URL -> same filename)
        assert url_to_filename(url) == filename
        # Different URLs should give different filenames
        assert url_to_filename("https://other.com/model.ckpt") != filename

    def test_compute_file_hash(self, tmp_path):
        """Test file hash computation."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")

        file_hash = compute_file_hash(test_file)

        # Verify it's a valid SHA256 hash
        assert len(file_hash) == 64
        assert all(c in "0123456789abcdef" for c in file_hash)

        # Should be deterministic
        assert compute_file_hash(test_file) == file_hash

    @patch("urllib.request.urlretrieve")
    def test_download_file(self, mock_retrieve, tmp_path):
        """Test basic file download."""
        url = "https://example.com/file.txt"
        output_path = tmp_path / "downloaded.txt"

        # Mock the download
        def fake_retrieve(url, path, reporthook=None):
            Path(path).write_text("downloaded content")

        mock_retrieve.side_effect = fake_retrieve

        download_file(url, output_path)

        assert output_path.exists()
        assert output_path.read_text() == "downloaded content"
        mock_retrieve.assert_called_once()

    @patch("urllib.request.urlretrieve")
    def test_download_file_with_hash_validation(self, mock_retrieve, tmp_path):
        """Test download with hash validation."""
        url = "https://example.com/file.txt"
        output_path = tmp_path / "downloaded.txt"
        content = "test content"
        expected_hash = hashlib.sha256(content.encode()).hexdigest()

        # Mock the download
        def fake_retrieve(url, path, reporthook=None):
            Path(path).write_text(content)

        mock_retrieve.side_effect = fake_retrieve

        # Should succeed with correct hash
        download_file(url, output_path, expected_hash=expected_hash)
        assert output_path.exists()

    @patch("urllib.request.urlretrieve")
    def test_download_file_hash_mismatch(self, mock_retrieve, tmp_path):
        """Test download with hash mismatch."""
        url = "https://example.com/file.txt"
        output_path = tmp_path / "downloaded.txt"
        wrong_hash = "0" * 64

        # Mock the download
        def fake_retrieve(url, path, reporthook=None):
            Path(path).write_text("content")

        mock_retrieve.side_effect = fake_retrieve

        # Should raise ValueError and remove file
        with pytest.raises(ValueError, match="hash mismatch"):
            download_file(url, output_path, expected_hash=wrong_hash)

        # File should be removed after failed validation
        assert not output_path.exists()

    @patch("spine.config.download.download_file")
    def test_download_from_url_caching(self, mock_download, tmp_path):
        """Test that files are cached and not re-downloaded."""
        url = "https://example.com/model.ckpt"
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        # Create a cached file
        filename = url_to_filename(url)
        cached_file = cache_dir / filename
        cached_file.write_text("cached content")

        # Should use cached file without downloading
        result = download_from_url(url, cache_dir=cache_dir)

        assert result == str(cached_file.absolute())
        mock_download.assert_not_called()

    @patch("spine.config.download.download_file")
    def test_download_from_url_downloads_if_missing(self, mock_download, tmp_path):
        """Test that missing files are downloaded."""
        url = "https://example.com/model.ckpt"
        cache_dir = tmp_path / "cache"

        # Mock the download
        def fake_download(url, path, expected_hash=None):
            path.write_text("downloaded")

        mock_download.side_effect = fake_download

        result = download_from_url(url, cache_dir=cache_dir)

        # Should have downloaded
        mock_download.assert_called_once()
        assert Path(result).exists()


class TestConfigLoaderDownload:
    """Test suite for !download YAML tag in config loader."""

    @patch("spine.config.download.download_file")
    def test_download_tag_simple_url(self, mock_download, tmp_path):
        """Test !download tag with simple URL string."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
model:
  weights: !download https://example.com/model.ckpt
""")

        # Mock download to create a file
        cache_dir = tmp_path / "weights"
        cache_dir.mkdir()
        weights_file = cache_dir / url_to_filename("https://example.com/model.ckpt")

        def fake_download(url, path, expected_hash=None):
            path.write_text("model weights")

        mock_download.side_effect = fake_download

        with patch("spine.config.download.get_cache_dir", return_value=cache_dir):
            cfg = load_config(str(config_file))

        assert "model" in cfg
        assert "weights" in cfg["model"]
        # Should be an absolute path
        assert Path(cfg["model"]["weights"]).is_absolute()

    @patch("spine.config.download.download_file")
    def test_download_tag_with_hash(self, mock_download, tmp_path):
        """Test !download tag with URL and hash."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
model:
  weights: !download
    url: https://example.com/model.ckpt
    hash: abc123def456
""")

        cache_dir = tmp_path / "weights"
        cache_dir.mkdir()

        def fake_download(url, path, expected_hash=None):
            path.write_text("model weights")

        mock_download.side_effect = fake_download

        with patch("spine.config.download.get_cache_dir", return_value=cache_dir):
            cfg = load_config(str(config_file))

        assert Path(cfg["model"]["weights"]).is_absolute()
        # Verify hash was passed to download (as third positional argument)
        mock_download.assert_called_once()
        call_args = mock_download.call_args[0]
        assert len(call_args) == 3
        assert call_args[2] == "abc123def456"  # expected_hash is third positional arg

    def test_download_tag_missing_url(self, tmp_path):
        """Test !download tag without URL raises error."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
model:
  weights: !download
    hash: abc123
""")

        with pytest.raises(Exception, match="url"):
            load_config(str(config_file))

    @patch("spine.config.download.download_file")
    def test_download_integration(self, mock_download, tmp_path):
        """Test full integration with config includes."""
        # Create base config with downloaded weights
        base_config = tmp_path / "base.yaml"
        base_config.write_text("""
model:
  name: uresnet
  weights: !download https://example.com/uresnet.ckpt
""")

        # Create main config that includes base
        main_config = tmp_path / "main.yaml"
        main_config.write_text("""
include: base.yaml

base:
  iterations: 1000
""")

        cache_dir = tmp_path / "weights"
        cache_dir.mkdir()

        def fake_download(url, path, expected_hash=None):
            path.write_text("weights")

        mock_download.side_effect = fake_download

        with patch("spine.config.download.get_cache_dir", return_value=cache_dir):
            cfg = load_config(str(main_config))

        assert cfg["model"]["name"] == "uresnet"
        assert Path(cfg["model"]["weights"]).exists()
        assert cfg["base"]["iterations"] == 1000
