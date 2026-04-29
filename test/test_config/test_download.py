"""Tests for config download functionality."""

import hashlib
import multiprocessing
import time
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError, URLError

import pytest

from spine.config.download import (
    _validate_cached_file,
    compute_file_hash,
    download_file,
    download_from_url,
    get_cache_dir,
    url_to_filename,
)
from spine.config.load import load_config_file


class TestDownloadUtilities:
    """Test suite for download utility functions."""

    def test_get_cache_dir_default(self, monkeypatch):
        """Test default cache directory with warning when env vars not set."""
        # Clear environment
        monkeypatch.delenv("SPINE_CACHE_DIR", raising=False)
        monkeypatch.delenv("SPINE_PROD_BASEDIR", raising=False)
        monkeypatch.delenv("SPINE_BASEDIR", raising=False)

        with pytest.warns(
            UserWarning, match="SPINE_BASEDIR and SPINE_PROD_BASEDIR not set"
        ):
            cache_dir = get_cache_dir()
            assert cache_dir == Path.cwd() / "weights"

    def test_get_cache_dir_spine_prod(self, monkeypatch):
        """Test cache directory from SPINE_PROD_BASEDIR."""
        monkeypatch.delenv("SPINE_CACHE_DIR", raising=False)
        monkeypatch.setenv("SPINE_PROD_BASEDIR", "/opt/spine-prod")
        monkeypatch.setenv("SPINE_BASEDIR", "/opt/spine")

        cache_dir = get_cache_dir()
        # Should prefer SPINE_PROD_BASEDIR
        assert cache_dir == Path("/opt/spine-prod") / ".cache" / "weights"

    def test_get_cache_dir_spine_basedir(self, monkeypatch):
        """Test cache directory from SPINE_BASEDIR."""
        monkeypatch.delenv("SPINE_CACHE_DIR", raising=False)
        monkeypatch.delenv("SPINE_PROD_BASEDIR", raising=False)
        monkeypatch.setenv("SPINE_BASEDIR", "/opt/spine")

        cache_dir = get_cache_dir()
        assert cache_dir == Path("/opt/spine") / ".cache" / "weights"

    def test_get_cache_dir_env_override(self, monkeypatch):
        """Test cache directory from environment variable override."""
        monkeypatch.setenv("SPINE_CACHE_DIR", "/custom/cache")
        monkeypatch.setenv("SPINE_PROD_BASEDIR", "/opt/spine-prod")

        cache_dir = get_cache_dir()
        # SPINE_CACHE_DIR should take precedence
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
    def test_download_file_progress_hook_prints_progress(
        self, mock_retrieve, tmp_path, capsys
    ):
        """Test progress reporting callback is exercised during download."""
        url = "https://example.com/file.txt"
        output_path = tmp_path / "downloaded.txt"

        def fake_retrieve(url, path, reporthook=None):
            if reporthook is not None:
                reporthook(1, 50, 100)
            Path(path).write_text("downloaded content")

        mock_retrieve.side_effect = fake_retrieve

        download_file(url, output_path)

        captured = capsys.readouterr()
        assert "Progress: 50%" in captured.out

    @patch("urllib.request.urlretrieve")
    def test_download_file_wraps_http_error(self, mock_retrieve, tmp_path):
        """Test HTTP errors are re-raised with a normalized message."""
        url = "https://example.com/file.txt"
        output_path = tmp_path / "downloaded.txt"
        mock_retrieve.side_effect = HTTPError(url, 404, "Not Found", None, None)

        with pytest.raises(HTTPError, match="HTTP Error 404: Not Found"):
            download_file(url, output_path)

    @patch("urllib.request.urlretrieve")
    def test_download_file_wraps_url_error(self, mock_retrieve, tmp_path):
        """Test URL errors are wrapped with the download URL."""
        url = "https://example.com/file.txt"
        output_path = tmp_path / "downloaded.txt"
        mock_retrieve.side_effect = URLError("connection refused")

        with pytest.raises(
            URLError, match="Failed to download https://example.com/file.txt"
        ):
            download_file(url, output_path)

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

    def test_concurrent_downloads_safe(self, tmp_path):
        """Test that concurrent downloads do not corrupt the cached file."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        url = "https://example.com/test_model.ckpt"

        num_workers = 5
        manager = multiprocessing.Manager()
        result_queue = manager.Queue()
        processes = []

        for i in range(num_workers):
            process = multiprocessing.Process(
                target=_download_worker, args=(url, cache_dir, i, result_queue)
            )
            processes.append(process)
            process.start()

        for process in processes:
            process.join(timeout=30)

        assert all(process.exitcode == 0 for process in processes)

        results = []
        while not result_queue.empty():
            results.append(result_queue.get())

        assert len(results) == num_workers
        assert all(status == "success" for _, status, _ in results)

        paths = [result for _, _, result in results]
        assert len(set(paths)) == 1

        final_path = cache_dir / url_to_filename(url)
        assert final_path.exists()
        assert not list(cache_dir.glob(".*.lock"))
        assert not list(cache_dir.glob(".*tmp*"))

    def test_download_from_url_timeout_waiting_for_lock(self, tmp_path, monkeypatch):
        """Test lock acquisition timeout raises a clear TimeoutError."""
        url = "https://example.com/model.ckpt"
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        time_values = iter([0.0, 0.1, 1.5])

        monkeypatch.setattr(
            "spine.config.download.time.time", lambda: next(time_values)
        )
        monkeypatch.setattr("spine.config.download.time.sleep", lambda _: None)

        def fake_flock(*args, **kwargs):
            raise BlockingIOError()

        monkeypatch.setattr("spine.config.download.fcntl.flock", fake_flock)

        with pytest.raises(
            TimeoutError, match="Timeout waiting for download lock after 1s"
        ):
            download_from_url(url, cache_dir=cache_dir, max_wait_seconds=1)

    def test_download_from_url_reports_long_waits(self, tmp_path, monkeypatch, capsys):
        """Test waiting messages switch to the minute-based status format."""
        url = "https://example.com/model.ckpt"
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        time_values = iter([0.0, 60.0, 60.0])
        flock_calls = {"count": 0}

        def fake_time():
            return next(time_values)

        def fake_flock(*args, **kwargs):
            flock_calls["count"] += 1
            if flock_calls["count"] == 1:
                raise BlockingIOError()
            return None

        monkeypatch.setattr("spine.config.download.time.time", fake_time)
        monkeypatch.setattr("spine.config.download.time.sleep", lambda _: None)
        monkeypatch.setattr("spine.config.download.fcntl.flock", fake_flock)

        with patch("spine.config.download.download_file") as mock_download:
            mock_download.side_effect = (
                lambda _url, path, expected_hash=None: path.write_text("downloaded")
            )
            download_from_url(url, cache_dir=cache_dir, max_wait_seconds=120)

        assert "Still waiting (60s elapsed)..." in capsys.readouterr().out

    @patch("spine.config.download.download_file")
    def test_download_from_url_cleans_temp_file_on_download_error(
        self, mock_download, tmp_path
    ):
        """Test failed downloads remove temporary files before re-raising."""
        url = "https://example.com/model.ckpt"
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        mock_download.side_effect = RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            download_from_url(url, cache_dir=cache_dir)

        filename = url_to_filename(url)
        assert not list(cache_dir.glob(f".{filename}*.tmp"))
        assert not (cache_dir / filename).exists()

    @patch("spine.config.download._validate_cached_file", return_value=True)
    def test_download_from_url_rechecks_cache_after_lock(self, mock_validate, tmp_path):
        """Test a file appearing before download starts is reused after locking."""
        url = "https://example.com/model.ckpt"
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        output_path = cache_dir / url_to_filename(url)

        call_count = {"count": 0}

        def fake_exists():
            call_count["count"] += 1
            if call_count["count"] >= 2:
                output_path.write_text("cached")
                return True
            return False

        with patch.object(Path, "exists", autospec=True) as mock_exists:

            def dispatch(path_self):
                if path_self == output_path:
                    return fake_exists()
                return Path.__dict__["exists"](path_self)

            mock_exists.side_effect = dispatch
            result = download_from_url(url, cache_dir=cache_dir)

        assert result == str(output_path.absolute())
        assert mock_validate.called

    def test_download_from_url_cleanup_tolerates_unlock_and_unlink_errors(
        self, tmp_path, monkeypatch
    ):
        """Test best-effort lock cleanup ignores unlock and lock-file unlink errors."""
        url = "https://example.com/model.ckpt"
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        filename = url_to_filename(url)
        lock_path = cache_dir / f".{filename}.lock"
        original_unlink = Path.unlink
        flock_calls = {"count": 0}

        def fake_flock(*args, **kwargs):
            flock_calls["count"] += 1
            if flock_calls["count"] > 1:
                raise OSError("unlock failed")
            return None

        def fake_unlink(self):
            if self == lock_path:
                raise OSError("busy")
            return original_unlink(self)

        monkeypatch.setattr("spine.config.download.fcntl.flock", fake_flock)
        monkeypatch.setattr(Path, "unlink", fake_unlink)

        with patch("spine.config.download.download_file") as mock_download:
            mock_download.side_effect = (
                lambda _url, path, expected_hash=None: path.write_text("downloaded")
            )
            result = download_from_url(url, cache_dir=cache_dir)

        assert Path(result).exists()

    def test_validate_cached_file_missing_returns_false(self, tmp_path):
        """Test missing cached files are treated as invalid."""
        assert _validate_cached_file(tmp_path / "missing.bin") is False

    def test_validate_cached_file_hash_mismatch_removes_file(self, tmp_path, capsys):
        """Test invalid cached files are removed and rejected."""
        file_path = tmp_path / "cached.bin"
        file_path.write_text("cached")

        result = _validate_cached_file(file_path, expected_hash="0" * 64)

        assert result is False
        assert not file_path.exists()
        assert "hash mismatch" in capsys.readouterr().out

    def test_validate_cached_file_hash_mismatch_tolerates_unlink_error(
        self, tmp_path, monkeypatch
    ):
        """Test unlink failures during mismatch cleanup are ignored."""
        file_path = tmp_path / "cached.bin"
        file_path.write_text("cached")
        monkeypatch.setattr(
            Path, "unlink", lambda self: (_ for _ in ()).throw(OSError("busy"))
        )

        result = _validate_cached_file(file_path, expected_hash="0" * 64)

        assert result is False

    def test_validate_cached_file_matching_hash_returns_true(self, tmp_path, capsys):
        """Test cached files with matching hashes are accepted."""
        file_path = tmp_path / "cached.bin"
        file_path.write_text("cached")
        expected_hash = hashlib.sha256(b"cached").hexdigest()

        result = _validate_cached_file(file_path, expected_hash=expected_hash)

        assert result is True
        assert "Using cached file" in capsys.readouterr().out

    def test_validate_cached_file_handles_oserror(self, tmp_path, monkeypatch, capsys):
        """Test validation errors fall back to re-download behavior."""
        file_path = tmp_path / "cached.bin"
        file_path.write_text("cached")
        monkeypatch.setattr(
            "spine.config.download.compute_file_hash",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("bad read")),
        )

        result = _validate_cached_file(file_path, expected_hash="abc")

        assert result is False
        assert "Error validating cached file" in capsys.readouterr().out


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
            cfg = load_config_file(str(config_file))

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
            cfg = load_config_file(str(config_file))

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
            load_config_file(str(config_file))

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
            cfg = load_config_file(str(main_config))

        assert cfg["model"]["name"] == "uresnet"
        assert Path(cfg["model"]["weights"]).exists()
        assert cfg["base"]["iterations"] == 1000


def _download_worker(url, cache_dir, worker_id, result_queue):
    """Attempt a cached download from a child process."""
    try:
        time.sleep(0.1 * worker_id)

        def fake_download(url, path, expected_hash=None):
            time.sleep(0.5)
            path.write_text(f"downloaded by worker {worker_id}")

        with patch("spine.config.download.download_file", side_effect=fake_download):
            result = download_from_url(url, cache_dir=cache_dir)
            result_queue.put((worker_id, "success", result))
    except Exception as exc:  # pragma: no cover - reported through the queue
        result_queue.put((worker_id, "error", str(exc)))
