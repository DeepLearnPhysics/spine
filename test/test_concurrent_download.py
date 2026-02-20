"""Test concurrent download protection."""

import multiprocessing
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

from spine.config.download import download_from_url, url_to_filename


def download_worker(url, cache_dir, worker_id, result_queue):
    """Worker function that attempts to download a file."""
    try:
        # Simulate variable timing
        time.sleep(0.1 * worker_id)

        # Mock the actual download to take some time
        def fake_download(url, path, expected_hash=None):
            time.sleep(0.5)  # Simulate slow download
            path.write_text(f"downloaded by worker {worker_id}")

        with patch("spine.config.download.download_file", side_effect=fake_download):
            result = download_from_url(url, cache_dir=cache_dir)
            result_queue.put((worker_id, "success", result))
    except Exception as e:
        result_queue.put((worker_id, "error", str(e)))


def test_concurrent_downloads_safe():
    """Test that concurrent downloads don't cause corruption or race conditions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        url = "https://example.com/test_model.ckpt"

        # Start multiple processes trying to download the same file
        num_workers = 5
        manager = multiprocessing.Manager()
        result_queue = manager.Queue()
        processes = []

        for i in range(num_workers):
            p = multiprocessing.Process(
                target=download_worker, args=(url, cache_dir, i, result_queue)
            )
            processes.append(p)
            p.start()

        # Wait for all processes to complete
        for p in processes:
            p.join(timeout=30)

        # Collect results
        results = []
        while not result_queue.empty():
            results.append(result_queue.get())

        # All workers should succeed
        assert len(results) == num_workers
        assert all(status == "success" for _, status, _ in results)

        # All should get the same file path
        paths = [result for _, _, result in results]
        assert len(set(paths)) == 1

        # File should exist exactly once
        filename = url_to_filename(url)
        final_path = cache_dir / filename
        assert final_path.exists()

        # No leftover lock files
        lock_files = list(cache_dir.glob(".*.lock"))
        assert len(lock_files) == 0

        # No leftover temp files
        temp_files = list(cache_dir.glob(".*tmp*"))
        assert len(temp_files) == 0


if __name__ == "__main__":
    test_concurrent_downloads_safe()
    print("✓ Concurrent download test passed!")
