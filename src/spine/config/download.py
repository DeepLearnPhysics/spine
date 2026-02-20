"""File download utilities for SPINE configuration system.

This module provides utilities for downloading model weights and other files
referenced in YAML configurations.
"""

import fcntl
import hashlib
import os
import sys
import tempfile
import time
import urllib.parse
import urllib.request
import warnings
from pathlib import Path
from typing import Optional
from urllib.error import HTTPError, URLError


def get_cache_dir() -> Path:
    """Get the directory for caching downloaded files.

    By default, creates a '.cache/weights/' directory in SPINE_BASEDIR or
    SPINE_PROD_BASEDIR (if running from spine-prod). This ensures downloads
    are cached centrally regardless of execution directory.

    Can be overridden with the SPINE_CACHE_DIR environment variable.

    Returns
    -------
    Path
        Path to cache directory
    """
    cache_dir = os.environ.get("SPINE_CACHE_DIR")
    if cache_dir:
        return Path(cache_dir)

    # Try SPINE_PROD_BASEDIR first (if running from spine-prod)
    spine_prod_base = os.environ.get("SPINE_PROD_BASEDIR")
    if spine_prod_base:
        return Path(spine_prod_base) / ".cache" / "weights"

    # Fall back to SPINE_BASEDIR
    spine_base = os.environ.get("SPINE_BASEDIR")
    if spine_base:
        return Path(spine_base) / ".cache" / "weights"

    # Last resort: use current directory (with warning)
    warnings.warn(
        "SPINE_BASEDIR and SPINE_PROD_BASEDIR not set. "
        "Using current directory for cache. "
        "Please source configure.sh for proper caching.",
        UserWarning,
        stacklevel=2,
    )
    return Path.cwd() / "weights"


def compute_file_hash(filepath: Path, algorithm: str = "sha256") -> str:
    """Compute hash of a file.

    Parameters
    ----------
    filepath : Path
        Path to file to hash
    algorithm : str, optional
        Hash algorithm to use (default: sha256)

    Returns
    -------
    str
        Hex digest of file hash
    """
    hasher = hashlib.new(algorithm)
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def url_to_filename(url: str) -> str:
    """Convert URL to a safe filename using hash.

    Preserves the original extension if present.

    Parameters
    ----------
    url : str
        URL to convert

    Returns
    -------
    str
        Safe filename
    """
    # Extract extension from URL
    url_path = urllib.parse.urlparse(url).path
    ext = Path(url_path).suffix

    # Hash the URL to create a unique filename
    url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]

    # Combine hash with extension
    return f"{url_hash}{ext}"


def download_file(
    url: str,
    output_path: Path,
    expected_hash: Optional[str] = None,
    hash_algorithm: str = "sha256",
) -> None:
    """Download a file from a URL with progress reporting.

    Parameters
    ----------
    url : str
        URL to download from
    output_path : Path
        Path where file should be saved
    expected_hash : str, optional
        Expected hash of downloaded file for validation
    hash_algorithm : str, optional
        Hash algorithm to use for validation (default: sha256)

    Raises
    ------
    HTTPError
        If download fails with HTTP error
    URLError
        If download fails with URL error
    ValueError
        If downloaded file hash doesn't match expected hash
    """
    print(f"Downloading: {url}")
    print(f"Destination: {output_path}")

    try:
        # Download with progress reporting
        def progress_hook(count, block_size, total_size):
            if total_size > 0:
                percent = min(100, count * block_size * 100 // total_size)
                sys.stdout.write(f"\rProgress: {percent}% ")
                sys.stdout.flush()

        urllib.request.urlretrieve(url, output_path, reporthook=progress_hook)
        print()  # New line after progress

    except HTTPError as e:
        raise HTTPError(
            url, e.code, f"HTTP Error {e.code}: {e.reason}", e.hdrs, e.fp
        ) from e

    except URLError as e:
        raise URLError(f"Failed to download {url}: {e.reason}") from e

    # Validate hash if provided
    if expected_hash:
        actual_hash = compute_file_hash(output_path, hash_algorithm)
        if actual_hash != expected_hash:
            output_path.unlink()  # Remove corrupted file
            raise ValueError(
                f"Downloaded file hash mismatch!\n"
                f"Expected: {expected_hash}\n"
                f"Got:      {actual_hash}\n"
                f"File has been removed. Please try again."
            )
        print(f"✓ Hash validated: {actual_hash[:16]}...")


def download_from_url(
    url: str,
    expected_hash: Optional[str] = None,
    cache_dir: Optional[Path] = None,
    max_wait_seconds: int = 3600,
) -> str:
    """Download a file from URL and return the cached path.

    If the file already exists in cache and passes hash validation (if provided),
    returns the cached path without re-downloading.

    This function is safe for concurrent access from multiple processes. It uses
    file locking to ensure only one process downloads at a time, while others wait.

    Parameters
    ----------
    url : str
        URL to download from
    expected_hash : str, optional
        Expected SHA256 hash of file for validation
    cache_dir : Path, optional
        Directory to cache downloaded files (default: from get_cache_dir())
    max_wait_seconds : int, optional
        Maximum time to wait for lock acquisition in seconds (default: 3600)

    Returns
    -------
    str
        Absolute path to cached file

    Raises
    ------
    HTTPError
        If download fails with HTTP error
    URLError
        If download fails with URL error
    ValueError
        If hash validation fails
    TimeoutError
        If unable to acquire lock within max_wait_seconds

    Examples
    --------
    >>> path = download_from_url(
    ...     "https://example.com/model.ckpt",
    ...     expected_hash="abc123..."
    ... )
    >>> print(f"Model downloaded to: {path}")
    """
    if cache_dir is None:
        cache_dir = get_cache_dir()

    # Create cache directory if it doesn't exist
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename from URL
    filename = url_to_filename(url)
    output_path = cache_dir / filename
    lock_path = cache_dir / f".{filename}.lock"

    # Quick check without lock (optimization)
    if output_path.exists() and _validate_cached_file(output_path, expected_hash):
        return str(output_path.absolute())

    # Acquire lock to prevent concurrent downloads
    lock_file = None
    try:
        lock_file = open(lock_path, "w", encoding="utf-8")

        # Try to acquire lock with timeout
        start_time = time.time()
        while True:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break  # Lock acquired
            except BlockingIOError as exc:
                # Lock is held by another process
                elapsed = time.time() - start_time
                if elapsed > max_wait_seconds:
                    raise TimeoutError(
                        f"Timeout waiting for download lock after {max_wait_seconds}s. "
                        f"Another process may be downloading {url}"
                    ) from exc

                # Wait and retry
                if elapsed < 10:
                    # During first 10 seconds, print immediately
                    print(
                        f"Waiting for another process to finish downloading {filename}..."
                    )
                elif int(elapsed) % 60 == 0:
                    # After that, print every minute
                    print(f"Still waiting ({int(elapsed)}s elapsed)...")

                time.sleep(1)

        # Double-check if file exists (another process may have downloaded it)
        if output_path.exists() and _validate_cached_file(output_path, expected_hash):
            return str(output_path.absolute())

        # Download to a temporary file first (atomic operation)
        temp_fd, temp_path = tempfile.mkstemp(
            dir=cache_dir, prefix=f".{filename}.", suffix=".tmp"
        )
        temp_path = Path(temp_path)

        try:
            os.close(temp_fd)  # Close fd, we'll use the path

            # Download the file
            download_file(url, temp_path, expected_hash)

            # Atomically move to final location (overwrites if exists)
            temp_path.replace(output_path)

            print(f"✓ Download complete: {output_path}")
            return str(output_path.absolute())

        except Exception:
            # Clean up temp file on error
            if temp_path.exists():
                temp_path.unlink()
            raise

    finally:
        # Release lock and cleanup
        if lock_file:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                lock_file.close()
            except OSError:
                pass  # Best effort cleanup

            # Remove lock file (best effort - may fail if other processes waiting)
            try:
                lock_path.unlink()
            except OSError:
                pass


def _validate_cached_file(
    file_path: Path,
    expected_hash: Optional[str] = None,
) -> bool:
    """Validate a cached file exists and has correct hash.

    Parameters
    ----------
    file_path : Path
        Path to file to validate
    expected_hash : str, optional
        Expected hash to validate against

    Returns
    -------
    bool
        True if file is valid, False otherwise
    """
    try:
        if not file_path.exists():
            return False

        # If no hash expected, just check existence
        if not expected_hash:
            print(f"✓ Using cached file: {file_path}")
            return True

        # Validate hash
        actual_hash = compute_file_hash(file_path)
        if actual_hash == expected_hash:
            print(f"✓ Using cached file: {file_path}")
            return True
        else:
            print("⚠ Cached file hash mismatch, will re-download...")
            try:
                file_path.unlink()
            except OSError:
                pass  # Best effort cleanup
            return False

    except OSError as e:
        print(f"⚠ Error validating cached file: {e}, will re-download...")
        return False
