"""Backend interface and source identity helpers for entry filtering."""

from __future__ import annotations

import glob
import os
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

__all__ = ["EntryInspector", "SourceFingerprint", "canonical_source", "resolve_sources"]


def canonical_source(source: str) -> str:
    """Normalize one local source path or preserve a remote URI.

    Parameters
    ----------
    source : str
        Local path or supported remote URI.

    Returns
    -------
    str
        Absolute normalized local path or unchanged remote URI.
    """
    expanded = os.path.expandvars(os.path.expanduser(source))
    if expanded.startswith(("root://", "xroot://")):
        return expanded
    return str(Path(expanded).resolve())


def resolve_sources(
    sources: str | Sequence[str] | None = None,
    source_list: str | None = None,
) -> list[str]:
    """Resolve source patterns or a source-list file canonically.

    Exactly one input form must be provided. Local globs are expanded, remote
    URIs are preserved, duplicates are removed and the result is sorted.

    Parameters
    ----------
    sources : str or sequence[str], optional
        Direct source paths or local glob expressions.
    source_list : str, optional
        Text file containing one source expression per line.

    Returns
    -------
    list[str]
        Unique canonical sources in lexicographic order.
    """
    if (sources is None) == (source_list is None):
        raise ValueError("Provide exactly one of `sources` or `source_list`.")

    if source_list is not None:
        list_path = Path(source_list).expanduser()
        if not list_path.is_file():
            raise FileNotFoundError(f"Source list does not exist: {source_list}")
        source_items = [
            line.strip()
            for line in list_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    elif isinstance(sources, str):
        source_items = [sources]
    else:
        assert sources is not None
        source_items = list(sources)

    resolved: list[str] = []
    for item in source_items:
        expanded = os.path.expandvars(os.path.expanduser(item))
        if expanded.startswith(("root://", "xroot://")):
            resolved.append(expanded)
            continue

        matches = glob.glob(expanded)
        if len(matches) == 0:
            raise FileNotFoundError(f"Source expression matched no files: {item}")
        resolved.extend(canonical_source(path) for path in matches)

    sources_out = sorted(set(resolved))
    if len(sources_out) == 0:
        raise ValueError("No input sources were resolved.")
    return sources_out


@dataclass(frozen=True)
class SourceFingerprint:
    """Stable identity information used to validate a scanned source.

    Parameters
    ----------
    path : str
        Canonical source path.
    size : int, optional
        Local file size, or ``None`` when unavailable.
    mtime_ns : int, optional
        Local modification time in nanoseconds, or ``None`` when unavailable.
    """

    path: str
    size: int | None = None
    mtime_ns: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a serialization-ready fingerprint mapping."""
        return asdict(self)


class EntryInspector(ABC):
    """Abstract interface implemented by format-specific entry inspectors."""

    name: str
    version: int = 1

    @staticmethod
    def canonical_source(source: str) -> str:
        """Normalize one local path or preserve a remote source URI."""
        return canonical_source(source)

    def resolve_sources(
        self,
        sources: str | Sequence[str] | None = None,
        source_list: str | None = None,
    ) -> list[str]:
        """Resolve source patterns or a source-list file canonically.

        Parameters
        ----------
        sources : str or sequence[str], optional
            Source paths or local glob expressions.
        source_list : str, optional
            Text file containing one source expression per line.

        Returns
        -------
        list[str]
            Unique canonical sources in lexicographic order.
        """
        return resolve_sources(sources, source_list)

    def fingerprint(self, source: str) -> SourceFingerprint:
        """Build the best available fingerprint for one source."""
        path = self.canonical_source(source)
        if path.startswith(("root://", "xroot://")):
            return SourceFingerprint(path=path)

        stat_result = os.stat(path)
        return SourceFingerprint(
            path=path,
            size=int(stat_result.st_size),
            mtime_ns=int(stat_result.st_mtime_ns),
        )

    @abstractmethod
    def inspect(
        self, source: str, measurements: Mapping[str, Mapping[str, Any]]
    ) -> tuple[int, dict[str, list[int]]]:
        """Measure every requested product for every entry in one source.

        Returns
        -------
        num_entries : int
            Common number of entries across all requested products.
        measurements : dict[str, list[int]]
            Per-measurement values ordered by file-local entry number.
        """
        raise NotImplementedError
