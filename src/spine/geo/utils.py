"""Shared helpers for geometry modules."""

from __future__ import annotations


def normalize_version(version: str | int | float | None) -> str | None:
    """Normalize a geometry version to the package comparison format.

    Parameters
    ----------
    version : str, int, float or None
        Geometry version value read from user input or a geometry YAML file.

    Returns
    -------
    str or None
        Version represented as a float-like string, e.g. ``"6.0"`` or
        ``"6.5"``. Returns ``None`` when the input is ``None``.
    """
    if version is None:
        return None

    return str(float(version))


def version_key(version: str | int | float) -> tuple[float, ...]:
    """Build a numeric sorting key for a geometry version.

    Parameters
    ----------
    version : str, int or float
        Geometry version value to sort.

    Returns
    -------
    tuple[float, ...]
        Numeric version components suitable for ordering, so ``"10.0"`` sorts
        after ``"6.5"``.
    """
    return tuple(float(part) for part in str(version).split("."))


def version_matches(
    current: str | int | float,
    requested: str | int | float | None,
) -> bool:
    """Check whether a current version satisfies a requested version.

    A major-only string request, e.g. ``"6"``, matches any ``6.x`` version.
    Numeric requests are normalized exactly, so ``6`` matches ``6.0``.

    Parameters
    ----------
    current : str, int or float
        Normalized or raw version currently associated with a geometry object.
    requested : str, int, float or None
        Requested version constraint. ``None`` matches any current version.

    Returns
    -------
    bool
        ``True`` if the current version satisfies the request.
    """
    if requested is None:
        return True

    requested_parts = str(requested).split(".")
    if isinstance(requested, str) and len(requested_parts) == 1:
        return str(current).split(".")[0] == requested_parts[0]

    return normalize_version(current) == normalize_version(requested)
