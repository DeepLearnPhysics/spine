"""Construct a geometry module class from its name."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .base import Geometry
from .utils import normalize_version, version_key

# Get config directory relative to this module
GEO_CONFIG_DIR = Path(__file__).parent / "config"

__all__ = ["geo_factory"]


def geo_dict() -> dict[Path, dict[str, str]]:
    """Builds a dictionary of available geometry modules.

    Returns
    -------
    dict
        Dictionary of available geometry modules
    """
    # Gather all geometry yaml files from the config directory
    options = {}
    for path in GEO_CONFIG_DIR.glob("*/*_geometry.yaml"):
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        version = normalize_version(cfg["version"])
        if version is None:
            raise ValueError(f"Geometry configuration is missing a version: {path}")
        options[path] = {"version": version}
        options[path].update({k: str(cfg[k]) for k in ("name", "tag")})

    return options


def geo_factory(
    detector: str,
    tag: str | None = None,
    version: str | int | float | None = None,
) -> Geometry:
    """Instantiates a geometry module from a name or

    Parameters
    ----------
    detector : str
        Name of the detector (e.g. "icarus", "2x2", "ndlar", "protodune-vd", etc.)
    tag : str, optional
        Geometry tag (e.g. "sbndv2", "mr5", etc.)
    version : str, optional
        Geometry version (e.g. "1", "6.5", etc.)

    Returns
    -------
    object
         Initialized geometry object
    """
    # Find a geometry configuration that matches the requested parameters
    options = geo_dict()
    requested_version = normalize_version(version)
    requested_version_parts = str(version).split(".") if version is not None else []
    paths, tags, versions = [], [], []
    for path, cfg in options.items():
        # If the detector name matches, store the path
        if cfg["name"].lower() == detector.lower():
            paths.append(path)
            tags.append(cfg.get("tag", None))
            versions.append(cfg.get("version", None))

    if len(paths) == 0:
        raise ValueError(f"No geometry found for detector '{detector}'.")

    # If a tag is specified, must find the exact tag or throw
    file_path = ""
    if tag is not None:
        if tag in tags:
            index = tags.index(tag)
            if requested_version is not None and requested_version != versions[index]:
                raise ValueError(
                    f"Geometry version '{version}' does not match found version "
                    f"'{versions[index]}' for detector '{detector}' with tag '{tag}'."
                )
            file_path = paths[index]
        else:
            raise ValueError(
                f"No geometry found for detector '{detector}' with tag '{tag}'. "
                f"Available tags are: {set(tags)}"
            )

    # If a version is specified, must match major revision if it is the only
    # one specified or both if major and minor are specified
    elif requested_version is not None:
        matches = []
        for i, ver in enumerate(versions):
            ver_parts = ver.split(".")
            if len(requested_version_parts) == 1:
                # Only major version specified
                if requested_version_parts[0] == ver_parts[0]:
                    matches.append(i)
            elif len(requested_version_parts) == 2:
                # Major and minor version specified
                if requested_version == ver:
                    matches.append(i)
        if not matches:
            raise ValueError(
                f"No geometry found for detector '{detector}' with version '{version}'. "
                f"Available versions are: {set(versions)}"
            )
        index = max(matches, key=lambda i: version_key(versions[i]))
        file_path = paths[index]

    # If no tag or version is specified, return the most recent version
    else:
        index = max(range(len(versions)), key=lambda i: version_key(versions[i]))
        file_path = paths[index]

    # Parse configuration file as a dictionary
    with open(file_path, "r", encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f)

    # Instantiate the geometry module
    return Geometry(**cfg)
