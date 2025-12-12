"""Construct a geometry module class from its name."""

from pathlib import Path
from typing import Dict, Optional, Union

import yaml

from .base import Geometry

# Get config directory relative to this module
GEO_CONFIG_DIR = Path(__file__).parent / "config"

__all__ = ["geo_factory"]


def geo_dict() -> Dict[Path, Dict[str, str]]:
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

        options[path] = {k: cfg[k] for k in ("name", "tag", "version")}
        options[path]["version"] = str(float(options[path]["version"]))

    return options


def geo_factory(
    detector: str,
    tag: Optional[str] = None,
    version: Optional[Union[str, int, float]] = None,
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
            assert version is None or version == versions[index], (
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
    elif version is not None:
        version_parts = str(version).split(".")
        matched = False
        for i, ver in enumerate(versions):
            ver_parts = ver.split(".")
            if len(version_parts) == 1:
                # Only major version specified
                if version_parts[0] == ver_parts[0]:
                    file_path = paths[i]
                    matched = True
                    break
            elif len(version_parts) == 2:
                # Major and minor version specified
                if (version_parts[0] == ver_parts[0]) and (
                    version_parts[1] == ver_parts[1]
                ):
                    file_path = paths[i]
                    matched = True
                    break
        if not matched:
            raise ValueError(
                f"No geometry found for detector '{detector}' with version '{version}'. "
                f"Available versions are: {set(versions)}"
            )

    # If no tag or version is specified, return the most recent version
    else:
        index = versions.index(max(versions))
        file_path = paths[index]

    # Parse configuration file as a dictionary
    with open(file_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Instantiate the geometry module
    return Geometry(**cfg)
