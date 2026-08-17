"""Tests for geometry factory helpers."""

import numpy as np
import pytest
import yaml

from spine.geo.factories import GEO_CONFIG_DIR, geo_dict, geo_factory
from spine.geo.utils import normalize_version, version_key, version_matches

from .conftest import write_geometry_config


def test_normalize_version_and_version_key():
    """Version helpers should compare numeric versions, not strings."""
    assert normalize_version(None) is None
    assert normalize_version(6) == "6.0"
    assert normalize_version("6.5") == "6.5"
    assert version_key("10.0") > version_key("6.5")
    assert version_matches("6.5", "6") is True
    assert version_matches("6.5", 6) is False


def test_geo_dict_reads_available_configs(geo_config_dir):
    """Geometry configs should be indexed by normalized metadata."""
    path = write_geometry_config(geo_config_dir, "demo", "v1", "1")

    options = geo_dict()

    assert options[path] == {"name": "demo", "tag": "v1", "version": "1.0"}


def test_geo_factory_accepts_deprecated_aliases(geo_config_dir):
    """Geometry aliases should preserve old names while naming replacements."""
    path = write_geometry_config(geo_config_dir, "DUNE-HD", "v1", "1")
    text = path.read_text(encoding="utf-8")
    path.write_text(
        text.replace("name: DUNE-HD", "name: DUNE-HD\naliases: [DUNE]"),
        encoding="utf-8",
    )

    with pytest.warns(DeprecationWarning, match="use 'DUNE-HD'"):
        geo = geo_factory("DUNE")

    assert geo.name == "DUNE-HD"


def test_packaged_geometries_declare_up_direction():
    """Every packaged detector should explicitly define its physical up axis."""
    paths = sorted(GEO_CONFIG_DIR.glob("*/*_geometry.yaml"))

    assert paths
    for path in paths:
        with open(path, "r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        expected = (
            [1.0, 0.0, 0.0]
            if path.parent.name in {"dunevd10kt-1x8x6", "protodune-vd"}
            else [0.0, 1.0, 0.0]
        )
        assert np.array_equal(config["up_dir"], expected), path


def test_packaged_dune_hd_geometry_preserves_old_name():
    """The clarified DUNE-HD name should retain its historical factory alias."""
    with pytest.warns(DeprecationWarning, match="DUNE-HD-10kt-1x2x6"):
        geo = geo_factory("DUNE10kt-1x2x6")

    assert geo.name == "DUNE-HD-10kt-1x2x6"


def test_geo_dict_rejects_missing_version(geo_config_dir):
    """Geometry configs should fail clearly when version is absent."""
    path = write_geometry_config(geo_config_dir, "demo", "v1", "null")
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace("version: null", "version:"), encoding="utf-8")

    with pytest.raises(ValueError, match="missing a version"):
        geo_dict()


def test_geo_factory_selects_numeric_latest_version(geo_config_dir):
    """Default geometry selection should use numeric version ordering."""
    write_geometry_config(geo_config_dir, "demo", "v6", "6.5")
    write_geometry_config(geo_config_dir, "demo", "v10", "10.0")

    geo = geo_factory("demo")

    assert geo.tag == "v10"
    assert geo.version == "10.0"


def test_geo_factory_selects_latest_major_version(geo_config_dir):
    """Major-only version requests should select the latest matching minor."""
    write_geometry_config(geo_config_dir, "demo", "v6-0", "6.0")
    write_geometry_config(geo_config_dir, "demo", "v6-5", "6.5")
    write_geometry_config(geo_config_dir, "demo", "v10", "10.0")

    geo = geo_factory("demo", version="6")

    assert geo.tag == "v6-5"
    assert geo.version == "6.5"


def test_geo_factory_selects_exact_minor_version(geo_config_dir):
    """Major-minor version requests should select exact normalized versions."""
    write_geometry_config(geo_config_dir, "demo", "v6-0", "6.0")
    write_geometry_config(geo_config_dir, "demo", "v6-5", "6.5")

    geo = geo_factory("demo", version="6.5")

    assert geo.tag == "v6-5"
    assert geo.version == "6.5"


def test_geo_factory_accepts_normalized_tag_version(geo_config_dir):
    """Tag and version selection should compare normalized versions."""
    write_geometry_config(geo_config_dir, "demo", "v6", "6")

    geo = geo_factory("demo", tag="v6", version=6.0)

    assert geo.tag == "v6"
    assert geo.version == "6.0"


def test_geo_factory_rejects_bad_requests(geo_config_dir):
    """Missing or inconsistent geometry requests should fail clearly."""
    write_geometry_config(geo_config_dir, "demo", "v1", "1")

    with pytest.raises(ValueError, match="No geometry found"):
        geo_factory("missing")

    with pytest.raises(ValueError, match="with tag"):
        geo_factory("demo", tag="missing")

    with pytest.raises(ValueError, match="does not match found version"):
        geo_factory("demo", tag="v1", version="2")

    with pytest.raises(ValueError, match="with version"):
        geo_factory("demo", version="2")
