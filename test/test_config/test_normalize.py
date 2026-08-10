"""Tests for runtime configuration normalization."""

import pytest

from spine.config import normalize_config


def test_normalize_config_relocates_legacy_train_block():
    """Legacy training configuration should warn and move without mutation."""
    cfg = {"base": {"seed": 1, "train": {"optimizer": {}}}}

    with pytest.warns(FutureWarning, match="base.train"):
        normalized = normalize_config(cfg)

    assert cfg["base"]["train"] == {"optimizer": {}}
    assert normalized == {
        "base": {"seed": 1},
        "train": {"optimizer": {}},
    }


def test_normalize_config_rejects_duplicate_train_blocks():
    """Training configuration should have one unambiguous owner."""
    with pytest.raises(ValueError, match="either at top level"):
        normalize_config({"base": {"train": {}}, "train": {}})
