"""Test that optional I/O dependencies are handled properly."""

import pytest


def test_torch_io_conditional_import():
    """Test that torch IO can be conditionally imported."""
    from spine.io import TORCH_IO_AVAILABLE

    if TORCH_IO_AVAILABLE:
        from spine.io import loader_factory

        assert callable(loader_factory)
    else:
        with pytest.raises(ImportError):
            from spine.io import loader_factory


def test_meta_import():
    """Test that Meta can be imported from spine.data.meta."""
    from spine.data import Meta

    assert Meta is not None


@pytest.mark.skipif(condition=True, reason="ROOT tests require optional dependencies")
def test_root_dependent_functionality():
    """Test ROOT-dependent functionality (skipped in CI)."""
    # This test would run locally but be skipped in CI
    # ROOT-specific tests would go here
    assert True
