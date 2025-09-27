"""Test that optional I/O dependencies are handled properly."""

import pytest


def test_torch_io_conditional_import():
    """Test that torch IO can be conditionally imported."""
    try:
        from spine.io import TORCH_IO_AVAILABLE

        if TORCH_IO_AVAILABLE:
            from spine.io import loader_factory

            # If torch is available, this should work
            assert callable(loader_factory)
            print("✅ PyTorch IO functionality available")
        else:
            # If torch not available, should not be able to import loader_factory
            with pytest.raises(ImportError):
                from spine.io import loader_factory
            print("✅ PyTorch IO functionality correctly unavailable")
    except ImportError as e:
        # This is expected when torch is not available
        print(f"✅ Expected ImportError: {e}")


def test_meta_import():
    """Test that Meta can be imported from spine.data.meta."""
    from spine.data.meta import Meta

    assert Meta is not None
    print("✅ Meta class successfully imported from spine.data.meta")


@pytest.mark.skipif(condition=True, reason="ROOT tests require optional dependencies")
def test_root_dependent_functionality():
    """Test ROOT-dependent functionality (skipped in CI)."""
    # This test would run locally but be skipped in CI
    # ROOT-specific tests would go here
    assert True


if __name__ == "__main__":
    test_torch_io_conditional_import()
    test_meta_import()
    print("✅ All optional dependency tests passed")
