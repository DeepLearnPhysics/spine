"""Test to verify Driver is always importable from main spine module."""


def test_driver_always_importable():
    """Test that Driver can always be imported from spine, regardless of PyTorch availability."""
    # Test direct import from spine
    from spine import Driver

    assert Driver is not None
    assert hasattr(Driver, "__init__")
    assert hasattr(Driver, "__doc__")

    # Test that it's the same as importing from spine.driver
    from spine.driver import Driver as DirectDriver

    assert Driver is DirectDriver

    # Test that the class is properly documented
    assert "Central SPINE driver" in Driver.__doc__
    assert "Processes global configuration" in Driver.__doc__


def test_driver_import_with_other_classes():
    """Test that Driver can be imported alongside other main spine exports."""
    from spine import Driver, __version__

    # All should be successfully imported
    assert Driver is not None
    assert __version__ is not None

    # Version should be a string
    assert isinstance(__version__, str)

    # Classes should be classes
    assert callable(Driver)

    # Test that we can also import other key components conditionally
    try:
        from spine.data.meta import Meta
        from spine.data.particle import Particle
        from spine.data.neutrino import Neutrino
        from spine.data.batch.tensor import TensorBatch
        
        assert callable(Meta)
        assert callable(Particle)
        assert callable(Neutrino) 
        assert callable(TensorBatch)
        print("✅ Data classes successfully imported")
    except ImportError as e:
        # This is acceptable if dependencies aren't available
        print(f"⚠️ Some data classes not available: {e}")
    assert callable(TensorBatch)


if __name__ == "__main__":
    test_driver_always_importable()
    test_driver_import_with_other_classes()
    print("✅ All Driver import tests passed!")
