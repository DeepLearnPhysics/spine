"""Tests for top-level IO package behavior."""

import pytest


def test_torch_io_conditional_import():
    """The top-level IO package should expose the torch availability flag and loader factory."""
    from spine.io import TORCH_IO_AVAILABLE, loader_factory

    if TORCH_IO_AVAILABLE:
        assert callable(loader_factory)
    else:
        assert callable(loader_factory)
        with pytest.raises(ImportError, match="PyTorch is required"):
            loader_factory(dataset={}, dtype="float32", batch_size=1)


def test_meta_import():
    """Meta should still be importable independently of the IO refactor."""
    from spine.data import Meta

    assert Meta is not None


def test_io_exports_read_write_factories():
    """The top-level IO package should still export the main factories."""
    from spine.io import reader_factory, writer_factory

    assert callable(reader_factory)
    assert callable(writer_factory)
