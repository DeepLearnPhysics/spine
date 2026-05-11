"""Tests for top-level IO package behavior."""

import pytest


def test_torch_io_conditional_import():
    """The top-level IO package should expose the torch availability flag."""
    from spine.io import TORCH_IO_AVAILABLE

    if TORCH_IO_AVAILABLE:
        from spine.io import loader_factory

        assert callable(loader_factory)
    else:
        with pytest.raises(ImportError):
            from spine.io import loader_factory


def test_meta_import():
    """Meta should still be importable independently of the IO refactor."""
    from spine.data import Meta

    assert Meta is not None


def test_io_exports_read_write_factories():
    """The top-level IO package should still export the main factories."""
    from spine.io import reader_factory, writer_factory

    assert callable(reader_factory)
    assert callable(writer_factory)
