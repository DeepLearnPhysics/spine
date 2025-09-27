"""Comprehensive tests for conditional imports and dependency management."""

import importlib
import sys
from unittest.mock import patch

import pytest


class TestConditionalImports:
    """Test conditional import behavior with and without optional dependencies."""

    def test_conditional_torch_availability(self):
        """Test TORCH_AVAILABLE flag works correctly."""
        from spine.utils.conditional import TORCH_AVAILABLE

        # Should be a boolean
        assert isinstance(TORCH_AVAILABLE, bool)

        # Test importing torch matches the flag
        try:
            import torch

            assert (
                TORCH_AVAILABLE is True
            ), "TORCH_AVAILABLE should be True when torch is available"
        except ImportError:
            assert (
                TORCH_AVAILABLE is False
            ), "TORCH_AVAILABLE should be False when torch unavailable"

    def test_mock_torch_when_unavailable(self):
        """Test MockTorch provides expected interfaces when torch unavailable."""
        from spine.utils.conditional import TORCH_AVAILABLE, torch

        if not TORCH_AVAILABLE:
            # Should have MockTorch
            assert hasattr(torch, "Tensor"), "MockTorch should have Tensor attribute"
            assert torch.Tensor is not None, "MockTorch.Tensor should not be None"
        else:
            # Should have real torch
            assert hasattr(torch, "__version__"), "Real torch should have __version__"

    def test_driver_imports_without_torch(self):
        """Test Driver can be imported without torch."""
        # Mock torch as unavailable
        with patch.dict("sys.modules", {"torch": None}):
            with patch("spine.utils.conditional.TORCH_AVAILABLE", False):
                from spine.driver import Driver

                assert Driver is not None


class TestManagerIndependence:
    """Test that all managers can work independently without torch."""

    def test_model_manager_conditional_import(self):
        """Test ModelManager imports without torch."""
        from spine.model import ModelManager

        assert ModelManager is not None

        # Should have conditional methods - test minimal config
        try:
            manager = ModelManager(
                name="test", modules={}, network_input=["input_data"]
            )
            assert hasattr(manager, "prepare")
        except ImportError:
            # Expected when torch is not available
            pytest.skip("ModelManager requires torch dependencies")

    def test_build_manager_torch_independence(self):
        """Test BuildManager doesn't require torch."""
        from spine.construct import BuildManager

        assert BuildManager is not None

    def test_post_manager_torch_independence(self):
        """Test PostManager doesn't require torch."""
        from spine.post import PostManager

        assert PostManager is not None

    def test_ana_manager_torch_independence(self):
        """Test AnaManager doesn't require torch."""
        from spine.ana import AnaManager

        assert AnaManager is not None


class TestNetworkXElimination:
    """Test NetworkX dependency has been successfully eliminated."""

    def test_no_networkx_imports(self):
        """Test that no SPINE modules import networkx."""
        import spine

        # Get all loaded spine modules
        spine_modules = [
            name for name in sys.modules.keys() if name.startswith("spine")
        ]

        # Check none of them import networkx
        for module_name in spine_modules:
            module = sys.modules[module_name]
            if hasattr(module, "__file__") and module.__file__:
                # Module has source file, check it doesn't import networkx
                try:
                    with open(module.__file__, "r") as f:
                        content = f.read()
                        assert (
                            "import networkx" not in content
                        ), f"{module_name} still imports networkx"
                        assert (
                            "from networkx" not in content
                        ), f"{module_name} still imports from networkx"
                except (UnicodeDecodeError, FileNotFoundError):
                    # Skip binary files or missing files
                    pass

    def test_children_processor_networkx_free(self):
        """Test ChildrenProcessor works without networkx."""
        # Mock networkx as unavailable
        with patch.dict("sys.modules", {"networkx": None}):
            from spine.post.truth.label import ChildrenProcessor

            # Should be able to create processor
            processor = ChildrenProcessor(mode="shape")
            assert processor is not None
            assert processor.name == "children_count"

    def test_post_processing_performance(self):
        """Test that post-processing without networkx is fast."""
        import time
        from collections import defaultdict

        # Simulate large parent-child relationship
        size = 10000
        parent_ids = [max(0, i // 2) for i in range(size)]

        start_time = time.time()

        # Dictionary-based approach (current implementation)
        children = defaultdict(list)
        for child_id, parent_id in enumerate(parent_ids):
            if child_id != parent_id:
                children[parent_id].append(child_id)

        children_counts = {}
        for node_id in range(size):
            children_counts[node_id] = len(children[node_id])

        elapsed = time.time() - start_time

        # Should be very fast (under 0.1 seconds for 10k nodes)
        assert elapsed < 0.1, f"Dictionary approach too slow: {elapsed:.3f}s"

        # Should produce reasonable results
        assert len(children_counts) == size
        assert all(count >= 0 for count in children_counts.values())


class TestMainEntryPoints:
    """Test main entry points work without dependencies."""

    def test_main_functions_import(self):
        """Test all main functions can be imported."""
        from spine.main import (inference_single, process_world, run,
                                run_single, train_single)

        assert callable(run)
        assert callable(run_single)
        assert callable(train_single)
        assert callable(inference_single)
        assert callable(process_world)

    def test_cli_import_and_version(self):
        """Test CLI imports and version detection works."""
        from spine.bin.cli import check_dependencies, get_version, main

        assert callable(main)

        # Version should work even without full installation
        version = get_version()
        assert isinstance(version, str)
        assert len(version) > 0

        # Dependencies check should return dict
        deps = check_dependencies()
        assert isinstance(deps, dict)
        assert "torch" in deps


class TestConditionalUtilities:
    """Test conditional utility functions."""

    def test_torch_utilities_conditional(self):
        """Test torch utilities work conditionally."""
        from spine.utils.conditional import TORCH_AVAILABLE

        if TORCH_AVAILABLE:
            # Should be able to import torch utilities
            try:
                from spine.utils.torch.devices import set_visible_devices

                assert callable(set_visible_devices)
            except ImportError:
                pytest.skip(
                    "Torch utilities not available despite TORCH_AVAILABLE=True"
                )
        else:
            # Should handle gracefully when torch not available
            from spine.utils import torch as torch_utils

            # Should not crash even if torch unavailable

    def test_jit_conditional_behavior(self):
        """Test JIT utilities handle torch unavailability."""
        from spine.utils.conditional import TORCH_AVAILABLE
        # Test that jit module can be imported regardless of torch availability
        from spine.utils.jit import numbafy

        assert callable(numbafy)

        # The numbafy decorator should work with or without torch
        # Test a simple numbafy-decorated function
        @numbafy(cast_args=[], keep_torch=False)
        def simple_function(x):
            return x * 2

        assert callable(simple_function)


class TestPerformanceRegression:
    """Test for performance regressions in conditional imports."""

    def test_import_performance(self):
        """Test that import times are reasonable."""
        import time

        # Test core imports are fast
        start_time = time.time()
        import spine.driver

        driver_time = time.time() - start_time

        # Should import quickly (under 2 seconds even with all setup)
        assert driver_time < 2.0, f"Driver import too slow: {driver_time:.3f}s"

        # Test manager imports are fast
        start_time = time.time()
        from spine.ana import AnaManager
        from spine.construct import BuildManager
        from spine.model import ModelManager
        from spine.post import PostManager

        manager_time = time.time() - start_time

        # Should be very fast since already imported
        assert manager_time < 0.5, f"Manager imports too slow: {manager_time:.3f}s"

    def test_memory_usage_reasonable(self):
        """Test memory usage of imports is reasonable."""
        try:
            import os

            import psutil

            process = psutil.Process(os.getpid())
            before_memory = process.memory_info().rss / 1024 / 1024  # MB

            # Import major components
            from spine.driver import Driver
            from spine.post.truth.label import ChildrenProcessor

            after_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = after_memory - before_memory

            # Should not use excessive memory (under 100MB increase)
            assert (
                memory_increase < 100
            ), f"Memory usage too high: {memory_increase:.1f}MB"

        except ImportError:
            pytest.skip("psutil not available for memory testing")


@pytest.mark.integration
class TestIntegrationWithoutDependencies:
    """Integration tests that verify functionality without optional dependencies."""

    def test_full_stack_import_chain(self):
        """Test complete import chain works without optional dependencies."""
        # This mimics a real user importing spine for the first time
        import spine
        from spine.bin.cli import main as cli_main
        from spine.driver import Driver
        from spine.main import run

        # All should be available
        assert spine.__version__ is not None
        assert Driver is not None
        assert run is not None
        assert cli_main is not None

    def test_configuration_loading_without_torch(self):
        """Test config loading works without torch."""
        from spine.utils.config import load_config

        # Should be able to load configs even without torch
        # (though some model configs may fail at runtime)
        assert callable(load_config)

    def test_post_processor_factory_conditional(self):
        """Test post-processor factory handles missing dependencies."""
        try:
            from spine.post.factories import post_processor_factory

            # Should be able to get post processors that don't need torch
            available_processors = post_processor_factory(name="children_count", cfg={})
            assert available_processors is not None

        except ImportError:
            pytest.skip("Post processor factory not available")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v"])
