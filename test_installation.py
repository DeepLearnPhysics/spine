#!/usr/bin/env python3
"""
Test script to validate SPINE package installation and imports.
This script tests different import scenarios for both full and core packages.
"""

import sys
import importlib
import traceback
from typing import List, Tuple


def test_import(module_name: str, description: str = "") -> Tuple[bool, str]:
    """Test importing a module and return success status with message."""
    try:
        importlib.import_module(module_name)
        return True, f"✓ {module_name} {description}"
    except ImportError as e:
        return False, f"✗ {module_name} {description} - {str(e)}"
    except Exception as e:
        return False, f"✗ {module_name} {description} - Unexpected error: {str(e)}"


def test_core_imports() -> List[Tuple[bool, str]]:
    """Test imports that should work with spine-ml-core."""
    tests = [
        ("spine", "- Main package"),
        ("spine.version", "- Version module"),
        ("spine.data", "- Data structures"),
        ("spine.utils", "- Utility functions"),
        ("spine.vis", "- Visualization tools"),
        ("spine.io", "- I/O functions"),
        ("spine.math", "- Math utilities"),
        ("spine.post", "- Post-processing"),
    ]

    results = []
    for module, desc in tests:
        results.append(test_import(module, desc))

    return results


def test_ml_imports() -> List[Tuple[bool, str]]:
    """Test imports that require ML dependencies (spine-ml[full])."""
    tests = [
        ("spine.model", "- ML models (requires torch)"),
        ("torch", "- PyTorch framework"),
        ("numba", "- JIT compilation"),
        ("sklearn", "- Scikit-learn"),
    ]

    results = []
    for module, desc in tests:
        results.append(test_import(module, desc))

    return results


def test_optional_imports() -> List[Tuple[bool, str]]:
    """Test optional dependencies that may not be installed."""
    tests = [
        ("plotly", "- Plotly visualization"),
        ("matplotlib", "- Matplotlib plotting"),
        ("seaborn", "- Seaborn statistical plots"),
        ("MinkowskiEngine", "- Sparse convolutions"),
        ("larcv", "- LArCV data I/O"),
    ]

    results = []
    for module, desc in tests:
        results.append(test_import(module, desc))

    return results


def test_functionality():
    """Test basic functionality."""
    try:
        # Test version access
        import spine

        version = spine.__version__
        print(f"✓ SPINE version: {version}")

        # Test basic data structures
        from spine.data import TensorBatch  # noqa: F401

        print("✓ Data structures accessible")

        # Test utilities
        from spine.utils.logger import logger  # noqa: F401

        print("✓ Utilities accessible")

        return True
    except Exception as e:  # noqa: BLE001
        print(f"✗ Functionality test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all import tests."""
    print("SPINE Package Import Test")
    print("=" * 50)

    print("\n1. Testing Core Package Imports (should work with spine-ml-core):")
    core_results = test_core_imports()
    core_success = 0
    for success, message in core_results:
        print(f"  {message}")
        if success:
            core_success += 1

    print(f"\n   Core imports: {core_success}/{len(core_results)} successful")

    print("\n2. Testing ML Package Imports (requires spine-ml[full]):")
    ml_results = test_ml_imports()
    ml_success = 0
    for success, message in ml_results:
        print(f"  {message}")
        if success:
            ml_success += 1

    print(f"\n   ML imports: {ml_success}/{len(ml_results)} successful")

    print("\n3. Testing Optional Dependencies:")
    opt_results = test_optional_imports()
    opt_success = 0
    for success, message in opt_results:
        print(f"  {message}")
        if success:
            opt_success += 1

    print(f"\n   Optional imports: {opt_success}/{len(opt_results)} successful")

    print("\n4. Testing Basic Functionality:")
    func_success = test_functionality()

    print("\n" + "=" * 50)
    print("SUMMARY:")
    print(f"  Core package functionality: {'✓' if core_success >= 6 else '✗'}")
    print(f"  ML package functionality: {'✓' if ml_success >= 2 else '✗'}")
    print(f"  Basic functionality: {'✓' if func_success else '✗'}")

    # Determine package variant
    if ml_success >= 2:
        print(f"\n  Detected: spine-ml (full package)")
    elif core_success >= 6:
        print(f"\n  Detected: spine-ml-core (minimal package)")
    else:
        print(f"\n  Warning: Package may not be properly installed")

    print("\nInstallation commands:")
    print("  pip install spine-ml-core          # Minimal package")
    print("  pip install spine-ml[full,viz]     # Full package with visualization")
    print("  pip install spine-ml[all]          # All optional dependencies")


if __name__ == "__main__":
    main()
