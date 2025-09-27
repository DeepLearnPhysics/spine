#!/usr/bin/env python3
"""
SPINE Test Discovery and Execution Script

Runs all tests in the structured test organization.
"""

import pytest
import sys
from pathlib import Path


def main():
    """Run all SPINE tests with proper discovery."""
    
    test_root = Path("test")
    
    # Test directories in dependency order
    test_dirs = [
        "test_utils",      # Core utilities first
        "test_data",       # Data structures
        "test_math",       # Mathematical operations
        "test_io",         # Input/Output
        "test_construct",  # Object construction
        "test_model",      # ML models
        "test_post",       # Post-processing
        "test_ana",        # Analysis
        "test_vis",        # Visualization
    ]
    
    print("🧪 SPINE Comprehensive Test Suite")
    print("=" * 50)
    
    total_passed = 0
    total_failed = 0
    
    for test_dir in test_dirs:
        test_path = test_root / test_dir
        
        if test_path.exists():
            print(f"\n🔍 Testing {test_dir}...")
            
            # Run pytest on this directory
            result = pytest.main([
                str(test_path),
                "-v",
                "--tb=short", 
                "--continue-on-collection-errors",
                "-x"  # Stop on first failure for debugging
            ])
            
            if result == 0:
                print(f"✅ {test_dir} tests passed")
                total_passed += 1
            else:
                print(f"❌ {test_dir} tests failed")
                total_failed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Summary: {total_passed} passed, {total_failed} failed")
    
    if total_failed == 0:
        print("🎉 All test modules passed!")
        return 0
    else:
        print("⚠️  Some test modules failed - check output above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
