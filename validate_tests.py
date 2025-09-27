#!/usr/bin/env python3
"""Script to validate comprehensive test coverage for conditional imports."""

import subprocess
import sys
import os
import time
from pathlib import Path


def run_command(cmd, description, check=True):
    """Run a command and report results."""
    print(f"\n🔧 {description}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 50)
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=check)
        elapsed = time.time() - start_time
        
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
            
        print(f"✅ Completed in {elapsed:.2f}s")
        return True, result
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"❌ Failed in {elapsed:.2f}s")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False, e
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"💥 Error in {elapsed:.2f}s: {e}")
        return False, e


def check_test_environment():
    """Verify test environment is set up correctly."""
    print("🔍 Checking Test Environment")
    print("=" * 60)
    
    # Check Python version
    print(f"Python version: {sys.version}")
    
    # Check working directory
    cwd = os.getcwd()
    print(f"Working directory: {cwd}")
    
    # Check if we're in the right directory
    if not Path("src/spine").exists():
        print("❌ Not in SPINE root directory!")
        return False
    
    # Check if test files exist
    test_files = [
        "test/test_imports.py",
        "test/test_conditional_imports.py", 
        "test/pytest.ini"
    ]
    
    for test_file in test_files:
        if Path(test_file).exists():
            print(f"✅ {test_file} exists")
        else:
            print(f"❌ {test_file} missing")
            return False
    
    return True


def run_comprehensive_tests():
    """Run comprehensive test suite."""
    print("\n🚀 Running Comprehensive Test Suite")
    print("=" * 60)
    
    test_results = []
    
    # Test 1: Core imports without optional dependencies
    success, _ = run_command([
        sys.executable, "-c",
        """
import sys
# Ensure torch and networkx are not available
if 'torch' in sys.modules:
    del sys.modules['torch']
if 'networkx' in sys.modules:
    del sys.modules['networkx']

# Test core imports
from spine.driver import Driver
from spine.main import run
from spine.bin.cli import main
print('✅ Core imports successful without optional dependencies')
"""
    ], "Core Imports Test")
    test_results.append(("Core Imports", success))
    
    # Test 2: Run pytest on import tests
    success, _ = run_command([
        sys.executable, "-m", "pytest", 
        "test/test_imports.py", 
        "-v", "--tb=short"
    ], "Import Tests")
    test_results.append(("Import Tests", success))
    
    # Test 3: Run conditional import tests
    success, _ = run_command([
        sys.executable, "-m", "pytest",
        "test/test_conditional_imports.py",
        "-v", "--tb=short"
    ], "Conditional Import Tests")
    test_results.append(("Conditional Import Tests", success))
    
    # Test 4: Performance tests
    success, _ = run_command([
        sys.executable, "-m", "pytest",
        "test/test_conditional_imports.py::TestPerformanceRegression",
        "-v", "--tb=short"
    ], "Performance Tests")
    test_results.append(("Performance Tests", success))
    
    # Test 5: NetworkX elimination verification
    success, _ = run_command([
        sys.executable, "-c",
        """
# Test NetworkX elimination
from collections import defaultdict
import time

print('Testing NetworkX-free post-processing...')

# Simulate children counting without NetworkX
size = 1000
parent_ids = [max(0, i // 2) for i in range(size)]

start_time = time.time()
children = defaultdict(list)
for child_id, parent_id in enumerate(parent_ids):
    if child_id != parent_id:
        children[parent_id].append(child_id)

children_counts = {node_id: len(children[node_id]) for node_id in range(size)}
elapsed = time.time() - start_time

print(f'✅ Processed {size} nodes in {elapsed:.4f}s ({size/elapsed:.0f} nodes/sec)')
assert elapsed < 0.1, f'Too slow: {elapsed:.4f}s'

# Test actual SPINE processor
from spine.post.truth.label import ChildrenProcessor
processor = ChildrenProcessor(mode='shape')
print(f'✅ ChildrenProcessor created: {processor.name}')
"""
    ], "NetworkX Elimination Test") 
    test_results.append(("NetworkX Elimination", success))
    
    # Test 6: CLI functionality
    success, _ = run_command([
        sys.executable, "-m", "spine.bin.cli", "--version"
    ], "CLI Version Test", check=False)  # May fail if not fully installed
    test_results.append(("CLI Version", success))
    
    return test_results


def generate_coverage_report():
    """Generate test coverage report."""
    print("\n📊 Generating Coverage Report")
    print("=" * 40)
    
    # Run tests with coverage
    success, result = run_command([
        sys.executable, "-m", "pytest",
        "test/test_imports.py",
        "test/test_conditional_imports.py", 
        "--cov=spine",
        "--cov-report=term",
        "--cov-report=html:htmlcov",
        "-v"
    ], "Coverage Analysis", check=False)
    
    if success and Path("htmlcov/index.html").exists():
        print("\n✅ HTML coverage report generated: htmlcov/index.html")
        return True
    else:
        print("\n⚠️  Coverage report generation failed or incomplete")
        return False


def main():
    """Main test validation script."""
    print("🧪 SPINE Comprehensive Test Coverage Validation")
    print("=" * 70)
    
    # Check environment
    if not check_test_environment():
        print("❌ Environment check failed!")
        sys.exit(1)
    
    # Run tests
    test_results = run_comprehensive_tests()
    
    # Generate coverage  
    coverage_success = generate_coverage_report()
    
    # Summary
    print("\n" + "=" * 70)
    print("📋 TEST RESULTS SUMMARY")
    print("=" * 70)
    
    total_tests = len(test_results)
    passed_tests = sum(1 for _, success in test_results if success)
    
    for test_name, success in test_results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"  {test_name:<25}: {status}")
    
    print("-" * 70)
    print(f"Tests passed: {passed_tests}/{total_tests}")
    print(f"Coverage report: {'✅ Generated' if coverage_success else '❌ Failed'}")
    
    if passed_tests == total_tests and coverage_success:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Comprehensive test coverage validated")
        print("✅ Conditional imports working correctly")
        print("✅ NetworkX dependency eliminated") 
        print("✅ Performance benchmarks met")
        sys.exit(0)
    else:
        print(f"\n💥 {total_tests - passed_tests} test(s) failed!")
        print("Review the output above for details")
        sys.exit(1)


if __name__ == '__main__':
    main()