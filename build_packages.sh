#!/bin/bash

# Build script for SPINE package
# Builds the spine package with proper environment detection

set -e  # Exit on any error

echo "=== Building SPINE Package ==="

# Detect Python environment and set executable
if [ -n "$VIRTUAL_ENV" ]; then
    echo "Using virtual environment: $VIRTUAL_ENV"
    PYTHON_CMD="$VIRTUAL_ENV/bin/python"
elif [ -f ".venv/bin/python" ]; then
    echo "Using local .venv virtual environment"
    PYTHON_CMD=".venv/bin/python"
else
    echo "Using system Python"
    PYTHON_CMD="python3"
fi

# Verify Python is available
if ! command -v $PYTHON_CMD &> /dev/null; then
    echo "Error: Python not found at $PYTHON_CMD"
    exit 1
fi

echo "Python executable: $PYTHON_CMD"

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf build/ dist/ *.egg-info/ src/*.egg-info/

# Build the package
echo "Building spine package..."
$PYTHON_CMD -m build

echo "=== Build Summary ==="
ls -la dist/

echo "=== Checking packages ==="
# Detect twine command
if command -v twine &> /dev/null; then
    TWINE_CMD="twine"
elif [ -n "$VIRTUAL_ENV" ] && [ -f "$VIRTUAL_ENV/bin/twine" ]; then
    TWINE_CMD="$VIRTUAL_ENV/bin/twine"
elif [ -f ".venv/bin/twine" ]; then
    TWINE_CMD=".venv/bin/twine"
else
    echo "Installing twine..."
    $PYTHON_CMD -m pip install twine
    TWINE_CMD="$PYTHON_CMD -m twine"
fi

$TWINE_CMD check dist/*

echo ""
echo "Build completed successfully!"
echo ""
echo "Package structure: src/spine/ (src layout)"
echo "Project: spine (published to PyPI)"
echo ""
echo "To upload to PyPI:"
echo "  # Test PyPI first:"
echo "  $TWINE_CMD upload --repository testpypi dist/*"
echo ""
echo "  # Production PyPI:"
echo "  $TWINE_CMD upload dist/*"
echo ""
echo "Installation commands for users:"
echo "  pip install spine                    # Core package (numpy, scipy, pandas, pyyaml, h5py, numba)"
echo "  pip install spine[viz]               # + Visualization (matplotlib, plotly, seaborn)"
echo "  pip install spine[dev]               # + Development tools (testing, linting, docs)"
echo "  pip install spine[all]               # + viz, dev (excludes PyTorch ecosystem)"
echo ""
echo "PyTorch ecosystem (torch, torch-geometric, torch-scatter, torch-cluster, MinkowskiEngine):"
echo "  Recommended: singularity pull spine.sif docker://deeplearnphysics/larcv2:ub2204-cu121-torch251-larndsim"
echo "  Manual install: See README.md for detailed instructions"
echo ""
echo "Command line usage:"
echo "  spine --config your_config.cfg --source data.h5 [--output results/]"
echo "  spine --help                            # Show all available options"
