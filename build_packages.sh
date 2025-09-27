#!/bin/bash

# Build script for SPINE package
# Builds the spine-ml package with proper environment detection

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
echo "Building spine-ml package..."
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
echo "Structure: src/spine/ (src layout)"
echo ""
echo "To upload to PyPI:"
echo "  # Test PyPI first:"
echo "  $TWINE_CMD upload --repository testpypi dist/*"
echo ""
echo "  # Production PyPI:"
echo "  $TWINE_CMD upload dist/*"
echo ""
echo "Installation commands for users:"
echo "  pip install spine-ml                    # Core package (numpy, scipy, pandas, yaml, h5py, numba)"
echo "  pip install spine-ml[torch]             # + PyTorch & MinkowskiEngine"
echo "  pip install spine-ml[ml]                # + scikit-learn"
echo "  pip install spine-ml[viz]               # + matplotlib & plotly"
echo "  pip install spine-ml[io]                # + larcv"
echo "  pip install spine-ml[all]               # All extras"
echo ""
echo "Usage: spine --config your_config.cfg --source data.h5"
