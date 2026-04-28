#!/bin/bash
# Build SPINE documentation locally

set -e

echo "=========================================="
echo "Building SPINE Documentation"
echo "=========================================="

# Navigate to docs directory
cd "$(dirname "$0")"

# Clean previous build
echo ""
echo "Cleaning previous build..."
make clean

# Build HTML documentation
echo ""
echo "Building HTML documentation..."
make html

# Report status
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Documentation built successfully!"
    echo "=========================================="
    echo ""
    echo "Open in browser:"
    echo "  file://$(pwd)/build/html/index.html"
    echo ""
    echo "Or run:"
    echo "  open build/html/index.html  # macOS"
    echo "  xdg-open build/html/index.html  # Linux"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "✗ Documentation build failed"
    echo "=========================================="
    exit 1
fi
