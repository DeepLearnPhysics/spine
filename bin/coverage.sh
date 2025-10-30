#!/bin/bash
# Script to run test coverage locally and generate an HTML report

set -e

# Run pytest with coverage
echo "Running tests with coverage..."
pytest --cov=spine --cov-report=term --cov-report=html --cov-report=xml

echo ""
echo "Coverage report generated:"
echo "  - Terminal output: shown above"
echo "  - HTML report: htmlcov/index.html"
echo "  - XML report: coverage.xml"
echo ""
echo "To view the HTML report, run: open htmlcov/index.html"
