#!/usr/bin/env bash

set -euo pipefail

test_path="${1:-test}"
coverage_target="${2:-spine}"

docker run --rm \
  -v "$PWD:/workspace" \
  -w /workspace \
  -e TEST_PATH="$test_path" \
  -e COVERAGE_TARGET="$coverage_target" \
  --platform linux/amd64 \
  ghcr.io/deeplearnphysics/spine:latest \
  bash -lc '
    PYTHONPATH="src:${PYTHONPATH:-}" python -m pytest "$TEST_PATH" \
      --override-ini addopts= \
      --cov="$COVERAGE_TARGET" \
      --cov-report=term-missing \
      --cov-report=xml:coverage.xml \
      --maxfail=10
  '
