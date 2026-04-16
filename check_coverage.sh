docker run --rm \
  -v "$PWD:/workspace" \
  -w /workspace \
  --platform linux/amd64 \
  ghcr.io/deeplearnphysics/spine:latest \
  bash -lc '
    PYTHONPATH="src:${PYTHONPATH:-}" python -m pytest test/test_data \
      --override-ini addopts="" \
      --cov=spine.data \
      --cov-report=term-missing \
      --cov-report=xml:coverage.xml \
      --maxfail=10
  '
