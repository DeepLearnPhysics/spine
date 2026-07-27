# Command-line Utilities

This directory contains source-checkout entry points and focused maintenance
utilities. Run commands from the repository root so imports and relative paths
match the examples below.

## Primary entry points

- `run.py` runs the main SPINE training, inference, or analysis CLI:

  ```bash
  python3 bin/run.py --config config/infer/example.yaml
  ```

- `config.py` inspects resolved configuration files. Use `--help` to see the
  available dump, diff, and validation operations:

  ```bash
  python3 bin/config.py --help
  ```

- `coverage.sh` runs the project test suite with coverage reporting.

## Specialized utilities

- [`calib/`](calib/README.md): calibration-map diagnostics
- [`geo/`](geo/README.md): geometry conversion tools
- [`larcv/`](larcv/README.md): LArCV ROOT inspection and maintenance
- [`output/`](output/README.md): SPINE HDF5 validation, comparison, and reduction

Every Python utility supports `--help`. The functions above the CLI wrappers
are also importable for programmatic use and carry type annotations.
