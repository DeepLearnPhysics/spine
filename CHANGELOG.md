# Changelog

## [0.7.5] - 2025-10-01

### Changed
- Reorganized binary scripts: moved non-package scripts to top-level `bin/`, added `bin/run.py` for CLI convenience.
- Renamed CLI entry point from `run.py` to `cli.py` in `src/spine/bin/`.
- Updated `pyproject.toml` to reference new CLI entry point.
 
### Fixed
- Switched to `spine.bin.cli` import for tests and user code.
- Pre-commit hooks and code formatting issues resolved.

### Other
- General codebase cleanup and documentation improvements.

## [0.7.4] - 2025-10-01

### Changed
- **Cluster feature extraction:** Improved memory handling and LAPACK buffer allocation in geometric feature routines.
- **Manual edits:** Updated and refactored `cluster.py` for stability and performance.

### Fixed
- **LAPACK/Numba errors:** Addressed buffer allocation and parallelization issues in cluster feature extraction.

### Other
- **Pre-commit checks:** All code changes pass pre-commit hooks.
- **Version bump:** Updated to v0.7.4 for release.

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.7.3] - 2025-09-30

### Changed
- **CLI improvements**: `-s/--source` and `-S/--source-list` are now mutually exclusive arguments.
- **Input validation**: CLI now checks that `source` is not only not None, but also not an empty list before overriding file keys.

### Fixed
- **Robust file key handling**: Prevents empty input lists from being used in configuration.

### Other
- **Pre-commit checks**: All code changes pass pre-commit hooks.
- **Version bump**: Updated to v0.7.3 for release.

## [0.7.2] - 2025-09-30

### Changed
- **CLI config argument**: The CLI now requires `-c/--config` to specify the configuration file (no more positional config argument).
- **CLI help and validation**: Improved help text and argument validation for configuration files.

### Fixed
- **Stopwatch/timing logic**: Improved equality checks and state handling in `Stopwatch` and `Time` classes to prevent timing errors and double-stop issues.

### Other
- **Code formatting**: Codebase fully formatted and linted (pre-commit checks enforced).
- **Version bump**: Updated to v0.7.2 for release.

## [0.7.1] - 2025-09-27

- **Driver import optimization**: Moved torch utilities to top-level imports for cleaner code structure
- **Eliminated conditional imports**: Reduced code complexity by removing scattered conditional imports

### Added
- **Stopwatch state properties**: Added `running` and `paused` properties for clean state checking
- **Enhanced documentation**: Professional styling, comprehensive API docs, and SPINE logo integration
- **Reset functionality**: Proper stopwatch reinitialization using clean instance replacement

## [0.7.0] - Previous Release

Initial release with core SPINE functionality.

---

**Note**: For detailed commit history, see the [GitHub repository](https://github.com/DeepLearnPhysics/spine).
