# Changelog

## [0.8.1] - 2025-12-19

### Fixed
- Fixed `AttributeError` in `LArCVReader` when `num_entries` was accessed before initialization

## [0.8.0] - 2025-12-17

### Added
- Automated LArSoft and FLOW (larnd-sim) geometry parsers with CRT/optical support
- New geometries: ProtoDUNE-VD/SP/HD, DUNE-FD 10kt-1x2x6
- Updated geometries: 2x2 (MR5/MR6), ND-LAr, FSD, ICARUS, SBND
- Cylinder/disk visualization tools for optical detectors
- Geometry metadata: name, tag, version, GDML field, CRS/LRS config
- Full geometry visualization option via `GeoDrawer.show()`
- Lite-format particle/interaction drawing with basic track/shower representations

### Changed
- **Breaking**: Complete geometry system overhaul to singleton `GeoManager` class
  - New API: `initialize()`, `initialize_or_get()`, `get_instance()`
  - Geometry initialized prior to IO in Driver, no re-initialization if instance exists
- Thorough cleanup of reader modules
- Improved optical volume organization and detector range handling

### Fixed
- Multiple Pylance type checking issues resolved
- Optical geometry styling when mixing detector types

## [0.7.9] - 2025-11-12

### Fixed
- Fixed pseudovertex computation to properly check for antiparallel particles
  - Prevents `np.inv` fail when particles are parallel but opposite in direction

## [0.7.8] - 2025-11-11

### Added
- Added test coverage tools and codecov badge to README
- Created `bin/coverage.sh` script for local coverage checking
- Added codecov.yml configuration with 1% coverage drop threshold

### Fixed
- Fixed camera synchronization in `dual_figure3d()` for Jupyter notebooks
  - Corrected callback signature from `(scene, camera)` to `(layout_obj, camera)`
  - Added mutex flag to prevent infinite loop between synchronized scenes
- Fixed issue with `np.inv` on singular matrices in the vertexer (issue with parallel directions)
- Fixed Codecov CI integration with proper token and verbose logging

### Changed
- Set codecov patch coverage target to 0% to allow untested code changes
- Enhanced GitHub issue templates with system information section
- Improved PR template with comprehensive checklist

## [0.7.7] - 2025-10-13

### Changed
- Small API clarifications in `post.manager`
- Updated the ReadTheDocs path to match the new spine.readthedocs.io URL

### Fixed
- Fixed an issue with the cathode crosser post-processor which did not return anything when there are no particles

### Other
- Version bump and release housekeeping.

## [0.7.6] - 2025-10-03

### Changed
- Improved eigen-decomposition handling in cluster feature extraction (fixed `.astype` usage).
- Clarified Plotly legend behavior and usage in visualization modules.
- Manual edits to cluster and visualization code for stability and clarity.

### Fixed
- Addressed errors related to `EighResult` and legend handling in Plotly.

### Other
- General codebase maintenance and documentation updates.

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
