# Changelog

## [0.9.2] - 2026-01-14

### Added
- **Path Resolution System**: `!path` YAML tag for resolving file paths relative to config files
  - Returns absolute path string (unlike `!include` which loads content)
  - Verifies file exists at load time (fail fast)
  - Useful for post-processor configs, model weights, data files
  - Solves path context issues when configs are included from different locations
- **SPINE_CONFIG_PATH**: Environment variable for config file search paths
  - Colon-separated list of directories (like `PATH` or `PYTHONPATH`)
  - Used by both `!include` and `!path` tags
  - Enables sharing configs across projects without absolute paths
  - Auto-adds `.yaml`/`.yml` extensions if not found
- **DataLoader flexibility**: Pass arbitrary kwargs to `torch.utils.data.DataLoader`
  - Added `**kwargs` support in `loader_factory()`
  - Enables `pin_memory`, `persistent_workers`, `prefetch_factor`, etc.
  - Forward compatible with future PyTorch DataLoader parameters
- **DDP file sharing strategy**: Added `file_sharing_strategy` option for distributed training
  - Controls how files are shared across processes in DDP mode

### Fixed
- Fixed track completeness analysis script
- Added run number offset option in run number injection utility

## [0.9.1] - 2026-01-11

### Added
- Added several 2x2 geometry tags to FLOW geometry parser
- Consolidated configuration documentation in `spine/config/README.md`
  - Integrated METADATA_GUIDE.md and REMOVING_KEYS_EXAMPLE.md content
  - Comprehensive coverage of composition, overrides, metadata, and compatibility features

### Changed
- **Breaking**: Fully separated `file_keys` and `file_list` configuration paths to guarantee proper parsing
  - `file_keys` must now be a list of file paths only
  - `file_list` must be a path to a text file containing file paths
  - Mixed usage is no longer supported
- Renamed CI workflow from 'Comprehensive Testing' to 'CI' (`.github/workflows/ci.yml`)
- Reordered README badges: CI → codecov → RTD → PyPI → Python

### Fixed
- Fixed syntax of FLOW geometry tags
- Fixed typos in reader docstrings (`LArCVReader`, `FlowReader`)
- Fixed issue with `skip_entry_list` parameter handling in dataset readers
- Fixed `GeoDrawer` to explicitly require detector geometry in constructor
- Fixed hard-coded scaling factor in batch unwrapper when merging volumes

## [0.9.0] - 2026-01-07

### Added
- **Advanced YAML Configuration System**: Complete configuration management with composition and validation
  - File includes via `include:` directive (single file or list) and `!include` tag
  - Parameter overrides with dot-notation syntax (e.g., `io.loader.batch_size: 8`)
  - Command-line configuration via `--set` flag
  - Recursive deep merging of configuration dictionaries
- **Configuration Metadata System**: Version control and compatibility checking via `__meta__` blocks
  - Version tracking with 6-digit YYMMDD format (e.g., "240719")
  - Compatibility constraints with operators (==, >=, <=, >, <, !=)
  - Deferred validation supporting forward references between components
  - Automatic component version inference from directory structure
  - Configurable behavior: `kind` (bundle/mod), `strict` mode (warn/error), `list_append` (append/unique)
  - Modifier metadata: `priority`, `applies_to`, `requires`, `conflicts_with`
  - Comprehensive METADATA_GUIDE.md documentation
- **Typed Exception Hierarchy**: 7 specialized exceptions for configuration errors
  - `ConfigError` (base), `ConfigIncludeError`, `ConfigCycleError`, `ConfigPathError`
  - `ConfigTypeError`, `ConfigOperationError`, `ConfigValidationError`
- **Modular Package Architecture**: New `spine/config/` package
  - `loader.py`: Include resolution, override processing, metadata validation
  - `meta.py`: Version parsing and compatibility checking
  - `errors.py`: Typed exception hierarchy
  - `api.py`: Configuration schema and constants

### Changed
- CLI now uses `load_config()` with full include/override support
- **Breaking**: Configuration loader moved from `spine.utils.config` to `spine.config` package
  - New import: `from spine.config import load_config`

### Removed
- CLI `--detect-anomaly` flag (use `--set model.detect_anomaly=true` instead)

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
