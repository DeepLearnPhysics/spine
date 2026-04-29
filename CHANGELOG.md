# Changelog

## [0.11.1] - 2026-04-29

### Added
- **Container tooling**: Ship `jupyterlab`, the classic `notebook` interface, and lightweight in-container editors (`vim`, `nano`) in the published SPINE image for tutorials and interactive debugging.
- **Testing**: Expand focused regression coverage for `spine.config`, `spine.data`, and the new constants package so those surfaces are now exercised end-to-end in CI and release validation.

### Changed
- **Constants package**: Consolidate shared labels, enums, physics values, sentinels, and column definitions under `spine.constants`, and remove the old `spine.utils.enums` compatibility shim.
- **Documentation**: Refresh the Sphinx API/reference structure, installation guide, quickstart, and container documentation; also add the missing Read the Docs dependency needed by the current docs build.
- **Docker build caching**: Move the SPINE source copy/install later in the Dockerfile so routine SPINE releases reuse the more stable notebook and flash-matching layers above.

### Fixed
- **Configuration metadata**: Ensure normalized metadata values survive validation even when raw `__meta__` entries are malformed.
- **Full-chain HDF5 output**: Correct full-chain output handling while keeping the related data/model/docs surfaces aligned with the current refactor.
- **Documentation builds**: Fix Read the Docs and Sphinx docstring formatting regressions introduced by the API/documentation refresh.

## [0.11.0] - 2026-04-27

### Added
- **Data structures**: Introduce typed `FieldMetadata` and decorator-based `@stored_property` / `@stored_alias` metadata for `spine.data` objects.
- **Testing**: Add comprehensive `spine.data`, HDF5 I/O, parser, and full-chain regression coverage, including deterministic full-chain reference checks.
- **Configuration**: Allow config parsing without resolving `!download` directives when desired.

### Changed
- **Data model**: Refactor `spine.data` around dataclass field metadata, clearer repr/equality behavior, and explicit stored-property serialization.
- **LArCV data layout**: Reorganize LArCV-backed classes under `spine.data.larcv`.
- **Metadata classes**: Prefer explicit `ImageMeta2D` / `ImageMeta3D` classes while keeping `Meta` as a compatibility surface.
- **HDF5 schema**: Normalize serialized object typing:
  - scalar booleans are now stored as `bool` instead of `uint8`
  - many object-member index/ID arrays now store as `int32` instead of `int64`
  - scalar numeric attributes now follow Python scalar typing
- **Units handling**: Standardize spatial attributes around `units="instance"` where values should follow `to_cm()` / `to_px()` conversions.
- **Truth matching**: Replace set-based overlap computation with a sorted-index intersection path for cleaner and more efficient overlap evaluation.

### Fixed
- **HDF5 writer**: Restore serialization of stored-property values for output data objects.
- **HDF5 reader**: Read legacy HDF5 files produced by older releases, including:
  - boolean fields stored as `uint8`
  - legacy `class_name="Meta"` payloads, now reconstructed as explicit metadata classes
- **Full-chain stability**: Tighten deterministic regression checks to tolerate architecture-level floating point noise while still flagging meaningful drift.
- **LArCV particle positions**: Fix position-like attributes so they can be expressed consistently in both pixel and detector coordinates.
- **Conditional imports**: Reduce overhead and test fragility in optional-import code paths.

## [0.10.13] - 2026-04-17

### Fixed
- **Docker**: Bundle OpT0Finder v1.0.0 and the ICARUS PhotonLibrary so likelihood flash matching works in the published image.

## [0.10.12] - 2026-04-17

### Fixed
- **Inference**: Run model-less inference jobs once so reader/build/post/writer workflows execute without requiring a `model` block.

## [0.10.11] - 2026-04-14

### Added
- **Configuration**: Add a `spine-config` command and `bin/config.py` proxy to load complex SPINE configs, dump the resolved YAML, and compare resolved configs.

### Changed
- **HDF5 Writer**: Refactor output file handling, split-output naming, dataset creation, and append/opening logic for clearer and more consistent writer behavior.
- **CSV Writer**: Align writer naming and type handling with the cleaned-up HDF5 writer behavior.

### Fixed
- **Docker**: Build `h5py` against the system HDF5 library used by LArCV to avoid HDF5 ABI mismatches when importing LArCV and `h5py` in the same process.
- **Full Chain**: Handle empty fragment-list group indexes without failing on an empty maximum reduction.

## [0.10.10] - 2026-04-13

### Changed
- **Package import**: Lazy-load `Driver` from the top-level `spine` package so lightweight imports such as `spine.config` do not load the full driver stack.

### Fixed
- **Tests**: Avoid mocking unavailable PyTorch with a `sys.modules` sentinel in conditional import tests.

## [0.10.9] - 2026-04-11

### Added
- **Configuration**: Support for environment variable expansion in configuration files
- **I/O**: Remote XRootD input path support for distributed file access
- **GrapPA**: Configurable feature outputs for GrapPA module ([#123](https://github.com/DeepLearnPhysics/spine/pull/123))
- **Inference**: Support for lists of inference weight paths for multi-model workflows ([#122](https://github.com/DeepLearnPhysics/spine/pull/122))

### Changed
- **Docker**: Enhanced CI/CD with Buildx and Docker layer caching
- **Truth Matching**: Track original point indexes for improved truth matching ([#121](https://github.com/DeepLearnPhysics/spine/pull/121))
- **Documentation**: Clarified Docker usage documentation
- **Validation**: Tightened module weight path validation ([#122](https://github.com/DeepLearnPhysics/spine/pull/122))

## [0.10.8] - 2026-04-06

### Added
- **Docker Containerization**: Complete Docker infrastructure for production deployments
  - Full ML stack with PyTorch 2.5.1, MinkowskiEngine v0.5.4, torch-geometric, ROOT, and LArCV2
  - Ubuntu 22.04 base with CUDA 12.1 toolkit (perfect version match with PyTorch)
  - XRootD client with SciTokens support for dCache streaming with token authentication
  - Multi-GPU architecture support: V100, A100, H100/H200, RTX 20xx/30xx/40xx (compute 7.0-9.0)
  - Automated GitHub Actions workflow for container builds and publishing to GHCR
  - Comprehensive documentation with Apptainer/Singularity usage examples
  - Build script for local development and testing

### Changed
- **Dependencies**: Removed torch-sparse dependency (no longer required)
- **Documentation**: Updated all Singularity references to Apptainer (current standard)
- **Sphinx**: Removed torch-sparse from autodoc mock imports
- **Docker**: Local Docker builds now force-refresh the base image with `--pull`

### Fixed
- **NumPy 2**: Avoid coercing `EventSparseTensor3D` lists into NumPy arrays in `Sparse3DParser`

## [0.10.6] - 2026-03-18

### Changed
- **CSV Writer**: Significantly improved CSV writer performance for analysis scripts ([#119](https://github.com/DeepLearnPhysics/spine/pull/119))
- **Multi-node training**: Fixed multi-node distributed training support ([#118](https://github.com/DeepLearnPhysics/spine/pull/118))

## [0.10.5] - 2026-03-04

### Fixed
- **Visualization**: Fixed raw drawing to behave correctly for truth data
- **Track analysis**: Fixed bug in track completeness algorithm

## [0.10.4] - 2026-03-01

### Added
- **CLI**: Added `--entry-list` and `--skip-entry-list` arguments for easy entry filtering
- **Multi-node training**: Enhanced main.py to support multi-node distributed training

### Fixed
- **File downloads**: Added file locking to prevent race conditions when multiple jobs download files concurrently
- **CLI**: Guard against null loader/reader to prevent crashes
- **Checkpointing**: Fixed epoch-based weight checkpointing bug
- **Vertexer**: Proper singularity check in vertex computation
- **Vertex utility**: Minor bug fix in vertex calculation

## [0.10.3] - 2026-02-16

### Added
- **Visualization**: Added support to draw `_sum` attributes in Drawer

### Fixed
- **Data objects**: Fixed default typing issue in `crt_times`/`crt_scores` attributes of Interaction objects
- **Visualization**: Do not make a single attribute in the drawer be the default color scale

### Changed
- **Visualization**: Use asdict to get object properties in Drawer (more complete)
- **Config loading**: Added explicit message when `SPINE_CONFIG_PATH` is not set and config include is not found

## [0.10.2] - 2026-02-12

### Added
- **Data augmentation**: Added masking (cut-out) and cropping augmentation
- **Data augmentation**: Enable using geometry to determine ranges

### Fixed
- **GrapPA**: Fixed bug in GrapPA geometric feature extraction when `use_numpy: false`
- **Visualization**: Drawer can now draw attributes for truth but not reco (or vice-versa)

### Changed
- **Config loading**: Use recursive loading in `!include` directive
- **Config loading**: Strip `__meta__` block from configs included through `!include`
- **Checkpointing**: More elegant approach to epoch-based checkpointing

## [0.10.1] - 2026-02-09

### Added
- **GrapPA edge utility**: Utility to compute edge length requirements for the GrapPA models

### Fixed
- **Primary labeling**: Fixed error in primary particle labeling logic
- **Neutrino interaction type**: Fixed problem with neutrino interaction type enumerator
- **Error messaging**: Improved error message clarity when run/subrun/event triplet is not found

### Changed
- **Config path resolution**: Removed requirement to provide `parent_path` to the FileManager
  - Absolute paths are now created automatically by spine.config on the fly
  - Simplifies configuration management
- **Batch size handling**: `batch_size` is now automatically fetched from `minibatch_size` if not explicitly provided

## [0.10.0] - 2026-02-06

### Changed
- **Package renamed from `spine-ml` to `spine` on PyPI**: The package has been renamed to use the simpler `spine` name on PyPI
  - All installation commands should now use `spine` instead of `spine-ml`
  - Examples: `pip install spine`, `pip install spine[all]`, etc.
  - The old `spine-ml` package name is deprecated and will no longer receive updates
  - **Migration for existing users**: Simply replace `spine-ml` with `spine` in your installation commands and requirements files
  - No code changes required - the Python import name remains `import spine`
  - This is the first release under the new `spine` package name

## [0.9.5] - 2026-02-01

### Added
- **String-based config loading**: `load_config()` now accepts YAML strings in addition to file paths
  - Enables dynamic config generation in notebooks and scripts
  - `load_config_file()` added for explicit file loading
  - Maintains full support for includes and SPINE_CONFIG_PATH resolution

### Changed
- **Config module refactoring**: Split large `loader.py` into focused modules for maintainability
  - `operations.py`: Utility functions (deep_merge, parse_value, apply_collection_operation, etc.)
  - `loader.py`: ConfigLoader class and YAML tag registration
  - `load.py`: Main loading functions (load_config, load_config_file, _load_config_recursive)
- **Exception handling**: More specific exception catching in download validation (OSError, IOError, ValueError instead of broad Exception)

## [0.9.4] - 2026-01-26

### Added
- **!download YAML tag**: Automatically download files referenced in configs
- **Centralized cache directory**: All downloaded files now use a shared cache location

### Changed
- **SPINE_CONFIG_PATH**: Can now use paths relative to SPINE_CONFIG_PATH for configuration files
- Formatting and documentation improvements

## [0.9.3] - 2026-01-20

### Added
- **tqdm dependency**: Progress bars now available for long-running operations
- **LArCV tree size measurement**: Script to measure tree sizes in each entry of a LArCV file
- **Run list support**: `larcv_inject_run_number` script can now take a run list (different run per file)
- **Gain from database**: Option to fetch gain calibration from database
- **Run ID in gain correction**: Pass run_id to the gain correction calibrator

### Changed
- **CI improvements**: Use pre-commit for CI linting and update to latest tool versions
- **Calibration cleanup**: Cleaned up calibration package around `CalibrationConstant` class
- **Config loader**: Allow for empty strings in configuration
- **Training visualization**: Move training legend outside figure for better visibility

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
