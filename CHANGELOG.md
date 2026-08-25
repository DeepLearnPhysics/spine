# Changelog

## [Unreleased]

### Fixed
- **Deposition hover labels**: Preserve each point-wise deposition field name
  in hover text so selecting fields such as `depositions` and `depositions_q`
  does not produce a redundant generic `Deposition` entry.

## [1.0.0] - 2026-08-25

### Added
- **Interaction vertex prediction**: Generalize the UResNet point-proposal decoder to support particle-point and interaction-vertex heads, either with independent decoders or a shared decoder, and provide maintained particle-only, vertex-only, and combined train/test configurations.
- **Full-chain vertexing**: Add configurable PPN- and GrapPA-based interaction vertex reducers, publish one selected `interaction_vertices` prediction per interaction, and populate reconstructed interaction vertices during object construction when the prediction is available.
- **Cluster-quality supervision**: Compute matched-instance IoU, purity, and efficiency and expose optional scalar or per-class thresholds across GrapPA node, edge, classification, regression, orientation, primary, and vertex losses as well as image classification and regression losses. Rejected targets use the existing ignore convention and report `count_rejected`.
- **Full-chain quality policy**: Apply a conservative `min_iou: 0.5` default to full-chain particle PID and primary classification while leaving standalone GrapPA configurations unrestricted unless explicitly configured.
- **Robust label adaptation**: Rework full-chain instance-label propagation around deterministic geometric fronts, coordinate-alignment validation, and optional distance weighting, with ambiguous assignments left unsupervised.
- **Grouped graph construction**: Add Numba-accelerated grouped radius graphs and connected-component helpers for building disjoint instance graphs in one pass, alongside benchmarked dense and spatial-hash construction paths.
- **Restored model capabilities**: Rehabilitate SPICE fragment construction and GrapPA voxel encoders under their owning model packages, with maintained configuration coverage and full-chain integration contracts.
- **Inference configuration conversion**: Add `spine.config.to_inference_config` and the CLI `--inference` option to convert training configurations into deterministic inference jobs before applying command-line overrides.
- **Shower-start dE/dx diagnostic**: Implement the shower analyzer with configurable spherical and direction-aware kernels, scalar or scanned neighborhood radii, reconstructed/truth point representations, and stable per-object CSV output.
- **Structured data-class reference**: Generate source-driven API pages that separate stored fields, computed properties, and methods, group inherited members by their declaring class, and expose defaults, types, units, and other SPINE field metadata without hard-coded member lists.
- **Documentation concepts**: Add pipeline-architecture and data-model guides alongside API pages for the new calibration, clustering, logging, and physics package boundaries.
- **Local inspection quick start**: Document core and visualization `pip` installs and provide a tested HDF5 reader/build workflow that produces drawable reconstructed particles without requiring a model runtime.

### Changed
- **Production-stable milestone**: Mark SPINE as production/stable and establish the documented package boundaries and configuration contracts as the public baseline for the 1.x release series.
- **Focused package ownership**: Reduce `spine.utils` to genuinely shared cross-layer helpers. Move cluster formation, features, directions, topology, and graph operations to `spine.cluster`; physical reconstruction routines and CSDA data to `spine.physics`; loggers and CSV output to `spine.logging`; configuration transforms to `spine.config`; and optimizer, Graph-SPICE, SPICE, and GrapPA utilities to their owning model packages.
- **Point-proposal architecture**: Replace the narrowly coupled PPN/vertex implementations with typed generic proposal-decoder contracts while keeping existing UResNet PPN configurations valid without modification.
- **Primary supervision**: Select the uniquely closest eligible shower and interaction fragment to the true start or vertex as primary and demote other broken pieces, avoiding multiple positive primary targets from one object.
- **Graph and matching utilities**: Expand `spine.math.graph` and `spine.math.match` with typed, publicly documented graph construction, connected-component, and overlap-matching interfaces used by clustering and supervision.
- **Startup presentation**: Consolidate the full banner under CLI execution, group runtime and resolved-configuration information consistently, and place weight download/cache messages within the startup section.
- **Warning-strict documentation**: Rebuild autosummary output from scratch and require warning-free Sphinx builds in CI and Read the Docs, with one aligned dependency set for local and hosted builds.
- **Documentation landing pages**: Align the README and Read the Docs opening narrative around the reconstruction schematic while keeping the hosted documentation concise and task-oriented.
- **Optional dependency extras**: Keep focused visualization and development extras; remove the empty `core`, incomplete `model`, overly narrow `tensorboard`, and misleading `all` extras. The platform-sensitive ML runtime remains provided by the released container or a manual compatible-stack installation.
- **Quality and typing standards**: Bring the reorganized clustering, physics, logging, parser, and model code under the repository's formatting, linting, type-hint, documentation, and complete statement-coverage expectations.

### Fixed
- **Graph-SPICE evaluation portability**: Evaluate Graph-SPICE metrics on the correct backend and restore fragment-production behavior required by model and full-chain consumers.
- **Label-front ambiguity**: Prevent arbitrary propagation through fronts reached equally by incompatible labels, distinguish face, edge, and corner neighbors when distance weighting is enabled, and preserve CPU/Torch input contracts without repeated GPU scalar operations.
- **Interaction construction**: Consume learned interaction-vertex products when present while retaining the established fallback and allowing explicit vertex post-processing to overwrite the result.
- **Container optional dependencies**: Install the declared visualization extra in release images and test that Dockerfiles cannot reference undefined package extras.
- **Visualization module namespace**: Ensure `spine.vis.scene` and the other documented visualization package attributes cannot be shadowed by modules leaked through wildcard convenience exports.
- **Core-only imports**: Keep `Driver`, `spine.main`, and `ModelManager` importable without PyTorch by avoiding eager optimizer implementation imports.
- **Documentation portability**: Keep strict HTML and EPUB builds warning-free, exclude unsupported favicon assets from EPUB, and use packaged image assets consistently across the README and hosted documentation.

### Removed
- **Obsolete utility namespaces**: Remove the legacy globals, clustering, GNN, inference, logging, and Torch-script utility modules after migrating retained functionality to canonical packages. Shared optical, ghost, and minimal PPN helpers remain in `spine.utils` because they serve multiple subsystems.

## [0.17.3] - 2026-08-18

### Added
- **Renderer-neutral visualization scenes**: Add typed point, marker, line,
  vector, mesh, and box layers, a backend registry, and a Plotly compatibility
  backend for notebook and application consumers.
- **Complete output-drawer scenes**: Expose raw depositions, endpoints,
  directions, vertices, optical responses, CRT hits, and detector geometry
  through `Drawer.get_scene`, with compact object boundaries and recoloring
  metadata for browser renderers.
- **Visualization field metadata**: Describe data fields as categorical,
  point-wise, index-like, or references to related object namespaces so generic
  drawers can offer type-appropriate hover and color controls.

### Changed
- **Portable mesh topology**: Resolve implicit Plotly convex hulls into explicit
  triangle faces at the scene boundary so non-Plotly backends can render lite
  showers, cylinders, cones, ellipsoids, and hulls directly.
- **Matched auxiliary colors**: Match particle endpoints and directions and
  interaction vertices to their parent-object colors by default, with an option
  to retain the previous fixed colors.
- **Detector orientation**: Declare the physical up direction in every packaged
  geometry and use it to orient default 3D cameras.
- **DUNE far-detector naming**: Rename the horizontal-drift configuration to
  `DUNE-HD-10kt-1x2x6`; the former `DUNE10kt-1x2x6` name remains available as a
  deprecated alias.

### Fixed
- **Plotly scene parity**: Preserve detector bounds, coordinate units, camera
  orientation, line hover labels, and batched arrow tips when rendering neutral
  scenes through the Plotly compatibility backend.

## [0.17.2] - 2026-08-17

### Fixed
- **Truth shower geometry**: Build aggregate truth-particle start, first-step, timing, end, and semantic properties from a retained visible fragment instead of allowing empty or excluded group progenitors to contribute invalid sentinel geometry, while preserving explicit Michel and delta shapes.
- **Fragment group-primary supervision**: Assign a positive group-primary label only to a retained, label-eligible group progenitor that is uniquely earliest among retained eligible fragments; leave groups without a clean observable primary unsupervised instead of promoting a visible daughter.

## [0.17.1] - 2026-08-14

### Added
- **Charge smearing calibration**: Add configurable Gaussian charge smearing with literal additive and multiplicative modes, mean and scale controls, optional lower clipping, and voxel- or image-scoped random factors shared consistently across TPC partitions.
- **Calibration map inputs**: Load transparency maps from ROOT histograms and support column-mapped per-TPC scalar calibration payloads with configurable value scaling for lifetime and drift-velocity databases.
- **Modifier conflict declarations**: Let configuration modifiers declare incompatible peers and reject conflicting combinations across nested includes.

### Fixed
- **Duplicate-coordinate geometry**: Prevent coincident calibrated voxels from causing divisions by zero in direction optimization and local dE/dx, and handle repeated longitudinal coordinates in spline-based track reconstruction.

### Removed
- **One-off container recovery workflow**: Remove the temporary manual GHCR repair workflow after the release-publishing path was hardened.

## [0.17.0] - 2026-08-10

### Added
- **Typed data products**: Introduce the self-describing `spine.data.product` hierarchy for event and batch tensors, indexes, edge indexes, objects, and cluster labels, with product-owned schemas, coordinate groups, named fields, spans, index shifts, duplicate policies, and NumPy/Torch conversion.
- **Structured cluster labels**: Store compact voxel association data separately from per-particle information, expose named particle, group, interaction, semantic, kinematic, and vertex accessors, and avoid expanding repeated particle quantities until requested.
- **Product-oriented HDF5 I/O**: Add a self-describing `/products` layout with product metadata, schemas, event offsets, primary values, and owned child products; split the reader and writer into focused product, schema, region, and orchestration modules while retaining legacy HDF5 compatibility.
- **Provider-based full chain**: Replace the monolithic full-chain implementation with an ordered provider pipeline built around `ChainState`, `StageResult`, provider specifications, dependency validation, and explicit product ownership. Segmentation, deghosting, calibration, fragmentation, aggregation, and particle-image tasks are now independently composable stages.
- **Rehabilitated model packages**: Establish maintained package-level implementations and train/test configurations for UResNet, PPN, Bayesian UResNet, SPICE, Graph-SPICE, GrapPA shower/track/interaction aggregation, image PID, primary classification, and energy regression.
- **Modular image tasks**: Add reusable object selection, image encoding, prediction heads, task losses, and ancestor-tree supervision for whole-image and object-level classification or regression.
- **On-the-fly validation**: Run deterministic validation at checkpoint boundaries against the live training weights, including distributed metric reduction, configurable validation fractions, restored validation state, and dedicated validation data sources derived from the training loader schema.
- **Early stopping and best checkpoints**: Add validation-metric early stopping and stable best-checkpoint promotion with configurable monitor, direction, minimum improvement, patience, and output path.
- **Checkpoint-bound schedulers**: Let learning-rate schedulers advance per optimization step or per checkpoint and optionally consume a monitored validation metric.
- **Rich checkpoint manifests**: Store a versioned manifest, creation time, SPINE/Python/PyTorch versions, Git revision and dirty state, normalized configuration, dataset provenance, distributed world size, artifact contents, scheduler state, validation state, and per-rank runtime state in each checkpoint.
- **Checkpoint integrity tools**: Write checkpoints atomically, record SHA-256 checksums, verify artifacts on load, inspect checkpoint metadata without constructing a model, and promote an existing snapshot to a best-checkpoint path without reserializing training state.
- **Exact continuation state**: Capture and restore Python, NumPy, PyTorch, CUDA, sampler, loader, and scheduler state for compatible distributed runs, with explicit diagnostics when older checkpoints cannot provide exact continuation.
- **Resume controls**: Add configuration and `--resume`/`--no-resume` CLI controls, automatic continuation when training is supplied a single checkpoint, strict explicit-resume validation, and parameter-only loading for intentional fine-tuning.
- **Distributed inference parity**: Support launcher-driven and internally spawned multi-GPU inference through the same worker path used by training, including externally supplied `RANK`, `LOCAL_RANK`, and `WORLD_SIZE` for multi-node execution.
- **Model configuration contracts**: Add construction and execution checks for every maintained model configuration, deterministic UResNet and full-chain regressions, default model-test collection, and complete `spine.model` statement coverage in CI.
- **Container recovery tooling**: Add recursive OCI-manifest verification and a manually dispatched GHCR recovery workflow that restores recoverable manifests, rebuilds only unhealthy historical releases, and repairs minor-version and `latest` aliases.

### Changed
- **Model package ownership**: Move CNN, sparse, GrapPA, Graph-SPICE, UResNet/PPN, image, full-chain, and common components out of the legacy `spine.model.layer` and top-level monoliths into the packages that own them, with clearer factories, typing, validation, and output contracts.
- **Training configuration**: Normalize training instructions into a canonical top-level `train` block while continuing to accept and relocate the historical `base.train` block with a deprecation warning. Validation configuration lives in a sibling top-level block.
- **Training progress accounting**: Persist epoch progress independently of global iteration so resumed runs retain the correct epoch when minibatch size or loader length changes.
- **Distributed training reporting**: Average scalar training outputs across ranks, gather checkpoint runtime state from every rank, and restrict TensorBoard ownership to rank zero while keeping validation and checkpoint decisions synchronized.
- **Joint dataset sampling**: Default joint datasets to deterministic sequential pairing when no sampler is configured, and derive validation samplers sequentially without removing configured overlay composition.
- **Sparse model contract**: Standardize sparse-network construction, tensor conversion, feature replacement, coordinate queries, duplicate restoration, pooling, empty inputs, and backend-neutral feature access around the public sparse interface.
- **Configuration library**: Reorganize maintained examples by model, pair train and inference configurations, standardize minibatch/epoch/checkpoint settings, and add dedicated full-chain provider documentation and regression configurations.
- **Construction and analysis consumers**: Replace raw numeric column slicing with typed product accessors throughout reconstruction, metrics, post-processing, model inputs, collation, overlay, augmentation, sampling, and unwrapping.
- **Container publishing**: Preserve OCI child manifests instead of deleting apparently untagged GHCR package versions, serialize release builds, verify every published child manifest, and use durable, isolated registry-backed caches for normal releases and historical repairs.

### Fixed
- **PPN supervision and feature alignment**: Pool coarse target masks onto each decoder layer's actual sparse coordinate map and align concatenated sparse features by coordinates rather than backend union order.
- **Model edge cases**: Harden empty clusters, graphs, batches, frozen networks, detached losses, ghost masks, endpoint classification, Graph-SPICE ownership, Delaunay fallback, and degenerate geometric encodings.
- **Vertex and metadata handling**: Use each batch entry's own metadata for vertex coordinate conversion, report mean per-node Euclidean vertex error, and tolerate expected float32 image-bound roundoff without hiding genuine metadata mismatches.
- **Cluster-label correctness**: Preserve feature dtypes, semantic precedence, aligned primary values, multi-volume scalar tensors, overlay charge, interaction IDs, and structured cluster metadata through parsing, collation, overlay, HDF5, construction, and model execution.
- **Distributed device handling**: Correct CUDA batch-ID construction and keep train, validation, and inference rank/device selection consistent under single-node and external multi-node launchers.
- **Checkpoint compatibility**: Load legacy weight files without requiring new metadata, provide actionable warnings for unavailable optimizer, scheduler, epoch, RNG, or loader state, and reject ambiguous training weight lists.
- **GHCR release integrity**: Stop deleting OCI children referenced by tagged indexes, normalize repository names for direct Buildx operations, and retain repair caches across historical rebuild jobs.

### Removed
- **Obsolete model implementations**: Remove the unmaintained experimental model tree, legacy transformer paths, duplicated top-level model modules, and superseded `spine.model.layer` implementations now owned by maintained packages.
- **Ambiguous dense cluster labels**: Remove runtime support for raw dense cluster-label tensors in favor of the structured cluster-label product contract.

### Migration notes
- Imports from `spine.model.layer.*` and superseded top-level model modules must move to their new owning packages.
- New full-chain configurations should use the ordered `chain.stages` provider plan; historical mode-matrix configurations remain translated for compatibility.
- Cluster parser legacy options remain accepted, but parsers now return structured cluster-label products. Consumers should use named accessors instead of hard-coded columns.
- Legacy HDF5 layouts and checkpoints remain readable. Newly written files and checkpoints carry richer product, provenance, validation, and continuation metadata.
- Existing `base.train` blocks continue to work for this release but are normalized to top-level `train` with a deprecation warning.

## [0.16.3] - 2026-08-06

### Added
- **Truth-object timing and filtering**: Expose particle creation time through a consistent derived attribute, derive truth-interaction time from neutrino information or the earliest constituent particle, and support inclusive truth-time windows in cluster metrics.
- **Point-completeness diagnostics**: Compare detector-level truth-associated points with Geant4 depositions using configurable spatial and time windows, reporting bidirectional purity and efficiency for fragments, particles, or interactions.
- **Stored optical hypotheses**: Persist per-channel likelihood predictions for eligible interactions, associate them with observed flashes, serialize them through HDF5, and overlay measured and predicted light in the event display with optional shared PE-based sizing.
- **Full flash-match candidates**: Optionally retain every positive-scoring OpT0Finder interaction/flash hypothesis and score while preserving the selected match for standard analysis and visualization.

### Changed
- **Flash-matching analysis outputs**: Support configurable matching directions, unmatched interaction counterparts, flash multiplicities, overlap scores, and additional reconstructed or truth attributes in the optical metric analyzer.

### Fixed
- **Point diagnostic robustness**: Handle empty and degenerate point collections safely in point- and track-completeness studies while preserving configurable truth point representations.
- **Invalid MCS fits**: Return `NaN` when kinetic-energy optimization fails or saturates a fit boundary, expose configurable fit limits, and provide an explicit legacy option to retain invalid optimizer values.

## [0.16.2] - 2026-08-06

### Added
- **Position-selected field maps**: Load suffixed field-map variants and select them by coordinate boundaries, including support for the SBND east/west dual map and matching calibration diagnostic plots.

### Fixed
- **LArCV index compatibility**: Infer the invalid particle-index sentinel per event, preserve valid historical index values, and handle NumPy integer index attributes during overlays.
- **Interaction overlay shifts**: Derive interaction spans consistently across cluster, particle, and neutrino products so overlaid interaction IDs remain aligned.
- **Catch-all cluster semantics**: Label unmatched LArCV `cluster3d` depositions with the valid unknown semantic shape instead of the generic `-1` sentinel.

## [0.16.1] - 2026-08-04

### Changed
- **Calibration module priorities**: Apply calibration modules in descending configured priority order.

### Fixed
- **Truth fragment selection**: Build truth particles from the first visible fragment using its first energy-deposition time and shape, rather than particle creation time or the group representative.
- **Low-energy particle labels**: Consistently exclude low-energy clusters from particle associations unless explicitly enabled, while preserving valid duplicate-voxel precedence handling.
- **Particle step timing**: Preserve first- and last-step timestamps in canonical particle objects and use first-step time for particle-coordinate labels and group-primary selection.

## [0.16.0] - 2026-07-27

### Added
- **Versioned HDF5 V2 layout**: Add an opt-in offset-based HDF5 schema that stores event and object boundaries as integer ranges and variable-length object attributes in flat typed datasets, while retaining the legacy V1 writer and automatic V1/V2 reader compatibility.
- **Columnar HDF5 analysis**: Add reader-controlled columnar loading, `process_columnar` analysis hooks, fixed-field-only reads, and stored scalar best-match identifiers so high-level exports can avoid rebuilding event objects and relationship arrays.
- **HDF5 maintenance tools**: Add V1/V2-aware output comparison and structural V2 litification utilities, including configurable product selection and fixed-only object reduction without event deserialization.
- **Backend-neutral sparse tensors**: Introduce a public sparse model abstraction and MinkowskiEngine adapter covering tensor construction, feature replacement, sparse modules, empty inputs, batching, and backend-independent model outputs.

### Changed
- **Sparse duplicate handling**: Coalesce duplicate coordinates through explicit `sum`, `mean`, or `first` policies while preserving source-row provenance, allowing UResNet, PPN, and GraphSPICE outputs to restore original input ordering and multiplicity.
- **Selective HDF5 performance**: Reduce V2 read and object-construction overhead through pooled range reads, trusted stored-object reconstruction, selective variable-field loading, and direct fixed-dataset access.
- **Analysis execution policy**: Let the HDF5 reader select event or columnar delivery while `AnaManager` validates module support and supplies each columnar analyzer's requested products and fields.
- **Command-line utilities**: Expand typing, parser entry points, tests, and usage documentation across calibration, geometry, LArCV, and SPINE output scripts.

### Fixed
- **Sparse decoder cardinality**: Prevent backend coordinate deduplication from shrinking row-aligned segmentation and point-proposal outputs relative to their input tensors.
- **Cross-product jitter alignment**: Apply one random displacement per unique event coordinate across products with different subsets, lengths, row orderings, and coordinate dtypes, while leaving collision reduction to downstream sparse consumers.

## [0.15.5] - 2026-07-23

### Added
- **Response calibration functions**: Add configurable NumExpr response transformations as a dedicated calibration stage, separate from detector gain calibration.

### Changed
- **Parsing and data-object performance**: Reduce allocations in LArCV cluster parsing and cache data-class reflection metadata used for field validation and serialization.
- **Data-object type enforcement**: Preserve declared array dtypes on assignment and reject undeclared attributes, keeping reconstructed objects internally type-consistent.

### Fixed
- **Field-corrected reconstructed positions**: Restore temporary module translations before returning calibrated coordinates and propagate field corrections to every declared reconstructed positional attribute, including start, end, PPN, and vertex points.
- **Mixed-precision reconstruction**: Keep coordinate, endpoint, and direction dtypes consistent so Numba tracking and vertexing kernels do not fail on mixed `float32`/`float64` inputs.

## [0.15.4] - 2026-07-21

### Added
- **NVIDIA P100 support**: Compile MinkowskiEngine for compute capability 6.0 and advertise Pascal P100 compatibility in the SPINE container and GPU diagnostics.

## [0.15.3] - 2026-07-20

### Added
- **CLI output options**: Add dedicated `--output-dir` and `--output-suffix` options alongside `--output`, and warn when output options are supplied without a configured `io.writer` block.

## [0.15.2] - 2026-07-20

### Fixed
- **CLI input source overrides**: Clear the configured `file_list` when `--source` is provided and clear `file_keys` when `--source-list` is provided, preventing mutually exclusive reader inputs from being passed together and improving the related validation errors.

## [0.15.1] - 2026-07-17

### Fixed
- **Drawer hovertext ordering**: Preserve the caller-provided attribute order when building reconstructed and truth object hovertext instead of iterating attributes in hash-dependent set order.

## [0.15.0] - 2026-07-17

### Added
- **Shared module manager**: Add a reusable `ModuleManager` for running ordered analysis and post-processing modules on individual entries and batches.
- **Repeated module instances**: Allow analysis and post-processing modules to be configured more than once through explicit `name` fields, with deterministic priority ordering and validation of malformed module blocks.
- **Configuration fragments**: Add `kind: fragment` for reusable, intentionally unversioned configuration pieces. Fragments can be included without missing-metadata warnings, do not register their own component version, and still propagate versions from nested components.
- **Analysis and post-processing coverage**: Add comprehensive unit coverage for managers, factories, metrics, diagnostics, reconstruction modules, optical and CRT matching, triggers, truth utilities, and CSV output.

### Changed
- **Analysis and post-processing infrastructure**: Refactor `AnaManager` and `PostManager` around the shared manager implementation, type them directly against their module base classes, and modernize module interfaces and documentation throughout both packages.
- **Configuration validation**: Preserve caller-provided module configurations and replace configuration-related assertions with explicit `ValueError`, `TypeError`, and `KeyError` exceptions.
- **GrapPA label points**: Make label-point ordering configurable and improve label-based full-chain aggregation, identity, group assignment, fragmentation, and truth-point selection.

### Fixed
- **Segmentation and truth propagation**: Correctly expand deghosted segmentation outputs to the original point set and propagate field-corrected coordinates through truth particles, interactions, and reference tensors.
- **Analysis diagnostics**: Avoid meaningless interaction shape metrics, correct mixed-shape graph indexing and distance lookup, fix detailed ghost scores, and make CSV attribute ordering deterministic.
- **Optical and CRT matching**: Correct charge-weighted barycenters, optical-coordinate indexing, flash-match score storage, and CRT matching behavior and validation.
- **Reconstruction edge cases**: Enable proton-to-point shower conversion distances and safely handle interactions without nonempty particles during calorimetric direction reconstruction.
- **Optional ROOT checks**: Avoid importing ROOT merely to probe optional dependency availability.

## [0.14.2] - 2026-06-30

### Fixed
- **CVMFS container setup**: Restore LArCV feature flags in `/opt/spine/setup.sh` so unpacked-image runtimes expose PyUtil bindings such as `larcv.fill_3d_voxels`, and make `/opt/spine/check-env.sh` validate that binding explicitly.
- **Stage-cache writer configuration**: Raise a clear configuration error when `stage_hdf5` is used without `base.split_output: true`.
- **Writer output directories**: Create configured output directories automatically for HDF5, staged HDF5, and CSV writers.
- **Staged cache provenance**: Preserve source entry metadata in staged caches and provide a fallback when reading older/minimal stage-cache files without explicit `source_file_entry_index`.
- **Remote source provenance**: Populate stable source provenance keys for XRootD-streamed inputs using sentinel values when file size and modification time are unavailable.

## [0.14.1] - 2026-06-27

### Fixed
- **GrapPA inference stability**: Normalize indexed cluster coordinate views before Numba-compiled distance, endpoint, node-feature, and edge-feature helpers so full-chain GrapPA inference no longer fails intermittently on arbitrary-layout arrays in batch jobs.
- **Regression coverage**: Add focused tests for arbitrary-layout Numba callers in distance helpers and GrapPA cluster/node/edge feature extraction.

## [0.14.0] - 2026-06-26

### Added
- **Public calibration package**: Promote calibration utilities from `spine.utils.calib` to the top-level `spine.calib` package, with focused coverage for calibration constants, databases, factories, managers, gain, lifetime, transparency, recombination, and field corrections.
- **SCE field-map calibration**: Add electric-field non-uniformity corrections through `FieldCalibrator` and `FieldMap`, including dense map interpolation, ROOT TH3 map loading, configurable out-of-bounds behavior, detector-volume transforms, and `bin/calib/sce_field_check.py` validation tooling.
- **Response calibration functions**: Add `ResponseCalibrator` with configurable NumExpr expressions such as `2.3 * exp(x) - 3` to transform deposition values independently of flat or database-backed gain calibration.
- **DUNE-VD geometry support**: Add DUNE-VD 10 kt geometry metadata and harden geometry parsing for nonterminal version tags in LArSoft/Flow geometry descriptions.
- **Calorimetric interaction directions**: Add `CalorimetricDirectionProcessor` for charge-weighted interaction direction reconstruction, with separate storage for true and reconstructed interaction direction fields.
- **Validation and metrics coverage**: Add configurable truth-index selection for cluster metrics, plus tests for calorimetric direction reconstruction, calibration modules, geometry parsing, LArCV helpers, and output validation scripts.

### Changed
- **Calibration namespace**: Remove the old `spine.utils.calib` namespace in favor of `spine.calib`; update calorimetry, full-chain, and post-processing imports accordingly.
- **Script organization**: Move utility scripts into domain-specific `bin/calib`, `bin/geo`, `bin/larcv`, and `bin/output` directories with lightweight README files and shared LArCV ROOT tree helpers.
- **Factory parsing**: Extend module factory parsing to support repeated modules with explicit names while preserving the legacy single-module configuration style.

### Fixed
- **MCS robustness**: Skip MCS kinetic-energy reconstruction for one-point tracks and prevent `bin_pca` segment PCA from running on one-point chunks, avoiding PCA assertion failures for degenerate track-like objects.
- **Shared download caches**: Normalize downloaded cache artifacts to group-readable permissions, preserve atomic downloads, and raise clear permission errors when cache directories, lock files, temporary files, or existing cached downloads cannot be accessed.
- **Field and calibration edge cases**: Add validation around field-map dimensions, bounds handling, calibration constant source selection, and response function expression inputs so configuration mistakes fail early with actionable errors.

## [0.13.3] - 2026-06-03

### Added
- **Container runtime setup script**: Add `/opt/spine/setup.sh` and `/opt/spine/check-env.sh` to the published container image so ROOT/LArCV/SPINE runtime environment setup is explicit, testable, and recoverable when unpacked-image runtimes fail to apply the container environment automatically.

## [0.13.2] - 2026-06-02

### Changed
- **Semantic overlay precedence**: Include the ghost semantic class at the end of the default shape precedence so ghost-inclusive semantic labels can use precedence-based duplicate cleanup safely during overlays.
- **LArCV overlay configuration**: Add dataset-level `overlay_methods` overrides to `LArCVDataset`, matching `HDF5Dataset`, so products such as `run_info` can use policies like `first`, `match`, or `cat` without changing parser defaults.

### Fixed
- **Overlay duplicate cleanup**: Preserve aligned feature-only tensors during overlay duplicate cleanup by letting tensors such as `sources` reuse the row selection from an explicit `overlay_reference`, and add sum/average aggregation support for duplicate sparse features.
- **Manager stopwatch recovery**: Reset active manager-owned stopwatches before new manager calls so exceptions in I/O, model, post-processing, or analysis do not leave watches stuck in a running state.

## [0.13.1] - 2026-06-01

### Added
- **Joint overlay datasets**: Add backend-agnostic `JointDataset` support for overlay training across independently sampled primary and secondary datasets, plus dedicated joint samplers and loader validation so tuple-based pairing is only used with joint datasets.

### Changed
- **Dataset documentation and coverage**: Expand dataset-layer docstrings, clarify aligned (`MixedDataset`) versus unaligned (`JointDataset`) composition semantics, and add focused tests for joint dataset construction, pairing, and loader/sampler validation.

### Fixed
- **On-demand driver usability**: Restore the ability to omit both `iterations` and `epochs` when using the driver and I/O manager in on-demand mode, while still rejecting the ambiguous case where both are specified.

## [0.13.0] - 2026-05-30

### Added
- **Visualization docs**: Add a dedicated `spine.vis` README covering the reorganized trace, drawer, metric, and layout structure introduced in [#131](https://github.com/DeepLearnPhysics/spine/pull/131).
- **Driver logging backends**: Add a structured `LogManager` with optional TensorBoard integration and CSV/timing/memory logging support as part of the driver refactor in [#132](https://github.com/DeepLearnPhysics/spine/pull/132).
- **Index-span metadata**: Add explicit parser payload classes and span-aware batch metadata for flat indexes, index lists, and edge indexes in [#133](https://github.com/DeepLearnPhysics/spine/pull/133).

### Changed
- **Math package cleanup**: Review and tighten `spine.math` typing, tests, and helper behavior, including the iterative pair-distance path now used by the full-chain regression baseline in [#130](https://github.com/DeepLearnPhysics/spine/pull/130).
- **Visualization package structure**: Reorganize `spine.vis` into explicit `trace`, `drawer`, `metric`, and `layout` subpackages, preserve direct import exposure through `spine.vis`, and restore comprehensive coverage in [#131](https://github.com/DeepLearnPhysics/spine/pull/131).
- **Driver and I/O ownership boundaries**: Refactor driver initialization, move batching/unwrapping responsibilities under `spine.io`, introduce an `IOManager`, and separate structured logging concerns through [#132](https://github.com/DeepLearnPhysics/spine/pull/132).
- **Index batching model**: Replace implicit global index shifts with explicit per-entry spans throughout parsing, collation, overlay, unwrapping, and cached HDF5 index handling in [#133](https://github.com/DeepLearnPhysics/spine/pull/133).
- **Container/runtime defaults**: Simplify container metadata handling and restore multi-rank training summaries so distributed runs emit one coherent per-rank progress table from the main process.

### Fixed
- **Closest/farthest pair utilities**: Fix pair-distance helper behavior used by GrapPA feature engineering and align the deterministic full-chain regression reference with the iterative implementation in [#130](https://github.com/DeepLearnPhysics/spine/pull/130).
- **Visualization regressions**: Fix CI regressions in the reorganized output drawer, restore coverage, and preserve existing behavior after the package shuffle in [#131](https://github.com/DeepLearnPhysics/spine/pull/131).
- **CLI and runtime polish**: Move banner printing to the CLI, reduce duplicate startup output, improve bin-package typing, and expand coverage around the main runtime and entrypoint helpers as part of [#132](https://github.com/DeepLearnPhysics/spine/pull/132).
- **Index list semantics**: Preserve list-backed `IndexBatch` behavior for object-array-backed cluster lists, fix downstream PPN cluster access after the span refactor, and require count metadata for HDF5 index parsers where spans must be reconstructed after [#133](https://github.com/DeepLearnPhysics/spine/pull/133).
- **Distributed training summaries**: Gather per-rank iteration rows onto rank 0 so training logs once again report timing, memory, loss, and accuracy for every process without duplicating the full header block.

## [0.12.4] - 2026-05-18

### Changed
- **Docker publishing**: Add a persistent Buildx registry cache for published container builds so release-tag builds can reuse expensive dependency layers across workflow runs instead of relying only on GitHub Actions cache scope.

### Fixed
- **Stored property metadata**: Add missing array metadata for output data derived attributes so values such as `module_ids` and truth direction vectors are correctly classified for scalar expansion and serialization introspection.
- **Truth particle units**: Correct `TruthParticle.ke` metadata from rest-mass units to kinetic-energy units.

## [0.12.3] - 2026-05-12

### Added
- **Data attribute introspection**: Add `DataBase.attr_names()` to expose the full valid attribute surface, including derived and serialization-skipped attributes by default.

### Fixed
- **Output visualization attributes**: Use `DataBase.attr_names()` when validating drawer hover attributes so derived quantities such as `RecoParticle.ke` can be displayed.

## [0.12.2] - 2026-05-12

### Fixed
- **Truth object units**: Rebuild per-class field metadata caches after multiprocessing worker unpickle so truth particle and interaction coordinates convert from pixel units to detector coordinates correctly during output construction.

## [0.12.1] - 2026-05-11

### Fixed
- **GrapPA cluster dE/dx**: Normalize mixed coordinate dtypes before anchored distance calls so GrapPA feature engineering no longer fails in Numba when `start` arrives as `float64` and voxel coordinates are `float32`.

## [0.12.0] - 2026-05-10

### Added
- **Staged HDF5 caching**: Add staged cache readers and writers that support one cache file per source file, per-stage completeness tracking, provenance validation, and staged cache reuse across sequential workflows ([#128](https://github.com/DeepLearnPhysics/spine/pull/128)).
- **Mixed dataset loading**: Add `MixedDataset` plus staged/flat `HDF5Dataset` support so live LArCV inputs can be aligned with cached HDF5 products in training and inference jobs ([#128](https://github.com/DeepLearnPhysics/spine/pull/128)).
- **Generic HDF5 parsers**: Add cached-tensor, cached-index, and cached-object parsers for HDF5-backed SPINE products, including feature ablation and cluster-tensor reconstruction support ([#128](https://github.com/DeepLearnPhysics/spine/pull/128)).
- **Data augmentation**: Add rotation and pixel-jitter augmentation plus broader augmentation test coverage and geometry-aware worker initialization ([#127](https://github.com/DeepLearnPhysics/spine/pull/127)).
- **Validation tooling**: Update `bin/output_check_valid.py` to prefer staged-cache completeness and provenance metadata when available while preserving legacy fallback checks.

### Changed
- **I/O package structure**: Reorganize `spine.io` around top-level readers, writers, parsers, datasets, augmentation, collation, overlay, and sampling utilities, replacing the older `core`/`torch` split ([#128](https://github.com/DeepLearnPhysics/spine/pull/128)).
- **Writers**: Extend HDF5, staged HDF5, and CSV writers with cleaner prefix/suffix/directory handling and driver-facing staged-writer integration ([#128](https://github.com/DeepLearnPhysics/spine/pull/128)).
- **Documentation**: Refresh the `spine.io` API docs to reflect the new staged-cache and dataset structure, and harden docs builds against missing optional ML dependencies.
- **Testing**: Expand `spine.io`, augmentation, bin-script, and staged-cache regression coverage substantially; restore full `spine.io` coverage after the refactor ([#127](https://github.com/DeepLearnPhysics/spine/pull/127), [#128](https://github.com/DeepLearnPhysics/spine/pull/128)).

### Fixed
- **GrapPA caching path**: Preserve the standard GrapPA path while supporting cached cluster/edge/feature inputs, and fix related geometric-feature and Numba indexing/reshape issues ([#128](https://github.com/DeepLearnPhysics/spine/pull/128)).
- **Cluster-label adaptation**: Switch full-chain cluster label adaptation to use `orig_index` provenance instead of dense ghost masks, enabling cached deghosted workflows with evolving segmentation predictions ([#128](https://github.com/DeepLearnPhysics/spine/pull/128)).
- **Optional imports**: Make optional-dependency proxies and docs builds robust when PyTorch and other heavy dependencies are absent or mocked.
- **Stage writer stability**: Isolate per-stage schema state correctly so staged cache writes do not leak product definitions across stages ([#128](https://github.com/DeepLearnPhysics/spine/pull/128)).

### Notes
- **Caching workflow maturity**: This release includes the core staged-caching infrastructure needed for sequential training and inference workflows. The end-to-end full-chain caching-enabled training workflow has not yet been exhaustively validated across every stage and may still require additional integration debugging.

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
