# Geometry Package

`spine.geo` provides detector geometry objects used to map detector sources,
translate points between detector modules, check containment, and describe
optional optical and CRT subsystems.

Geometry definitions live in:

```text
src/spine/geo/config/<detector>/<detector>_<tag>_geometry.yaml
```

They are loaded through `geo_factory` or the process-wide `GeoManager`
singleton.

## Loading Geometry

Use the factory when you need a standalone geometry object:

```python
from spine.geo.factories import geo_factory

geo = geo_factory("icarus")
geo = geo_factory("2x2", version="6")
geo = geo_factory("sbnd", tag="sbnd_v2-6")
```

Use the manager when code paths need shared geometry access:

```python
from spine.geo import GeoManager

geo = GeoManager.initialize_or_get("icarus")
same_geo = GeoManager.get_instance()
```

Selection rules:

- `detector` is case-insensitive.
- `tag` selects an exact configuration tag.
- `version="6"` selects the latest matching `6.x` configuration.
- `version="6.5"` selects the exact normalized version.
- If no tag or version is provided, the latest numeric version is selected.

## YAML Schema

Every geometry file must provide:

```yaml
name: 2x2
tag: mr6-5
version: 6.5
tpc:
  dimensions: [30.6, 129.6, 64.0]
  positions:
    - [15.6, 0.0, 0.0]
    - [-15.6, 0.0, 0.0]
  module_ids: [0, 0]
```

Top-level optional fields:

- `gdml`: GDML filename associated with the geometry.
- `crs_files`: Charge readout system geometry file references.
- `lrs_file`: Light readout system geometry file reference.
- `optical`: Optical detector configuration.
- `crt`: CRT detector configuration.

## TPC Configuration

The `tpc` block is parsed by `TPCDetector`.

Required fields:

- `dimensions`: One `[x, y, z]` size shared by all TPCs, or one size per TPC.
- `positions`: TPC center positions in centimeters.
- `module_ids`: Module ID for each TPC position.

Optional fields:

- `det_ids`: Mapping from logical TPC IDs to physical TPC IDs.
- `drift_dirs`: One axis-aligned drift direction per TPC. If omitted, two-TPC
  modules infer drift from the two TPC centers.
- `limits`: Active-region bounding planes.

Example with detector ID mapping and active limits:

```yaml
tpc:
  dimensions: [148.2, 316.82, 1789.902]
  positions:
    - [-284.39, -23.45, 0.0]
    - [-136.04, -23.45, 0.0]
    - [136.04, -23.45, 0.0]
    - [284.39, -23.45, 0.0]
  module_ids: [0, 0, 1, 1]
  det_ids: [0, 0, 1, 1]
  limits:
    intercepts:
      - [0.0, 1650.6114, 0.0]
    norms:
      - [0.0, 0.5, -0.866025]
```

## Optical Configuration

The `optical` block is parsed by `OptDetector`.

Required fields:

- `volume`: Either `tpc` or `module`. Determines which TPC geometry centers are
  used as optical volume offsets.
- `shape`: One of `ellipsoid`, `box`, `disk`, or a list of those shapes.
- `dimensions`: One `[x, y, z]` shape size, or one size per shape.
- `positions`: Optical detector positions relative to each optical volume.

Optional fields:

- `shape_ids`: Shape index for each physical detector when multiple shapes are
  listed.
- `det_ids`: Mapping from readout channel to physical optical detector. This
  allows multiple channels per detector.
- `global_index`: If `true`, optical channel indexes refer to the whole
  detector instead of one optical volume.
- `mirror`: If `true`, mirror detector positions across alternating TPC volumes.

Example:

```yaml
optical:
  volume: tpc
  shape: [box, box]
  mirror: true
  dimensions:
    - [48.0, 31.02, 1.0]
    - [48.0, 10.34, 1.0]
  shape_ids: [1, 1, 1, 0]
  det_ids: [0, 0, 1, 1]
  positions:
    - [0.0, -149.93, -49.0]
    - [0.0, -139.59, -49.0]
    - [0.0, -129.25, -49.0]
    - [0.0, -108.57, -49.0]
```

## CRT Configuration

The `crt` block is parsed by `CRTDetector`.

Required fields:

- `dimensions`: One `[x, y, z]` size per CRT plane.
- `positions`: One center position per CRT plane.
- `normals`: Normal-axis index for each CRT plane.

Optional fields:

- `logical_ids`: Mapping from logical CRT channel IDs to physical plane IDs.

Example:

```yaml
crt:
  dimensions:
    - [100.0, 5.0, 500.0]
  positions:
    - [0.0, 250.0, 0.0]
  normals: [1]
  logical_ids: [42]
```

## Common Queries

```python
import numpy as np

from spine.geo.factories import geo_factory

geo = geo_factory("icarus")

boundaries = geo.get_boundaries(with_optical=False, with_crt=False)
closest_tpc = geo.get_closest_tpc(np.array([[0.0, 0.0, 0.0]]))
contributors = geo.get_contributors(np.array([[0, 0], [0, 1]]))

definition = geo.define_containment_volumes(
    margin=5.0,
    mode="module",
    include_limits=True,
)
contained = geo.check_containment(
    definition,
    np.array([[0.0, 0.0, 0.0]]),
)
```

Containment modes:

- `tpc`: points must be contained in one TPC volume.
- `module`: points must be contained in one detector module.
- `detector`: points must be contained in the full detector envelope.
- `source`: source IDs define the contributing TPC volumes.

## Generating Geometry Files

Two helper scripts exist for geometry extraction:

- `bin/geo/parse_larsoft_geometry.py`: parses LArSoft geometry dumps.
- `bin/geo/parse_flow_geometry.py`: parses FLOW HDF5 geometry metadata.

Typical LArSoft workflow:

```bash
lar -c dump_<detector>_geometry.fcl > geometry_dump.txt
python bin/geo/parse_larsoft_geometry.py \
  --source geometry_dump.txt \
  --output src/spine/geo/config/<detector>/<detector>_<tag>_geometry.yaml
```

Typical FLOW workflow:

```bash
python bin/geo/parse_flow_geometry.py \
  --source detector.flow.hdf5 \
  --tag mr6-5 \
  --opdet-thickness 1.0 \
  --output src/spine/geo/config/<detector>/<detector>_<tag>_geometry.yaml
```

After adding a geometry, validate that `geo_factory("<detector>", tag="<tag>")`
loads it and that expected TPC, optical, and CRT counts match the source
detector description.

## Design Notes

- `GeoManager` is intentionally a singleton for shared runtime geometry. Tests
  or scripts that need isolation should call `GeoManager.reset()`.
- Configuration loading is file-based and intentionally lightweight. The
  factory rereads available YAML files on demand.
- Detector constructors still validate several configuration invariants at
  runtime. Prefer explicit exceptions for new validations so checks remain
  active under optimized Python.
