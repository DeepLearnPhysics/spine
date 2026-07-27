# Geometry Scripts

Utilities for extracting or converting detector geometry descriptions.

## Convert FLOW geometry

`parse_flow_geometry.py` reads detector and optical geometry stored in a FLOW
HDF5 file and emits a SPINE geometry YAML file.

```bash
python3 bin/geo/parse_flow_geometry.py \
  --source FSD_CosmicRun3.flow.0000000.FLOW.hdf5 \
  --tag cr3 \
  --output fsd_cr3.yaml \
  --opdet-thickness 1.0
```

`--tag` identifies the geometry revision and is used to derive the version.
`--opdet-thickness` is required only when the stored optical-detector bounds
have zero thickness. If `--output` is omitted, the script replaces the input
`.hdf5` suffix with `_geometry.yaml`.

## Convert a LArSoft geometry dump

First produce a text dump using the appropriate LArSoft geometry configuration,
then convert it:

```bash
lar -c dump_icarus_geometry.fcl
python3 bin/geo/parse_larsoft_geometry.py \
  --source icarus-geometry.txt \
  --output icarus.yaml
```

`--cathode-thickness` and `--pixel-size` adjust active-volume dimensions when
the text dump does not encode those effects directly. `--crt-mapping` accepts
a YAML mapping from CRT module names to stable logical IDs. Without `--output`,
the input `.txt` suffix is replaced with `_tpc.yaml`.

Both converters print the generated YAML before writing it, making their output
easy to inspect before installing it as detector geometry.
