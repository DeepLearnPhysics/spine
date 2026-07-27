# Output Scripts

Utilities for checking SPINE processing outputs.

## Validate campaign outputs

`output_check_valid.py` maps each source input to its expected SPINE output and
writes missing or incomplete source paths to a text file suitable for
resubmission:

```bash
python3 bin/output/output_check_valid.py \
  --source-list inputs.txt \
  --dest /path/to/outputs \
  --suffix spine \
  --output retry.txt
```

For modern HDF5 files it checks the writer's completion marker and source-file
provenance. Older HDF5 files fall back to entry-count validation. Use
`--larcv-output` when the campaign produced ROOT rather than HDF5, and
`--tree-name` to select the reference LArCV tree. `--event-list` accepts
whitespace- or comma-separated `(run, subrun, event)` triplets when only a
subset was requested.

## Compare two HDF5 outputs

`output_compare.py` compares two SPINE HDF5 files event by event. It supports
both the legacy region-reference/VLEN layout (format version 1) and the
offset-based layout (format version 2), including direct V1-to-V2 comparisons.
Physical references, offsets, dataset order, and object-field pools are
normalized before comparison. Integer, boolean, and string values must match
exactly; floating-point values use relative and absolute tolerances.

```bash
python3 bin/output/output_compare.py reference.h5 candidate.h5
```

The default floating-point tolerances are `rtol=1e-4` and `atol=1e-6`. Use
exact comparison for deterministic CPU validation:

```bash
python3 bin/output/output_compare.py reference_cpu.h5 candidate_cpu.h5 --exact
```

Individual products can be selected or omitted with `--keys` and
`--skip-keys`, respectively. A mismatch returns exit status 1, which makes the
script suitable for automated validation.

## Create a lite V2 HDF5 output

`output_litify.py` structurally reduces an offset-based format-V2 file without
deserializing events or rebuilding SPINE objects. With no product selection it
uses the standard production lite product list:

```bash
python3 bin/output/output_litify.py full.h5 lite.h5
```

Products can be supplied directly:

```bash
python3 bin/output/output_litify.py full.h5 lite.h5 \
  --keys run_info meta reco_particles truth_particles
```

The script can also read either a small YAML file containing a top-level
`keys` list or an existing SPINE configuration containing `io.writer.keys`:

```bash
python3 bin/output/output_litify.py full.h5 lite.h5 --config litify.yaml
```

Use `--mode fixed_only` to discard every variable-length object attribute.
The default `lite` mode exactly follows each stored class's established
`lite=True` policy and therefore retains small relationship arrays.
