# Output Scripts

Utilities for checking SPINE processing outputs.

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
