# Output Scripts

Utilities for checking SPINE processing outputs.

## Compare two HDF5 outputs

`output_compare.py` compares two SPINE HDF5 files event by event. HDF5 region
references are dereferenced before comparison, so differences in file layout or
dataset order do not affect the result. Integer, boolean, and string values must
match exactly; floating-point values use relative and absolute tolerances.

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
