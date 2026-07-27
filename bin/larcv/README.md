# LArCV Scripts

Utilities for inspecting, validating, and editing LArCV ROOT files.

All commands accept input paths either directly with `--source`/`-s` or from a
text file containing one path per line with `--source-list`/`-S`. These options
are mutually exclusive. ROOT and LArCV Python bindings must be available.

## Validate files before merging

`larcv_check_valid.py` verifies that every `*_tree` in each file has the same
non-zero entry count and that all files expose the same trees. It writes bad
input paths to a text file.

```bash
python3 bin/larcv/larcv_check_valid.py \
  -S files.txt --output bad_files.txt
```

## Count events

`larcv_count_entries.py` reports per-file and total event counts. By default it
uses the first LArCV tree; `--tree-name` selects a short product name without
the `_tree` suffix.

```bash
python3 bin/larcv/larcv_count_entries.py \
  -s input_0.root input_1.root --tree-name sparse3d_pcluster
```

## Find duplicate files

`larcv_find_duplicates.py` groups likely duplicates using the entry count and
the first `(run, subrun, event)` triplet. All but the first file in each group
are written to the requested output list.

```bash
python3 bin/larcv/larcv_find_duplicates.py \
  -S files.txt --output duplicates.txt
```

This is a fast campaign-level heuristic; it does not compare every event or
payload byte.

## Select files from one run

`larcv_find_run.py` writes files whose first event carries a requested run
number.

```bash
python3 bin/larcv/larcv_find_run.py \
  -S files.txt --run-number 10536 --output run_10536.txt
```

## Inject run numbers

`larcv_inject_run_number.py` rewrites event IDs across all products. Choose one
output policy (`--dest` or `--overwrite`) and one numbering policy:
`--run-number`, `--run-list`, or `--offset`.

```bash
python3 bin/larcv/larcv_inject_run_number.py \
  -S files.txt --dest updated --run-number 10536 --suffix run_fixed
```

A run number of `-1` assigns each input file its zero-based position in the
input list. `--run-list` consumes one integer per input file, while `--offset`
adds a constant to each existing run number. `--overwrite` replaces original
files after LArCV has successfully finalized temporary outputs, so it should be
used deliberately.

## Measure product sizes per event

`larcv_tree_sizes.py` writes one CSV per input file with the number of elements
in each requested product for every event.

```bash
python3 bin/larcv/larcv_tree_sizes.py \
  -S files.txt \
  --tree-names sparse3d_pcluster particle_pcluster \
  --suffix counts
```

CSV files are written in the current directory as
`<input-stem>_<suffix>.csv`. Existing files are skipped unless `--replace` is
provided.
