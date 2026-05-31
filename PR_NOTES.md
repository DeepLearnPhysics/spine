# PR Notes: feature/index-span

## Summary

This PR refactors index-style parser outputs and batch objects so index spans are preserved explicitly through parsing, collation, overlay, unwrapping, and downstream model utilities. It also moves the unwrapper into the IO package, tightens batch typing/static analysis, and expands coverage for the affected data and IO paths.

## Motivation

Index-like parser products were previously treated too much like generic tensors. That made it hard to distinguish plain tensor features from flat indexes, index lists, and edge indexes, and it meant downstream rebatching/unwrapping had to infer parent spans indirectly. This PR gives those concepts first-class parser and batch representations so cached HDF5 indexes can round-trip with the span metadata needed to recover event-local indexing.

## Main Changes

- Adds explicit parser payload classes for index products:
  - `ParserIndex` for flat index arrays.
  - `ParserIndexList` for lists of indexes.
  - `ParserEdgeIndex` for graph edge indexes.
- Updates HDF5 index parsers to require and consume the matching count dataset so they can compute and store the parent index span.
- Adds `spans` and derived `offsets` to `IndexBatch` and `EdgeIndexBatch`.
- Updates collation, overlay, and rebatching logic to propagate spans instead of relying on inferred offsets.
- Moves `Unwrapper` from `spine.utils.unwrap` to `spine.io.unwrap`, where it belongs with the rest of IO/batching logic.
- Updates full-chain/model utility call sites to import the unwrapper from `spine.io`.
- Extends unwrapping to export span metadata for index-like batches under `<name>_spans` when that key is not already present.
- Handles multi-volume index unwrapping by combining per-volume spans and adjusting offsets event-wise.
- Adds modern type annotations to `spine.data.batch` classes and resolves pyright/Pylance issues around optional `torch` and `MinkowskiEngine` types.
- Adds a structural `SparseTensorLike` protocol plus `is_sparse_tensor_like(...)` helper for optional MinkowskiEngine sparse tensors without requiring ME stubs or import-time ME availability.

## Behavioral Notes

- HDF5 index parsers now require the corresponding count event input. This is intentional: the count information is needed to compute the index span safely.
- Index-like batches now carry explicit `spans`, so merging and unwrapping can validate compatible parent spans and preserve correct offsets.
- The unwrapper now adds `<name>_spans` for `IndexBatch` and `EdgeIndexBatch` outputs unless that key already exists in the input dictionary.
- Existing direct imports from `spine.utils.unwrap` should be updated to `spine.io.unwrap`.

## Typing and Static Analysis

- `BatchBase`, `TensorBatch`, `IndexBatch`, and `EdgeIndexBatch` now have substantially better annotations.
- Optional `torch.Tensor` aliases are hidden behind `TYPE_CHECKING` so importing `spine.data.batch.base` does not force a runtime torch import.
- Real `MinkowskiEngine.SparseTensor` objects are detected through `is_sparse_tensor_like(...)` rather than a runtime-checkable protocol, avoiding false rejections of valid ME tensors.
- Focused pyright check passes for the touched batch and unwrap modules:
  - `src/spine/data/batch/base.py`
  - `src/spine/data/batch/tensor.py`
  - `src/spine/data/batch/index.py`
  - `src/spine/data/batch/edge_index.py`
  - `src/spine/io/unwrap.py`

## Tests and Coverage

- Expanded batch tests for:
  - `IndexBatch` spans and offsets.
  - `EdgeIndexBatch` spans and offsets.
  - sparse-like tensor validation.
  - ME sparse tensor compatibility.
  - internal narrowing guard paths used for static analysis.
- Moved unwrap tests under `test/test_io/test_unwrap.py`.
- Reworked unwrap tests to avoid global geometry singleton leakage from other IO tests.
- Added unwrap coverage for:
  - tensor batch lists.
  - single-volume and multi-volume tensors.
  - multi-volume index and edge-index offsets.
  - exported `<name>_spans` metadata.
  - tensor-like span conversion.
  - error paths for missing geometry/metadata and unsupported data types.

Validated locally with:

- `.venv/bin/pyright --pythonpath .venv/bin/python src/spine/data/batch/index.py src/spine/data/batch/tensor.py src/spine/data/batch/edge_index.py src/spine/data/batch/base.py src/spine/io/unwrap.py`
- `.venv/bin/pytest test/test_io/test_unwrap.py -q`
- `.venv/bin/pytest test/test_io -q --tb=short`
- `bash check_coverage.sh test/test_io/test_unwrap.py spine.io.unwrap`
- `bash check_coverage.sh test/test_data/test_batch spine.data`

Relevant results:

- `test/test_io/test_unwrap.py`: 18 passed.
- `test/test_io`: 356 passed, 237 skipped.
- `src/spine/io/unwrap.py`: 100% coverage.
- `src/spine/data/batch/base.py`, `edge_index.py`, `index.py`, and `tensor.py`: 100% coverage under the batch coverage target.

## Compatibility and Risk

- This is a structural refactor of index handling, so the main compatibility risk is external code that imports `Unwrapper` from `spine.utils.unwrap` or assumes parser index products are generic tensor payloads.
- The HDF5 index parser input contract is stricter because count events are now required.
- Runtime behavior for standard tensor parsing/collation should be unchanged.
- ME sparse tensor support was explicitly checked after the typing refactor to avoid rejecting valid `MinkowskiEngine.SparseTensor` objects.

## Follow-Up

- Update any downstream imports still using `spine.utils.unwrap`.
- Confirm that older cached HDF5 samples include the count datasets needed by the new index parser contract.
- Consider documenting the parser payload distinction in the IO README or parser developer notes.
