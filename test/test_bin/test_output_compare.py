"""Tests for the SPINE HDF5 output comparison utility."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import h5py
import numpy as np


def load_output_compare_module():
    """Import ``bin/output/output_compare.py`` as a test module."""
    script_path = (
        Path(__file__).resolve().parents[2] / "bin" / "output" / "output_compare.py"
    )
    spec = importlib.util.spec_from_file_location("output_compare", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_output_file(
    path: Path,
    values: list[np.ndarray],
    *,
    labels: list[np.ndarray] | None = None,
) -> None:
    """Create a minimal multi-event SPINE-style HDF5 output."""
    ref_dtype = h5py.special_dtype(ref=h5py.RegionReference)
    event_dtype = [("values", ref_dtype)]
    if labels is not None:
        event_dtype.append(("labels", ref_dtype))

    with h5py.File(path, "w") as out_file:
        events = out_file.create_dataset(
            "events", (len(values),), dtype=np.dtype(event_dtype)
        )
        flat_values = np.concatenate(values)
        value_ds = out_file.create_dataset("values", data=flat_values, maxshape=(None,))
        value_ds.attrs["scalar"] = False

        labels_ds = None
        if labels is not None:
            flat_labels = np.concatenate(labels)
            labels_ds = out_file.create_dataset(
                "labels", data=flat_labels, maxshape=(None,)
            )
            labels_ds.attrs["scalar"] = False

        value_offset = 0
        label_offset = 0
        for idx, event_values in enumerate(values):
            event = np.zeros(1, dtype=np.dtype(event_dtype))
            value_end = value_offset + len(event_values)
            event["values"][0] = value_ds.regionref[value_offset:value_end]
            value_offset = value_end
            if labels is not None and labels_ds is not None:
                label_end = label_offset + len(labels[idx])
                event["labels"][0] = labels_ds.regionref[label_offset:label_end]
                label_offset = label_end
            events[idx] = event[0]


def test_compare_files_exact_match(tmp_path):
    """Identical semantic content should pass exact comparison."""
    module = load_output_compare_module()
    reference = tmp_path / "reference.h5"
    candidate = tmp_path / "candidate.h5"
    values = [
        np.asarray([1.0, np.nan], dtype=np.float32),
        np.asarray([], dtype=np.float32),
    ]
    make_output_file(reference, values)
    make_output_file(candidate, values)

    result = module.compare_files(reference, candidate, exact=True)

    assert result.agrees
    assert result.num_events == 2
    assert result.num_products == 2
    assert result.num_values == 2


def test_compare_files_float_tolerance_and_exact(tmp_path):
    """Tolerant comparison should accept noise that exact mode rejects."""
    module = load_output_compare_module()
    reference = tmp_path / "reference.h5"
    candidate = tmp_path / "candidate.h5"
    make_output_file(reference, [np.asarray([1.0, 0.0], dtype=np.float32)])
    make_output_file(candidate, [np.asarray([1.0 + 5.0e-5, 5.0e-7], dtype=np.float32)])

    tolerant = module.compare_files(reference, candidate, rtol=1.0e-4, atol=1.0e-6)
    exact = module.compare_files(reference, candidate, exact=True)

    assert tolerant.agrees
    assert not exact.agrees
    assert exact.num_mismatches == 1
    assert "event[0].values[0]" in exact.differences[0]


def test_compare_files_exact_non_float_and_schema(tmp_path):
    """Non-floating data and event-product schemas must agree exactly."""
    module = load_output_compare_module()
    reference = tmp_path / "reference.h5"
    candidate = tmp_path / "candidate.h5"
    make_output_file(
        reference,
        [np.asarray([1.0], dtype=np.float32)],
        labels=[np.asarray([1, 2], dtype=np.int64)],
    )
    make_output_file(
        candidate,
        [np.asarray([1.0], dtype=np.float32)],
        labels=[np.asarray([1, 3], dtype=np.int64)],
    )

    result = module.compare_files(reference, candidate)

    assert not result.agrees
    assert result.num_mismatches == 1
    assert "event[0].labels[1]" in result.differences[0]


def test_compare_files_shape_and_event_count(tmp_path):
    """Shape and event-count differences should produce clear failures."""
    module = load_output_compare_module()
    reference = tmp_path / "reference.h5"
    shape_candidate = tmp_path / "shape.h5"
    count_candidate = tmp_path / "count.h5"
    make_output_file(reference, [np.asarray([1.0, 2.0], dtype=np.float32)])
    make_output_file(shape_candidate, [np.asarray([1.0], dtype=np.float32)])
    make_output_file(
        count_candidate,
        [
            np.asarray([1.0, 2.0], dtype=np.float32),
            np.asarray([3.0], dtype=np.float32),
        ],
    )

    shape_result = module.compare_files(reference, shape_candidate)
    count_result = module.compare_files(reference, count_candidate)

    assert "shape differs" in shape_result.differences[0]
    assert "count differs" in count_result.differences[0]


def test_compare_files_key_selection(tmp_path):
    """Key inclusion and exclusion should limit compared products."""
    module = load_output_compare_module()
    reference = tmp_path / "reference.h5"
    candidate = tmp_path / "candidate.h5"
    make_output_file(
        reference,
        [np.asarray([1.0], dtype=np.float32)],
        labels=[np.asarray([1], dtype=np.int64)],
    )
    make_output_file(
        candidate,
        [np.asarray([2.0], dtype=np.float32)],
        labels=[np.asarray([1], dtype=np.int64)],
    )

    included = module.compare_files(reference, candidate, keys=["labels"])
    skipped = module.compare_files(reference, candidate, skip_keys=["values"])
    missing = module.compare_files(reference, candidate, keys=["unknown"])

    assert included.agrees
    assert skipped.agrees
    assert not missing.agrees
    assert "missing products" in missing.differences[0]


def test_compare_values_structured_and_object_fields():
    """Structured variable-length object fields should compare recursively."""
    module = load_output_compare_module()
    dtype = np.dtype(
        [("id", np.int64), ("scores", h5py.vlen_dtype(np.dtype("float32")))]
    )
    reference = np.empty(1, dtype=dtype)
    candidate = np.empty(1, dtype=dtype)
    reference["id"] = 4
    candidate["id"] = 4
    reference["scores"][0] = np.asarray([0.1, 0.2], dtype=np.float32)
    candidate["scores"][0] = np.asarray([0.1, 0.3], dtype=np.float32)
    result = module.ComparisonResult()

    module.compare_values(
        reference,
        candidate,
        "event[0].objects",
        result,
        rtol=0.0,
        atol=0.0,
        exact=True,
    )

    assert not result.agrees
    assert "objects.scores[0][1]" in result.differences[0]


def test_compare_files_rejects_invalid_options(tmp_path):
    """Negative tolerances and report bounds should be rejected."""
    module = load_output_compare_module()
    path = tmp_path / "output.h5"
    make_output_file(path, [np.asarray([1.0], dtype=np.float32)])

    for kwargs in ({"rtol": -1.0}, {"atol": -1.0}, {"max_differences": -1}):
        try:
            module.compare_files(path, path, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected ValueError for {kwargs}")


def test_format_result():
    """The text report should expose status, mode, and bounded differences."""
    module = load_output_compare_module()
    result = module.ComparisonResult(max_differences=1)
    result.add_difference("event[0].x", "first")
    result.add_difference("event[0].y", "second")

    report = module.format_result(
        result, "reference.h5", "candidate.h5", rtol=1.0e-4, atol=1.0e-6, exact=False
    )

    assert report.startswith("FAIL:")
    assert "rtol=0.0001, atol=1e-06" in report
    assert "1 additional mismatches not shown" in report
