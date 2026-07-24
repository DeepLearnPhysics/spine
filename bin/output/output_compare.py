#!/usr/bin/env python3
"""Compare the event content of two SPINE HDF5 output files.

The comparison is semantic rather than bytewise. Legacy region references and
version-2 integer offsets are normalized, structured data products are compared
field by field, and HDF5 layout details such as dataset order, object addresses,
and variable-field pools are ignored.

Integer, boolean, and string values must always agree exactly. Floating-point
values are compared with configurable absolute and relative tolerances, unless
``--exact`` is requested.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml


@dataclass
class ComparisonResult:
    """Accumulate comparison statistics and human-readable differences.

    Attributes
    ----------
    num_events : int
        Number of events compared.
    num_products : int
        Number of event-level data products compared.
    num_values : int
        Number of scalar leaf values compared.
    num_mismatches : int
        Total number of comparison failures. A failure may describe a shape,
        schema, or one-or-more-value mismatch.
    max_abs_diff : float
        Largest absolute floating-point difference observed.
    max_rel_diff : float
        Largest relative floating-point difference observed. This is infinite
        when a nonzero candidate value is compared with a zero reference.
    differences : list[str]
        First ``max_differences`` human-readable mismatch descriptions.
    max_differences : int
        Maximum number of descriptions retained while counting all failures.
    """

    num_events: int = 0
    num_products: int = 0
    num_values: int = 0
    num_mismatches: int = 0
    max_abs_diff: float = 0.0
    max_rel_diff: float = 0.0
    differences: list[str] = field(default_factory=list)
    max_differences: int = 20

    @property
    def agrees(self) -> bool:
        """Return whether no comparison failures were found."""
        return self.num_mismatches == 0

    def add_difference(self, path: str, message: str) -> None:
        """Record one mismatch while bounding retained report text.

        Parameters
        ----------
        path : str
            Event/product path at which the mismatch occurred.
        message : str
            Concise description of the mismatch.
        """
        self.num_mismatches += 1
        if len(self.differences) < self.max_differences:
            self.differences.append(f"{path}: {message}")


def get_format_version(in_file: h5py.File) -> int:
    """Return the declared SPINE HDF5 layout version."""
    version = 1
    if "info" in in_file:
        version = int(in_file["info"].attrs.get("format_version", 1))
    if version not in (1, 2):
        raise ValueError(f"Unsupported HDF5 format version {version}.")
    return version


def get_product_keys(in_file: h5py.File, format_version: int) -> set[str]:
    """Return the event data products exposed by one SPINE HDF5 file."""
    events = in_file["events"]
    assert isinstance(events, h5py.Dataset)
    if format_version == 1:
        return set(events.dtype.names or ())
    return set(in_file.keys()) - {"events", "info"}


def _decode_attribute(value: Any) -> Any:
    """Decode byte-valued HDF5 attributes."""
    return value.decode() if isinstance(value, bytes) else value


def _require_dataset(parent: h5py.File | h5py.Group, name: str) -> h5py.Dataset:
    """Return a named child dataset or fail with a schema error."""
    child = parent[name]
    if not isinstance(child, h5py.Dataset):
        raise TypeError(f"Expected '{child.name}' to be an HDF5 dataset.")
    return child


def _require_group(parent: h5py.File | h5py.Group, name: str) -> h5py.Group:
    """Return a named child group or fail with a schema error."""
    child = parent[name]
    if not isinstance(child, h5py.Group):
        raise TypeError(f"Expected '{child.name}' to be an HDF5 group.")
    return child


def load_objects_v2(group: h5py.Group, event_index: int) -> Any:
    """Load V2 object rows into their logical V1 structured representation."""
    fixed = group["fixed"]
    event_offsets = group["event_offsets"]
    variables = group["variables"]
    assert isinstance(fixed, h5py.Dataset)
    assert isinstance(event_offsets, h5py.Dataset)
    assert isinstance(variables, h5py.Group)

    first, last = (int(value) for value in event_offsets[event_index : event_index + 2])
    rows = fixed[first:last]
    fixed_names = [
        name for name in rows.dtype.names or () if not name.startswith("_var_offsets_")
    ]
    dtype_specs: list[tuple[str, Any]] = [
        (name, rows.dtype.fields[name][0]) for name in fixed_names
    ]
    pools = sorted(variables.items(), key=lambda item: int(item[0].split("_")[-1]))
    for _, pool in pools:
        fields = yaml.safe_load(pool.attrs["fields"])
        kind = _decode_attribute(pool.attrs["kind"])
        values = pool["values"]
        dtype = (
            h5py.string_dtype() if kind == "string" else h5py.vlen_dtype(values.dtype)
        )
        dtype_specs.extend((name, dtype) for name in fields)

    result = np.empty(len(rows), dtype=np.dtype(dtype_specs))
    for name in fixed_names:
        result[name] = rows[name]

    for pool_name, pool in pools:
        fields = yaml.safe_load(pool.attrs["fields"])
        kind = _decode_attribute(pool.attrs["kind"])
        values = pool["values"]
        pool_index = int(pool_name.split("_")[-1])
        bounds = rows[f"_var_offsets_{pool_index}"]
        base = int(bounds[0, 0]) if len(bounds) else 0
        terminal = int(bounds[-1, -1]) if len(bounds) else base
        event_values = values[base:terminal]
        for field_index, name in enumerate(fields):
            for row_index in range(len(rows)):
                start = int(bounds[row_index, field_index]) - base
                stop = int(bounds[row_index, field_index + 1]) - base
                value = event_values[start:stop]
                if kind == "string":
                    value = value.tobytes().decode("utf-8")
                result[name][row_index] = value

    if bool(group.attrs["scalar"]):
        return result[0]
    return result


def load_event_value(
    in_file: h5py.File,
    event_index: int,
    key: str,
    format_version: int,
) -> Any:
    """Load one event-level value from a SPINE HDF5 file.

    This mirrors the dereferencing performed by
    :class:`spine.io.read.HDF5Reader`, but deliberately leaves serialized
    objects as NumPy structured arrays. Avoiding class reconstruction makes
    the utility usable across SPINE versions and exposes precise field paths
    in mismatch reports.

    Parameters
    ----------
    in_file : h5py.File
        Open SPINE HDF5 file.
    event_index : int
        Event row index.
    key : str
        Event data-product name.
    format_version : int
        Physical SPINE HDF5 layout version.

    Returns
    -------
    Any
        Dereferenced event value.
    """
    if format_version == 2:
        group = in_file[key]
        if not isinstance(group, h5py.Group) or "kind" not in group.attrs:
            raise ValueError(f"V2 product '{key}' is not a recognized product group.")

        kind = _decode_attribute(group.attrs["kind"])
        if kind in {"array", "string"}:
            values = _require_dataset(group, "values")
            offsets = _require_dataset(group, "event_offsets")
            start, stop = (
                int(value) for value in offsets[event_index : event_index + 2]
            )
            value = values[start:stop]
            if kind == "string":
                return value.tobytes().decode("utf-8")
            if bool(group.attrs["scalar"]):
                value = value[0]
            return value

        if kind == "objects":
            return load_objects_v2(group, event_index)

        if kind == "list":
            values = _require_dataset(group, "values")
            element_offsets = _require_dataset(group, "element_offsets")
            event_offsets = _require_dataset(group, "event_offsets")
            first, last = (
                int(value) for value in event_offsets[event_index : event_index + 2]
            )
            bounds = element_offsets[first : last + 1]
            result = np.empty(last - first, dtype=object)
            base = int(bounds[0]) if len(bounds) else 0
            terminal = int(bounds[-1]) if len(bounds) else base
            event_values = values[base:terminal]
            for index in range(last - first):
                start = int(bounds[index]) - base
                stop = int(bounds[index + 1]) - base
                result[index] = event_values[start:stop]
            return result

        if kind == "multi_list":
            result = []
            elements = sorted(
                group.items(), key=lambda item: int(item[0].split("_")[-1])
            )
            for name, _ in elements:
                element = _require_group(group, name)
                values = _require_dataset(element, "values")
                offsets = _require_dataset(element, "event_offsets")
                start, stop = (
                    int(value) for value in offsets[event_index : event_index + 2]
                )
                result.append(values[start:stop])
            return result

        raise ValueError(f"Unrecognized V2 product kind '{kind}' for key '{key}'.")

    events = in_file["events"]
    assert isinstance(events, h5py.Dataset)
    event = events[event_index]
    region_ref = event[key]
    dataset = in_file[key]

    if isinstance(dataset, h5py.Dataset):
        value = dataset[region_ref]
        if bool(dataset.attrs.get("scalar", False)):
            value = value[0]
        elif dataset.ndim > 1:
            value = value.reshape(-1, dataset.shape[1])
        return value

    if isinstance(dataset, h5py.Group):
        index = _require_dataset(dataset, "index")
        element_refs = index[region_ref].flatten()
        if index.ndim == 1:
            elements = _require_dataset(dataset, "elements")
            values = np.empty(len(element_refs), dtype=object)
            for idx, reference in enumerate(element_refs):
                value = elements[reference]
                if elements.ndim > 1:
                    value = value.reshape(-1, elements.shape[1])
                values[idx] = value
            return values

        values = []
        for idx, reference in enumerate(element_refs):
            elements = _require_dataset(dataset, f"element_{idx}")
            value = elements[reference]
            if elements.ndim > 1:
                value = value.reshape(-1, elements.shape[1])
            values.append(value)
        return values

    raise TypeError(f"Data product '{key}' is neither a dataset nor a group.")


def _format_index(index: tuple[int, ...]) -> str:
    """Format a NumPy index for inclusion in a comparison path."""
    return "".join(f"[{value}]" for value in index)


def _compare_float_arrays(
    reference: np.ndarray,
    candidate: np.ndarray,
    path: str,
    result: ComparisonResult,
    rtol: float,
    atol: float,
    exact: bool,
) -> None:
    """Compare floating-point arrays and update numeric diagnostics."""
    result.num_values += reference.size
    if not reference.size:
        return

    finite_pairs = np.isfinite(reference) & np.isfinite(candidate)
    if np.any(finite_pairs):
        abs_diff = np.abs(candidate[finite_pairs] - reference[finite_pairs])
        result.max_abs_diff = max(result.max_abs_diff, float(np.max(abs_diff)))
        denominator = np.abs(reference[finite_pairs])
        rel_diff = np.divide(
            abs_diff,
            denominator,
            out=np.full(abs_diff.shape, np.inf, dtype=np.float64),
            where=denominator != 0,
        )
        zero_equal = (denominator == 0) & (abs_diff == 0)
        rel_diff[zero_equal] = 0.0
        result.max_rel_diff = max(result.max_rel_diff, float(np.max(rel_diff)))

    if exact:
        matches = (reference == candidate) | (np.isnan(reference) & np.isnan(candidate))
    else:
        matches = np.isclose(reference, candidate, rtol=rtol, atol=atol, equal_nan=True)

    if np.all(matches):
        return

    mismatch_count = int(np.count_nonzero(~matches))
    mismatch_index = tuple(np.argwhere(~matches)[0])
    indexed_path = f"{path}{_format_index(mismatch_index)}"
    ref_value = reference[mismatch_index]
    candidate_value = candidate[mismatch_index]
    result.add_difference(
        indexed_path,
        (
            f"floating values differ ({ref_value!r} != {candidate_value!r}); "
            f"{mismatch_count}/{reference.size} values outside tolerance"
        ),
    )


def compare_values(
    reference: Any,
    candidate: Any,
    path: str,
    result: ComparisonResult,
    rtol: float,
    atol: float,
    exact: bool,
) -> None:
    """Recursively compare two dereferenced HDF5 values.

    Parameters
    ----------
    reference, candidate : Any
        Values loaded from the reference and candidate files.
    path : str
        Human-readable path to the values.
    result : ComparisonResult
        Mutable accumulator for statistics and differences.
    rtol, atol : float
        Relative and absolute floating-point tolerances.
    exact : bool
        If `True`, require bit-for-bit floating-point equality, with NaNs at
        matching positions treated as equal.
    """
    reference_sequence = isinstance(reference, (list, tuple))
    candidate_sequence = isinstance(candidate, (list, tuple))
    if reference_sequence or candidate_sequence:
        if not reference_sequence or not candidate_sequence:
            result.add_difference(
                path,
                (
                    "container type differs "
                    f"({type(reference).__name__} != {type(candidate).__name__})"
                ),
            )
            return
        if len(reference) != len(candidate):
            result.add_difference(
                path, f"length differs ({len(reference)} != {len(candidate)})"
            )
            return
        for index, (ref_value, candidate_value) in enumerate(zip(reference, candidate)):
            compare_values(
                ref_value,
                candidate_value,
                f"{path}[{index}]",
                result,
                rtol,
                atol,
                exact,
            )
        return

    if isinstance(reference, (bytes, np.bytes_)):
        reference = bytes(reference).decode("utf-8")
    if isinstance(candidate, (bytes, np.bytes_)):
        candidate = bytes(candidate).decode("utf-8")

    reference = np.asarray(reference)
    candidate = np.asarray(candidate)

    if reference.shape != candidate.shape:
        result.add_difference(
            path, f"shape differs ({reference.shape} != {candidate.shape})"
        )
        return

    ref_fields = reference.dtype.names
    candidate_fields = candidate.dtype.names
    if ref_fields is not None or candidate_fields is not None:
        if set(ref_fields or ()) != set(candidate_fields or ()):
            result.add_difference(
                path, f"structured fields differ ({ref_fields} != {candidate_fields})"
            )
            return
        for name in ref_fields or ():
            compare_values(
                reference[name],
                candidate[name],
                f"{path}.{name}",
                result,
                rtol,
                atol,
                exact,
            )
        return

    if reference.dtype != candidate.dtype:
        result.add_difference(
            path, f"dtype differs ({reference.dtype} != {candidate.dtype})"
        )
        return

    if reference.dtype.kind == "O":
        for index in np.ndindex(reference.shape):
            compare_values(
                reference[index],
                candidate[index],
                f"{path}{_format_index(index)}",
                result,
                rtol,
                atol,
                exact,
            )
        return

    if reference.dtype.kind in "fc" and candidate.dtype.kind in "fc":
        _compare_float_arrays(reference, candidate, path, result, rtol, atol, exact)
        return

    result.num_values += reference.size
    matches = reference == candidate
    if np.all(matches):
        return

    mismatch_count = int(np.count_nonzero(~matches))
    mismatch_index = tuple(np.argwhere(~matches)[0])
    indexed_path = f"{path}{_format_index(mismatch_index)}"
    result.add_difference(
        indexed_path,
        (
            f"values differ ({reference[mismatch_index]!r} != "
            f"{candidate[mismatch_index]!r}); "
            f"{mismatch_count}/{reference.size} values differ"
        ),
    )


def compare_files(
    reference_path: str | Path,
    candidate_path: str | Path,
    *,
    rtol: float = 1.0e-4,
    atol: float = 1.0e-6,
    exact: bool = False,
    keys: list[str] | None = None,
    skip_keys: list[str] | None = None,
    max_differences: int = 20,
) -> ComparisonResult:
    """Compare the event content of two SPINE HDF5 output files.

    Parameters
    ----------
    reference_path, candidate_path : str or pathlib.Path
        Paths to the reference and candidate SPINE HDF5 files.
    rtol : float, default 1e-4
        Relative tolerance for floating-point values.
    atol : float, default 1e-6
        Absolute tolerance for floating-point values near zero.
    exact : bool, default False
        Require exact floating-point equality. The tolerance arguments are
        ignored when this option is enabled.
    keys : list[str], optional
        Restrict comparison to these event data products.
    skip_keys : list[str], optional
        Event data products to omit from the comparison.
    max_differences : int, default 20
        Maximum number of mismatch descriptions retained in the result.

    Returns
    -------
    ComparisonResult
        Comparison statistics and mismatch descriptions.
    """
    if rtol < 0 or atol < 0:
        raise ValueError("Floating-point tolerances must be nonnegative.")
    if max_differences < 0:
        raise ValueError("max_differences must be nonnegative.")

    result = ComparisonResult(max_differences=max_differences)
    with (
        h5py.File(reference_path, "r") as reference_file,
        h5py.File(candidate_path, "r") as candidate_file,
    ):
        for label, in_file in (
            ("reference", reference_file),
            ("candidate", candidate_file),
        ):
            if "events" not in in_file:
                result.add_difference(label, "file has no 'events' dataset")
                return result
            if not isinstance(in_file["events"], h5py.Dataset):
                result.add_difference(label, "'events' is not a dataset")
                return result

        reference_events = _require_dataset(reference_file, "events")
        candidate_events = _require_dataset(candidate_file, "events")
        if len(reference_events) != len(candidate_events):
            result.add_difference(
                "events",
                f"count differs ({len(reference_events)} != {len(candidate_events)})",
            )
            return result

        try:
            reference_version = get_format_version(reference_file)
            candidate_version = get_format_version(candidate_file)
        except ValueError as error:
            result.add_difference("format", str(error))
            return result

        reference_keys = get_product_keys(reference_file, reference_version)
        candidate_keys = get_product_keys(candidate_file, candidate_version)
        requested_keys = set(keys) if keys is not None else reference_keys
        skipped_keys = set(skip_keys or ())
        selected_keys = requested_keys - skipped_keys

        missing_reference = selected_keys - reference_keys
        missing_candidate = selected_keys - candidate_keys
        if missing_reference:
            result.add_difference(
                "events",
                f"reference is missing products {sorted(missing_reference)}",
            )
        if missing_candidate:
            result.add_difference(
                "events",
                f"candidate is missing products {sorted(missing_candidate)}",
            )
        if keys is None:
            extra_candidate = (candidate_keys - reference_keys) - skipped_keys
            if extra_candidate:
                result.add_difference(
                    "events",
                    f"candidate has extra products {sorted(extra_candidate)}",
                )

        common_keys = sorted(selected_keys & reference_keys & candidate_keys)
        result.num_events = len(reference_events)
        result.num_products = len(reference_events) * len(common_keys)
        for event_idx in range(len(reference_events)):
            for key in common_keys:
                path = f"event[{event_idx}].{key}"
                try:
                    reference = load_event_value(
                        reference_file, event_idx, key, reference_version
                    )
                    candidate = load_event_value(
                        candidate_file, event_idx, key, candidate_version
                    )
                    compare_values(
                        reference,
                        candidate,
                        path,
                        result,
                        rtol,
                        atol,
                        exact,
                    )
                except (KeyError, TypeError, ValueError) as error:
                    result.add_difference(path, str(error))

    return result


def format_result(
    result: ComparisonResult,
    reference_path: str | Path,
    candidate_path: str | Path,
    *,
    rtol: float,
    atol: float,
    exact: bool,
) -> str:
    """Format a complete command-line report for a comparison result."""
    mode = "exact" if exact else f"rtol={rtol:g}, atol={atol:g}"
    status = "PASS" if result.agrees else "FAIL"
    lines = [
        f"{status}: {reference_path} vs {candidate_path}",
        f"Mode: {mode}",
        (
            f"Compared: {result.num_events} events, "
            f"{result.num_products} event products, "
            f"{result.num_values} scalar values"
        ),
        (
            f"Maximum floating difference: abs={result.max_abs_diff:.8g}, "
            f"rel={result.max_rel_diff:.8g}"
        ),
    ]
    if not result.agrees:
        lines.append(f"Mismatches: {result.num_mismatches}")
        lines.extend(f"- {difference}" for difference in result.differences)
        hidden = result.num_mismatches - len(result.differences)
        if hidden > 0:
            lines.append(f"- ... {hidden} additional mismatches not shown")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Compare the event content of two SPINE HDF5 output files."
    )
    parser.add_argument("reference", help="Reference SPINE HDF5 file.")
    parser.add_argument("candidate", help="Candidate SPINE HDF5 file.")
    parser.add_argument(
        "--rtol",
        type=float,
        default=1.0e-4,
        help="Relative floating-point tolerance (default: %(default)g).",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1.0e-6,
        help="Absolute floating-point tolerance (default: %(default)g).",
    )
    parser.add_argument(
        "--exact",
        action="store_true",
        help="Require exact floating-point agreement; ignore tolerances.",
    )
    parser.add_argument(
        "--keys",
        nargs="+",
        help="Compare only the listed event data products.",
    )
    parser.add_argument(
        "--skip-keys",
        nargs="+",
        help="Omit the listed event data products.",
    )
    parser.add_argument(
        "--max-differences",
        type=int,
        default=20,
        help="Maximum mismatch descriptions to print (default: %(default)d).",
    )
    return parser


def main() -> int:
    """Run the command-line HDF5 comparison."""
    parser = build_parser()
    args = parser.parse_args()
    try:
        result = compare_files(
            args.reference,
            args.candidate,
            rtol=args.rtol,
            atol=args.atol,
            exact=args.exact,
            keys=args.keys,
            skip_keys=args.skip_keys,
            max_differences=args.max_differences,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))

    print(
        format_result(
            result,
            args.reference,
            args.candidate,
            rtol=args.rtol,
            atol=args.atol,
            exact=args.exact,
        )
    )
    return 0 if result.agrees else 1


if __name__ == "__main__":
    raise SystemExit(main())
