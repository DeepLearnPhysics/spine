#!/usr/bin/env python3
"""Compare the event content of two SPINE HDF5 output files.

The comparison is semantic rather than bytewise. Event region references are
dereferenced, structured data products are compared field by field, and HDF5
layout details such as dataset order and object addresses are ignored.

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


def load_event_value(in_file: h5py.File, event: np.void, key: str) -> Any:
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
    event : numpy.void
        Structured row from the file's ``events`` dataset.
    key : str
        Event data-product name.

    Returns
    -------
    Any
        Dereferenced event value.
    """
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
        index = dataset["index"]
        element_refs = index[region_ref].flatten()
        if index.ndim == 1:
            elements = dataset["elements"]
            values = np.empty(len(element_refs), dtype=object)
            for idx, reference in enumerate(element_refs):
                value = elements[reference]
                if elements.ndim > 1:
                    value = value.reshape(-1, elements.shape[1])
                values[idx] = value
            return values

        values = []
        for idx, reference in enumerate(element_refs):
            elements = dataset[f"element_{idx}"]
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
        if ref_fields != candidate_fields:
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

        reference_events = reference_file["events"]
        candidate_events = candidate_file["events"]
        if len(reference_events) != len(candidate_events):
            result.add_difference(
                "events",
                f"count differs ({len(reference_events)} != {len(candidate_events)})",
            )
            return result

        reference_keys = set(reference_events.dtype.names or ())
        candidate_keys = set(candidate_events.dtype.names or ())
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
        for event_idx, (reference_event, candidate_event) in enumerate(
            zip(reference_events, candidate_events)
        ):
            for key in common_keys:
                path = f"event[{event_idx}].{key}"
                try:
                    reference = load_event_value(reference_file, reference_event, key)
                    candidate = load_event_value(candidate_file, candidate_event, key)
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
