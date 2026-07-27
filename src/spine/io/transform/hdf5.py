"""Structural transformations for offset-based SPINE HDF5 files."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Sequence
from typing import Any

import h5py
import numpy as np
import yaml

import spine.data

__all__ = ["DEFAULT_LITE_KEYS", "litify_hdf5"]


DEFAULT_LITE_KEYS = (
    "run_info",
    "meta",
    "reco_particles",
    "truth_particles",
    "reco_interactions",
    "truth_interactions",
)
"""Products retained by the standard production lite-output workflow."""

_ADMINISTRATIVE_KEYS = (
    "index",
    "source_file_index",
    "source_file_entry_index",
)


def _copy_attrs(source: h5py.AttributeManager, target: h5py.AttributeManager) -> None:
    """Copy HDF5 attributes without interpreting their values."""
    for key, value in source.items():
        target[key] = value


def _require_v2(in_file: h5py.File, path: str) -> None:
    """Validate that an input file uses the structural V2 layout."""
    if "events" not in in_file or "info" not in in_file:
        raise ValueError(f"'{path}' is not a complete SPINE HDF5 output.")
    version = int(in_file["info"].attrs.get("format_version", 1))
    if version != 2:
        raise ValueError(
            "Structural litification requires HDF5 format version 2; "
            f"'{path}' uses version {version}."
        )


def _lite_skip_fields(class_name: str) -> set[str]:
    """Return fields omitted by a data class's established lite policy."""
    cls = getattr(spine.data, class_name, None)
    if cls is None or not hasattr(cls, "attr_names"):
        raise ValueError(
            f"Cannot resolve stored object class '{class_name}' to determine "
            "its lite attribute policy."
        )
    full = set(
        cls.attr_names(
            include_derived=True,
            include_skipped=False,
            lite=False,
        )
    )
    lite = set(
        cls.attr_names(
            include_derived=True,
            include_skipped=False,
            lite=True,
        )
    )
    return full.difference(lite)


def _decode_fields(pool: h5py.Group) -> tuple[str, ...]:
    """Decode and validate a V2 variable pool's ordered field names."""
    value = pool.attrs.get("fields")
    if isinstance(value, bytes):
        value = value.decode()
    if not isinstance(value, str):
        raise TypeError(f"Variable pool '{pool.name}' has no string field metadata.")
    fields = yaml.safe_load(value)
    if not isinstance(fields, list) or not all(
        isinstance(field, str) for field in fields
    ):
        raise TypeError(
            f"Variable pool '{pool.name}' fields must decode to a string list."
        )
    return tuple(fields)


def _dataset_kwargs(dataset: h5py.Dataset) -> dict[str, Any]:
    """Return reusable physical creation options from a source dataset."""
    kwargs: dict[str, Any] = {}
    if dataset.chunks is not None:
        kwargs["chunks"] = dataset.chunks
    if dataset.compression is not None:
        kwargs["compression"] = dataset.compression
        kwargs["compression_opts"] = dataset.compression_opts
    if dataset.shuffle:
        kwargs["shuffle"] = True
    if dataset.fletcher32:
        kwargs["fletcher32"] = True
    if dataset.scaleoffset is not None:
        kwargs["scaleoffset"] = dataset.scaleoffset
    return kwargs


def _copy_object_group(
    source: h5py.Group,
    target: h5py.Group,
    *,
    mode: str,
    block_size: int,
) -> None:
    """Copy one object product while omitting selected variable attributes."""
    _copy_attrs(source.attrs, target.attrs)
    class_name = source.attrs.get("class_name")
    if isinstance(class_name, bytes):
        class_name = class_name.decode()
    if not isinstance(class_name, str):
        raise TypeError(f"Object product '{source.name}' has no class name.")

    source_fixed = source["fixed"]
    source_offsets = source["event_offsets"]
    source_variables = source["variables"]
    if not isinstance(source_fixed, h5py.Dataset):
        raise TypeError(f"Object product '{source.name}' has no fixed dataset.")
    if not isinstance(source_offsets, h5py.Dataset):
        raise TypeError(f"Object product '{source.name}' has no event offsets.")
    if not isinstance(source_variables, h5py.Group):
        raise TypeError(f"Object product '{source.name}' has no variable group.")

    drop_fields = (
        set().union(*(_decode_fields(pool) for pool in source_variables.values()))
        if mode == "fixed_only" and len(source_variables)
        else _lite_skip_fields(class_name)
    )
    ordinary_names = tuple(
        name
        for name in source_fixed.dtype.names or ()
        if not name.startswith("_var_offsets_")
    )
    fixed_dtype: list[tuple[Any, ...]] = [
        (name, source_fixed.dtype.fields[name][0]) for name in ordinary_names
    ]

    target_variables = target.create_group("variables")
    pool_specs: list[
        tuple[h5py.Group, h5py.Group, str, str, tuple[int, ...], tuple[str, ...]]
    ] = []
    for source_pool_name, source_pool_obj in source_variables.items():
        if not isinstance(source_pool_obj, h5py.Group):
            raise TypeError(f"Variable pool '{source_pool_name}' is not a group.")
        source_fields = _decode_fields(source_pool_obj)
        keep_indices = tuple(
            i for i, field in enumerate(source_fields) if field not in drop_fields
        )
        if not keep_indices:
            continue

        target_pool_name = f"pool_{len(pool_specs)}"
        target_pool = target_variables.create_group(target_pool_name)
        _copy_attrs(source_pool_obj.attrs, target_pool.attrs)
        kept_fields = tuple(source_fields[i] for i in keep_indices)
        target_pool.attrs["fields"] = yaml.safe_dump(list(kept_fields))

        source_values = source_pool_obj["values"]
        if not isinstance(source_values, h5py.Dataset):
            raise TypeError(f"Variable pool '{source_pool_obj.name}' has no values.")
        target_values = target_pool.create_dataset(
            "values",
            (0,),
            maxshape=(None,),
            dtype=source_values.dtype,
            **_dataset_kwargs(source_values),
        )
        _copy_attrs(source_values.attrs, target_values.attrs)

        old_pool_index = int(source_pool_name.rsplit("_", 1)[-1])
        source_helper = f"_var_offsets_{old_pool_index}"
        target_helper = f"_var_offsets_{len(pool_specs)}"
        if source_helper not in (source_fixed.dtype.names or ()):
            raise KeyError(
                f"Object product '{source.name}' is missing helper "
                f"column '{source_helper}'."
            )
        fixed_dtype.append((target_helper, np.int64, len(kept_fields) + 1))
        pool_specs.append(
            (
                source_pool_obj,
                target_pool,
                source_helper,
                target_helper,
                keep_indices,
                kept_fields,
            )
        )

    target_fixed = target.create_dataset(
        "fixed",
        source_fixed.shape,
        maxshape=source_fixed.maxshape,
        dtype=np.dtype(fixed_dtype),
        **_dataset_kwargs(source_fixed),
    )
    _copy_attrs(source_fixed.attrs, target_fixed.attrs)
    source.copy(source_offsets, target, name="event_offsets")

    num_rows = len(source_fixed)
    for first in range(0, num_rows, block_size):
        last = min(first + block_size, num_rows)
        source_rows = source_fixed[first:last]
        target_rows: Any = np.empty(last - first, dtype=target_fixed.dtype)
        for name in ordinary_names:
            target_rows[name] = source_rows[name]

        for (
            source_pool,
            target_pool,
            source_helper,
            target_helper,
            keep_indices,
            _,
        ) in pool_specs:
            source_values = source_pool["values"]
            target_values = target_pool["values"]
            assert isinstance(source_values, h5py.Dataset)
            assert isinstance(target_values, h5py.Dataset)

            source_bounds = source_rows[source_helper]
            target_bounds = np.empty(
                (last - first, len(keep_indices) + 1), dtype=np.int64
            )
            cursor = len(target_values)
            chunks = []
            for row_idx, bounds in enumerate(source_bounds):
                target_bounds[row_idx, 0] = cursor
                for field_idx, source_field_idx in enumerate(keep_indices):
                    start = int(bounds[source_field_idx])
                    stop = int(bounds[source_field_idx + 1])
                    chunk = source_values[start:stop]
                    chunks.append(chunk)
                    cursor += len(chunk)
                    target_bounds[row_idx, field_idx + 1] = cursor

            combined = (
                np.concatenate(chunks)
                if chunks
                else np.empty(0, dtype=target_values.dtype)
            )
            old_size = len(target_values)
            target_values.resize(old_size + len(combined), axis=0)
            if len(combined):
                target_values[old_size:] = combined
            target_rows[target_helper] = target_bounds

        target_fixed[first:last] = target_rows


def litify_hdf5(
    source_path: str,
    target_path: str,
    *,
    keys: Sequence[str] = DEFAULT_LITE_KEYS,
    mode: str = "lite",
    overwrite: bool = False,
    block_size: int = 4096,
) -> None:
    """Create a structurally reduced copy of a V2 SPINE HDF5 output.

    Parameters
    ----------
    source_path : str
        Input SPINE HDF5 path.
    target_path : str
        Destination path. The input is never modified in place.
    keys : sequence[str], default DEFAULT_LITE_KEYS
        Physics products to retain. Administrative index/provenance products
        are retained automatically when available.
    mode : {'lite', 'fixed_only'}, default 'lite'
        ``lite`` reproduces the class-level ``lite=True`` storage policy.
        ``fixed_only`` removes every variable-length object attribute.
    overwrite : bool, default False
        Replace an existing destination atomically.
    block_size : int, default 4096
        Number of object rows processed per fixed-table block.
    """
    if mode not in ("lite", "fixed_only"):
        raise ValueError(f"Unsupported litification mode '{mode}'.")
    if block_size <= 0:
        raise ValueError("`block_size` must be positive.")

    source_path = os.path.abspath(os.path.expanduser(source_path))
    target_path = os.path.abspath(os.path.expanduser(target_path))
    if source_path == target_path:
        raise ValueError("Input and output paths must be different.")
    if not os.path.isfile(source_path):
        raise FileNotFoundError(f"Input HDF5 file not found: {source_path}")
    if os.path.exists(target_path) and not overwrite:
        raise FileExistsError(f"Output already exists: {target_path}")

    target_dir = os.path.dirname(target_path)
    os.makedirs(target_dir, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(target_path)}.",
        suffix=".tmp",
        dir=target_dir,
    )
    os.close(descriptor)

    try:
        with (
            h5py.File(source_path, "r") as source,
            h5py.File(temporary_path, "w") as target,
        ):
            _require_v2(source, source_path)
            _copy_attrs(source.attrs, target.attrs)
            source.copy("events", target)
            source.copy("info", target)
            info = target["info"]
            assert isinstance(info, h5py.Group)
            info.attrs["complete"] = False

            requested = tuple(dict.fromkeys(keys))
            missing = [key for key in requested if key not in source]
            if missing:
                raise KeyError(f"Requested products are missing: {missing}.")

            selected = list(requested)
            for key in _ADMINISTRATIVE_KEYS:
                if key in source and key not in selected:
                    selected.append(key)
            if "source" in source and "source" not in selected:
                selected.append("source")

            for key in selected:
                product = source[key]
                if (
                    isinstance(product, h5py.Group)
                    and product.attrs.get("kind") == "objects"
                ):
                    target_product = target.create_group(key)
                    _copy_object_group(
                        product,
                        target_product,
                        mode=mode,
                        block_size=block_size,
                    )
                else:
                    source.copy(key, target)

            info.attrs["litified"] = True
            info.attrs["litify_mode"] = mode
            info.attrs["litify_keys"] = yaml.safe_dump(list(requested))
            info.attrs["complete"] = True
            target.flush()

        os.chmod(temporary_path, os.stat(source_path).st_mode & 0o666)
        os.replace(temporary_path, target_path)
    except Exception:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
        raise
