"""Decoder for the self-describing HDF5 product-group storage layout."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import h5py
import numpy as np
import yaml

import spine.data

from .common import (
    contiguous_runs,
    decode_string_attribute,
    require_dataset,
    require_group,
    resolve_object_class,
)


class ProductGroupBackend:
    """Read products stored as self-describing groups and offset arrays.

    The class is an implementation mixin for :class:`HDF5Reader`. It owns the
    current product layout's decoding rules and declares the reader state it
    consumes. File selection, projection, and handle lifetime remain shared by
    the public reader.
    """

    build_classes: bool
    fixed_only: bool
    keep_open: bool
    _product_object_schemas: dict[
        tuple[str, str],
        tuple[
            str,
            bool,
            tuple[str, ...],
            tuple[tuple[str, int, bool, tuple[str, ...]], ...],
        ],
    ]
    _product_object_handles: dict[
        tuple[str, str],
        tuple[h5py.Dataset, h5py.Dataset, tuple[h5py.Dataset, ...]],
    ]
    _product_handles: dict[tuple[str, str], "_ProductHandles"]

    def _initialize_product_backend(self) -> None:
        """Initialize schema metadata and process-local handle caches."""
        self._product_object_schemas = {}
        self._product_object_handles = {}
        self._product_handles = {}

    def _clear_product_handles(self) -> None:
        """Drop product handles tied to closed or inherited HDF5 files."""
        self._product_object_handles = {}
        self._product_handles = {}

    def _load_product_columnar_objects(
        self,
        container: h5py.File | h5py.Group,
        key: str,
        entries: np.ndarray,
        requested_fields: tuple[str, ...] | None,
    ) -> dict[str, Any]:
        """Project fixed object columns and event boundaries.

        Parameters
        ----------
        container : h5py.File or h5py.Group
            Container holding the logical product.
        key : str
            Object-product group name.
        entries : numpy.ndarray
            File-local event indexes to project.
        requested_fields : tuple[str, ...], optional
            Fixed object fields to read. All fields are returned when omitted.

        Returns
        -------
        dict
            Projected columns and their chunk-local ``event_offsets``.
        """
        group = require_group(container, key)
        kind = decode_string_attribute(group.attrs["kind"], "kind")
        if kind != "objects":
            raise TypeError(
                f"Columnar product `{key}` must be an object collection, got `{kind}`."
            )

        fixed = require_dataset(group, "fixed")
        offsets = require_dataset(group, "event_offsets")
        available = tuple(
            name
            for name in fixed.dtype.names or ()
            if not name.startswith("_var_offsets_")
        )
        fields_to_read = available if requested_fields is None else requested_fields
        missing = set(fields_to_read).difference(available)
        if missing:
            raise KeyError(
                f"Columnar product `{key}` is missing fixed fields "
                f"{sorted(missing)}."
            )

        # Merge each contiguous disk span before exposing ordinary field arrays
        # to the analyzer-facing columnar interface.
        rows = []
        counts = []
        for first, last in contiguous_runs(entries):
            bounds = offsets[first : last + 1]
            start, stop = int(bounds[0]), int(bounds[-1])
            if fields_to_read:
                rows.append(fixed.fields(fields_to_read)[start:stop])
            counts.extend(np.diff(bounds).astype(np.int64, copy=False))

        result = {}
        if fields_to_read:
            combined = (
                np.concatenate(rows)
                if rows
                else np.empty(
                    0,
                    dtype=np.dtype(
                        [(name, fixed.dtype[name]) for name in fields_to_read]
                    ),
                )
            )
            result.update({name: combined[name] for name in fields_to_read})
        result["event_offsets"] = np.concatenate(
            ([0], np.cumsum(counts, dtype=np.int64))
        )
        return result

    def reconstruct_products(
        self,
        products: h5py.Group,
        entry_idx: int,
        data: dict[str, Any],
    ) -> None:
        """Rebuild typed event products from their stored metadata.

        Parameters
        ----------
        products : h5py.Group
            Logical-product root for the current file.
        entry_idx : int
            File-local event index used to load auxiliary values.
        data : dict
            Mutable event dictionary containing decoded primary payloads.

        Raises
        ------
        ValueError
            If metadata names an unsupported product type or an empty object
            list lacks class information.
        """
        # Only primary products already admitted by reader-level projection are
        # reconstructed. Auxiliary children remain private to their owner.
        for key in tuple(data):
            if key not in products:
                continue
            group = products[key]
            if not isinstance(group, h5py.Group):
                continue
            if "product_metadata" not in group.attrs:
                continue

            encoded = decode_string_attribute(
                group.attrs["product_metadata"], "product_metadata"
            )
            metadata = yaml.safe_load(encoded) or {}
            product_type = metadata.get("product_type")

            # Recombine primary arrays with product-owned tensor metadata.
            if product_type == "tensor":
                values = np.asarray(data[key])
                width = int(metadata.get("coordinate_width", 0))
                schema = spine.data.TensorSchema.from_dict(metadata.get("schema", {}))
                coords = values[:, :width] if width else None
                features = values[:, width:] if width else values
                meta = (
                    self._load_product_child(group, "meta", entry_idx)
                    if metadata.get("has_meta", False)
                    else None
                )
                data[key] = spine.data.TensorData(
                    features, coords, meta=meta, schema=schema
                )

            # Rebuild compact voxel labels and their optional particle table.
            elif product_type == "cluster_label":
                meta = (
                    self._load_product_child(group, "meta", entry_idx)
                    if metadata.get("has_meta", False)
                    else None
                )
                particles = None
                if metadata.get("has_particles", False):
                    particle_objects = self._load_product_child(
                        group, "particles", entry_idx
                    )
                    particle_fields = ()
                    if len(particle_objects):
                        particle_fields = tuple(particle_objects[0].as_dict())
                    elif hasattr(particle_objects, "default"):
                        particle_fields = tuple(particle_objects.default.as_dict())
                    particles = {
                        name: np.asarray(
                            [getattr(particle, name) for particle in particle_objects]
                        )
                        for name in particle_fields
                    }
                data[key] = spine.data.ClusterLabelData(
                    np.asarray(data[key]), particles=particles, meta=meta
                )

            # Restore index semantics from the event-level node span.
            elif product_type in {"index", "index_list", "edge_index"}:
                span = int(self._load_product_child(group, "spans", entry_idx))
                if product_type == "index":
                    data[key] = spine.data.IndexData(np.asarray(data[key]), span)
                elif product_type == "index_list":
                    values = [np.asarray(value) for value in data[key]]
                    data[key] = spine.data.IndexListData(values, span)
                else:
                    data[key] = spine.data.EdgeIndexData(
                        np.asarray(data[key]).T,
                        span,
                        bool(metadata.get("directed", True)),
                    )

            # Restore object-list index shifts and the empty-list class.
            elif product_type == "object_list":
                values = data[key]
                shifts = np.asarray(
                    self._load_product_child(group, "index_shifts", entry_idx),
                    dtype=np.int64,
                )
                shift_fields = metadata.get("index_shift_fields")
                index_shifts: int | dict[str, int]
                if shift_fields is None:
                    index_shifts = int(shifts[0])
                else:
                    index_shifts = {
                        name: int(value) for name, value in zip(shift_fields, shifts)
                    }
                default = getattr(values, "default", None)
                if default is None:
                    if len(values) == 0:
                        raise ValueError(
                            f"Cannot reconstruct empty object product `{key}` "
                            "without a stored default class."
                        )
                    default = type(values[0])()
                data[key] = spine.data.ObjectListData(
                    list(values), default, index_shifts=index_shifts
                )

            else:
                raise ValueError(f"Unknown product type `{product_type}` for `{key}`.")

    def _load_product_child(
        self,
        group: h5py.Group,
        name: str,
        entry_idx: int,
    ) -> Any:
        """Load one auxiliary value owned by a logical product group."""
        if name not in group:
            raise KeyError(f"Product `{group.name}` is missing child `{name}`.")

        values: dict[str, Any] = {}
        self.load_product(group, entry_idx, values, name)
        return values[name]

    def load_product(
        self,
        container: h5py.Group,
        entry_idx: int,
        data: dict[str, Any],
        key: str,
    ) -> None:
        """Load one event value from a self-describing product group.

        Product groups advertise a physical ``kind``:

        - ``array`` and ``string`` use values plus event offsets;
        - ``objects`` use compound fixed rows and variable-field pools;
        - ``list`` maps event offsets to element offsets and shared values;
        - ``multi_list`` owns one values/offset pair per list position.

        Parameters
        ----------
        container : h5py.Group
            Group containing the requested logical product.
        entry_idx : int
            File-local event index.
        data : dict
            Event dictionary to update.
        key : str
            Product-group name relative to ``container``.
        """
        group = container[key]
        if not isinstance(group, h5py.Group) or "kind" not in group.attrs:
            raise ValueError(
                f"Product '{group.name}' is not a recognized product group."
            )

        group_name = group.name
        if group_name is None:
            raise ValueError("Product groups must have an HDF5 path.")
        cache_key = (os.fspath(container.file.filename), group_name)
        product = self._product_handles.get(cache_key) if self.keep_open else None
        if product is None:
            kind = decode_string_attribute(group.attrs["kind"], "kind")

            # Resolve immutable physical datasets once for persistent readers.
            if kind in {"array", "string"}:
                product = _ProductHandles(
                    kind=kind,
                    values=require_dataset(group, "values"),
                    event_offsets=require_dataset(group, "event_offsets"),
                    scalar=bool(group.attrs["scalar"]),
                )
            elif kind == "objects":
                product = _ProductHandles(kind=kind, object_group=group)
            elif kind == "list":
                product = _ProductHandles(
                    kind=kind,
                    values=require_dataset(group, "values"),
                    element_offsets=require_dataset(group, "element_offsets"),
                    event_offsets=require_dataset(group, "event_offsets"),
                )
            elif kind == "multi_list":
                elements = []
                element_names = []
                for name in group:
                    if name is not None and name.startswith("element_"):
                        element_names.append(name)
                for name in sorted(
                    element_names, key=lambda item: int(item.split("_")[-1])
                ):
                    element = require_group(group, name)
                    elements.append(
                        (
                            require_dataset(element, "values"),
                            require_dataset(element, "event_offsets"),
                        )
                    )
                product = _ProductHandles(kind=kind, elements=tuple(elements))
            else:
                raise ValueError(
                    f"Unrecognized product kind '{kind}' for group '{group.name}'."
                )

            if self.keep_open:
                self._product_handles[cache_key] = product

        kind = product.kind
        if kind in {"array", "string"}:
            values = product.values
            offsets = product.event_offsets
            if values is None or offsets is None:
                raise RuntimeError(
                    f"Incomplete cached handles for product kind `{kind}`."
                )
            start, stop = (int(value) for value in offsets[entry_idx : entry_idx + 2])
            result = values[start:stop]
            if kind == "string":
                data[key] = result.tobytes().decode("utf-8")
            else:
                data[key] = result[0] if product.scalar else result
            return

        if kind == "objects":
            if product.object_group is None:
                raise RuntimeError("Object product cache is missing its group handle.")
            self.load_product_objects(product.object_group, entry_idx, data, key)
            return

        if kind == "list":
            values = product.values
            element_offsets = product.element_offsets
            event_offsets = product.event_offsets
            if values is None or element_offsets is None or event_offsets is None:
                raise RuntimeError("List product cache is missing required datasets.")

            # Map the event to logical elements, then each element to its slice
            # in the shared values dataset.
            first, last = (
                int(value) for value in event_offsets[entry_idx : entry_idx + 2]
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
            data[key] = result
            return

        if kind == "multi_list":
            result = []
            for values, offsets in product.elements:
                start, stop = (
                    int(value) for value in offsets[entry_idx : entry_idx + 2]
                )
                result.append(values[start:stop])
            data[key] = result

    def load_product_objects(
        self,
        group: h5py.Group,
        entry_idx: int,
        data: dict[str, Any],
        key: str,
    ) -> None:
        """Load an object collection and optionally rebuild its classes.

        Fixed-width fields occupy one compound row per object. Variable fields
        are pooled by physical dtype and addressed by absolute offsets stored
        in helper columns on each fixed row. Only the spans touched by this
        event are read.

        Parameters
        ----------
        group : h5py.Group
            Object-product group.
        entry_idx : int
            File-local event index.
        data : dict
            Event dictionary to update.
        key : str
            Name under which to store the reconstructed collection.
        """
        file_name = os.fspath(group.file.filename)
        group_name = group.name
        if group_name is None:
            raise ValueError("Object product group must have an HDF5 path.")
        schema_key = (file_name, group_name)

        # Cache decoded schema attributes independently of h5py handles so
        # keep_open=False and process-safe handle reopening remain supported.
        schema = self._product_object_schemas.get(schema_key)
        if schema is None:
            fixed = require_dataset(group, "fixed")
            class_name = decode_string_attribute(
                group.attrs["class_name"], "class_name"
            )
            scalar = bool(group.attrs["scalar"])
            fixed_names = tuple(
                name
                for name in fixed.dtype.names or ()
                if not name.startswith("_var_offsets_")
            )
            decoded_pool_specs: list[tuple[str, int, bool, tuple[str, ...]]] = []
            if not self.fixed_only:
                variables = require_group(group, "variables")
                for pool_name, pool in sorted(
                    variables.items(),
                    key=lambda item: int(item[0].split("_")[-1]),
                ):
                    if not isinstance(pool, h5py.Group):
                        raise TypeError(f"Variable pool '{pool_name}' must be a group.")
                    pool_index = int(pool_name.split("_")[-1])
                    kind = decode_string_attribute(pool.attrs["kind"], "kind")
                    fields_value = yaml.safe_load(
                        decode_string_attribute(pool.attrs["fields"], "fields")
                    )
                    if not isinstance(fields_value, list) or not all(
                        isinstance(name, str) for name in fields_value
                    ):
                        raise TypeError(
                            f"Variable pool '{pool_name}' fields must be a "
                            "list of strings."
                        )
                    decoded_pool_specs.append(
                        (
                            pool_name,
                            pool_index,
                            kind == "string",
                            tuple(fields_value),
                        )
                    )
            schema = (
                class_name,
                scalar,
                fixed_names,
                tuple(decoded_pool_specs),
            )
            self._product_object_schemas[schema_key] = schema

        class_name, scalar, fixed_names, pool_specs = schema

        # Persistent readers also cache direct dataset handles. These caches
        # are cleared together with their owning file handles.
        handles = (
            self._product_object_handles.get(schema_key) if self.keep_open else None
        )
        if handles is None:
            fixed = require_dataset(group, "fixed")
            event_offsets = require_dataset(group, "event_offsets")
            pool_values = []
            for pool_name, _, _, _ in pool_specs:
                pool_values.append(
                    require_dataset(group, f"variables/{pool_name}/values")
                )
            handles = (fixed, event_offsets, tuple(pool_values))
            if self.keep_open:
                self._product_object_handles[schema_key] = handles

        fixed, event_offsets, pool_values = handles
        first, last = (int(value) for value in event_offsets[entry_idx : entry_idx + 2])
        rows = fixed[first:last]
        obj_class = resolve_object_class(class_name, rows)

        # Decode each variable pool once for the enclosing event, then divide
        # that in-memory span among its objects and named fields.
        variable_values: dict[str, list[Any]] = {}
        for values, (_, pool_index, is_string, pool_fields) in zip(
            pool_values, pool_specs
        ):
            bounds = rows[f"_var_offsets_{pool_index}"]
            base = int(bounds[0, 0]) if len(bounds) else 0
            terminal = int(bounds[-1, -1]) if len(bounds) else base
            event_values = values[base:terminal]
            for field_index, name in enumerate(pool_fields):
                loaded = []
                for object_index in range(last - first):
                    start = int(bounds[object_index, field_index]) - base
                    stop = int(bounds[object_index, field_index + 1]) - base
                    value = event_values[start:stop]
                    if is_string:
                        value = value.tobytes().decode("utf-8")
                    loaded.append(value)
                variable_values[name] = loaded

        # Helper offsets are physical metadata and are excluded from the
        # reconstructed logical object dictionaries.
        result = []
        for object_index, row in enumerate(rows):
            obj_dict = {name: row[name] for name in fixed_names}
            obj_dict.update(
                {name: values[object_index] for name, values in variable_values.items()}
            )
            if self.build_classes:
                result.append(obj_class.from_dict_trusted(obj_dict))
            else:
                result.append(obj_dict)
        data[key] = result[0] if scalar else result


@dataclass(frozen=True)
class _ProductHandles:
    """Resolved HDF5 objects needed to load one product group."""

    kind: str
    values: h5py.Dataset | None = None
    event_offsets: h5py.Dataset | None = None
    scalar: bool = False
    object_group: h5py.Group | None = None
    element_offsets: h5py.Dataset | None = None
    elements: tuple[tuple[h5py.Dataset, h5py.Dataset], ...] = ()
