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

ObjectPoolSpec = tuple[str, int, bool, tuple[str, ...]]
ObjectSchema = tuple[str, bool, tuple[str, ...], tuple[ObjectPoolSpec, ...]]
ObjectHandles = tuple[h5py.Dataset, h5py.Dataset, tuple[h5py.Dataset, ...]]


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
    _product_object_schemas: dict[tuple[str, str], ObjectSchema]
    _product_object_handles: dict[tuple[str, str], ObjectHandles]
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
            children = {
                name: self._load_product_child(group, name, entry_idx)
                for name in self._product_child_names(metadata)
            }
            self._reconstruct_product(data, key, metadata, children)

    def reconstruct_products_many(
        self,
        products: h5py.Group,
        first: int,
        last: int,
        data: list[dict[str, Any]],
    ) -> None:
        """Rebuild typed products for one contiguous run of V2 events.

        Product-owned auxiliary values are decoded in contiguous batches just
        like their parent products. This avoids falling back to scalar HDF5
        reads for tensor metadata, index spans, particle tables, or shifts.

        Parameters
        ----------
        products : h5py.Group
            Logical-product root for the current file.
        first, last : int
            Inclusive-exclusive file-local event range.
        data : list[dict]
            Mutable event dictionaries populated by :meth:`load_product_many`.
        """
        if not data:
            return

        for key in tuple(data[0]):
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
            child_names = self._product_child_names(metadata)
            children = [dict() for _ in data]
            for name in child_names:
                self.load_product_many(group, first, last, children, name)

            for index, event in enumerate(data):
                self._reconstruct_product(event, key, metadata, children[index])

    @staticmethod
    def _product_child_names(metadata: dict[str, Any]) -> tuple[str, ...]:
        """Return auxiliary groups required to reconstruct a typed product."""
        product_type = metadata.get("product_type")
        if product_type == "tensor":
            return ("meta",) if metadata.get("has_meta", False) else ()
        if product_type == "cluster_label":
            names = []
            if metadata.get("has_meta", False):
                names.append("meta")
            if metadata.get("has_particles", False):
                names.append("particles")
            return tuple(names)
        if product_type in {"index", "index_list", "edge_index"}:
            return ("spans",)
        if product_type == "object_list":
            return ("index_shifts",)
        return ()

    @staticmethod
    def _reconstruct_product(
        data: dict[str, Any],
        key: str,
        metadata: dict[str, Any],
        children: dict[str, Any],
    ) -> None:
        """Reconstruct one typed event product from decoded V2 payloads."""
        product_type = metadata.get("product_type")

        # Recombine primary arrays with product-owned tensor metadata.
        if product_type == "tensor":
            values = np.asarray(data[key])
            width = int(metadata.get("coordinate_width", 0))
            schema = spine.data.TensorSchema.from_dict(metadata.get("schema", {}))
            coords = values[:, :width] if width else None
            features = values[:, width:] if width else values
            data[key] = spine.data.TensorData(
                features,
                coords,
                meta=children.get("meta"),
                schema=schema,
            )

        # Rebuild compact voxel labels and their optional particle table.
        elif product_type == "cluster_label":
            particles = None
            if "particles" in children:
                particle_objects = children["particles"]
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
                np.asarray(data[key]),
                particles=particles,
                meta=children.get("meta"),
            )

        # Restore index semantics from the event-level node span.
        elif product_type in {"index", "index_list", "edge_index"}:
            span = int(children["spans"])
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
            shifts = np.asarray(children["index_shifts"], dtype=np.int64)
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

    def _get_product_handles(
        self,
        container: h5py.Group,
        key: str,
    ) -> "_ProductHandles":
        """Resolve and optionally cache datasets backing one V2 product."""
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
        if product is not None:
            return product

        kind = decode_string_attribute(group.attrs["kind"], "kind")
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
            element_names = [
                name
                for name in group
                if name is not None and name.startswith("element_")
            ]
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
        return product

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
        product = self._get_product_handles(container, key)
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

    def load_product_many(
        self,
        container: h5py.Group,
        first: int,
        last: int,
        data: list[dict[str, Any]],
        key: str,
    ) -> None:
        """Load one V2 product for a contiguous range of events.

        Offset vectors and their corresponding payload spans are each read
        once, then divided into event values in memory. The decoded values are
        identical to repeated :meth:`load_product` calls.

        Parameters
        ----------
        container : h5py.Group
            Group containing the requested logical product.
        first, last : int
            Inclusive-exclusive file-local event range.
        data : list[dict]
            Event dictionaries to update. Its length must equal ``last-first``.
        key : str
            Product-group name relative to ``container``.

        Raises
        ------
        ValueError
            If the destination count does not match the requested event range.
        """
        if len(data) != last - first:
            raise ValueError(
                "Batch product destination count must match the event range."
            )
        if not data:
            return

        product = self._get_product_handles(container, key)
        kind = product.kind
        if kind in {"array", "string"}:
            values = product.values
            offsets = product.event_offsets
            if values is None or offsets is None:
                raise RuntimeError(
                    f"Incomplete cached handles for product kind `{kind}`."
                )
            bounds = np.asarray(offsets[first : last + 1], dtype=np.int64)
            base, terminal = int(bounds[0]), int(bounds[-1])
            batch_values = values[base:terminal]
            for index, event in enumerate(data):
                start = int(bounds[index]) - base
                stop = int(bounds[index + 1]) - base
                result = batch_values[start:stop]
                if kind == "string":
                    event[key] = result.tobytes().decode("utf-8")
                else:
                    event[key] = result[0] if product.scalar else result
            return

        if kind == "objects":
            if product.object_group is None:
                raise RuntimeError("Object product cache is missing its group handle.")
            self.load_product_objects_many(product.object_group, first, last, data, key)
            return

        if kind == "list":
            values = product.values
            element_offsets = product.element_offsets
            event_offsets = product.event_offsets
            if values is None or element_offsets is None or event_offsets is None:
                raise RuntimeError("List product cache is missing required datasets.")

            event_bounds = np.asarray(event_offsets[first : last + 1], dtype=np.int64)
            element_base = int(event_bounds[0])
            element_terminal = int(event_bounds[-1])
            element_bounds = np.asarray(
                element_offsets[element_base : element_terminal + 1],
                dtype=np.int64,
            )
            value_base = int(element_bounds[0]) if len(element_bounds) else 0
            value_terminal = (
                int(element_bounds[-1]) if len(element_bounds) else value_base
            )
            batch_values = values[value_base:value_terminal]

            for index, event in enumerate(data):
                element_start = int(event_bounds[index]) - element_base
                element_stop = int(event_bounds[index + 1]) - element_base
                result = np.empty(element_stop - element_start, dtype=object)
                for output_index, element_index in enumerate(
                    range(element_start, element_stop)
                ):
                    start = int(element_bounds[element_index]) - value_base
                    stop = int(element_bounds[element_index + 1]) - value_base
                    result[output_index] = batch_values[start:stop]
                event[key] = result
            return

        if kind == "multi_list":
            results = [[] for _ in data]
            for values, offsets in product.elements:
                bounds = np.asarray(offsets[first : last + 1], dtype=np.int64)
                base, terminal = int(bounds[0]), int(bounds[-1])
                batch_values = values[base:terminal]
                for index, result in enumerate(results):
                    start = int(bounds[index]) - base
                    stop = int(bounds[index + 1]) - base
                    result.append(batch_values[start:stop])
            for event, result in zip(data, results):
                event[key] = result

    def _get_object_product(
        self,
        group: h5py.Group,
    ) -> tuple[ObjectSchema, ObjectHandles]:
        """Resolve schema metadata and physical datasets for an object product."""
        file_name = os.fspath(group.file.filename)
        group_name = group.name
        if group_name is None:
            raise ValueError("Object product group must have an HDF5 path.")
        schema_key = (file_name, group_name)

        # Schema attributes remain valid across transient file handles.
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
            decoded_pool_specs: list[ObjectPoolSpec] = []
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

        _, _, _, pool_specs = schema
        handles = (
            self._product_object_handles.get(schema_key) if self.keep_open else None
        )
        if handles is None:
            fixed = require_dataset(group, "fixed")
            event_offsets = require_dataset(group, "event_offsets")
            pool_values = tuple(
                require_dataset(group, f"variables/{pool_name}/values")
                for pool_name, _, _, _ in pool_specs
            )
            handles = (fixed, event_offsets, pool_values)
            if self.keep_open:
                self._product_object_handles[schema_key] = handles

        return schema, handles

    def _decode_object_rows(
        self,
        rows: np.ndarray,
        class_name: str,
        fixed_names: tuple[str, ...],
        pool_specs: tuple[ObjectPoolSpec, ...],
        pool_values: tuple[h5py.Dataset, ...],
    ) -> list[Any]:
        """Decode one contiguous compound-row span into logical objects."""
        obj_class = resolve_object_class(class_name, rows)

        # Each variable pool is also contiguous across the selected rows.
        variable_values: dict[str, list[Any]] = {}
        for values, (_, pool_index, is_string, pool_fields) in zip(
            pool_values, pool_specs
        ):
            bounds = rows[f"_var_offsets_{pool_index}"]
            base = int(bounds[0, 0]) if len(bounds) else 0
            terminal = int(bounds[-1, -1]) if len(bounds) else base
            batch_values = values[base:terminal]
            for field_index, name in enumerate(pool_fields):
                loaded = []
                for object_index in range(len(rows)):
                    start = int(bounds[object_index, field_index]) - base
                    stop = int(bounds[object_index, field_index + 1]) - base
                    value = batch_values[start:stop]
                    if is_string:
                        value = value.tobytes().decode("utf-8")
                    loaded.append(value)
                variable_values[name] = loaded

        # Physical helper offsets are deliberately omitted from each object.
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
        return result

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
        schema, handles = self._get_object_product(group)
        class_name, scalar, fixed_names, pool_specs = schema
        fixed, event_offsets, pool_values = handles
        first, last = (int(value) for value in event_offsets[entry_idx : entry_idx + 2])
        rows = fixed[first:last]
        result = self._decode_object_rows(
            rows, class_name, fixed_names, pool_specs, pool_values
        )
        data[key] = result[0] if scalar else result

    def load_product_objects_many(
        self,
        group: h5py.Group,
        first: int,
        last: int,
        data: list[dict[str, Any]],
        key: str,
    ) -> None:
        """Load an object product for one contiguous V2 event range.

        The compound rows and each variable-value pool are read as single
        spans. Objects are reconstructed once and then partitioned using the
        event boundary vector.
        """
        schema, handles = self._get_object_product(group)
        class_name, scalar, fixed_names, pool_specs = schema
        fixed, event_offsets, pool_values = handles

        bounds = np.asarray(event_offsets[first : last + 1], dtype=np.int64)
        base, terminal = int(bounds[0]), int(bounds[-1])
        rows = fixed[base:terminal]
        objects = self._decode_object_rows(
            rows, class_name, fixed_names, pool_specs, pool_values
        )

        for index, event in enumerate(data):
            start = int(bounds[index]) - base
            stop = int(bounds[index + 1]) - base
            result = objects[start:stop]
            event[key] = result[0] if scalar else result


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
