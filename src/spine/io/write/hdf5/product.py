"""Product-group storage backend for the SPINE HDF5 writer."""

from __future__ import annotations

from typing import Any, TypeVar

import h5py
import numpy as np
import yaml

import spine.data

from .common import DataFormat, require_group

ProductType = TypeVar("ProductType", bound=spine.data.DataProduct)


class ProductGroupBackend:
    """Write self-describing products using flat values and offset arrays.

    Logical products own one group containing their primary payload, schema
    metadata, and any auxiliary reconstruction values. The backend appends
    complete batches collectively to minimize resize and small-write overhead.
    """

    keys: set[str] | None
    skip_keys: list[str] | None
    ready: bool
    lite: bool
    type_dict: dict[str, DataFormat] | None
    event_dtype: Any
    product_metadata: dict[str, dict[str, Any]]
    product_children: dict[str, tuple[str, str]]

    @staticmethod
    def _typed_entries(
        entries: list[spine.data.DataProduct],
        product_type: type[ProductType],
        key: str,
    ) -> list[ProductType]:
        """Narrow a homogeneous event-product list to one concrete class.

        Parameters
        ----------
        entries : list[DataProduct]
            Event-level products normalized from one batched input.
        product_type : type[ProductType]
            Concrete class expected by the serialization branch.
        key : str
            Product key used in validation errors.

        Returns
        -------
        list[ProductType]
            Entries narrowed to the requested product class.
        """
        if any(type(entry) is not type(entries[0]) for entry in entries):
            raise TypeError(f"Product `{key}` mixes event data classes.")
        if not isinstance(entries[0], product_type):
            raise TypeError(
                f"Product `{key}` is not a `{product_type.__name__}` collection."
            )
        return [entry for entry in entries if isinstance(entry, product_type)]

    def prepare_products(self, data: dict[str, Any]) -> dict[str, Any]:
        """Lower self-describing products and register reconstruction metadata.

        Logical products retain their public key, while event-varying metadata
        is assigned temporary private keys that map to owned child groups.

        Parameters
        ----------
        data : dict
            Batched arrays and self-describing SPINE products.

        Returns
        -------
        dict
            Physical payloads consumed by the product-group append backend.
        """
        prepared = dict(data)

        # Discovery state must describe exactly the current batch. It is later
        # compared with the initialized schema to reject incompatible appends.
        previous_metadata = self.product_metadata
        previous_children = self.product_children
        self.product_metadata = {}
        self.product_children = {}
        for key, value in tuple(data.items()):
            if self.keys is not None and key not in self.keys:
                continue
            if self.skip_keys is not None and key in self.skip_keys:
                continue

            # Normalize supported batched representations to event products.
            entries: list[spine.data.DataProduct]
            if isinstance(value, spine.data.TensorBatch):
                entries = [value.event(index) for index in range(value.batch_size)]
            elif isinstance(value, spine.data.IndexBatch):
                entries = [value.event(index) for index in range(value.batch_size)]
            elif isinstance(value, spine.data.EdgeIndexBatch):
                entries = [value.event(index) for index in range(value.batch_size)]
            elif isinstance(value, spine.data.ClusterLabelBatch):
                entries = [value[index] for index in range(value.batch_size)]
            elif (
                isinstance(value, list)
                and value
                and all(isinstance(entry, spine.data.DataProduct) for entry in value)
            ):
                entries = [
                    entry
                    for entry in value
                    if isinstance(entry, spine.data.DataProduct)
                ]
            else:
                continue

            reference = entries[0]

            # Tensor rows carry a named schema and optional spatial metadata.
            if isinstance(reference, spine.data.TensorData):
                typed = self._typed_entries(entries, spine.data.TensorData, key)
                if any(entry.schema != reference.schema for entry in typed[1:]):
                    raise ValueError(f"Tensor schemas differ for product `{key}`.")
                has_meta = reference.meta is not None
                if any((entry.meta is not None) != has_meta for entry in typed):
                    raise ValueError(f"Tensor metadata is inconsistent for `{key}`.")
                self.product_metadata[key] = reference.metadata(reference.schema) | {
                    "coordinate_width": (
                        0
                        if reference.coordinate_data is None
                        else reference.coordinate_data.shape[1]
                    ),
                    "has_meta": has_meta,
                }
                prepared[key] = [entry.data for entry in typed]
                if has_meta:
                    self._add_product_child(
                        prepared, key, "meta", [entry.meta for entry in typed]
                    )

            # Cluster labels own optional particle tables and image metadata.
            elif isinstance(reference, spine.data.ClusterLabelData):
                typed = self._typed_entries(entries, spine.data.ClusterLabelData, key)
                has_particles = reference.particles is not None
                has_meta = reference.meta is not None
                if any(
                    (entry.particles is not None) != has_particles for entry in typed
                ):
                    raise ValueError(
                        f"Cluster-label particle tables are inconsistent for `{key}`."
                    )
                if any((entry.meta is not None) != has_meta for entry in typed):
                    raise ValueError(
                        f"Cluster-label metadata is inconsistent for `{key}`."
                    )
                self.product_metadata[key] = reference.metadata(has_particles) | {
                    "has_particles": has_particles,
                    "has_meta": has_meta,
                }
                prepared[key] = [entry.data for entry in typed]
                if has_particles:
                    self._add_product_child(
                        prepared,
                        key,
                        "particles",
                        self._serialize_particle_tables(typed),
                    )
                if has_meta:
                    self._add_product_child(
                        prepared, key, "meta", [entry.meta for entry in typed]
                    )

            # Index products store their feature arrays plus reconstruction span.
            elif isinstance(reference, spine.data.EdgeIndexData):
                typed = self._typed_entries(entries, spine.data.EdgeIndexData, key)
                self.product_metadata[key] = reference.metadata() | {
                    "directed": reference.directed
                }
                prepared[key] = [entry.features.T for entry in typed]
                self._add_product_child(
                    prepared, key, "spans", [entry.span for entry in typed]
                )
            elif isinstance(reference, spine.data.IndexListData):
                typed = self._typed_entries(entries, spine.data.IndexListData, key)
                self.product_metadata[key] = reference.metadata()
                prepared[key] = [entry.features for entry in typed]
                self._add_product_child(
                    prepared, key, "spans", [entry.span for entry in typed]
                )
            elif isinstance(reference, spine.data.IndexData):
                typed = self._typed_entries(entries, spine.data.IndexData, key)
                self.product_metadata[key] = reference.metadata()
                prepared[key] = [entry.features for entry in typed]
                self._add_product_child(
                    prepared, key, "spans", [entry.span for entry in typed]
                )

            # Object lists store index shifts separately from their object rows.
            elif isinstance(reference, spine.data.ObjectListData):
                typed = self._typed_entries(entries, spine.data.ObjectListData, key)
                shift_fields = None
                shifts = []
                if isinstance(reference.index_shifts, dict):
                    shift_fields = tuple(reference.index_shifts)
                    for entry in typed:
                        entry_shifts = entry.index_shifts
                        if (
                            not isinstance(entry_shifts, dict)
                            or tuple(entry_shifts) != shift_fields
                        ):
                            raise ValueError(
                                f"Object-list index shifts differ for product `{key}`."
                            )
                        shifts.append(
                            np.asarray(
                                [entry_shifts[name] for name in shift_fields],
                                dtype=np.int64,
                            )
                        )
                else:
                    for entry in typed:
                        entry_shifts = entry.index_shifts
                        if isinstance(entry_shifts, dict):
                            raise ValueError(
                                f"Object-list index shifts differ for product `{key}`."
                            )
                        shifts.append(np.asarray([entry_shifts], dtype=np.int64))

                self.product_metadata[key] = reference.metadata() | {
                    "index_shift_fields": shift_fields
                }
                prepared[key] = [entry.to_object_list for entry in typed]
                self._add_product_child(prepared, key, "index_shifts", shifts)

        if self.ready and (
            self.product_metadata != previous_metadata
            or self.product_children != previous_children
        ):
            raise ValueError(
                "The product classes or schemas changed after the product-group "
                "writer was initialized."
            )
        return prepared

    def _add_product_child(
        self,
        prepared: dict[str, Any],
        parent: str,
        name: str,
        values: Any,
    ) -> None:
        """Register event-varying values owned by one logical V2 product."""
        key = f"__spine_v2_aux__{parent}__{name}"
        if key in prepared:
            raise KeyError(
                f"Input product `{key}` conflicts with an internal V2 storage key."
            )

        prepared[key] = values
        self.product_children[key] = (parent, name)

        # Selected public products implicitly select their private children
        if self.keys is not None and parent in self.keys:
            self.keys.add(key)

    @staticmethod
    def _serialize_particle_tables(
        entries: list[spine.data.ClusterLabelData],
    ) -> list[spine.data.ObjectList]:
        """Convert named particle tables into serializable object lists."""
        # Convert column-oriented particle tables to row-oriented data objects
        particle_arrays = []
        for entry in entries:
            if entry.particles is None:
                raise ValueError(
                    "Particle information must be consistent across an HDF5 batch."
                )

            num_particles = len(next(iter(entry.particles.values()), ()))
            objects = []
            for particle_id in range(num_particles):
                values = {
                    name: field[particle_id] for name, field in entry.particles.items()
                }
                objects.append(spine.data.ParticleLabel(**values))
            # Preserve the object class even for events with no particles
            particle_arrays.append(
                spine.data.ObjectList(objects, default=spine.data.ParticleLabel())
            )

        return particle_arrays

    def initialize_product_datasets(
        self, out_file: h5py.Group, type_dict: dict[str, DataFormat]
    ) -> None:
        """Create the offset-based version-2 dataset layout.

        Every logical product is represented by a group under ``/products``.
        Event-varying implementation details are nested below their owning
        product instead of appearing as sibling products. Each payload group
        has a ``kind`` attribute which determines the required flat datasets
        and offset levels:

        - Arrays and strings have ``values`` and ``event_offsets``.
        - Object collections have compound ``fixed`` rows, ``event_offsets``,
          and one or more dtype-homogeneous variable pools.
        - Lists of same-width arrays add ``element_offsets`` between the event
          and value levels.
        - Fixed-length lists of differently shaped arrays use one child group
          per list position.

        All offset datasets begin with zero. Appending ``N`` logical items adds
        ``N`` terminal offsets, preserving the invariant
        ``len(offsets) == num_items + 1``. Equal adjacent offsets represent an
        empty item.

        Parameters
        ----------
        out_file : h5py.File
            Newly created output file.
        type_dict : dict[str, DataFormat]
            Logical product formats inferred from the first input batch.
        """
        self.event_dtype = np.dtype(np.int64)

        # Create the public logical-product namespace before owned auxiliaries
        products = out_file.create_group("products")
        for key in type_dict:
            if key not in self.product_children:
                products.create_group(key)

        # Initialize primary payloads and private child payloads uniformly
        for key, val in type_dict.items():
            if key in self.product_children:
                parent, name = self.product_children[key]
                parent_group = products[parent]
                assert isinstance(parent_group, h5py.Group)
                group = parent_group.create_group(name)
            else:
                group = products[key]
                assert isinstance(group, h5py.Group)

            group.attrs["scalar"] = val.scalar
            if key in self.product_metadata:
                group.attrs["product_metadata"] = yaml.safe_dump(
                    self.product_metadata[key]
                )

            if val.class_name is not None:
                # Fixed and derived attributes stay in a normal compound
                # dataset. Only attributes represented as VLEN in the logical
                # dtype are moved into flat pools.
                group.attrs["kind"] = "objects"
                group.attrs["class_name"] = val.class_name
                assert isinstance(val.dtype, list)
                fixed_dtype, variable_pools = self.split_object_dtype(val.dtype)
                storage_dtype: list[tuple[Any, ...]] = list(fixed_dtype)
                storage_dtype.extend(
                    (f"_var_offsets_{i}", np.int64, len(fields) + 1)
                    for i, (_, _, fields) in enumerate(variable_pools)
                )
                group.create_dataset(
                    "fixed", (0,), maxshape=(None,), dtype=storage_dtype
                )
                group.create_dataset(
                    "event_offsets",
                    data=np.zeros(1, dtype=np.int64),
                    maxshape=(None,),
                )
                variables = group.create_group("variables")
                for i, (dtype, is_string, fields) in enumerate(variable_pools):
                    # Pooling fields by dtype limits dataset count while
                    # retaining one deterministic ordered field list.
                    pool = variables.create_group(f"pool_{i}")
                    pool.attrs["kind"] = "string" if is_string else "array"
                    pool.attrs["fields"] = yaml.safe_dump(fields)
                    value_dtype = np.uint8 if is_string else dtype
                    pool.create_dataset(
                        "values", (0,), maxshape=(None,), dtype=value_dtype
                    )

            elif not isinstance(val.width, list):
                # A simple product needs one event-to-value offset level.
                # Strings are encoded explicitly so V2 contains no HDF5 VLEN
                # datatype anywhere in its product tree.
                dtype = np.dtype(val.dtype)
                is_string = h5py.check_dtype(vlen=dtype) is str
                group.attrs["kind"] = "string" if is_string else "array"
                shape = (0, val.width) if val.width else (0,)
                maxshape = (None, val.width) if val.width else (None,)
                value_dtype = np.uint8 if is_string else val.dtype
                group.create_dataset(
                    "values", shape, maxshape=maxshape, dtype=value_dtype
                )
                group.create_dataset(
                    "event_offsets",
                    data=np.zeros(1, dtype=np.int64),
                    maxshape=(None,),
                )

            elif val.merge:
                # The event contains a variable number of arrays which share a
                # width and can therefore occupy one values dataset.
                group.attrs["kind"] = "list"
                width = val.width[0]
                shape = (0, width) if width else (0,)
                maxshape = (None, width) if width else (None,)
                group.create_dataset(
                    "values", shape, maxshape=maxshape, dtype=val.dtype
                )
                group.create_dataset(
                    "element_offsets",
                    data=np.zeros(1, dtype=np.int64),
                    maxshape=(None,),
                )
                group.create_dataset(
                    "event_offsets",
                    data=np.zeros(1, dtype=np.int64),
                    maxshape=(None,),
                )

            else:
                # Differently shaped list positions cannot share a rectangular
                # values dataset. Each position gets an independent event span.
                group.attrs["kind"] = "multi_list"
                for i, width in enumerate(val.width):
                    element = group.create_group(f"element_{i}")
                    shape = (0, width) if width else (0,)
                    maxshape = (None, width) if width else (None,)
                    element.create_dataset(
                        "values", shape, maxshape=maxshape, dtype=val.dtype
                    )
                    element.create_dataset(
                        "event_offsets",
                        data=np.zeros(1, dtype=np.int64),
                        maxshape=(None,),
                    )

        # V2 retains an event axis for counting, global indexing and completeness
        out_file.create_dataset(
            "events", (0,), maxshape=(None,), dtype=self.event_dtype
        )

    def _product_group(self, out_file: h5py.Group, key: str) -> h5py.Group:
        """Resolve a temporary write key to its physical V2 product group."""
        products = require_group(out_file, "products")
        if key in self.product_children:
            parent, name = self.product_children[key]
            parent_group = require_group(products, parent)
            return require_group(parent_group, name)

        return require_group(products, key)

    @staticmethod
    def split_object_dtype(
        obj_dtype: list[tuple[str, type]],
    ) -> tuple[list[tuple[str, type]], list[tuple[np.dtype, bool, list[str]]]]:
        """Partition a logical object dtype into fixed columns and flat pools.

        ``get_object_dtype`` expresses variable arrays and strings using HDF5
        VLEN dtypes because that description is also consumed by V1. V2 uses
        the VLEN base dtype only as schema information; no VLEN dtype is
        created on disk. Variable fields with the same base dtype share one
        values pool, while strings use a distinct ``uint8`` UTF-8 pool.

        Parameters
        ----------
        obj_dtype : list[tuple[str, type]]
            Ordered logical object-field specifications.

        Returns
        -------
        fixed_dtype : list[tuple[str, type]]
            Scalar and fixed-width compound-dataset fields.
        variable_pools : list[tuple[np.dtype, bool, list[str]]]
            One tuple per flat pool containing its value dtype, string flag,
            and ordered logical field names.
        """
        fixed_dtype = []
        pool_map: dict[tuple[str, bool], tuple[np.dtype, bool, list[str]]] = {}
        for spec in obj_dtype:
            dtype = np.dtype(spec[1])
            base = h5py.check_dtype(vlen=dtype)
            if base is None:
                fixed_dtype.append(spec)
                continue
            is_string = base is str
            base_dtype = np.dtype(np.uint8 if is_string else base)
            pool_key = (base_dtype.str, is_string)
            if pool_key not in pool_map:
                pool_map[pool_key] = (base_dtype, is_string, [])
            pool_map[pool_key][2].append(spec[0])
        return fixed_dtype, list(pool_map.values())

    def append_product_entry(
        self, out_file: h5py.Group, data: dict[str, Any], batch_id: int
    ) -> None:
        """Append one entry through the collective V2 implementation.

        This compatibility wrapper keeps :meth:`append_entry` useful to
        callers which explicitly write individual entries. The physical write
        path remains batch-oriented, with a one-element batch.

        Parameters
        ----------
        out_file : h5py.File
            Output file initialized with the V2 schema.
        data : dict
            Batched data-product dictionary.
        batch_id : int
            Index of the entry within ``data``.
        """
        self.append_product_entries(
            out_file, data, np.asarray([batch_id], dtype=np.int64)
        )

    def append_product_entries(
        self, out_file: h5py.Group, data: dict[str, Any], batch_ids: np.ndarray
    ) -> None:
        """Append selected batch entries using collective V2 writes.

        Products are committed first and the authoritative ``events`` axis is
        extended last. During normal operation the writer's ``complete=False``
        marker protects readers from observing a partially written batch. On
        successful finalization every product has exactly one event boundary
        per row in ``events``.

        Parameters
        ----------
        out_file : h5py.File
            Output file initialized with the V2 schema.
        data : dict
            Batched data-product dictionary.
        batch_ids : np.ndarray
            Ordered indexes of entries to append. Split output uses a subset of
            the input batch here.
        """
        assert self.keys is not None
        for key in self.keys:
            self.append_product_batch(out_file, data, key, batch_ids)

        # Rows carry their own monotonic IDs today. Their primary contract is
        # the stable event count/axis; product lookup uses product offsets.
        events = out_file["events"]
        assert isinstance(events, h5py.Dataset)
        event_id = len(events)
        events.resize(event_id + len(batch_ids), axis=0)
        events[event_id:] = np.arange(
            event_id, event_id + len(batch_ids), dtype=np.int64
        )

    def append_product_batch(
        self,
        out_file: h5py.Group,
        data: dict[str, Any],
        key: str,
        batch_ids: np.ndarray,
    ) -> None:
        """Append multiple events for one V2 product in collective slices.

        The product group's ``kind`` is the only physical-layout dispatch.
        Logical :class:`DataFormat` metadata is used to normalize scalar versus
        collection inputs before they enter the common offset helpers.

        Parameters
        ----------
        out_file : h5py.File
            Output file containing the product group.
        data : dict
            Batched data-product dictionary.
        key : str
            Product to append.
        batch_ids : np.ndarray
            Ordered indexes of entries to append.
        """
        assert self.type_dict is not None
        val = self.type_dict[key]
        group = self._product_group(out_file, key)
        kind = group.attrs["kind"]

        if kind == "objects":
            # Normalize scalar object products to one-object collections so the
            # storage helper only needs one representation.
            batches = []
            for batch_id in batch_ids:
                obj = data[key] if np.isscalar(data[key]) else data[key][batch_id]
                batches.append([obj] if val.scalar else obj)
            self.store_object_batches(group, batches, self.lite)
            return

        if kind in {"array", "string"}:
            # Build one array per event, then concatenate and resize once.
            arrays = []
            for batch_id in batch_ids:
                if np.isscalar(data[key]):
                    value = data[key]
                else:
                    value = data[key][batch_id]
                if kind == "string":
                    array = np.frombuffer(str(value).encode("utf-8"), dtype=np.uint8)
                else:
                    array = np.asarray([value]) if val.scalar else np.asarray(value)
                arrays.append(array)
            self.append_array_batch(group, arrays)
            return

        array_lists = [data[key][batch_id] for batch_id in batch_ids]
        if kind == "list":
            # Flatten event -> element -> value. The two offset levels preserve
            # both collection boundaries without region references.
            arrays = [array for array_list in array_lists for array in array_list]
            values = group["values"]
            element_offsets = group["element_offsets"]
            event_offsets = group["event_offsets"]
            assert isinstance(values, h5py.Dataset)
            assert isinstance(element_offsets, h5py.Dataset)
            assert isinstance(event_offsets, h5py.Dataset)
            self.append_values_with_offsets(values, element_offsets, arrays)
            counts = [len(array_list) for array_list in array_lists]
            self.append_lengths(event_offsets, counts)
            return

        assert kind == "multi_list"
        # Each list position owns a separate rectangular values dataset.
        elements = []
        for name in group:
            if name is not None and name.startswith("element_"):
                elements.append(name)
        for i, name in enumerate(
            sorted(elements, key=lambda item: int(item.split("_")[-1]))
        ):
            element = group[name]
            assert isinstance(element, h5py.Group)
            self.append_array_batch(
                element, [array_list[i] for array_list in array_lists]
            )

    @staticmethod
    def append_lengths(offsets: h5py.Dataset, lengths: Any) -> None:
        """Extend a boundary array from a sequence of item lengths.

        The existing final offset is the absolute base of the append. For
        lengths ``[a, b]``, the method appends ``base + a`` and
        ``base + a + b``. Zero lengths intentionally repeat the previous
        boundary and represent empty items.

        Parameters
        ----------
        offsets : h5py.Dataset
            One-dimensional monotonic ``int64`` boundary dataset whose initial
            value is zero.
        lengths : array-like
            Number of values contributed by each newly appended logical item.
        """
        lengths = np.asarray(lengths, dtype=np.int64)
        if not len(lengths):
            return
        first = len(offsets) - 1
        base = int(offsets[first])
        offsets.resize(len(offsets) + len(lengths), axis=0)
        offsets[first + 1 :] = base + np.cumsum(lengths)

    @classmethod
    def append_values_with_offsets(
        cls,
        values: h5py.Dataset,
        offsets: h5py.Dataset,
        arrays: list[np.ndarray],
    ) -> None:
        """Append variable arrays and their boundaries with collective resizes.

        Arrays are concatenated in logical order. The values dataset and offset
        dataset are each resized once, which is the central write-side
        performance advantage over per-object region references/VLEN payloads.

        Parameters
        ----------
        values : h5py.Dataset
            Flat destination dataset.
        offsets : h5py.Dataset
            Boundary dataset corresponding to ``values``.
        arrays : list[np.ndarray]
            Ordered variable-length arrays to append.
        """
        lengths = np.asarray([len(array) for array in arrays], dtype=np.int64)
        first = len(values)
        combined = np.concatenate(arrays) if arrays else np.empty(0, dtype=values.dtype)
        values.resize(first + len(combined), axis=0)
        if len(combined):
            values[first:] = combined
        cls.append_lengths(offsets, lengths)

    @classmethod
    def append_array_batch(cls, group: h5py.Group, arrays: list[np.ndarray]) -> None:
        """Append one array per event to a simple V2 product group.

        Parameters
        ----------
        group : h5py.Group
            Group containing ``values`` and ``event_offsets``.
        arrays : list[np.ndarray]
            Ordered event payloads.
        """
        values = group["values"]
        offsets = group["event_offsets"]
        assert isinstance(values, h5py.Dataset)
        assert isinstance(offsets, h5py.Dataset)
        cls.append_values_with_offsets(values, offsets, arrays)

    @classmethod
    def store_object_batches(
        cls, group: h5py.Group, batches: list[Any], lite: bool
    ) -> None:
        """Store object batches in fixed rows and dtype-specific value pools.

        Objects are flattened in event order. Each logical object contributes
        one compound ``fixed`` row. For a variable pool containing ``F``
        fields, the corresponding fixed-row helper column contains ``F + 1``
        absolute offsets; adjacent boundaries delimit each field in the pool's
        shared values dataset. The final ``event_offsets`` update maps events
        back to their ranges of fixed rows.

        Derived properties returned by ``obj.as_dict`` are stored alongside
        ordinary fixed attributes. This is intentional: consumers which read
        HDF5 directly, without SPINE classes, retain access to the advertised
        object summaries.

        Parameters
        ----------
        group : h5py.Group
            V2 object product group.
        batches : list
            Ordered per-event object collections.
        lite : bool
            Passed to ``as_dict`` to omit configured heavy attributes.
        """
        fixed = group["fixed"]
        event_offsets = group["event_offsets"]
        variables = group["variables"]
        assert isinstance(fixed, h5py.Dataset)
        assert isinstance(event_offsets, h5py.Dataset)
        assert isinstance(variables, h5py.Group)

        # Flatten once so fixed columns and every variable pool share exactly
        # the same object-row order.
        object_dicts = [obj.as_dict(lite) for batch in batches for obj in batch]
        rows: Any = np.empty(len(object_dicts), dtype=fixed.dtype)
        if object_dicts:
            for name in fixed.dtype.names or ():
                if not name.startswith("_var_offsets_"):
                    rows[name] = [obj[name] for obj in object_dicts]

        for pool_name, pool in variables.items():
            assert isinstance(pool, h5py.Group)
            values = pool["values"]
            assert isinstance(values, h5py.Dataset)
            is_string = pool.attrs["kind"] == "string"
            fields_attr = pool.attrs["fields"]
            if isinstance(fields_attr, bytes):
                fields_attr = fields_attr.decode()
            if not isinstance(fields_attr, str):
                raise TypeError(
                    f"V2 variable pool '{pool_name}' fields must be a string."
                )
            fields = yaml.safe_load(fields_attr)
            if not isinstance(fields, list) or not all(
                isinstance(name, str) for name in fields
            ):
                raise TypeError(
                    f"V2 variable pool '{pool_name}' fields must decode "
                    "to a list of strings."
                )
            chunks = []
            offset_rows = np.empty((len(object_dicts), len(fields) + 1), dtype=np.int64)
            # Offsets are absolute in the full pool, not relative to this
            # batch. This makes appends independent and permits direct slicing.
            cursor = len(values)
            for i, obj in enumerate(object_dicts):
                offset_rows[i, 0] = cursor
                for j, name in enumerate(fields):
                    value = obj[name]
                    if is_string:
                        chunk = np.frombuffer(value.encode("utf-8"), dtype=np.uint8)
                    else:
                        chunk = np.asarray(value)
                        if chunk.ndim != 1:
                            raise ValueError(
                                f"V2 variable object field '{name}' must be "
                                f"one-dimensional, got shape {chunk.shape}."
                            )
                    chunks.append(chunk)
                    cursor += len(chunk)
                    offset_rows[i, j + 1] = cursor

            first_value = len(values)
            combined = (
                np.concatenate(chunks) if chunks else np.empty(0, dtype=values.dtype)
            )
            values.resize(first_value + len(combined), axis=0)
            if len(combined):
                values[first_value:] = combined

            if len(object_dicts):
                pool_index = int(pool_name.split("_")[-1])
                rows[f"_var_offsets_{pool_index}"] = offset_rows

        # Commit fixed rows only after their variable offset columns have been
        # populated. Event boundaries are appended last.
        first_object = len(fixed)
        fixed.resize(first_object + len(object_dicts), axis=0)
        if len(object_dicts):
            fixed[first_object:] = rows

        cls.append_lengths(event_offsets, [len(batch) for batch in batches])
