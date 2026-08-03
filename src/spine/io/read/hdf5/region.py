"""Decoder for the region-reference HDF5 storage layout."""

from __future__ import annotations

from dataclasses import fields as dataclass_fields
from typing import Any

import h5py
import numpy as np

from .common import require_dataset, resolve_object_class


class RegionReferenceBackend:
    """Read legacy products addressed through event-level region references.

    The class is an implementation mixin for :class:`HDF5Reader`. Reader
    configuration remains owned by the public reader, while the physical
    decoding rules live here in isolation from the product-group backend.
    """

    build_classes: bool
    skip_unknown_attrs: bool

    def _load_region_columnar_objects(
        self,
        in_file: h5py.File,
        key: str,
        entries: np.ndarray,
        requested_fields: tuple[str, ...] | None,
    ) -> dict[str, Any]:
        """Project compound object fields through region references.

        Parameters
        ----------
        in_file : h5py.File
            Open input file containing the requested entries.
        key : str
            Object-product dataset name.
        entries : numpy.ndarray
            File-local entry indexes to project.
        requested_fields : tuple[str, ...], optional
            Fixed object fields to read. All fields are returned when omitted.

        Returns
        -------
        dict
            Projected columns and their chunk-local ``event_offsets``.
        """
        dataset = require_dataset(in_file, key)
        available = tuple(dataset.dtype.names or ())
        fields_to_read = available if requested_fields is None else requested_fields
        missing = set(fields_to_read).difference(available)
        if missing:
            raise KeyError(
                f"Region-reference product `{key}` is missing fields "
                f"{sorted(missing)}."
            )

        # Region references must be resolved event by event because this
        # layout has no explicit event-offset dataset.
        events = require_dataset(in_file, "events")
        rows = []
        counts = []
        for entry in entries:
            event = events[int(entry)]
            names = getattr(getattr(event, "dtype", None), "names", ())
            if key not in names:
                raise KeyError(f"Event does not reference product `{key}`.")
            if fields_to_read:
                values = dataset.fields(fields_to_read)[event[key]]
                rows.append(values)
                counts.append(len(values))
            else:
                counts.append(len(dataset[event[key]]))

        # Return ordinary arrays so downstream columnar consumers do not need
        # to understand HDF5 compound dtypes or region references.
        result = {}
        if fields_to_read:
            combined = (
                np.concatenate(rows)
                if rows
                else np.empty(
                    0,
                    dtype=np.dtype(
                        [(name, dataset.dtype[name]) for name in fields_to_read]
                    ),
                )
            )
            result.update({name: combined[name] for name in fields_to_read})
        result["event_offsets"] = np.concatenate(
            ([0], np.cumsum(counts, dtype=np.int64))
        )
        return result

    def load_region_product(
        self,
        in_file: h5py.File | h5py.Group,
        event: dict[str, Any],
        data: dict[str, Any],
        key: str,
    ) -> None:
        """Load one event product through its stored region reference.

        Parameters
        ----------
        in_file : h5py.File or h5py.Group
            Open input file or stage group containing the product.
        event : dict
            Structured event row containing product region references.
        data : dict
            Event dictionary to update.
        key : str
            Product name to load.
        """
        region_ref = event[key]
        dataset = in_file[key]

        # Plain datasets store either arrays or compound object rows.
        if isinstance(dataset, h5py.Dataset):
            names = getattr(getattr(dataset, "dtype", None), "names", None)
            if not names:
                data[key] = dataset[region_ref]
                if dataset.attrs["scalar"]:
                    data[key] = data[key][0]
                if len(dataset.shape) > 1:
                    data[key] = data[key].reshape(-1, dataset.shape[1])
                return

            # Compound rows describe SPINE objects. Resolve their class before
            # optionally filtering fields unknown to the current installation.
            array = dataset[region_ref]
            class_name = dataset.attrs["class_name"]
            if not isinstance(class_name, str):
                raise TypeError("Dataset is missing a string 'class_name' attribute.")
            obj_class = resolve_object_class(class_name, array)
            known_attrs = (
                {field.name for field in dataclass_fields(obj_class)}
                if self.skip_unknown_attrs
                else None
            )

            objects = []
            names = array.dtype.names or ()
            for element in array:
                if known_attrs is not None:
                    obj_dict = {
                        name: element[index]
                        for index, name in enumerate(names)
                        if name in known_attrs
                    }
                else:
                    obj_dict = dict(zip(names, element))
                objects.append(
                    obj_class.from_dict(obj_dict) if self.build_classes else obj_dict
                )
            data[key] = objects[0] if dataset.attrs["scalar"] else objects
            return

        # Group products encode one list, or a fixed list of differently shaped
        # arrays, using region references into child datasets.
        if isinstance(dataset, h5py.Group):
            index = require_dataset(dataset, "index")
            element_refs = index[region_ref].flatten()
            if len(index.shape) == 1:
                elements = require_dataset(dataset, "elements")
                result = np.empty(len(element_refs), dtype=object)
                result[:] = [elements[ref] for ref in element_refs]
                if len(elements.shape) > 1:
                    for index in range(len(element_refs)):
                        result[index] = result[index].reshape(-1, elements.shape[1])
            else:
                element_datasets = [
                    require_dataset(dataset, f"element_{index}")
                    for index in range(len(element_refs))
                ]
                result = []
                for index, element in enumerate(element_datasets):
                    result.append(element[element_refs[index]])
                    if len(element.shape) > 1:
                        result[index] = result[index].reshape(-1, element.shape[1])

            data[key] = result
            return

        raise ValueError(f"Product '{key}' is neither an HDF5 group nor dataset.")
