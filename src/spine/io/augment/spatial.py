"""Explicit spatial-product contracts used by geometric augmentation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from spine.data import (
    ClusterLabelData,
    DataProduct,
    EdgeIndexData,
    IndexData,
    IndexListData,
    Meta,
    ObjectListData,
    TensorData,
)


@dataclass(frozen=True)
class SpatialField:
    """Description of one three-dimensional point field.

    Parameters
    ----------
    name : str
        Product-local name used to write transformed values back.
    values : numpy.ndarray
        Point array with shape ``(N, 3)``.
    discrete : bool
        Whether rows identify voxel cells rather than continuous points.
    primary : bool, default True
        Whether rows own the product's row-aligned features and may therefore
        be used by crop, mask and other support-changing augmentations.
    units : {'px', 'cm'}, default 'px'
        Coordinate system used by :attr:`values`.
    """

    name: str
    values: np.ndarray
    discrete: bool
    primary: bool = True
    units: str = "px"


class SpatialAdapter(ABC):
    """Mutation contract between a data product and spatial augmenters.

    Parameters
    ----------
    key : str
        Event-dictionary key associated with the product.
    product : object
        Product whose spatial fields are exposed by the adapter.
    """

    def __init__(self, key: str, product: Any) -> None:
        self.key = key
        self.product = product

    @property
    @abstractmethod
    def fields(self) -> tuple[SpatialField, ...]:
        """Return all point fields carried by the product.

        Returns
        -------
        tuple[SpatialField, ...]
            Current spatial field descriptions.
        """

    @abstractmethod
    def set_field(self, name: str, values: np.ndarray) -> None:
        """Replace one point field.

        Parameters
        ----------
        name : str
            Product-local spatial field name.
        values : numpy.ndarray
            Replacement array with shape ``(N, 3)``.
        """

    @abstractmethod
    def select_rows(self, mask: np.ndarray) -> None:
        """Restrict rows aligned with the primary coordinate field.

        Parameters
        ----------
        mask : numpy.ndarray
            Boolean mask over primary rows.
        """

    @property
    def primary_fields(self) -> tuple[SpatialField, ...]:
        """Return fields which own row-aligned product features.

        Returns
        -------
        tuple[SpatialField, ...]
            Primary coordinate fields.
        """
        return tuple(field for field in self.fields if field.primary)

    @property
    def features(self) -> np.ndarray:
        """Return features aligned with the primary point rows.

        Returns
        -------
        numpy.ndarray
            Product feature array.
        """
        return np.asarray(self.product.features)

    def set_meta(self, meta: Meta) -> None:
        """Attach updated image metadata to the product.

        Parameters
        ----------
        meta : Meta
            Metadata describing the transformed image frame.
        """
        self.product.meta = meta

    def validate_row_selection(self, operation: str) -> None:
        """Require an unambiguous row-owning coordinate field.

        Parameters
        ----------
        operation : str
            Human-readable augmentation name used in error messages.

        Raises
        ------
        ValueError
            If zero or multiple coordinate groups claim ownership of rows.
        """
        if len(self.primary_fields) != 1:
            names = tuple(field.name for field in self.primary_fields)
            raise ValueError(
                f"{operation} cannot select rows of spatial product `{self.key}` "
                f"with primary coordinate groups {names}."
            )


class TensorSpatialAdapter(SpatialAdapter):
    """Spatial contract for a :class:`TensorData`.

    Every coordinate group declared by the tensor schema is exposed as an
    independent point field. All groups remain primary because they share the
    tensor's feature rows.
    """

    def __init__(self, key: str, product: TensorData) -> None:
        """Validate and initialize a tensor spatial adapter.

        Parameters
        ----------
        key : str
            Event-dictionary key associated with the tensor.
        product : TensorData
            Coordinate-bearing tensor product.

        Raises
        ------
        ValueError
            If coordinate groups are absent or are not three-dimensional.
        """
        super().__init__(key, product)
        groups = product.coordinate_groups
        if not groups:
            raise ValueError(
                f"TensorData `{key}` carries coordinates but declares no coordinate groups."
            )
        for name, columns in groups.items():
            if len(columns) != 3:
                raise ValueError(
                    f"Spatial coordinate group `{key}.{name}` must have exactly "
                    f"three columns, got {len(columns)}."
                )

    @property
    def fields(self) -> tuple[SpatialField, ...]:
        """Expose every schema coordinate group as a spatial field.

        Returns
        -------
        tuple[SpatialField, ...]
            Coordinate-group views with their declared sampling modes.
        """
        coords = self.product.coordinate_data
        assert coords is not None
        return tuple(
            SpatialField(
                name,
                coords[:, columns],
                self.product.schema.coordinate_modes[name] == "discrete",
            )
            for name, columns in self.product.coordinate_groups.items()
        )

    def set_field(self, name: str, values: np.ndarray) -> None:
        """Write transformed values into one coordinate group.

        Parameters
        ----------
        name : str
            Coordinate-group name from the tensor schema.
        values : numpy.ndarray
            Replacement point array with shape ``(N, 3)``.
        """
        coords = self.product.coordinate_data
        assert coords is not None
        columns = self.product.coordinate_groups[name]
        coords[:, columns] = values

    def select_rows(self, mask: np.ndarray) -> None:
        """Select coordinates and their row-aligned features together.

        Parameters
        ----------
        mask : numpy.ndarray
            Boolean row-selection mask.
        """
        coords = self.product.coordinate_data
        assert coords is not None
        self.product.coordinate_data = coords[mask]
        self.product.features = self.product.features[mask]


class ClusterSpatialAdapter(SpatialAdapter):
    """Spatial contract for compact cluster voxels and particle vertices.

    Cluster voxels own the packed feature rows and are always discrete, even
    though the packed table may use a floating storage dtype. Particle-table
    vertices are continuous auxiliary points with independent row ownership.
    """

    @property
    def fields(self) -> tuple[SpatialField, ...]:
        """Return cluster voxels and optional particle vertices.

        Returns
        -------
        tuple[SpatialField, ...]
            Primary voxel coordinates followed by auxiliary vertices, when
            particle information is available.
        """
        # Voxel semantics are structural and must not be inferred from the
        # floating dtype of the compact cluster-label table.
        fields = [
            SpatialField("voxels", np.asarray(self.product.coords), discrete=True)
        ]
        particles = self.product.particles
        if particles is not None and "vertex" in particles:
            fields.append(
                SpatialField(
                    "particle_vertices",
                    np.asarray(particles["vertex"]),
                    discrete=False,
                    primary=False,
                )
            )
        return tuple(fields)

    def set_field(self, name: str, values: np.ndarray) -> None:
        """Write transformed cluster voxels or particle vertices.

        Parameters
        ----------
        name : {'voxels', 'particle_vertices'}
            Cluster spatial field to replace.
        values : numpy.ndarray
            Replacement point array with shape ``(N, 3)``.

        Raises
        ------
        KeyError
            If the field name is not part of the cluster spatial contract.
        """
        if name == "voxels":
            self.product.data[:, self.product.coord_cols] = values
            return
        if name == "particle_vertices":
            assert self.product.particles is not None
            self.product.particles["vertex"] = values
            return
        raise KeyError(name)

    def select_rows(self, mask: np.ndarray) -> None:
        """Select complete packed cluster-label rows.

        Parameters
        ----------
        mask : numpy.ndarray
            Boolean mask over cluster voxel rows.
        """
        self.product.data = self.product.data[mask]


class ObjectListSpatialAdapter(SpatialAdapter):
    """Spatial contract for position fields stored on typed object lists.

    Position attributes are transformed as auxiliary points because object
    rows do not share the sparse row-selection semantics used by crop or mask.
    Extent-like vectors remain unsupported until their reflection and rotation
    rules are explicitly defined.
    """

    def __init__(self, key: str, product: ObjectListData) -> None:
        """Validate and initialize an object-list spatial adapter.

        Parameters
        ----------
        key : str
            Event-dictionary key associated with the object list.
        product : ObjectListData
            Typed list containing positional data objects.

        Raises
        ------
        NotImplementedError
            If an object carries extent or normalized-vector fields.
        ValueError
            If objects mix runtime types or coordinate units.
        """
        super().__init__(key, product)
        representative = product[0] if len(product) else product.default
        self.position_attrs = tuple(getattr(representative, "_pos_attrs", ()))
        normed_vectors = tuple(getattr(representative, "_normed_vec_attrs", ()))
        if normed_vectors:
            raise NotImplementedError(
                f"Spatial ObjectListData `{key}` ({type(representative).__name__}) "
                "contains extent/vector fields with undefined augmentation "
                f"semantics: {normed_vectors}."
            )
        units = getattr(representative, "units", None)
        if units not in ("px", "cm"):
            raise ValueError(
                f"Spatial ObjectListData `{key}` must declare `px` or `cm` units."
            )
        for item in product:
            if type(item) is not type(representative) or item.units != units:
                raise ValueError(
                    f"Spatial ObjectListData `{key}` must contain one object type "
                    "in one coordinate frame."
                )
        self.units = units

    @property
    def fields(self) -> tuple[SpatialField, ...]:
        """Collect each positional attribute into a point matrix.

        Returns
        -------
        tuple[SpatialField, ...]
            Auxiliary point fields, one per positional object attribute.
        """
        fields = []
        for name in self.position_attrs:
            if len(self.product):
                values = np.asarray([getattr(item, name) for item in self.product])
            else:
                # Preserve the declared width and dtype for empty typed lists.
                default = np.asarray(getattr(self.product.default, name))
                values = np.empty((0, 3), dtype=default.dtype)
            fields.append(
                SpatialField(
                    name,
                    values,
                    discrete=False,
                    primary=False,
                    units=self.units,
                )
            )
        return tuple(fields)

    def set_field(self, name: str, values: np.ndarray) -> None:
        """Scatter a transformed position matrix back to the objects.

        Parameters
        ----------
        name : str
            Positional object attribute.
        values : numpy.ndarray
            Replacement positions in object-list order.
        """
        for item, value in zip(self.product, values):
            setattr(item, name, value)

    def select_rows(self, mask: np.ndarray) -> None:
        """Reject sparse row selection for independent physics objects.

        Parameters
        ----------
        mask : numpy.ndarray
            Proposed row-selection mask, which cannot be interpreted for this
            product type.

        Raises
        ------
        ValueError
            Always, because object-list row ownership is undefined.
        """
        raise ValueError(
            f"Row selection is undefined for spatial object list `{self.key}`."
        )

    def set_meta(self, meta: Meta) -> None:
        """Leave object units unchanged after a frame transformation.

        Parameters
        ----------
        meta : Meta
            Updated image metadata. Objects do not store this metadata because
            their ``units`` attribute defines their coordinate representation.
        """
        # Objects retain their declared unit system rather than carrying image meta.
        return None


_NONSPATIAL_PRODUCTS = (IndexData, IndexListData, EdgeIndexData)


def discover_spatial_products(data: dict[str, Any]) -> dict[str, SpatialAdapter]:
    """Discover supported spatial products and reject ambiguous products.

    Parameters
    ----------
    data : dict
        Dictionary of event data products.

    Returns
    -------
    dict[str, SpatialAdapter]
        Spatial adapters keyed by their event-dictionary names.

    Raises
    ------
    NotImplementedError
        If a data product has not been explicitly classified by the spatial
        contract or contains unsupported spatial fields.
    ValueError
        If a supported product carries an invalid spatial declaration.
    """
    adapters: dict[str, SpatialAdapter] = {}
    for key, value in data.items():
        # Index products have topology but no physical or image coordinates.
        if isinstance(value, (Meta, *_NONSPATIAL_PRODUCTS)):
            continue
        if isinstance(value, TensorData):
            if value.coordinate_data is not None:
                adapters[key] = TensorSpatialAdapter(key, value)
            continue
        if isinstance(value, ClusterLabelData):
            adapters[key] = ClusterSpatialAdapter(key, value)
            continue
        if isinstance(value, ObjectListData):
            representative = value[0] if len(value) else value.default
            position_attrs = getattr(representative, "_pos_attrs", ())
            vector_attrs = getattr(representative, "_normed_vec_attrs", ())
            if position_attrs or vector_attrs:
                adapters[key] = ObjectListSpatialAdapter(key, value)
            continue
        if isinstance(value, DataProduct):
            raise NotImplementedError(
                f"Data product `{key}` ({type(value).__name__}) is not classified "
                "by the augmentation contract."
            )

    return adapters


def field_to_cm(field: SpatialField, meta: Meta) -> np.ndarray:
    """Express one point field in detector centimeters.

    Parameters
    ----------
    field : SpatialField
        Point field in its declared coordinate system.
    meta : Meta
        Image metadata used for pixel-to-centimeter conversion.

    Returns
    -------
    numpy.ndarray
        Point coordinates in detector centimeters.
    """
    if field.units == "cm":
        return field.values.copy()
    return meta.to_cm(field.values, center=field.discrete)


def field_from_cm(field: SpatialField, values: np.ndarray, meta: Meta) -> np.ndarray:
    """Express detector-centimeter points in a field's declared units.

    Parameters
    ----------
    field : SpatialField
        Field whose unit and sampling semantics should be restored.
    values : numpy.ndarray
        Point coordinates in detector centimeters.
    meta : Meta
        Output image metadata.

    Returns
    -------
    numpy.ndarray
        Coordinates in the field's declared units and storage dtype.
    """
    if field.units == "cm":
        return values.astype(field.values.dtype, copy=False)
    return meta.to_px(values, floor=field.discrete).astype(field.values.dtype)


def field_to_px(field: SpatialField, meta: Meta) -> np.ndarray:
    """Express one point field in image coordinates.

    Parameters
    ----------
    field : SpatialField
        Point field in its declared coordinate system.
    meta : Meta
        Image metadata used for centimeter-to-pixel conversion.

    Returns
    -------
    numpy.ndarray
        Discrete voxel indexes or continuous image coordinates.
    """
    if field.units == "px":
        return field.values.copy()
    return meta.to_px(field.values, floor=field.discrete)


def field_from_px(field: SpatialField, values: np.ndarray, meta: Meta) -> np.ndarray:
    """Express image points in a field's declared units.

    Parameters
    ----------
    field : SpatialField
        Field whose unit and sampling semantics should be restored.
    values : numpy.ndarray
        Discrete or continuous image coordinates.
    meta : Meta
        Output image metadata.

    Returns
    -------
    numpy.ndarray
        Coordinates in the field's declared units and storage dtype.
    """
    if field.units == "px":
        return values.astype(field.values.dtype, copy=False)
    return meta.to_cm(values, center=field.discrete).astype(field.values.dtype)
