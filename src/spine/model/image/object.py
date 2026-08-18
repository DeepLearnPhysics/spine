"""Construction of whole-image and object-level image samples."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from spine.cluster.formation import form_clusters_batch
from spine.constants import LOWES_SHP
from spine.constants.factory import enum_factory
from spine.data import ClusterLabelBatch, IndexBatch, TensorBatch

__all__ = ["ImageObjectBuilder"]


class ImageObjectBuilder:
    """Build the voxel indexes represented by each image-model sample.

    Samples may represent complete batch entries or objects identified by any
    cluster-label column recognized by :func:`enum_factory`. Explicit indexes
    always take precedence, allowing inference to consume reconstructed
    objects without consulting truth labels.
    """

    def __init__(
        self,
        source: str | int = "image",
        shapes: Sequence[str | int] | None = None,
        min_size: int = -1,
    ) -> None:
        """Initialize the object construction policy.

        Parameters
        ----------
        source : str, default "image"
            ``"image"`` creates one sample per batch entry. ``"explicit"``
            requires indexes to be passed to :meth:`__call__`. Other strings
            select a cluster-label column such as ``cluster``, ``group`` or
            ``ancestor``.
        shapes : sequence of str or int, optional
            Semantic shapes retained when constructing labeled objects.
            Defaults to every class below ``lowe``, matching GrapPA.
        min_size : int, default -1
            Minimum number of voxels in a labeled object.

        Raises
        ------
        ValueError
            If the source, shape list, or minimum size is invalid.
        """
        # Validate the common object selection parameters
        if min_size < -1:
            raise ValueError(f"`min_size` must be at least -1, got {min_size}.")

        # Resolve label-based sources while retaining direct source names.
        if not isinstance(source, str):
            raise TypeError("Image-object `source` must be a named field.")
        aliases = {"clust": "cluster", "part": "particle", "inter": "interaction"}
        self.source_name = aliases.get(source, source)
        direct_source = self.source_name in {"image", "explicit"}
        self.source = None if direct_source else self.source_name

        # Normalize the optional semantic-shape selection
        if shapes is None:
            self.shapes: list[int] | None = (
                None if direct_source else list(range(LOWES_SHP))
            )
        else:
            if isinstance(shapes, (str, bytes)) or np.isscalar(shapes):
                raise ValueError("Semantic shapes must be provided as a sequence.")
            self.shapes = [
                enum_factory("shape", shape) if isinstance(shape, str) else int(shape)
                for shape in shapes
            ]
        if direct_source and self.shapes is not None:
            raise ValueError(
                f"`source: {self.source_name}` samples cannot be filtered by shape."
            )

        self.min_size = min_size

    @staticmethod
    def _validate_objects(data: TensorBatch, objects: IndexBatch) -> None:
        """Check that explicit indexes address the supplied input batch."""
        # Validate the high-level batching contract
        if not objects.is_list:
            raise TypeError("Image objects must be an IndexBatch of index lists.")
        if objects.batch_size != data.batch_size:
            raise ValueError("Image-object and input batch sizes must match.")

        spans = objects.to_numpy().spans
        counts = data.to_numpy().counts
        if not np.array_equal(spans, counts):
            raise ValueError("Image-object spans must match the input data counts.")

        # Check that every object stays within its owning event span
        objects_numpy = objects.to_numpy()
        edges = np.concatenate(([0], np.cumsum(counts)))
        for batch_id in range(data.batch_size):
            lower, upper = objects_numpy.edges[batch_id : batch_id + 2]
            for index in objects_numpy.index_list[lower:upper]:
                if len(index) == 0:
                    raise ValueError("Image objects cannot contain empty indexes.")
                if np.any(index < edges[batch_id]) or np.any(
                    index >= edges[batch_id + 1]
                ):
                    raise IndexError(
                        "An image-object index lies outside its owning batch entry."
                    )

    @staticmethod
    def _whole_images(data: TensorBatch) -> IndexBatch:
        """Create one full-span object for each nonempty batch entry."""
        # Convert each nonempty event span into one explicit object index
        counts = data.to_numpy().counts
        edges = np.concatenate(([0], np.cumsum(counts)))
        indexes = [
            np.arange(edges[batch_id], edges[batch_id + 1], dtype=np.int64)
            for batch_id in range(data.batch_size)
            if counts[batch_id] > 0
        ]
        object_counts = (counts > 0).astype(np.int64)
        single_counts = np.asarray([len(index) for index in indexes], dtype=np.int64)
        return IndexBatch(indexes, counts, object_counts, single_counts)

    def __call__(
        self,
        data: TensorBatch,
        objects: IndexBatch | None = None,
        object_data: ClusterLabelBatch | None = None,
    ) -> IndexBatch:
        """Construct or validate image-object indexes.

        Parameters
        ----------
        data : TensorBatch
            Sparse coordinate-feature input to be encoded.
        objects : IndexBatch, optional
            Explicit object indexes. These override the configured source.
        object_data : TensorBatch, optional
            Label-rich voxel data used only to construct objects. It must be
            aligned with ``data`` and is unnecessary for whole-image or
            explicit-index operation.

        Returns
        -------
        IndexBatch
            One index list per image-model sample, grouped by original event.
        """
        # Explicit reconstructed objects always override configured construction
        if objects is not None:
            self._validate_objects(data, objects)
            return objects

        # Handle the two direct construction policies
        if self.source_name == "explicit":
            raise ValueError("`source: explicit` requires `objects` input.")
        if self.source_name == "image":
            return self._whole_images(data)

        # Validate the label-rich input required for truth-based construction
        if object_data is None:
            raise ValueError(
                f"`source: {self.source_name}` requires `object_data` or explicit "
                "`objects`."
            )
        if object_data.batch_size != data.batch_size or not np.array_equal(
            object_data.to_numpy().counts,
            data.to_numpy().counts,
        ):
            raise ValueError("`object_data` must be voxel-aligned with `data`.")

        if self.source is None:  # Constructor narrowing for static analysis.
            raise RuntimeError("A labeled object source was not initialized.")

        # Delegate labeled clustering to the GrapPA-compatible implementation
        return form_clusters_batch(
            object_data.to_numpy(),
            self.min_size,
            self.source,
            self.shapes,
        )
