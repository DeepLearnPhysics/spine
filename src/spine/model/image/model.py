"""Generic whole-image and object-image prediction model."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from spine.data import IndexBatch, TensorBatch

from ..common.factories import final_factory
from ..registry import ModelSpec
from .encoder import image_encoder_factory
from .loss import ImageLoss
from .object import ImageObjectBuilder

__all__ = ["ImageModel", "ImageLoss", "MODEL_SPEC"]


class ImageModel(torch.nn.Module):
    """Encode complete images or reconstructed objects and predict tasks.

    Object construction, feature encoding, and prediction heads are
    independent configuration layers. The same encoder and heads can therefore
    operate on whole events, truth-defined clusters during standalone
    training, or explicit reconstructed indexes supplied by the full chain.
    """

    def __init__(self, image: dict[str, Any]) -> None:
        """Initialize the modular image model.

        Parameters
        ----------
        image : dict
            Configuration containing ``objects``, ``encoder`` and one or more
            named ``heads``. Integer head values create linear heads with that
            output width; mappings are forwarded to ``final_factory``.
        """
        # Initialize the parent class
        super().__init__()

        # Extract the independently configurable model components
        config = dict(image)
        object_config = config.pop("objects", {})
        try:
            encoder_config = config.pop("encoder")
        except KeyError as err:
            raise ValueError("Image model configuration requires `encoder`.") from err
        try:
            head_configs = config.pop("heads")
        except KeyError as err:
            raise ValueError("Image model configuration requires `heads`.") from err
        self.return_features = bool(config.pop("return_features", False))

        # Validate the top-level configuration before constructing modules
        if config:
            unknown = ", ".join(sorted(config))
            raise ValueError(f"Unknown image model options: {unknown}.")
        if not isinstance(object_config, Mapping):
            raise TypeError("Image `objects` configuration must be a mapping.")
        if not isinstance(encoder_config, Mapping):
            raise TypeError("Image `encoder` configuration must be a mapping.")
        if not isinstance(head_configs, Mapping) or not head_configs:
            raise ValueError("Image `heads` must be a nonempty mapping.")

        # Initialize the object construction and shared encoder stages
        self.object_builder = ImageObjectBuilder(**dict(object_config))
        self.encoder = image_encoder_factory(dict(encoder_config))

        # Initialize one independently configured prediction head per task
        self.heads = torch.nn.ModuleDict()
        self.head_sizes: dict[str, int] = {}
        for label, head_config in head_configs.items():
            if not isinstance(label, str) or not label:
                raise ValueError("Image head names must be nonempty strings.")
            if label in {"objects", "features", "loss", "accuracy"}:
                raise ValueError(f"Image head name `{label}` is reserved.")

            if isinstance(head_config, int):
                if head_config < 1:
                    raise ValueError("Image head widths must be positive.")
                output_size = head_config
                head_cfg: dict[str, Any] = {
                    "name": "linear",
                    "out_channels": output_size,
                }
            elif isinstance(head_config, Mapping):
                head_cfg = dict(head_config)
                head_cfg.setdefault("name", "linear")
                try:
                    output_size = int(head_cfg["out_channels"])
                except KeyError as err:
                    raise ValueError(
                        f"Image head `{label}` requires `out_channels`."
                    ) from err
                if output_size < 1:
                    raise ValueError("Image head widths must be positive.")
            else:
                raise TypeError(f"Image head `{label}` must be an integer or mapping.")

            self.heads[label] = final_factory(
                self.encoder.feature_size,
                **head_cfg,
            )
            self.head_sizes[label] = output_size

    @staticmethod
    def _objectize(data: TensorBatch, objects: IndexBatch) -> TensorBatch:
        """Rebatch selected voxels so every object becomes one encoder entry."""
        # Normalize the index representation and handle an empty object batch
        data_tensor = data.torch_tensor()
        objects_numpy = objects.to_numpy()
        if len(objects_numpy.index_list) == 0:
            empty = data_tensor.new_empty((0, data_tensor.shape[1]))
            return TensorBatch(empty, counts=np.empty(0, dtype=np.int64))

        # Gather all selected voxels in object order
        full_index = np.concatenate(objects_numpy.index_list)
        index = torch.as_tensor(full_index, dtype=torch.long, device=data_tensor.device)
        object_data = data_tensor[index].clone()

        # Replace event IDs with contiguous object IDs for the shared encoder
        object_ids = torch.repeat_interleave(
            torch.arange(len(objects_numpy.index_list), device=data_tensor.device),
            torch.as_tensor(
                objects_numpy.single_counts,
                dtype=torch.long,
                device=data_tensor.device,
            ),
        )
        object_data[:, 0] = object_ids.to(object_data.dtype)
        return TensorBatch(object_data, objects_numpy.single_counts)

    def forward(
        self,
        data: TensorBatch,
        objects: IndexBatch | None = None,
        object_data: TensorBatch | None = None,
    ) -> dict[str, Any]:
        """Encode image objects and evaluate every configured head.

        Parameters
        ----------
        data : TensorBatch
            Sparse coordinate-feature rows used by the encoder.
        objects : IndexBatch, optional
            Explicit object indexes. When present, these override configured
            label-based object construction.
        object_data : TensorBatch, optional
            Voxel-aligned label data used only to construct objects.

        Returns
        -------
        dict
            ``objects`` plus one ``<head>_pred`` TensorBatch per prediction
            head, and optionally the shared encoded ``features``.
        """
        # Construct samples and rebatch their voxels as independent images
        objects = self.object_builder(data, objects, object_data)
        objectized_data = self._objectize(data, objects)

        # Encode every object once, sharing the representation across tasks
        num_objects = len(objects.index_list)
        if num_objects:
            encoded = self.encoder(objectized_data)
            if encoded.shape != (num_objects, self.encoder.feature_size):
                raise RuntimeError(
                    "Image encoder returned shape "
                    f"{tuple(encoded.shape)}; expected "
                    f"({num_objects}, {self.encoder.feature_size})."
                )
        else:
            data_tensor = data.torch_tensor()
            encoded = data_tensor.new_empty((0, self.encoder.feature_size))

        # Preserve original event ownership on all feature and head outputs
        features = TensorBatch(encoded, objects.counts)
        result: dict[str, Any] = {"objects": objects}
        if self.return_features:
            result["features"] = features
        for label, head in self.heads.items():
            result[f"{label}_pred"] = head(features)
        return result


MODEL_SPEC = ModelSpec("image", ImageModel, ImageLoss)
