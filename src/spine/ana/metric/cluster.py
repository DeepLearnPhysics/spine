"""Analysis script used to evaluate the clustering accuracy."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

import spine.utils.metrics
from spine.ana.base import AnaBase
from spine.constants import CLUST_COL, GROUP_COL, INTER_COL, LOWES_SHP, SHAPE_COL
from spine.constants.factory import enum_factory

__all__ = ["ClusterAna"]


class ClusterAna(AnaBase):
    """Compute clustering metrics at different aggregation stages:

    - fragments
    - particles
    - interactions
    """

    # Name of the analysis script (as specified in the configuration)
    name = "cluster_eval"

    # Label column to use for each clustering label_col
    _label_cols = (
        ("fragment", CLUST_COL),
        ("particle", GROUP_COL),
        ("interaction", INTER_COL),
    )

    def __init__(
        self,
        obj_type: str | Sequence[str] | None = None,
        use_objects: bool = False,
        per_object: bool = True,
        per_shape: bool = True,
        metrics: Sequence[str] = ("pur", "eff", "ari"),
        label_key: str = "clust_label_adapt",
        label_col: str | None = None,
        truth_index_mode: str = "index_adapt",
        **kwargs: Any,
    ) -> None:
        """Initialize the analysis script.

        Parameters
        ----------
        obj_type : str or Sequence[str], optional
            Name or list of names of the object types to process
        use_objects : bool, default False
            If `True`, rebuild the clustering assignments for truth and reco
            from the set of truth and reco particles
        per_object : bool, default True
            Evaluate the clustering accuracy for each object type (not relevant
            if running GrapPA standalone)
        per_shape : bool, default True
            Evaluate the clustering accuracy for each object shape (not
            relevant in the case of interactions)
        metrics : Sequence[str], default ('pur', 'eff', 'ari')
            List of clustering metrics to evaluate
        label_key : str, default 'clust_label_adapt'
            Name of the tensor which contains the cluster labels, when
            using the raw reconstruction output
        label_col : str, optional
            Column name in the label tensor specifying the aggregation label_col
        truth_index_mode : str, default 'index_adapt'
            Name of the truth object index attribute to use when rebuilding
            truth labels from objects
        **kwargs : dict, optional
            Additional arguments to pass to :class:`AnaBase`
        """
        # Check parameters
        if obj_type is None and per_object:
            raise ValueError(
                "If evaluating clustering metrics per object, provide a list "
                "of object types to evaluate the clustering for."
            )
        if not per_object and label_col is None:
            raise ValueError(
                "If evaluating clustering standalone (not per object), must "
                "provide the name of the target clustering label column."
            )
        if not per_object and use_objects:
            raise ValueError(
                "If evaluating clustering standalone (not per object), cannot "
                "use objects to evaluate it."
            )
        standalone_label_col = label_col if not per_object else None

        # Initialize the parent class
        super().__init__(
            obj_type=obj_type,
            run_mode="both",
            truth_index_mode=truth_index_mode,
            **kwargs,
        )

        # If the clustering is not done per object, fix target
        if standalone_label_col is not None:
            self.obj_type = [standalone_label_col]

        # Store the basic parameters
        self.use_objects = use_objects
        self.per_object = per_object
        self.per_shape = per_shape
        self.label_key = label_key

        # Parse the label_col column, if necessary
        self.label_col: int | None = (
            enum_factory("cluster", label_col) if label_col is not None else None
        )

        # Convert metric strings to functions
        self.metrics: dict[str, Callable[..., float]] = {
            m: getattr(spine.utils.metrics, m) for m in metrics
        }

        # If objects are not used, remove them from the required keys
        keys = self.keys
        if not use_objects:
            for key in self.obj_keys:
                del keys[key]

        # List other necessary data products
        if self.per_object:
            if not self.use_objects:
                # Store the labels and the clusters output by the reco chain
                keys[label_key] = True
                for obj in self.obj_type:
                    keys[f"{obj}_clusts"] = True
                    if obj != "interaction":
                        keys[f"{obj}_shapes"] = True

            else:
                keys["points"] = True

        else:
            keys[label_key] = True
            keys["clusts"] = True
            keys["group_pred"] = True

        self.keys = keys

        # Initialize the output
        for obj in self.obj_type:
            self.initialize_writer(obj)

    @property
    def label_cols(self) -> dict[str, int]:
        """Dictionary of (key, column_id) pairs which determine which column
        in the label tensor corresponds to a specific clustering target.

        Returns
        -------
        dict[str, int]
            Dictionary of (key, column_id) mapping from name to label column
        """
        return dict(self._label_cols)

    def process(self, data: Mapping[str, Any]) -> None:
        """Store the clustering metrics for one entry.

        Parameters
        ----------
        data : dict
            Dictionary of data products
        """
        # Loop over the different object types
        for obj_type in self.obj_type:
            shapes: NDArray[np.int32] | None = None

            # Build the cluster labels for this object type
            if not self.use_objects:
                # Fetch the right label column
                label_col = self.label_col or self.label_cols[obj_type]
                num_points = len(data[self.label_key])
                labels = data[self.label_key][:, label_col].astype(
                    np.int32,
                    copy=False,
                )
                if obj_type != "interaction":
                    shapes = data[self.label_key][:, SHAPE_COL].astype(
                        np.int32,
                        copy=False,
                    )
                num_truth = len(np.unique(labels[labels > -1]))

            else:
                # Rebuild the labels
                num_points = len(data["points"])
                labels = np.full(num_points, -1, dtype=np.int32)
                num_truth = len(data[f"truth_{obj_type}s"])
                for i, obj in enumerate(data[f"truth_{obj_type}s"]):
                    labels[self.get_index(obj)] = i

            # Build the cluster predictions for this object type
            preds = np.full(num_points, -1, dtype=np.int32)
            if self.per_object:
                pred_shapes = np.full(num_points, -LOWES_SHP, dtype=np.int32)
                shapes = pred_shapes
                if not self.use_objects:
                    # Use clusters directly from the full chain output
                    num_reco = len(data[f"{obj_type}_clusts"])
                    for i, index in enumerate(data[f"{obj_type}_clusts"]):
                        preds[index] = i
                        if obj_type != "interaction":
                            pred_shapes[index] = data[f"{obj_type}_shapes"][i]

                else:
                    # Use clusters from the object indexes
                    num_reco = len(data[f"reco_{obj_type}s"])
                    for i, obj in enumerate(data[f"reco_{obj_type}s"]):
                        preds[obj.index] = i
                        if obj_type != "interaction":
                            pred_shapes[obj.index] = obj.shape

            else:
                num_reco = len(data["clusts"])
                for i, index in enumerate(data["clusts"]):
                    preds[index] = int(data["group_pred"][i])

            # Evaluate clustering metrics
            row_dict: dict[str, int | float] = {
                "num_points": num_points,
                "num_truth": num_truth,
                "num_reco": num_reco,
            }
            for metric, func in self.metrics.items():
                valid_index = np.where((preds > -1) & (labels > -1))[0]
                row_dict[metric] = func(labels[valid_index], preds[valid_index])
                if self.per_shape and obj_type != "interaction":
                    assert shapes is not None
                    for shape in range(LOWES_SHP):
                        shape_index = np.where((shapes == shape) & (labels > -1))[0]
                        row_dict[f"{metric}_{shape}"] = func(
                            labels[shape_index], preds[shape_index]
                        )

            self.append(obj_type, **row_dict)
