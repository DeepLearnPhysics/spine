"""Analysis script used to evaluate the clustering accuracy."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

import spine.math.metrics
from spine.ana.base import AnaBase
from spine.constants import LOWES_SHP

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
        ("fragment", "cluster"),
        ("particle", "group"),
        ("interaction", "interaction"),
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
        time_window: Sequence[float] | None = None,
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
        time_window : Sequence[float], optional
            Truth-object time window in nanoseconds. Truth objects outside the
            inclusive window are excluded from the clustering evaluation.
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
        normalized_time_window: tuple[float, float] | None = None
        if time_window is not None:
            if not isinstance(time_window, Sequence) or len(time_window) != 2:
                raise ValueError(
                    "Time window must be specified as an array of two values."
                )
            if time_window[0] > time_window[1]:
                raise ValueError(
                    "Time window lower bound must not exceed its upper bound."
                )
            normalized_time_window = (time_window[0], time_window[1])
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
        self.time_window = normalized_time_window

        # Parse the label_col column, if necessary
        self.label_col = label_col

        # Identify the truth object type used to time-mask standalone labels
        self.time_obj_type: str | None = None
        if self.time_window is not None and not self.per_object:
            for name, column in self._label_cols:
                if self.label_col == column:
                    self.time_obj_type = name
                    break
            if self.time_obj_type is None:
                raise ValueError(
                    "Time filtering in standalone mode requires a fragment, "
                    "particle or interaction clustering label column."
                )

        # Convert metric strings to functions
        self.metrics: dict[str, Callable[..., float]] = {
            m: getattr(spine.math.metrics, m) for m in metrics
        }

        # If objects are not used, remove them from the required keys
        keys = self.keys
        if not use_objects:
            for key in self.obj_keys:
                del keys[key]

        # Time filtering always requires the corresponding truth objects
        if self.time_window is not None:
            time_obj_types = self.obj_type if self.per_object else [self.time_obj_type]
            for obj in time_obj_types:
                keys[f"truth_{obj}s"] = True

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
    def label_cols(self) -> dict[str, str]:
        """Dictionary of (key, column_id) pairs which determine which column
        in the label tensor corresponds to a specific clustering target.

        Returns
        -------
        dict[str, str]
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
            truth_shapes: NDArray[np.int32] | None = None

            # Build the cluster labels for this object type
            if not self.use_objects:
                # Fetch the right label column
                label_col = self.label_col or self.label_cols[obj_type]
                cluster_label = data[self.label_key]
                num_points = len(cluster_label)
                labels = cluster_label.voxel_field(label_col).astype(
                    np.int32,
                    copy=False,
                )
                if self.per_shape and obj_type != "interaction":
                    truth_shapes = cluster_label.shapes.astype(
                        np.int32,
                        copy=False,
                    )
                num_truth = len(np.unique(labels[labels > -1]))

            else:
                # Rebuild the labels
                num_points = len(data["points"])
                labels = np.full(num_points, -1, dtype=np.int32)
                if self.per_shape and obj_type != "interaction":
                    truth_shapes = np.full(num_points, LOWES_SHP, dtype=np.int32)
                num_truth = len(data[f"truth_{obj_type}s"])
                for i, obj in enumerate(data[f"truth_{obj_type}s"]):
                    index = self.get_index(obj)
                    labels[index] = i
                    if truth_shapes is not None:
                        truth_shapes[index] = obj.shape

            # Restrict the truth evaluation domain to the requested time window
            if self.time_window is not None:
                time_obj_type = obj_type if self.per_object else self.time_obj_type
                assert time_obj_type is not None
                truth_objects = data[f"truth_{time_obj_type}s"]
                valid_mask = np.zeros(num_points, dtype=bool)
                lower, upper = self.time_window
                for obj in truth_objects:
                    if lower <= obj.time <= upper:
                        valid_mask[self.get_index(obj)] = True

                # Raw labels may be a view into the input label tensor
                labels = labels.copy()
                labels[~valid_mask] = -1
                num_truth = len(np.unique(labels[labels > -1]))

            # Build the cluster predictions for this object type
            preds = np.full(num_points, -1, dtype=np.int32)
            reco_shapes: NDArray[np.int32] | None = None
            if self.per_object:
                if self.per_shape and obj_type != "interaction":
                    reco_shapes = np.full(num_points, LOWES_SHP, dtype=np.int32)
                if not self.use_objects:
                    # Use clusters directly from the full chain output
                    num_reco = len(data[f"{obj_type}_clusts"])
                    for i, index in enumerate(data[f"{obj_type}_clusts"]):
                        preds[index] = i
                        if reco_shapes is not None:
                            reco_shapes[index] = data[f"{obj_type}_shapes"][i]

                else:
                    # Use clusters from the object indexes
                    num_reco = len(data[f"reco_{obj_type}s"])
                    for i, obj in enumerate(data[f"reco_{obj_type}s"]):
                        preds[obj.index] = i
                        if reco_shapes is not None:
                            reco_shapes[obj.index] = obj.shape

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
            truth_mask = labels > -1
            reco_mask = preds > -1
            for metric, func in self.metrics.items():
                self._record_metric(
                    row_dict,
                    metric,
                    func,
                    labels,
                    preds,
                    truth_mask,
                    reco_mask,
                )
                if self.per_shape and obj_type != "interaction":
                    assert truth_shapes is not None
                    for shape in range(LOWES_SHP):
                        # Evaluate each truth class without allowing missing
                        # predictions to masquerade as a cluster labeled -1.
                        shape_reco_mask = reco_mask & (truth_shapes == shape)
                        if reco_shapes is not None:
                            shape_reco_mask = reco_mask & (reco_shapes == shape)
                        self._record_metric(
                            row_dict,
                            f"{metric}_{shape}",
                            func,
                            labels,
                            preds,
                            truth_mask & (truth_shapes == shape),
                            shape_reco_mask,
                        )

            self.append(obj_type, **row_dict)

    @staticmethod
    def _record_metric(
        row_dict: dict[str, int | float],
        name: str,
        func: Callable[..., float],
        labels: NDArray[np.int32],
        preds: NDArray[np.int32],
        truth_mask: NDArray[np.bool_],
        reco_mask: NDArray[np.bool_],
    ) -> None:
        """Evaluate one metric and record the support used to define it.

        Truth and reconstruction counts describe the available point-level
        assignments independently. The comparable count is their overlap and
        is the population passed to the metric. This makes an absent truth
        class distinguishable from a class that exists but was not rebuilt.

        Parameters
        ----------
        row_dict : dict
            Analyzer row to update.
        name : str
            Output metric name, including a semantic suffix when applicable.
        func : callable
            Clustering metric evaluated on comparable assignments.
        labels, preds : np.ndarray
            Truth and reconstructed cluster assignments.
        truth_mask, reco_mask : np.ndarray
            Point masks defining the truth and reconstruction populations.
        """
        comparable_mask = truth_mask & (preds > -1)
        value = float(func(labels[comparable_mask], preds[comparable_mask]))

        row_dict[name] = value
        row_dict[f"{name}_valid"] = int(np.isfinite(value))
        row_dict[f"{name}_num_truth_points"] = int(np.count_nonzero(truth_mask))
        row_dict[f"{name}_num_reco_points"] = int(np.count_nonzero(reco_mask))
        row_dict[f"{name}_num_comparable_points"] = int(
            np.count_nonzero(comparable_mask)
        )
