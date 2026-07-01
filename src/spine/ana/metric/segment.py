"""Analysis script used to evaluate the semantic segmentation accuracy."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy.special import softmax

from spine.ana.base import AnaBase
from spine.constants import GHOST_SHP, LOWES_SHP, SHAPE_COL

__all__ = ["SegmentAna"]


class SegmentAna(AnaBase):
    """Compute semantic segmentation confusion summaries or per-pixel rows."""

    # Name of the analysis script (as specified in the configuration)
    name = "segment_eval"

    def __init__(
        self,
        summary: bool = True,
        num_classes: int = GHOST_SHP,
        ghost: bool = False,
        use_fragments: bool = False,
        use_particles: bool = False,
        label_key: str = "seg_label",
        **kwargs: Any,
    ) -> None:
        """Initialize the analysis script.

        Parameters
        ----------
        summary : bool, default True
            If `True`, summarize the confusion matrix for each entry. If
            `False`, store one row per pixel in the image (extremely memory
            intensive but gives details about pixel scores).
        num_classes : int, default 5
            Number of pixel classes, excluding the ghost class
        ghost : bool, default False
            Evaluate deghosting performance
        use_fragments : bool, default False
            If `True`, rebuild the segmentation for truth and reco from the
            shape of truth and reco fragments. This method is exact, as long as
            there is no ghost points and the cluster label tensor is untouched.
            If the label tensor is adapted, the original fragment boundaries are
            lost.
        use_particles : bool, default False
            If `True`, rebuild the segmentation for truth and reco from the
            shape of truth and reco particles. This method is imperfect, as the
            shape of showers is determined by the primary shape, which may not
            match the secondary fragment shapes in the original segmentation.
            This method is not compatible with ghost points.
        label_key : str, default 'seg_label'
            Name of the tensor which contains the segmentation labels
        **kwargs : dict, optional
            Additional arguments to pass to :class:`AnaBase`
        """
        # Initialize the parent class
        super().__init__(**kwargs)

        # Store the basic parameters
        self.summary = summary
        self.num_classes = num_classes
        self.ghost = ghost
        self.use_fragments = use_fragments
        self.use_particles = use_particles
        self.label_key = label_key
        self.object_collection: str | None = None

        # Basic logic checks
        if use_fragments and use_particles:
            raise ValueError("Cannot use both fragments and particles.")
        if (use_fragments or use_particles) and not summary:
            raise ValueError(
                "Cannot store detailed score information from fragments/particles."
            )
        if (use_fragments or use_particles) and ghost:
            raise ValueError("Cannot produce ghost metrics from fragments/particles.")

        # List the necessary data products
        keys = self.keys
        if not use_fragments and not use_particles:
            keys[label_key] = True
            keys["segmentation"] = True
            if ghost:
                self.num_classes += 1
                keys["ghost"] = True

        else:
            keys["points"] = True
            self.object_collection = "fragments" if use_fragments else "particles"
            for prefix in ["reco", "truth"]:
                keys[f"{prefix}_{self.object_collection}"] = True

        self.keys = keys

        # Initialize the output
        if summary:
            self.initialize_writer("summary")
        else:
            self.initialize_writer("pixel")

    def process(self, data: Mapping[str, Any]) -> None:
        """Store the semantic segmentation metrics for one entry.

        Parameters
        ----------
        data : dict
            Dictionary of data products
        """
        # Fetch the list of labels and predictions for each pixel in the image
        seg_scores = None
        if not self.use_fragments and not self.use_particles:
            # Get the label/predictions from the raw reconstruction output
            seg_label = data[self.label_key][:, SHAPE_COL].astype(np.int32)
            seg_pred = np.argmax(data["segmentation"], axis=1).astype(np.int32)
            deghost_mask = None
            if self.ghost:
                # If there are ghost, must combine the predictions
                full_seg_pred = np.full_like(seg_label, GHOST_SHP, dtype=np.int32)
                deghost_mask = np.argmax(data["ghost"], axis=1) == 0
                full_seg_pred[deghost_mask] = seg_pred[deghost_mask]
                seg_pred = full_seg_pred

            if not self.summary:
                # If requested, fetch the individual class softmax scores
                seg_scores = softmax(data["segmentation"], axis=1)
                if deghost_mask is not None:
                    # If there are ghosts, interpret the non-ghost score
                    # as a shared score for all other classes.
                    full_seg_scores = np.zeros(
                        (len(seg_scores), self.num_classes), dtype=seg_scores.dtype
                    )
                    ghost_scores = softmax(data["ghost"], axis=1)
                    full_seg_scores[:, :-1] = ghost_scores[:, 0, None] / (
                        self.num_classes - 1
                    )
                    full_seg_scores[:, -1] = ghost_scores[:, 1]

                    full_seg_scores[deghost_mask, :-1] = seg_scores[deghost_mask]
                    seg_scores = full_seg_scores

        else:
            if self.object_collection is None:
                raise ValueError("Object collection mode was not initialized.")
            # Rebuild the labels/predictions from the fragment/particle objects
            seg_label = np.full(len(data["points"]), LOWES_SHP, dtype=np.int32)
            for obj in data[f"truth_{self.object_collection}"]:
                if len(obj.index) == 0:
                    raise ValueError(
                        "The `index` of true fragments is not filled, indicating "
                        "that the original label tensor was modified. Cannot use "
                        "modified fragments to rebuild semantic labels."
                    )
                seg_label[obj.index] = obj.shape

            seg_pred = np.full_like(seg_label, LOWES_SHP)
            for obj in data[f"reco_{self.object_collection}"]:
                seg_pred[obj.index] = obj.shape

        # Store the information
        if not self.summary:
            # Store one row per pixel in the image, including pixel scores
            if seg_scores is None:
                raise ValueError("Segment scores not available for detailed storage.")
            for i, (seg_l, seg_p) in enumerate(zip(seg_label, seg_pred)):
                row_dict = {"label": seg_l, "pred": seg_p}
                for s in range(self.num_classes):
                    row_dict[f"score_{s}"] = seg_scores[i, s]

                self.append("pixel", **row_dict)

        else:
            # Store a summary of the confusion per entry (confusion matrix counts)
            count = np.histogram2d(
                seg_pred,
                seg_label,
                bins=[self.num_classes, self.num_classes],
                range=[[0, self.num_classes], [0, self.num_classes]],
            )[0]

            row_dict = {}
            for i in range(self.num_classes):
                for j in range(self.num_classes):
                    row_dict[f"count_{i}{j}"] = int(count[i][j])

            self.append("summary", **row_dict)
