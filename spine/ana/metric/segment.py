"""Analysis script used to evaluate the semantic segmentation accuracy."""

from warnings import warn

import numpy as np
from scipy.special import softmax

from spine.ana.base import AnaBase

from spine.utils.globals import SHAPE_COL, LOWES_SHP, GHOST_SHP

__all__ = ['SegmentAna']


class SegmentAna(AnaBase):
    """Class which computes and stores the necessary data to build a
    semantic segmentation confusion matrix.
    """

    # Name of the analysis script (as specified in the configuration)
    name = 'segment_eval'

    def __init__(self, summary=True, num_classes=GHOST_SHP, ghost=False,
                 use_fragments=False, use_particles=False,
                 label_key='seg_label', **kwargs):
        """Initialize the analysis script.

        Parameters
        ----------
        summary : bool, default True
            If `True`, summarize the confusion matrix for each entry. If
            `False`, store one row per pixel in the image (extremely memory
            intensive but gives details about pixel scores).
        num_classes : int, default 5
            Number of pixel classses, excluding the ghost class
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

        # Basic logic checks
        assert not use_fragments or not use_particles, (
                "Cannot use both fragments and particles.")
        assert not (use_fragments or use_particles) or summary, (
                "Cannot store detailed score information from fragments/particles.")
        assert not (use_fragments or use_particles) or not ghost, (
                "Cannot produce ghost metrics from fragments/particles.")
        
        # List the necessary data products
        keys = self.keys
        if not use_fragments and not use_particles:
            self.obj_source = None
            keys[label_key] = True
            keys['segmentation'] = True
            if ghost:
                self.num_classes += 1
                keys['ghost'] = True

        else:
            keys['points'] = True
            self.obj_type = 'fragments' if use_fragments else 'particles'
            for prefix in ['reco', 'truth']:
                keys[f'{prefix}_{self.obj_type}'] = True

        self.keys = keys

        # Initialize the output
        if summary:
            self.initialize_writer('summary')
        else:
            self.initialize_writer('pixel')

    def process(self, data):
        """Store the semantic segmentation metrics for one entry.

        Parameters
        ----------
        data : dict
            Dictionary of data products
        """
        # Fetch the list of labels and predictions for each pixel in the image
        if not self.use_fragments and not self.use_particles:
            # Get the label/predictions from the raw reconstruction output
            seg_label = data[self.label_key][:, SHAPE_COL].astype(np.int32)
            seg_pred = np.argmax(data['segmentation'], axis=1).astype(np.int32)
            if self.ghost:
                # If there are ghost, must combine the predictions
                full_seg_pred = np.full_like(seg_label, GHOST_SHP, dtype=np.int32)
                deghost_mask = data['ghost'][:, 0] > data['ghost'][:, 1]
                full_seg_pred[deghost_mask] = seg_pred
                seg_pred = full_seg_pred

            if not self.summary:
                # If requested, fetch the individual class softmax scores
                seg_scores = softmax(data['segmentation'], axis=1)
                if self.ghost:
                    # If there are ghosts, interpret the non-ghost score
                    # as a shared score for all other classes.
                    full_seg_scores = np.zeros(
                            (len(seg_scores), self.num_classes), dtype=np.int32)
                    ghost_scores = softmax(data['ghost'], axis=1)
                    full_seg_scores[:, :-1] = ghost_scores[: 0]/(self.num_classes - 1)
                    full_seg_scores[:, -1] = ghost_scores[:, 1]

                    full_seg_scores[deghost_mask, :-1] = seg_scores

        else:
            # Rebuild the labels/predictions from the fragment/particle objects
            seg_label = np.full(len(data['points']), LOWES_SHP, dtype=np.int32)
            for obj in data[f'truth_{self.obj_type}']:
                assert len(obj.index > 0), (
                        "The `index` of true fragments is not filled, indicating "
                        "that the original label tensor was modified. Cannot use "
                        "modified fragments to rebuild semantic labels.")
                seg_label[obj.index] = obj.shape

            seg_pred = np.full_like(seg_label, LOWES_SHP)
            for obj in data[f'reco_{self.obj_type}']:
                seg_pred[obj.index] = obj.shape

        # Store the information
        if not self.summary:
            # Store one row per pixel in the image, including pixel scores
            for i in range(len(seg_label)):
                row_dict = {'label': seg_label[i], 'pred': seg_pred[i]}
                for s in range(self.num_classes):
                    row_dict[f'score_{s}'] = seg_scores[i, s]

                self.append('pixel', **row_dict)

        else:
            # Store a summary of the confusion per entry (confusion matrix counts)
            count = np.histogram2d(
                seg_pred, seg_label, bins=[self.num_classes, self.num_classes],
                range=[[0, self.num_classes], [0, self.num_classes]])[0]

            row_dict = {}
            for i in range(self.num_classes):
                for j in range(self.num_classes):
                    row_dict[f'count_{i}{j}'] = int(count[i][j])

            self.append('summary', **row_dict)
