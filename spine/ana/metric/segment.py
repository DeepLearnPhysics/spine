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
    name = 'segment_eval'
    keys = {'index': True, 'run_info': False}

    def __init__(self, summary=True, num_classes=GHOST_SHP, ghost=False,
                 use_particles=False, label_key='seg_label', **kwargs):
        """Initialize the analysis script.

        Parameters
        ----------
        summary : bool, default True
            If `True`, summarize the confusion matrix for each entry. If
            `False`, store one row per pixel in the image (extremely memory
            intensive but gives details about pixel scores).
        ghost : bool, default False
            Evaluate deghosting performance
        use_particles : bool, default False
            If `True`, rebuild the segmentation for truth and reco
            from the shape of truth and reco particles. This method is exact,
            as long as there is no ghost points (particles do not retain
            the full input tensor which includes ghosts, for the sake of
            memory consumption in an output file).
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
        self.use_particles = use_particles
        self.label_key = label_key

        # Basic logic checks
        assert not self.use_particles or not self.summary, (
                "Cannot store detailed score information from particles.")
        assert not self.use_particles or not self.ghost, (
                "Cannot produce ghost metrics from particles.")
        
        # List the necessary data products, intialize writer
        if not self.use_particles:
            self.initialize_writer('summary')
            self.keys[label_key] = True
            self.keys['segmentation'] = True
            if ghost:
                self.num_classes += 1
                self.keys['ghost'] = True

        else:
            self.initialize_writer('pixel')
            self.keys['points']
            for prefix in ['reco', 'truth']:
                self.keys[f'{prefix}_particles'] = True

        # Initialize the output

    def process(self, data):
        """Store the semantic segmentation metrics for one entry.

        Parameters
        ----------
        data : dict
            Dictionary of data products
        """
        # Extract basic information to store in every row
        # TODO add file index + index within the file?
        base_dict = {'index': data['index']}
        if 'run_info' in data:
            base_dict.update(**data['run_info'].scalar_dict())
        else:
            warn("`run_info` is missing; will not be included in CSV file.")

        # Fetch the list of labels and predictions for each pixel in the image
        if not self.use_particles:
            # Rebuild the labels from the particle objects
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
            seg_label = np.full(len(data['points']), LOWES_SHP, dtype=np.int32)
            for part in data['truth_particles']:
                seg_pred[part.index] = part.shape

            seg_pred = np.full_like(seg_label, LOWES_SHP)
            for part in data['reco_particles']:
                seg_pred[part.index] = part.shape

        # Store the information
        if not self.summary:
            # Store one row per pixel in the image, including pixel scores
            for i in range(len(seg_label)):
                row_dict = {**base_dict, 'label': seg_label[i], 'pred': seg_pred[i]}
                for s in range(self.num_classes):
                    row_dict[f'score_{s}'] = seg_scores[i, s]

                self.writers['pixel'].append(row_dict)

        else:
            # Store a summary of the confusion per entry (confusion matrix counts)
            count = np.histogram2d(
                seg_pred, seg_label, bins=[self.num_classes, self.num_classes],
                range=[[0, self.num_classes], [0, self.num_classes]])[0]

            print(count)
            row_dict = {**base_dict}
            for i in range(self.num_classes):
                for j in range(self.num_classes):
                    row_dict[f'count_{i}{j}'] = int(count[i][j])

            self.writers['summary'].append(row_dict)
