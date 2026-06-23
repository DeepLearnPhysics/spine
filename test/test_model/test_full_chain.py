import torch

from spine.constants import GHOST_SHP, SHAPE_COL
from spine.data import IndexBatch, TensorBatch
from spine.model.full_chain import FullChainLoss


class RecordingLoss:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "loss": torch.tensor(0.0),
            "accuracy": 1.0,
        }


def test_full_chain_loss_uses_orig_index_to_align_cached_segmentation_labels():
    """Cached deghosted segmentation inputs should be aligned via orig_index."""
    chain = {
        "deghosting": None,
        "charge_rescaling": None,
        "segmentation": "uresnet",
        "point_proposal": None,
        "fragmentation": None,
        "shower_aggregation": None,
        "shower_primary": None,
        "track_aggregation": None,
        "particle_aggregation": None,
        "inter_aggregation": None,
        "particle_identification": None,
        "primary_identification": None,
        "orientation_identification": None,
        "calibration": None,
    }
    loss = FullChainLoss(chain=chain)
    recorder = RecordingLoss()
    loss.uresnet_loss = recorder

    seg_label_tensor = torch.zeros((5, 2), dtype=torch.float32)
    seg_label_tensor[:, 0] = 0
    seg_label_tensor[:, SHAPE_COL] = torch.tensor([0, GHOST_SHP, 1, 2, GHOST_SHP])
    seg_label = TensorBatch(seg_label_tensor, batch_size=1, has_batch_col=True)

    segmentation = TensorBatch(
        torch.tensor(
            [
                [2.0, 0.1, 0.0],
                [0.1, 3.0, 0.0],
                [0.2, 0.3, 4.0],
            ],
            dtype=torch.float32,
        ),
        counts=torch.tensor([3]),
    )
    orig_index = IndexBatch(
        torch.tensor([0, 2, 4], dtype=torch.long),
        spans=torch.tensor([0], dtype=torch.long),
        counts=torch.tensor([3], dtype=torch.long),
    )

    loss(seg_label=seg_label, segmentation=segmentation, orig_index=orig_index)

    assert len(recorder.calls) == 1
    seg_label_used = recorder.calls[0]["seg_label"]
    segmentation_used = recorder.calls[0]["segmentation"]

    assert seg_label_used.counts.tolist() == [2]
    assert segmentation_used.counts.tolist() == [2]
    assert seg_label_used.tensor.shape[0] == 2
    assert segmentation_used.tensor.shape[0] == 2
    assert torch.equal(seg_label_used.tensor[:, SHAPE_COL], torch.tensor([0.0, 1.0]))
    assert torch.equal(segmentation_used.tensor, segmentation.tensor[:2])
