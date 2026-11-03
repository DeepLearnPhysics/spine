"""Fragment-level semantic prediction and supervision for shared UResNet models."""

from __future__ import annotations

import torch

from spine.data import IndexBatch, TensorBatch

__all__ = ["FragmentSemanticHead", "FragmentSemanticLoss", "resolve_semantics"]


def resolve_semantics(coupling: str, fragment_semantics: str | None) -> str:
    """Resolve the final semantic assignment policy for one coupling mode.

    Parameters
    ----------
    coupling : {'hard', 'independent', 'fragment_first', 'soft'}
        Relationship between preliminary semantics and graph construction.
    fragment_semantics : {'routed', 'pooled', 'learned'}, optional
        Explicit final fragment classification policy. If omitted, hard routing
        uses its routed classes, fragment-first uses a learned classifier, and
        the independent and soft modes pool voxel probabilities.

    Returns
    -------
    str
        Validated fragment semantic policy.

    Raises
    ------
    ValueError
        If a mode is unknown or the selected policies are incompatible.
    """
    # Infer a coherent default while allowing controlled classifier comparisons
    defaults = {
        "hard": "routed",
        "independent": "pooled",
        "fragment_first": "learned",
        "soft": "pooled",
    }
    if coupling not in defaults:
        raise ValueError(
            f"Unknown semantic_coupling {coupling!r}; choose {tuple(defaults)}."
        )
    policy = defaults[coupling] if fragment_semantics is None else fragment_semantics

    # Routed labels exist only when the graph was partitioned by semantic class
    if policy not in {"routed", "pooled", "learned"}:
        raise ValueError(f"Unknown fragment_semantics {policy!r}.")
    if policy == "routed" and coupling != "hard":
        raise ValueError("Routed fragment semantics require hard semantic coupling.")
    if coupling == "fragment_first" and policy != "learned":
        raise ValueError("Fragment-first coupling requires learned fragment semantics.")

    return policy


def pool_features(features: torch.Tensor, fragments: IndexBatch) -> torch.Tensor:
    """Mean-pool voxel features over a disjoint fragment partition.

    Parameters
    ----------
    features : torch.Tensor
        (N, F) Voxel features in the original input row domain.
    fragments : IndexBatch
        Jagged fragment indexes in that same domain, grouped by event.

    Returns
    -------
    torch.Tensor
        (K, F) Mean fragment features. The empty result remains connected to
        the input computation graph.
    """
    # Construct one destination row per member voxel without padding fragments
    counts = torch.as_tensor(fragments.single_counts, device=features.device)
    indexes = torch.as_tensor(fragments.full_index, device=features.device).long()
    owners = torch.repeat_interleave(
        torch.arange(len(counts), device=features.device), counts
    )

    # Sum and normalize while retaining autograd through the selected features
    pooled = features.new_zeros((len(counts), features.shape[1]))
    pooled = pooled.index_add(0, owners, features[indexes])
    return pooled / counts.clamp_min(1).unsqueeze(1)


class FragmentSemanticHead(torch.nn.Module):
    """Assign one class to each predicted fragment.

    The pooled policy averages voxel class probabilities. The learned policy
    averages shared decoder features and applies a fragment classifier. Both
    operate on discrete predicted fragments; gradients reach the pooled values
    but do not propagate through the connected-component assignment.
    """

    def __init__(self, num_features: int, num_classes: int, mode: str) -> None:
        """Initialize fragment classification.

        Parameters
        ----------
        num_features : int
            Number of shared decoder feature channels.
        num_classes : int
            Number of output semantic classes.
        mode : {'pooled', 'learned'}
            Source of the fragment-level semantic evidence.
        """
        # Initialize the parent class and record the final classification policy
        super().__init__()
        if mode not in {"pooled", "learned"}:
            raise ValueError(
                "FragmentSemanticHead supports pooled or learned semantics."
            )
        self.mode = mode
        self.num_classes = num_classes

        # Only learned semantics introduce parameters beyond the shared backbone
        self.classifier = (
            torch.nn.Linear(num_features, num_classes) if mode == "learned" else None
        )

    def forward(
        self,
        features: TensorBatch,
        segmentation: TensorBatch,
        fragments: IndexBatch,
    ) -> TensorBatch:
        """Classify every candidate fragment before output-class filtering.

        Parameters
        ----------
        features : TensorBatch
            (N, F) Shared decoder features in the original voxel domain.
        segmentation : TensorBatch
            (N, C) Preliminary voxel semantic logits.
        fragments : IndexBatch
            (K) Candidate fragments, including classes excluded from final output.

        Returns
        -------
        TensorBatch
            (K, C) Fragment logits. For pooled semantics these are logarithms
            of mean probabilities, with the same softmax interpretation.
        """
        # Pool features or probabilities according to the configured policy
        if self.classifier is not None:
            pooled = pool_features(features.torch_tensor(), fragments)
            logits = self.classifier(pooled)
        else:
            probabilities = torch.softmax(segmentation.torch_tensor(), dim=1)
            pooled = pool_features(probabilities, fragments)
            logits = torch.log(pooled.clamp_min(torch.finfo(pooled.dtype).tiny))

        return TensorBatch(logits, fragments.counts)


class FragmentSemanticLoss(torch.nn.Module):
    """Supervise learned fragment classes using their truth-class composition.

    Predicted fragments can contain multiple truth classes, particularly early
    in training. Their target is the class distribution of valid member voxels,
    rather than an arbitrary single winning label. This exposes mixed fragments
    through a purity metric without allowing their classification objective to
    claim that the discrete fragment partition has been corrected.
    """

    def forward(
        self,
        seg_label: TensorBatch,
        fragment_candidates: IndexBatch,
        fragment_logits: TensorBatch,
    ) -> dict[str, torch.Tensor | float | int]:
        """Compute mean soft-target cross entropy over supervised fragments.

        Parameters
        ----------
        seg_label : TensorBatch
            Original voxel semantic truth, aligned to the shared input domain.
        fragment_candidates : IndexBatch
            Predicted candidate fragments before final class filtering.
        fragment_logits : TensorBatch
            (K, C) Learned logits for the same candidates.

        Returns
        -------
        dict
            Loss, majority-class accuracy, mean target purity, and number of
            fragments containing at least one valid semantic target.
        """
        # Build class histograms while excluding invalid or ghost semantic labels
        logits = fragment_logits.torch_tensor()
        classes = seg_label.values.torch_tensor().long().flatten()
        valid = (classes >= 0) & (classes < logits.shape[1])
        encoded = torch.nn.functional.one_hot(
            classes.clamp(0, logits.shape[1] - 1), logits.shape[1]
        ).to(logits.dtype)
        encoded = encoded * valid.unsqueeze(1)
        targets = pool_features(encoded, fragment_candidates)

        # Renormalize after removing invalid labels and supervise each fragment
        # equally, independently of its size or its predicted output class.
        mass = targets.sum(dim=1)
        supervised = mass > 0
        targets = targets / mass.clamp_min(1e-12).unsqueeze(1)
        count = int(supervised.sum())
        if count:
            selected = logits[supervised]
            target = targets[supervised]
            loss = -(target * torch.log_softmax(selected, dim=1)).sum(dim=1).mean()
            accuracy = float((selected.argmax(1) == target.argmax(1)).float().mean())
            purity = float(target.max(dim=1).values.mean())
        else:
            loss = logits.sum() * 0.0
            accuracy, purity = 1.0, 1.0

        return {"loss": loss, "accuracy": accuracy, "purity": purity, "count": count}
