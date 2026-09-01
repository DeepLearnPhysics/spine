"""Canonical class selection and aggregation for metric reports.

Report configurations refer to semantic categories by their stable SPINE
names. Display labels and integer IDs remain owned by :mod:`spine.constants`,
so report YAML files do not duplicate that source of truth.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from spine.constants import PID_LABELS, SHAPE_LABELS, ParticlePID, ParticleShape

PRIMARY_LABELS = {0: "Secondary", 1: "Primary"}


def _canonical_labels(kind: str) -> Mapping[int, str]:
    """Return the canonical integer-to-display-label map for one class kind."""
    if kind == "shape":
        return SHAPE_LABELS
    if kind == "pid":
        return PID_LABELS
    if kind == "primary":
        return PRIMARY_LABELS
    raise ValueError(f"Unknown report class kind `{kind}`.")


def _aliases(kind: str) -> dict[str, int]:
    """Build case-insensitive enum and display-label aliases."""
    labels = _canonical_labels(kind)
    aliases = {
        label.lower().replace(" ", "_"): class_id
        for class_id, label in labels.items()
        if class_id >= 0
    }
    if kind == "shape":
        aliases.update(
            {
                member.name.lower(): int(member)
                for member in ParticleShape
                if int(member) >= 0
            }
        )
        aliases["low_energy"] = int(ParticleShape.LOWE)
    elif kind == "pid":
        aliases.update(
            {
                member.name.lower(): int(member)
                for member in ParticlePID
                if int(member) >= 0
            }
        )
    return aliases


def class_id(value: int | str, kind: str) -> int:
    """Resolve an integer ID or canonical class name to an integer ID."""
    if isinstance(value, (int, np.integer)):
        return int(value)
    normalized = str(value).strip().lower().replace(" ", "_")
    aliases = _aliases(kind)
    if normalized not in aliases:
        raise ValueError(f"Unknown {kind} class `{value}`.")
    return aliases[normalized]


def infer_class_kind(column: str) -> str:
    """Infer shape, PID, or primary classes from a save-record column name."""
    if column.endswith("_shape"):
        return "shape"
    if column.endswith("_pid"):
        return "pid"
    if column.endswith("_is_primary") or column.endswith("_group_primary"):
        return "primary"
    raise ValueError(
        f"Cannot infer a class kind from `{column}`; configure `class_type`."
    )


def resolve_class_groups(
    config: Mapping[str, Any],
    *,
    kind: str,
    default_ids: Sequence[int],
) -> list[dict[str, Any]]:
    """Resolve configured class restriction or many-to-one aggregation.

    ``classes`` restricts output to a sequence of canonical IDs or names.
    ``class_mapping`` maps each requested display label to one or more source
    classes. The two options are mutually exclusive. The legacy
    ``class_names`` option remains accepted as a display-label override.
    """
    classes = config.get("classes")
    class_mapping = config.get("class_mapping")
    class_names = config.get("class_names")
    configured = sum(value is not None for value in (classes, class_mapping))
    if configured > 1:
        raise ValueError("Use either `classes` or `class_mapping`, not both.")

    labels = _canonical_labels(kind)
    if class_mapping is not None:
        if not isinstance(class_mapping, Mapping) or not class_mapping:
            raise TypeError("`class_mapping` must be a non-empty mapping.")
        groups = []
        for name, source_values in class_mapping.items():
            if isinstance(source_values, (str, bytes)) or not isinstance(
                source_values, Sequence
            ):
                raise TypeError("Each class mapping value must be a sequence.")
            groups.append(
                {
                    "name": str(name),
                    "source_ids": [class_id(value, kind) for value in source_values],
                }
            )
    else:
        if classes is not None and (
            isinstance(classes, (str, bytes)) or not isinstance(classes, Sequence)
        ):
            raise TypeError("`classes` must be a sequence of class IDs or names.")
        source_ids = (
            [class_id(value, kind) for value in classes]
            if classes is not None
            else [int(value) for value in default_ids]
        )
        groups = [
            {"name": labels.get(source_id, str(source_id)), "source_ids": [source_id]}
            for source_id in source_ids
        ]

    if not groups or any(not group["source_ids"] for group in groups):
        raise ValueError("Report class groups must be non-empty.")
    flattened = [source_id for group in groups for source_id in group["source_ids"]]
    if len(flattened) != len(set(flattened)):
        raise ValueError("A source class may appear in only one report class group.")
    if any(source_id < 0 for source_id in flattened):
        raise ValueError("Report class groups cannot include negative sentinel IDs.")

    if class_names is not None:
        if class_mapping is not None:
            raise ValueError("`class_names` cannot be combined with `class_mapping`.")
        if len(class_names) != len(groups):
            raise ValueError("Must provide one class name per selected class.")
        for group, name in zip(groups, class_names):
            group["name"] = str(name)
    return groups


def map_class_values(
    values: np.ndarray,
    groups: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    """Map raw categorical values onto report groups and return validity."""
    mapped = np.full(len(values), -1, dtype=np.int64)
    for target, group in enumerate(groups):
        mapped[np.isin(values, group["source_ids"])] = target
    return mapped, mapped >= 0


def aggregate_confusion(
    matrix: np.ndarray,
    groups: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    """Aggregate a raw confusion matrix over selected or mapped classes."""
    size = len(matrix)
    output = np.zeros((len(groups), len(groups)), dtype=np.int64)
    for prediction, prediction_group in enumerate(groups):
        prediction_ids = prediction_group["source_ids"]
        for truth, truth_group in enumerate(groups):
            truth_ids = truth_group["source_ids"]
            if any(value >= size for value in (*prediction_ids, *truth_ids)):
                raise ValueError(
                    f"Configured class ID exceeds confusion matrix size {size}."
                )
            output[prediction, truth] = matrix[np.ix_(prediction_ids, truth_ids)].sum()
    return output


__all__ = [
    "PRIMARY_LABELS",
    "aggregate_confusion",
    "class_id",
    "infer_class_kind",
    "map_class_values",
    "resolve_class_groups",
]
