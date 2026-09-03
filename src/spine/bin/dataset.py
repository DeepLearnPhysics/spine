"""Shared command-line dataset selection configuration.

This module keeps the ordinary and validation CLI surfaces symmetric while
centralizing how source and entry-selection overrides reach dataset configs.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass

from .source import (
    SourceValues,
    apply_source_overrides,
    apply_validation_source_overrides,
    get_input_config,
)

__all__ = [
    "DatasetSelection",
    "add_dataset_arguments",
    "apply_dataset_selection",
    "apply_validation_dataset_selection",
]

ENTRY_KEYS = (
    "n_entry",
    "n_skip",
    "entry_list",
    "skip_entry_list",
    "run_event_list",
    "skip_run_event_list",
    "entry_fraction_range",
)


@dataclass
class DatasetSelection:
    """Dataset source and entry-selection overrides from the CLI.

    This normalized representation is shared by the ordinary input and
    on-the-fly validation paths. Fields left as ``None`` do not alter the
    corresponding configuration value.

    Attributes
    ----------
    source : list[str], optional
        Direct paths, optionally qualified for a composite dataset
    source_list : str or list[str], optional
        File-list paths, optionally qualified for a composite dataset
    n_entry : int, optional
        Maximum number of entries to select
    n_skip : int, optional
        Number of initial entries to skip
    entry_list : str, optional
        File containing explicit entry indexes to select
    skip_entry_list : str, optional
        File containing explicit entry indexes to reject
    run_event_list : str, optional
        File containing run/subrun/event triplets to select
    skip_run_event_list : str, optional
        File containing run/subrun/event triplets to reject
    entry_fraction_range : tuple[float, float], optional
        Half-open fractional range of the resolved entry order to select
    """

    source: list[str] | None = None
    source_list: SourceValues = None
    n_entry: int | None = None
    n_skip: int | None = None
    entry_list: str | None = None
    skip_entry_list: str | None = None
    run_event_list: str | None = None
    skip_run_event_list: str | None = None
    entry_fraction_range: tuple[float, float] | None = None

    @property
    def configured(self) -> bool:
        """Whether the command line supplied any dataset override.

        Returns
        -------
        bool
            ``True`` when at least one source or entry selector is set.
        """
        return any(value is not None for value in vars(self).values())

    @property
    def entry_overrides(self) -> dict[str, object]:
        """Collect configured entry selectors under their YAML keys.

        Source fields are deliberately excluded because they require separate
        routing for ordinary and composite datasets.

        Returns
        -------
        dict[str, object]
            Non-null entry-selection overrides ready to merge into a reader
            or dataset configuration.
        """
        return {
            key: value
            for key, value in vars(self).items()
            if key in ENTRY_KEYS and value is not None
        }

    @classmethod
    def from_namespace(
        cls, namespace: argparse.Namespace, *, validation: bool = False
    ) -> "DatasetSelection":
        """Build one selection from an argparse namespace.

        Parameters
        ----------
        namespace : argparse.Namespace
            Parsed SPINE command-line arguments
        validation : bool, default False
            Read attributes prefixed with ``val_`` instead of the ordinary
            input attributes.

        Returns
        -------
        DatasetSelection
            Normalized source and entry overrides. The two-element fractional
            range produced by ``argparse`` is converted to a tuple.
        """
        prefix = "val_" if validation else ""

        def get(name: str):
            return getattr(namespace, f"{prefix}{name}")

        fraction_range = get("entry_fraction_range")
        if fraction_range is not None:
            fraction_range = tuple(fraction_range)

        return cls(
            source=get("source"),
            source_list=get("source_list"),
            n_entry=get("num_entries"),
            n_skip=get("nskip"),
            entry_list=get("entry_list"),
            skip_entry_list=get("skip_entry_list"),
            run_event_list=get("run_event_list"),
            skip_run_event_list=get("skip_run_event_list"),
            entry_fraction_range=fraction_range,
        )


def add_dataset_arguments(
    parser: argparse.ArgumentParser, *, validation: bool = False
) -> None:
    """Register one complete dataset-override CLI surface.

    Calling this function once for the ordinary input and once with
    ``validation=True`` keeps both option sets symmetric. Historical short
    aliases are retained only for the ordinary input; validation options use
    the explicit ``--val-*`` names.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to update
    validation : bool, default False
        Add ``--val-*`` options and ``val_*`` namespace destinations
    """
    option_prefix = "val-" if validation else ""
    dest_prefix = "val_" if validation else ""
    label = "Validation " if validation else "Input "

    def option(name: str) -> str:
        return f"--{option_prefix}{name}"

    source_flags = [option("source")]
    source_list_flags = [option("source-list")]
    num_entry_flags = [option("num-entries")]
    skip_entry_flags = [option("skip-entries")]
    if not validation:
        source_flags.insert(0, "-s")
        source_list_flags.insert(0, "-S")
        num_entry_flags.insert(0, "-n")
        skip_entry_flags.append("--nskip")

    parser.add_argument(
        *source_flags,
        dest=f"{dest_prefix}source",
        nargs="+",
        metavar="[TARGET=]PATH",
        help=f"{label}paths, optionally qualified by a composite source name",
    )
    parser.add_argument(
        *source_list_flags,
        dest=f"{dest_prefix}source_list",
        nargs="+",
        metavar="[TARGET=]LIST",
        help=f"{label}file lists, optionally qualified by composite source names",
    )
    parser.add_argument(
        *num_entry_flags,
        dest=f"{dest_prefix}num_entries",
        type=int,
        help=f"Number of {label.lower()}dataset entries to load",
    )
    parser.add_argument(
        *skip_entry_flags,
        dest=f"{dest_prefix}nskip",
        type=int,
        help=f"Number of {label.lower()}dataset entries to skip",
    )
    parser.add_argument(
        option("entry-list"),
        dest=f"{dest_prefix}entry_list",
        help=f"Path to {label.lower()}entry indexes to process",
    )
    parser.add_argument(
        option("skip-entry-list"),
        dest=f"{dest_prefix}skip_entry_list",
        help=f"Path to {label.lower()}entry indexes to skip",
    )
    parser.add_argument(
        option("run-event-list"),
        dest=f"{dest_prefix}run_event_list",
        help=f"Path to {label.lower()}run/subrun/event triplets to process",
    )
    parser.add_argument(
        option("skip-run-event-list"),
        dest=f"{dest_prefix}skip_run_event_list",
        help=f"Path to {label.lower()}run/subrun/event triplets to skip",
    )
    parser.add_argument(
        option("entry-fraction-range"),
        dest=f"{dest_prefix}entry_fraction_range",
        type=float,
        nargs=2,
        metavar=("START", "STOP"),
        help=f"Half-open fractional range of the {label.lower()}dataset to process",
    )


def _apply_entry_overrides(
    io_cfg: MutableMapping, overrides: dict[str, object]
) -> None:
    """Apply entry selectors to the configuration that owns traversal.

    Ordinary readers and datasets receive selectors directly. A mixed
    dataset also receives them at its root so both aligned children observe
    the same selection. For a joint dataset, only the primary source controls
    traversal; inherited primary filters are explicitly masked before the new
    selectors are installed.

    Parameters
    ----------
    io_cfg : MutableMapping
        Top-level ``io`` configuration to update in place.
    overrides : dict[str, object]
        Entry-selection values keyed by reader configuration name.

    Raises
    ------
    TypeError
        If a joint dataset does not define its primary source inline.
    """
    if not overrides:
        return

    input_cfg, is_dataset = get_input_config(io_cfg)
    dataset_name = input_cfg.get("name") if is_dataset else None

    # Mixed inputs forward root options to both aligned children. Joint input
    # traversal is instead defined solely by the primary source.
    target_cfg = input_cfg
    inherited_keys: set[str] = set()
    if dataset_name == "joint":
        target_cfg = input_cfg.get("primary")
        if not isinstance(target_cfg, MutableMapping):
            raise TypeError(
                "CLI entry overrides require an inline joint `primary` block."
            )
        base_cfg = input_cfg.get("base")
        if isinstance(base_cfg, Mapping):
            inherited_keys = set(base_cfg).intersection(ENTRY_KEYS)

    for key in ENTRY_KEYS:
        if key in inherited_keys:
            # A source-level None masks a stale filter inherited from joint
            # base without changing the independent secondary source.
            target_cfg[key] = None
        else:
            target_cfg.pop(key, None)
    target_cfg.update(overrides)


def apply_dataset_selection(
    io_cfg: MutableMapping, selection: DatasetSelection
) -> None:
    """Apply CLI overrides to the primary runtime dataset.

    Source overrides are resolved first because they have composite-specific
    syntax. Entry selectors are then applied to the block responsible for
    traversing the resulting input.

    Parameters
    ----------
    io_cfg : MutableMapping
        Top-level ``io`` configuration to update in place.
    selection : DatasetSelection
        Normalized source and entry overrides.
    """
    apply_source_overrides(io_cfg, selection.source, selection.source_list)
    _apply_entry_overrides(io_cfg, selection.entry_overrides)


def apply_validation_dataset_selection(
    validation_cfg: MutableMapping,
    io_cfg: MutableMapping,
    selection: DatasetSelection,
) -> None:
    """Apply CLI overrides to an on-the-fly validation dataset.

    Validation sources are materialized using the shape of the main input.
    Entry selectors remain in the validation block until
    :class:`~spine.model.validation.ValidationManager` derives its loader,
    where they are routed with the same ordinary, mixed, and joint semantics
    as the primary dataset.

    Parameters
    ----------
    validation_cfg : MutableMapping
        Top-level validation configuration to update in place.
    io_cfg : MutableMapping
        Main ``io`` configuration used to derive the validation input shape.
    selection : DatasetSelection
        Normalized validation source and entry overrides.
    """
    apply_validation_source_overrides(
        validation_cfg,
        io_cfg,
        selection.source,
        selection.source_list,
    )

    # ValidationManager routes these filters after deriving the dataset shape.
    entry_overrides = selection.entry_overrides
    if entry_overrides:
        for key in ENTRY_KEYS:
            validation_cfg.pop(key, None)
        validation_cfg.update(entry_overrides)
