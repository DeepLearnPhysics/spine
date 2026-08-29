"""Parse and apply CLI input-source overrides.

Standard SPINE datasets accept one unqualified ``file_keys`` or ``file_list``
selector. Composite datasets instead contain named source blocks, so their
command-line values use ``target=path`` syntax. This module owns the parsing,
validation and configuration routing for both forms.
"""

from __future__ import annotations

from collections.abc import MutableMapping

__all__ = [
    "apply_source_overrides",
    "apply_validation_source_overrides",
    "get_input_config",
    "parse_source_overrides",
]


SourceValues = str | list[str] | None
SourceOverride = dict[str, str | list[str]]
SourceOverrides = dict[str | None, SourceOverride]


# These are configuration slot names, not necessarily dataset backends. Joint
# datasets use roles because both inputs may have the same underlying type.
_COMPOSITE_SOURCE_KEYS = {
    "joint": ("primary", "secondary"),
    "mixed": ("larcv", "hdf5"),
}


def _normalize_source_values(values: SourceValues) -> list[str]:
    """Normalize argparse and direct-call source values.

    Parameters
    ----------
    values : str, list[str] or None
        One direct-call value, multiple values produced by ``argparse``, or no
        override.

    Returns
    -------
    list[str]
        Source values in a common list representation.
    """
    if values is None:
        return []
    if isinstance(values, str):
        return [values]
    return values


def parse_source_overrides(
    source: list[str] | None,
    source_list: SourceValues,
    *,
    source_option: str = "--source",
    source_list_option: str = "--source-list",
) -> SourceOverrides:
    """Parse flat or target-qualified input source overrides.

    Unqualified values preserve the original single-source CLI contract and
    are stored under a ``None`` target. Qualified values are grouped by their
    composite source name. Direct paths accumulate within a target, whereas a
    target may have at most one file list.

    Parameters
    ----------
    source : list[str], optional
        Direct paths written as ``path`` or ``target=path``.
    source_list : str or list[str], optional
        File-list paths written as ``path`` or ``target=path``. A string is
        accepted for compatibility with direct callers of :func:`cli.main`.
    source_option : str, default "--source"
        Direct-source option name used in validation errors.
    source_list_option : str, default "--source-list"
        File-list option name used in validation errors.

    Returns
    -------
    dict
        Mapping from a target name, or ``None`` for a flat input, to the
        corresponding ``file_keys`` or ``file_list`` override.

    Raises
    ------
    ValueError
        If a value is malformed, qualified and unqualified forms are mixed,
        or one target receives conflicting or duplicate selectors.
    """
    direct_values = _normalize_source_values(source)
    list_values = _normalize_source_values(source_list)
    parsed: list[tuple[str, str | None, str]] = []

    # Split once so a qualified path may contain additional ``=`` characters.
    for option, values in (
        (source_option, direct_values),
        (source_list_option, list_values),
    ):
        for value in values:
            target = None
            path = value
            if "=" in value:
                target, path = value.split("=", 1)
                if not target or not path:
                    raise ValueError(
                        f"Invalid {option} value '{value}'. Expected "
                        "'[TARGET=]PATH'."
                    )
            parsed.append((option, target, path))

    if not parsed:
        return {}

    # A single invocation must select either the flat or composite grammar.
    qualified = [target is not None for _, target, _ in parsed]
    if any(qualified) and not all(qualified):
        raise ValueError(
            f"Qualified and unqualified {source_option}/{source_list_option} "
            "values cannot be mixed."
        )

    # Preserve the original mutually exclusive flat-input contract.
    if not any(qualified):
        if direct_values and list_values:
            raise ValueError(
                f"{source_option} and {source_list_option} are mutually exclusive."
            )
        if len(list_values) > 1:
            raise ValueError(
                f"Unqualified {source_list_option} accepts exactly one list file."
            )
        if direct_values:
            return {None: {"file_keys": direct_values}}
        return {None: {"file_list": list_values[0]}}

    # Accumulate direct paths but retain one selector type per named source.
    overrides: SourceOverrides = {}
    for option, target, path in parsed:
        assert target is not None
        target_override = overrides.setdefault(target, {})
        if option == source_option:
            target_override.setdefault("file_keys", [])
            file_keys = target_override["file_keys"]
            assert isinstance(file_keys, list)
            file_keys.append(path)
        else:
            if "file_keys" in target_override:
                raise ValueError(
                    f"Source target '{target}' cannot be provided through both "
                    f"{source_option} and {source_list_option}."
                )
            if "file_list" in target_override:
                raise ValueError(
                    f"Source target '{target}' has multiple "
                    f"{source_list_option} values."
                )
            target_override["file_list"] = path

    return overrides


def get_input_config(io_cfg: MutableMapping) -> tuple[MutableMapping, bool]:
    """Locate the mutable configuration block that owns input options.

    Parameters
    ----------
    io_cfg : MutableMapping
        Top-level ``io`` configuration containing either a reader or loader.

    Returns
    -------
    MutableMapping
        Inline reader configuration or loader dataset configuration.
    bool
        ``True`` when the returned block is a dataset, ``False`` for a reader.

    Raises
    ------
    TypeError
        If the selected reader, loader, or dataset is an external string
        configuration rather than an inline mutable mapping.
    AssertionError
        If a loader does not contain its required dataset block.
    KeyError
        If the I/O configuration contains neither a reader nor a loader.
    """
    reader = io_cfg.get("reader")
    if reader is not None:
        if not isinstance(reader, MutableMapping):
            raise TypeError("CLI source overrides require an inline `io.reader` block.")
        return reader, False

    loader = io_cfg.get("loader")
    if loader is not None:
        if not isinstance(loader, MutableMapping):
            raise TypeError("The `io.loader` block must be a mapping.")
        assert (
            "dataset" in loader
        ), "Missing `dataset` block in `io.loader` for input configuration."
        dataset = loader["dataset"]
        if not isinstance(dataset, MutableMapping):
            raise TypeError(
                "CLI source overrides require an inline `io.loader.dataset` block."
            )
        return dataset, True

    raise KeyError("Must specify `loader` or `reader` in the `io` block.")


def apply_source_overrides(
    io_cfg: MutableMapping,
    source: list[str] | None,
    source_list: SourceValues,
) -> None:
    """Apply flat or composite source overrides to an I/O configuration.

    Flat inputs update the configured reader or ordinary loader dataset.
    Qualified inputs update the matching source blocks of an inline ``joint``
    or ``mixed`` dataset. Unmentioned composite sources remain unchanged.

    Parameters
    ----------
    io_cfg : MutableMapping
        Top-level ``io`` configuration to update in place.
    source : list[str], optional
        Direct flat or target-qualified input paths.
    source_list : str or list[str], optional
        Flat or target-qualified paths to files containing input paths.

    Raises
    ------
    ValueError
        If flat syntax targets a composite dataset, qualified syntax targets
        an ordinary input, or a qualifier is invalid for the dataset type.
    KeyError
        If a requested composite source block is absent.
    TypeError
        If an input configuration or composite source is not inline and
        mutable.
    """
    overrides = parse_source_overrides(source, source_list)
    if not overrides:
        return

    # Identify the ordinary input block or composite dataset definition.
    input_cfg, is_dataset = get_input_config(io_cfg)
    name_value = input_cfg.get("name") if is_dataset else None
    dataset_name = name_value if isinstance(name_value, str) else None
    source_keys = (
        _COMPOSITE_SOURCE_KEYS.get(dataset_name) if dataset_name is not None else None
    )

    # Flat sources retain their historical reader or ordinary-dataset target.
    if None in overrides:
        if source_keys is not None:
            raise ValueError(
                f"The '{dataset_name}' dataset requires target-qualified "
                "--source/--source-list values."
            )
        override = overrides[None]
        input_cfg["file_keys"] = override.get("file_keys")
        input_cfg["file_list"] = override.get("file_list")
        return

    if source_keys is None:
        raise ValueError(
            "Target-qualified --source/--source-list values require an inline "
            "joint or mixed loader dataset."
        )

    # Route each qualified selector to its named composite source block.
    for target, override in overrides.items():
        assert target is not None
        if target not in source_keys:
            expected = ", ".join(source_keys)
            raise ValueError(
                f"Unknown source target '{target}' for '{dataset_name}' dataset. "
                f"Expected one of: {expected}."
            )
        if target not in input_cfg:
            raise KeyError(
                f"The '{dataset_name}' dataset has no `{target}` source block."
            )
        target_cfg = input_cfg[target]
        if not isinstance(target_cfg, MutableMapping):
            raise TypeError(
                f"CLI source overrides require an inline `{target}` source block."
            )

        # None masks an alternate selector inherited by a joint source from
        # the shared base configuration.
        target_cfg["file_keys"] = override.get("file_keys")
        target_cfg["file_list"] = override.get("file_list")


def apply_validation_source_overrides(
    validation_cfg: MutableMapping,
    io_cfg: MutableMapping,
    source: list[str] | None,
    source_list: SourceValues,
) -> None:
    """Apply flat or composite validation-source overrides.

    Ordinary validation datasets store their selector directly in the
    ``validation`` block. Composite validation datasets instead store named
    selectors under ``validation.sources``. Existing named selectors are
    retained when the CLI overrides only part of a composite source set.

    Parameters
    ----------
    validation_cfg : MutableMapping
        Validation configuration to update in place.
    io_cfg : MutableMapping
        Training I/O configuration used to identify the dataset topology.
    source : list[str], optional
        Direct flat or target-qualified validation paths.
    source_list : str or list[str], optional
        Flat or target-qualified validation file-list paths.

    Raises
    ------
    ValueError
        If selector syntax does not match the training dataset topology or the
        resulting composite source set is incomplete.
    TypeError
        If configured composite validation sources are not inline mappings.
    """
    overrides = parse_source_overrides(
        source,
        source_list,
        source_option="--val-source",
        source_list_option="--val-source-list",
    )
    if not overrides:
        return

    input_cfg, is_dataset = get_input_config(io_cfg)
    name_value = input_cfg.get("name") if is_dataset else None
    dataset_name = name_value if isinstance(name_value, str) else None
    source_keys = (
        _COMPOSITE_SOURCE_KEYS.get(dataset_name) if dataset_name is not None else None
    )

    # Flat validation selectors live directly alongside validation policies.
    if None in overrides:
        if source_keys is not None:
            raise ValueError(
                f"The '{dataset_name}' validation dataset requires "
                "target-qualified --val-source/--val-source-list values."
            )
        override = overrides[None]
        validation_cfg.pop("sources", None)
        validation_cfg.pop("file_keys", None)
        validation_cfg.pop("file_list", None)
        validation_cfg.update(override)
        return

    if source_keys is None:
        raise ValueError(
            "Target-qualified --val-source/--val-source-list values require an "
            "inline joint or mixed loader dataset."
        )

    # Merge requested targets with configured validation sources, then require
    # the exact topology consumed by the validation manager.
    configured_sources = validation_cfg.get("sources", {})
    if not isinstance(configured_sources, MutableMapping):
        raise TypeError("The `validation.sources` block must be an inline mapping.")
    merged_sources = dict(configured_sources)
    for target, override in overrides.items():
        assert target is not None
        if target not in source_keys:
            expected = ", ".join(source_keys)
            raise ValueError(
                f"Unknown validation source target '{target}' for "
                f"'{dataset_name}' dataset. Expected one of: {expected}."
            )
        merged_sources[target] = override

    if set(merged_sources) != set(source_keys):
        expected = ", ".join(source_keys)
        raise ValueError(
            f"Validation sources for '{dataset_name}' dataset must provide "
            f"exactly: {expected}."
        )

    validation_cfg.pop("file_keys", None)
    validation_cfg.pop("file_list", None)
    validation_cfg["sources"] = merged_sources
