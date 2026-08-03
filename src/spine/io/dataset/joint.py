"""Dataset wrapper for overlaying events from two independent sources."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

from ..overlay import Overlayer
from .base import BaseDataset, DataDict

__all__ = ["JointDataset"]


class JointDataset(BaseDataset):
    """Torch dataset that overlays unaligned primary/secondary events.

    This class is intentionally different from :class:`MixedDataset`:

    - ``MixedDataset`` is an aligned merge of products that describe the same
      event across backends.
    - ``JointDataset`` is an unaligned merge of products that describe
      different events and should be overlaid for training.

    ``JointDataset`` does not decide which secondary event to use. A joint
    sampler provides indexes of the form ``(primary_idx, secondary_idx)``. If
    ``secondary_idx`` is ``None`` or if a scalar primary index is provided, the
    dataset returns the primary sample unchanged. This keeps pairing policy and
    pair probability in the sampler, while this class only instantiates source
    datasets and applies the existing :class:`spine.io.overlay.Overlayer`.

    The first implementation supports one primary event plus at most one
    secondary event per sample. Both sources must expose the same data keys,
    collate types, and overlay methods so that the overlayer can operate on a
    common product schema.
    """

    name: ClassVar[str] = "joint"
    joint: ClassVar[bool] = True
    primary: BaseDataset
    secondary: BaseDataset
    reader: Any

    def __init__(
        self,
        primary: Mapping[str, Any] | str | BaseDataset,
        secondary: Mapping[str, Any] | str | BaseDataset,
        base: Mapping[str, Any] | str | None = None,
        dtype: str | None = None,
        augment: Mapping[str, Any] | None = None,
    ) -> None:
        """Instantiate the joint overlay dataset.

        Parameters
        ----------
        primary : mapping, str or BaseDataset
            Primary dataset configuration, overrides to a shared ``base``
            configuration, or already-instantiated dataset. The primary
            controls the length and primary index order.
        secondary : mapping, str or BaseDataset
            Secondary dataset configuration, overrides to a shared ``base``
            configuration, or already-instantiated dataset.
        base : mapping or str, optional
            Shared dataset configuration merged into both source configs before
            instantiation. Source blocks override values from ``base``. Use
            this for common schema/parser options, and put source-specific
            values such as file paths or entry filters in ``primary`` and
            ``secondary``.
        dtype : str, optional
            Floating-point dtype forwarded when instantiating configured
            datasets.
        augment : mapping, optional
            Augmentation applied after the primary sample is returned or after
            the primary/secondary samples are overlaid.
        """
        # Initialize parent class
        super().__init__()

        # Instantiate the source datasets. The optional `base` block lets users
        # define a common schema once while keeping paths and filters local to
        # each source block.
        self.primary = self.build_dataset(
            self.resolve_source_config(base, primary),
            dtype,
        )
        self.secondary = self.build_dataset(
            self.resolve_source_config(base, secondary), dtype
        )

        # Expose the primary reader for compatibility with code that inspects
        # the dataset's main source. Secondary access is internal to overlays.
        self.reader = getattr(self.primary, "reader", None)
        if len(self.primary) < 1:
            raise ValueError("The primary dataset must expose at least one entry.")
        if len(self.secondary) < 1:
            raise ValueError("The secondary dataset must expose at least one entry.")

        # The overlayer expects the same logical products on both sides. It uses
        # the primary metadata after compatibility is validated.
        self.validate_keys(self.primary.data_keys, self.secondary.data_keys)
        self.validate_metadata(
            "overlay", self.primary.overlay_methods, self.secondary.overlay_methods
        )

        self.overlayer = Overlayer(
            multiplicity=2,
            mode="constant",
            data_keys=self.primary.data_keys,
            methods=self.primary.overlay_methods,
        )

        # Initialize the augmenter
        self.build_augmenter(augment)

    @staticmethod
    def resolve_source_config(
        base: Mapping[str, Any] | str | None,
        source: Mapping[str, Any] | str | BaseDataset,
    ) -> Mapping[str, Any] | str | BaseDataset:
        """Merge one source override block into the shared base config.

        Already-instantiated datasets are returned unchanged. String configs
        cannot be merged with ``base`` because there is no mapping to update.

        Parameters
        ----------
        base : mapping or str, optional
            Shared source configuration.
        source : mapping, str or BaseDataset
            Source-specific overrides, configuration path, or dataset object.

        Returns
        -------
        mapping, str or BaseDataset
            Resolved source specification ready for instantiation.

        Raises
        ------
        ValueError
            If mapping overrides are combined with a non-mapping base.
        """
        if base is None or not isinstance(source, Mapping):
            return source
        if not isinstance(base, Mapping):
            raise ValueError(
                "A shared `base` config can only be merged with source "
                "override mappings."
            )

        return {**dict(base), **dict(source)}

    @staticmethod
    def build_dataset(
        source: Mapping[str, Any] | str | BaseDataset,
        dtype: str | None,
    ) -> BaseDataset:
        """Instantiate one source dataset unless an object is already provided.

        Source-specific options, including entry filters, must be present in
        the source config itself before this method is called.

        Parameters
        ----------
        source : mapping, str or BaseDataset
            Dataset configuration or an existing dataset instance.
        dtype : str, optional
            Floating-point dtype forwarded to the dataset factory.

        Returns
        -------
        BaseDataset
            Instantiated or pass-through source dataset.
        """
        if isinstance(source, (Mapping, str)):
            from ..factories import dataset_factory

            return dataset_factory(source, dtype=dtype)

        return source

    def __len__(self) -> int:
        """Return the number of primary entries.

        Joint samplers iterate over the primary source and independently choose
        secondary indexes to pair with those primary entries.
        """
        return len(self.primary)

    def __getitem__(self, idx: int | tuple[int, int | None]) -> DataDict:
        """Return one primary sample or one primary/secondary overlay.

        Parameters
        ----------
        idx : int or tuple[int, int or None]
            A scalar index returns the corresponding primary sample without an
            overlay. A tuple ``(primary_idx, secondary_idx)`` overlays the two
            source samples. A tuple ``(primary_idx, None)`` returns the primary
            sample without touching the secondary source.

        Returns
        -------
        dict
            Primary sample, optionally overlaid with the selected secondary
            event and then augmented.
        """
        primary_idx, secondary_idx = self.resolve_pair_index(idx)
        primary = self.primary[primary_idx]
        if secondary_idx is None:
            return self.apply_augmenter(primary)

        overlaid = self.overlayer([primary, self.secondary[secondary_idx]])
        assert len(overlaid) == 1, "Joint overlays should produce one sample."
        return self.apply_augmenter(overlaid[0])

    def resolve_pair_index(
        self, idx: int | tuple[int, int | None]
    ) -> tuple[int, int | None]:
        """Resolve the primary and optional secondary indexes for one sample.

        Parameters
        ----------
        idx : int or tuple[int, int or None]
            Scalar primary index or explicit primary/secondary pair.

        Returns
        -------
        tuple[int, int or None]
            Normalized primary index and validated optional secondary index.

        Raises
        ------
        ValueError
            If a tuple does not contain exactly two indexes or the secondary
            index is outside the dataset.
        """
        if isinstance(idx, tuple):
            if len(idx) != 2:
                raise ValueError("JointDataset tuple indexes must have length 2.")
            primary_idx, secondary_idx = idx
            if secondary_idx is not None:
                secondary_idx = self.validate_secondary_index(int(secondary_idx))
            return int(primary_idx), secondary_idx

        return idx, None

    def validate_secondary_index(self, secondary_idx: int) -> int:
        """Validate one secondary index produced by a joint sampler.

        Parameters
        ----------
        secondary_idx : int
            Candidate index into the secondary dataset.

        Returns
        -------
        int
            Validated secondary index.

        Raises
        ------
        ValueError
            If the index falls outside the secondary dataset.
        """
        if secondary_idx < 0 or secondary_idx >= len(self.secondary):
            raise ValueError("Secondary index is outside of bounds.")

        return secondary_idx

    @staticmethod
    def validate_metadata(
        label: str,
        primary: Mapping[str, str | None],
        secondary: Mapping[str, str | None],
    ) -> None:
        """Ensure both datasets expose compatible metadata for overlaying.

        The current joint overlay implementation is schema-preserving: every
        product emitted by the primary must also be emitted by the secondary
        with the same collate type and overlay method.

        Parameters
        ----------
        label : str
            Metadata category used in error messages.
        primary : mapping
            Metadata exposed by the primary dataset.
        secondary : mapping
            Metadata exposed by the secondary dataset.

        Raises
        ------
        ValueError
            If either the keys or their configured values differ.
        """
        primary_keys = set(primary)
        secondary_keys = set(secondary)
        if primary_keys != secondary_keys:
            missing = sorted(primary_keys - secondary_keys)
            extra = sorted(secondary_keys - primary_keys)
            raise ValueError(
                f"JointDataset {label} keys must match between primary and "
                f"secondary datasets. Missing in secondary: {missing}. "
                f"Extra in secondary: {extra}."
            )

        for key in primary:
            if primary[key] != secondary[key]:
                raise ValueError(
                    f"JointDataset {label} mismatch for '{key}': "
                    f"{primary[key]!r} vs {secondary[key]!r}."
                )

    @staticmethod
    def validate_keys(primary: tuple[str, ...], secondary: tuple[str, ...]) -> None:
        """Ensure both sources expose the same product names.

        Parameters
        ----------
        primary : tuple[str, ...]
            Product names exposed by the primary dataset.
        secondary : tuple[str, ...]
            Product names exposed by the secondary dataset.

        Raises
        ------
        ValueError
            If the two product-name sets differ.
        """
        if set(primary) != set(secondary):
            missing = sorted(set(primary) - set(secondary))
            extra = sorted(set(secondary) - set(primary))
            raise ValueError(
                "JointDataset data keys must match between sources. "
                f"Missing in secondary: {missing}. Extra in secondary: {extra}."
            )

    @property
    def overlay_methods(self) -> dict[str, str | None]:
        """Return overlay methods for any downstream batch-level overlay."""
        return dict(self.primary.overlay_methods)

    @property
    def data_keys(self) -> tuple[str, ...]:
        """Return the names of all products emitted by joint samples."""
        return tuple(self.primary.data_keys)
