"""Transactional sidecar publication for staged HDF5 caches."""

from __future__ import annotations

import os
import shutil
import tempfile
from typing import Any, Callable

import h5py
import numpy as np

from ..hdf5.common import require_dataset, require_group

__all__ = ["StageSidecarMixin"]


class StageSidecarMixin:
    """Build, validate, and atomically publish staged-cache sidecars.

    New products are written into a small neighboring HDF5 file while the
    canonical cache remains read-only. At finalization, the sidecar stage is
    inserted into a temporary canonical copy and published with
    :func:`os.replace`. Existing readers can therefore finish against the old
    inode while newly opened readers observe the completed stage.
    """

    # Concrete-writer interface required by this mixin. Keeping the contract
    # explicit lets static analyzers validate cross-mixin access without
    # runtime placeholder methods or broad ``type: ignore`` directives.
    sidecar: bool
    _handles: dict[str, h5py.File]
    _active_stages: set[tuple[str, str]]
    _canonical_files: set[str]
    _completed_stages: dict[str, set[str]]
    _initialized_files: set[str]
    _known_files: set[str]
    _sidecar_paths: dict[tuple[str, str], str]
    _sidecar_replace: dict[tuple[str, str], bool]

    _read_source_info: Callable[[h5py.File, str], dict[str, Any]]
    _source_identity: Callable[[dict[str, Any]], tuple[str, int, int]]
    _validate_stage_file: Callable[[h5py.File, str], None]

    @staticmethod
    def _temporary_path(target_path: str, label: str) -> str:
        """Reserve a temporary path beside a canonical cache file.

        Parameters
        ----------
        target_path : str
            Canonical path whose directory should hold the temporary file.
        label : str
            Short role included in the hidden temporary filename.

        Returns
        -------
        str
            Reserved neighboring path on the target filesystem.
        """
        directory = os.path.dirname(os.path.abspath(target_path))
        os.makedirs(directory, exist_ok=True)
        descriptor, temp_path = tempfile.mkstemp(
            prefix=f".spine-{label}-",
            suffix=".h5",
            dir=directory,
        )
        os.close(descriptor)
        return temp_path

    def _close_path_handle(self, file_path: str) -> None:
        """Close and forget one persistent writer handle, if present.

        Parameters
        ----------
        file_path : str
            Exact path used as the persistent-handle cache key.
        """
        handle = self._handles.pop(file_path, None)
        if handle is not None:
            try:
                handle.close()
            except (OSError, RuntimeError, ValueError):
                pass

    def _get_stage_write_path(
        self,
        target_path: str,
        stage: str,
        source_info: dict[str, Any],
        overwrite_stage: bool,
    ) -> str:
        """Resolve the physical write path for one canonical file and stage.

        In direct mode this returns the canonical path unchanged. In sidecar
        mode it validates the canonical file once and allocates a temporary
        neighboring staged cache for the new products.

        Parameters
        ----------
        target_path : str
            Canonical cache path that will eventually own the stage.
        stage : str
            Stage being written.
        source_info : dict
            File-level source provenance for the routed batch subset.
        overwrite_stage : bool
            Whether a completed canonical stage may be replaced.

        Returns
        -------
        str
            Canonical path in direct mode or temporary sidecar path in sidecar
            mode.

        Raises
        ------
        RuntimeError
            If source provenance differs or a completed stage already exists
            without replacement permission.
        """
        if not self.sidecar:
            return target_path

        target_path = os.path.abspath(target_path)
        self._canonical_files.add(target_path)
        stage_key = (target_path, stage)
        if stage_key in self._sidecar_paths:
            return self._sidecar_paths[stage_key]

        replace = False
        if os.path.exists(target_path):
            with h5py.File(target_path, "r") as target_file:
                self._validate_stage_file(target_file, target_path)
                target_source = self._read_source_info(target_file, target_path)
                if self._source_identity(target_source) != self._source_identity(
                    source_info
                ):
                    raise RuntimeError(
                        f"Cache source mismatch for '{target_path}' while "
                        "preparing a sidecar stage."
                    )

                stages = require_group(target_file, "stages")
                if stage in stages:
                    stage_group = require_group(stages, stage)
                    info = stage_group.get("info")
                    complete = bool(
                        isinstance(info, h5py.Group)
                        and info.attrs.get("complete", False)
                    )
                    replace = not complete or overwrite_stage
                    if complete and not overwrite_stage:
                        raise RuntimeError(
                            f"Stage '{stage}' is already complete in "
                            f"'{target_path}'. Set overwrite_stage=True to "
                            "rebuild it."
                        )

        sidecar_path = self._temporary_path(target_path, "stage")
        self._sidecar_paths[stage_key] = sidecar_path
        self._sidecar_replace[stage_key] = replace
        return sidecar_path

    def _prepare_merged_file(
        self,
        target_path: str,
        sidecar_path: str,
        stage: str,
        replace: bool,
    ) -> str:
        """Build and validate the file that will atomically replace a target.

        The returned path is either the sidecar itself for a new canonical
        cache, or a temporary copy of an existing canonical cache with the
        completed sidecar stage inserted.

        Parameters
        ----------
        target_path : str
            Canonical cache which will eventually be replaced.
        sidecar_path : str
            Completed temporary cache containing ``stage``.
        stage : str
            Stage to copy into the canonical cache.
        replace : bool
            Whether an existing stage with this name may be deleted.

        Returns
        -------
        str
            Fully prepared neighboring file ready for atomic publication.

        Raises
        ------
        RuntimeError
            If the sidecar is incomplete, missing its stage, conflicts with a
            concurrently published stage, or has different source provenance.
        ValueError
            If the new stage is not aligned with its canonical siblings.
        """
        with h5py.File(sidecar_path, "r") as sidecar_file:
            self._validate_stage_file(sidecar_file, sidecar_path)
            sidecar_stages = require_group(sidecar_file, "stages")
            if stage not in sidecar_stages:
                raise RuntimeError(
                    f"Sidecar '{sidecar_path}' does not contain stage '{stage}'."
                )

            sidecar_stage = require_group(sidecar_stages, stage)
            sidecar_info = require_group(sidecar_stage, "info")
            if not bool(sidecar_info.attrs.get("complete", False)):
                raise RuntimeError(
                    f"Sidecar stage '{stage}' is incomplete in '{sidecar_path}'."
                )

        # A new cache needs no merge: the self-contained sidecar can become
        # the canonical file directly.
        if not os.path.exists(target_path):
            return sidecar_path

        merged_path = self._temporary_path(target_path, "merge")
        try:
            # Copy first so the canonical inode remains read-only while worker
            # processes may still hold open handles to it.
            shutil.copy2(target_path, merged_path)
            with (
                h5py.File(merged_path, "a") as merged_file,
                h5py.File(sidecar_path, "r") as sidecar_file,
            ):
                self._validate_stage_file(merged_file, target_path)
                self._validate_stage_file(sidecar_file, sidecar_path)

                target_source = self._read_source_info(merged_file, target_path)
                sidecar_source = self._read_source_info(sidecar_file, sidecar_path)
                if self._source_identity(target_source) != self._source_identity(
                    sidecar_source
                ):
                    raise RuntimeError(
                        f"Cannot merge sidecar '{sidecar_path}' into "
                        f"'{target_path}': source provenance differs."
                    )

                merged_stages = require_group(merged_file, "stages")
                sidecar_stages = require_group(sidecar_file, "stages")
                self._validate_stage_alignment(
                    merged_stages,
                    require_group(sidecar_stages, stage),
                    target_path,
                    stage,
                )
                if stage in merged_stages:
                    if not replace:
                        raise RuntimeError(
                            f"Stage '{stage}' appeared in '{target_path}' "
                            "while its sidecar was being written."
                        )
                    del merged_stages[stage]

                sidecar_file.copy(
                    sidecar_stages[stage],
                    merged_stages,
                    name=stage,
                )
                merged_file.flush()

            return merged_path
        except Exception:
            if os.path.exists(merged_path):
                os.remove(merged_path)
            raise

    @staticmethod
    def _stage_source_entries(stage_group: h5py.Group) -> np.ndarray | None:
        """Return a stage's persisted source-entry axis when available.

        Parameters
        ----------
        stage_group : h5py.Group
            Stage group containing an event axis and V2 products.

        Returns
        -------
        numpy.ndarray or None
            Source-local entry indexes, or ``None`` for legacy or malformed
            products that cannot provide a scalar alignment axis.
        """
        products = require_group(stage_group, "products")
        key = "source_file_entry_index"
        if key not in products:
            return None

        product = require_group(products, key)
        if not bool(product.attrs.get("scalar", False)):
            return None
        return require_dataset(product, "values")[:]

    @classmethod
    def _validate_stage_alignment(
        cls,
        canonical_stages: h5py.Group,
        sidecar_stage: h5py.Group,
        target_path: str,
        stage: str,
    ) -> None:
        """Require a sidecar event axis to match every sibling stage.

        A stage being replaced is excluded because an incomplete prior attempt
        may legitimately contain fewer entries. All preserved siblings must
        agree in both event count and source-entry order when that provenance
        product is available.

        Parameters
        ----------
        canonical_stages : h5py.Group
            Canonical ``/stages`` group containing preserved siblings.
        sidecar_stage : h5py.Group
            Completed stage proposed for publication.
        target_path : str
            Canonical cache path used in diagnostics.
        stage : str
            Name under which the sidecar stage will be published.

        Raises
        ------
        TypeError
            If a canonical stage child is not an HDF5 group.
        ValueError
            If event counts or available source-entry indexes differ.
        """
        sidecar_events = require_dataset(sidecar_stage, "events")
        sidecar_entries = cls._stage_source_entries(sidecar_stage)
        for sibling_name, sibling in canonical_stages.items():
            if sibling_name == stage:
                continue
            if not isinstance(sibling, h5py.Group):
                raise TypeError(
                    f"Stage '{sibling_name}' in '{target_path}' must be a group."
                )

            sibling_events = require_dataset(sibling, "events")
            if len(sibling_events) != len(sidecar_events):
                raise ValueError(
                    f"Cannot merge stage '{stage}' into '{target_path}': it has "
                    f"{len(sidecar_events)} entries while sibling stage "
                    f"'{sibling_name}' has {len(sibling_events)}."
                )

            sibling_entries = cls._stage_source_entries(sibling)
            if (
                sidecar_entries is not None
                and sibling_entries is not None
                and not np.array_equal(sidecar_entries, sibling_entries)
            ):
                raise ValueError(
                    f"Cannot merge stage '{stage}' into '{target_path}': its "
                    "source entry order differs from sibling stage "
                    f"'{sibling_name}'."
                )

    def _merge_sidecar_stage(self, stage: str) -> None:
        """Atomically publish a completed sidecar stage to each target file.

        Parameters
        ----------
        stage : str
            Stage to merge across all canonical files touched by this writer.

        Notes
        -----
        Every merged file is prepared and validated before publication begins.
        Publication is atomic per canonical path, but multiple paths do not
        form one filesystem-wide transaction.
        """
        sidecars = [
            (target_path, sidecar_path)
            for (target_path, stage_name), sidecar_path in self._sidecar_paths.items()
            if stage_name == stage
        ]
        prepared: list[tuple[str, str, str]] = []
        try:
            # Validate every candidate before publishing any of them. This
            # catches schema and provenance failures ahead of the commit phase.
            for target_path, sidecar_path in sorted(sidecars):
                self._close_path_handle(sidecar_path)
                stage_key = (target_path, stage)
                merged_path = self._prepare_merged_file(
                    target_path,
                    sidecar_path,
                    stage,
                    self._sidecar_replace[stage_key],
                )
                prepared.append((target_path, sidecar_path, merged_path))
        except Exception:
            for _, sidecar_path, merged_path in prepared:
                if merged_path != sidecar_path and os.path.exists(merged_path):
                    os.remove(merged_path)
            raise

        # Atomic replacement lets existing readers finish on the old inode;
        # readers opened after publication see the newly completed stage.
        for index, (target_path, sidecar_path, merged_path) in enumerate(prepared):
            try:
                os.replace(merged_path, target_path)
            except Exception:
                # Remove unpublished merge copies. Their original sidecars
                # remain mapped and are discarded by ``close`` after failure.
                for _, pending_sidecar, pending_merge in prepared[index:]:
                    if pending_merge != pending_sidecar and os.path.exists(
                        pending_merge
                    ):
                        try:
                            os.remove(pending_merge)
                        except OSError:
                            pass
                raise

            if merged_path != sidecar_path and os.path.exists(sidecar_path):
                try:
                    os.remove(sidecar_path)
                except OSError:
                    pass

            # Forget only the successfully published file-stage pair. If a
            # later publication fails, remaining sidecars stay recoverable by
            # normal writer cleanup.
            stage_key = (target_path, stage)
            self._sidecar_paths.pop(stage_key, None)
            self._sidecar_replace.pop(stage_key, None)
            self._known_files.discard(sidecar_path)
            self._initialized_files.discard(sidecar_path)
            self._active_stages.discard((sidecar_path, stage))
            self._completed_stages.pop(sidecar_path, None)
