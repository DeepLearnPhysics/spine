"""Version 2 production allocation pause/outcome contract.

Environment paths are scoped to an immutable scheduler attempt. This helper
publishes only after the driver finishes cleanup; the production controller
also requires successful launcher exit and confirmed scheduler accounting.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
from pathlib import Path


class Continuation:
    """Publish verified training outcomes for one scheduler allocation.

    Marker and outcome paths belong to the submission controller. This helper
    handles the compute-side contract; it never submits or resubmits jobs.
    """

    def __init__(
        self, token: str, pause_file: str | Path, outcome_file: str | Path
    ) -> None:
        """Bind this allocation to controller-owned marker and outcome paths.

        Parameters
        ----------
        token : str
            Unique attempt identity supplied by the submission controller.
        pause_file : str or pathlib.Path
            Marker whose existence requests a recoverable allocation pause.
        outcome_file : str or pathlib.Path
            Destination for the atomic, checksum-verified outcome record.
        """
        self.token = token
        self.pause_file = Path(pause_file)
        self.outcome_file = Path(outcome_file)

    @classmethod
    def from_environment(cls) -> Continuation | None:
        """Read allocation paths, or return None when continuation is disabled.

        Returns
        -------
        Continuation or None
            Allocation contract when ``SPINE_CONTINUATION_TOKEN`` is set.

        Raises
        ------
        KeyError
            A token is present without its required marker or outcome path.
        """
        token = os.environ.get("SPINE_CONTINUATION_TOKEN")
        if not token:
            return None
        return cls(
            token, os.environ["SPINE_PAUSE_FILE"], os.environ["SPINE_OUTCOME_FILE"]
        )

    @property
    def pause_requested(self) -> bool:
        """Return whether the controller has created the pause marker."""
        return self.pause_file.is_file()

    def publish(
        self,
        state: str,
        reason: str,
        checkpoint: str | Path | None,
        pending_validation: bool = False,
    ) -> None:
        """Verify a durable checkpoint and atomically publish its outcome.

        Parameters
        ----------
        state : str
            Logical outcome: paused or completed.
        reason : str
            Allocation deadline or scientific completion reason.
        checkpoint : str or pathlib.Path
            Numeric or initial snapshot with a committed SHA-256 sidecar.
        pending_validation : bool, default False
            Whether validation must replay before the next optimizer step.

        Raises
        ------
        RuntimeError
            No restart checkpoint was produced.
        ValueError
            The snapshot name or integrity sidecar is invalid.
        OSError
            Checkpoint verification or durable outcome publication failed.
        """
        if checkpoint is None:
            raise RuntimeError("Continuation completed without a restart checkpoint")
        checkpoint = Path(checkpoint).resolve()
        match = re.fullmatch(r"snapshot-(\d+|initial)\.ckpt", checkpoint.name)
        if match is None:
            raise ValueError("Continuation requires a numeric snapshot checkpoint")

        # Verify the artifact rather than trusting its sidecar alone
        digest = hashlib.sha256()
        with checkpoint.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        checksum = digest.hexdigest()
        fields = Path(str(checkpoint) + ".sha256").read_text().split()
        if (
            len(fields) != 2
            or fields[0] != checksum
            or fields[1].lstrip("*") != checkpoint.name
        ):
            raise ValueError("Checkpoint integrity sidecar is not committed")
        result = dict(
            contract=1,
            token=self.token,
            state=state,
            reason=reason,
            checkpoint=str(checkpoint),
            sha256=checksum,
            iteration=-1 if match[1] == "initial" else int(match[1]),
            committed_at=time.time(),
            pending_validation=pending_validation,
        )

        # Commit the outcome only after checkpoint verification. Sync both
        # contents and the directory entry so restart readers see one record.
        fd, temporary = tempfile.mkstemp(
            dir=self.outcome_file.parent, prefix="outcome."
        )
        try:
            with os.fdopen(fd, "w") as stream:
                json.dump(result, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.outcome_file)
            directory = os.open(self.outcome_file.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
