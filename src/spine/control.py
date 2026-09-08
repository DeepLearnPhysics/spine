"""Process-control primitives for graceful training completion."""

from __future__ import annotations

import signal
from pathlib import Path
from types import FrameType
from typing import Any

__all__ = ["RunControl"]


class RunControl:
    """Track asynchronous requests to complete training gracefully.

    A request may come from the installed ``SIGUSR1`` handler or from a marker
    file created by an external batch-system wrapper. The signal handler
    deliberately performs no I/O, model work or distributed communication.
    Marker polling likewise happens only when the driver reaches a safe
    iteration boundary.
    """

    signal_number = signal.SIGUSR1

    def __init__(self, graceful_stop_file: str | None = None) -> None:
        """Initialize an idle process controller.

        Parameters
        ----------
        graceful_stop_file : str, optional
            Marker file whose presence requests graceful completion. Only the
            main training rank should receive this path so that shared storage
            is polled once per minibatch rather than once per rank.
        """
        self.graceful_stop_file = (
            None if graceful_stop_file is None else Path(graceful_stop_file)
        )
        self._graceful_stop_requested = False
        self._graceful_stop_source: str | None = None
        self._previous_handler: Any | None = None

    @property
    def installed(self) -> bool:
        """Whether this controller currently owns the process signal handler."""
        return self._previous_handler is not None

    @property
    def graceful_stop_requested(self) -> bool:
        """Whether a signal or marker has requested graceful completion.

        Marker detection is sticky: once observed, removing the file cannot
        withdraw a request which may already have propagated to other ranks.
        """
        if (
            not self._graceful_stop_requested
            and self.graceful_stop_file is not None
            and self.graceful_stop_file.is_file()
        ):
            self._graceful_stop_requested = True
            self._graceful_stop_source = "file"

        return self._graceful_stop_requested

    @property
    def graceful_stop_description(self) -> str | None:
        """Human-readable description of the observed request source."""
        if self._graceful_stop_source == "signal":
            return "SIGUSR1"
        if self._graceful_stop_source == "file":
            return f"marker file: {self.graceful_stop_file}"
        return None

    @property
    def completion_metadata(self) -> dict[str, str] | None:
        """Checkpoint metadata describing the observed graceful request."""
        if self._graceful_stop_source == "signal":
            return {"reason": "graceful_stop", "signal": "SIGUSR1"}
        if self._graceful_stop_source == "file":
            assert self.graceful_stop_file is not None
            return {
                "reason": "graceful_stop",
                "file": str(self.graceful_stop_file),
            }
        return None

    def request_graceful_stop(
        self,
        signum: int | None = None,
        frame: FrameType | None = None,
    ) -> None:
        """Record a graceful-stop request without doing asynchronous work.

        Parameters
        ----------
        signum : int, optional
            Delivered signal number. It is accepted for signal-handler
            compatibility and otherwise unused.
        frame : FrameType, optional
            Interrupted Python frame. It is accepted for signal-handler
            compatibility and otherwise unused.
        """
        self._graceful_stop_requested = True
        self._graceful_stop_source = "signal"

    def install(self) -> None:
        """Install the ``SIGUSR1`` handler, preserving the previous handler."""
        if self._previous_handler is not None:
            return

        self._previous_handler = signal.getsignal(self.signal_number)
        signal.signal(self.signal_number, self.request_graceful_stop)

    def restore(self) -> None:
        """Restore the signal handler that preceded :meth:`install`."""
        if self._previous_handler is None:
            return

        signal.signal(self.signal_number, self._previous_handler)
        self._previous_handler = None
