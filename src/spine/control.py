"""Process-control primitives for graceful training completion."""

from __future__ import annotations

import signal
from types import FrameType
from typing import Any

__all__ = ["RunControl"]


class RunControl:
    """Track asynchronous requests to complete training gracefully.

    The installed ``SIGUSR1`` handler deliberately performs no I/O, model work
    or distributed communication. It only records the request for the driver
    to observe at its next safe iteration boundary.
    """

    signal_number = signal.SIGUSR1

    def __init__(self) -> None:
        """Initialize an idle controller with no installed signal handler."""
        self.graceful_stop_requested = False
        self._previous_handler: Any | None = None

    @property
    def installed(self) -> bool:
        """Whether this controller currently owns the process signal handler."""
        return self._previous_handler is not None

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
        self.graceful_stop_requested = True

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
