"""Tests for asynchronous process-control primitives."""

from __future__ import annotations

import os
import signal
import subprocess
import sys

from spine.control import RunControl


def test_run_control_records_and_restores_sigusr1_handler():
    """SIGUSR1 should set only the sticky request and restore prior handling."""
    previous = signal.getsignal(signal.SIGUSR1)
    control = RunControl()

    assert not control.installed
    control.install()
    control.install()
    assert control.installed
    try:
        os.kill(os.getpid(), signal.SIGUSR1)
        assert control.graceful_stop_requested
    finally:
        control.restore()
        control.restore()

    assert not control.installed
    assert signal.getsignal(signal.SIGUSR1) == previous


def test_sigusr1_handler_allows_normal_subprocess_exit():
    """A handled user signal should not become a signal-coded process exit."""
    script = """
import os
import signal
from spine.control import RunControl

control = RunControl()
control.install()
os.kill(os.getpid(), signal.SIGUSR1)
raise SystemExit(0 if control.graceful_stop_requested else 1)
"""
    result = subprocess.run([sys.executable, "-c", script], check=False)

    assert result.returncode == 0
