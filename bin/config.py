#!/usr/bin/env python3
"""Run the SPINE configuration inspection command from a source checkout."""

from __future__ import annotations

import os
import sys

# Add src directory to PYTHONPATH
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Import and run the config CLI
from spine.bin.config import cli


def main() -> int:
    """Execute the configuration CLI and return its process status."""
    return cli()


if __name__ == "__main__":
    sys.exit(main())
