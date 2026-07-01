"""Analysis scripts and performance evaluation tools.

The analysis package runs configurable scripts on reconstruction and
post-processing outputs, usually to write CSV summaries for diagnostics,
calibration studies, and reconstruction-quality metrics.

``AnaManager`` orchestrates configured analysis modules and handles batched
input dictionaries in the driver workflow.
"""

from .manager import AnaManager
