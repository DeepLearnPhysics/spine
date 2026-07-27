"""Analysis scripts and performance evaluation tools.

The analysis package runs configurable scripts on reconstruction and
post-processing outputs, usually to write CSV summaries for diagnostics,
calibration studies, and reconstruction-quality metrics.

``AnaManager`` orchestrates configured analysis modules and handles batched
input dictionaries in the driver workflow. Individual analyzers may also
implement an optional columnar hook for projected-product execution
without changing their default event-oriented behavior.
"""

from .manager import AnaManager
