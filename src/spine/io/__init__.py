"""Input/output tools for SPINE.

Readers and writers are storage-format adapters. Parsers convert source
formats into framework-neutral parser products. Datasets, collation,
augmentation, and sampling are generic data pipeline tools used by training
and inference configurations.
"""

from spine.utils.conditional import TORCH_AVAILABLE

from .factories import *
from .read import *
from .write import *

TORCH_IO_AVAILABLE = TORCH_AVAILABLE
