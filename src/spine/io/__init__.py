"""I/O tools for reading and writing neutrino physics data.

This module provides comprehensive data input/output capabilities for various file formats
commonly used in neutrino physics experiments, with support for both PyTorch-dependent
and independent operations.

**Core I/O Functionality:**
- `core`: Framework-independent I/O operations (LArCV, HDF5, ROOT)
- `torch`: PyTorch-specific data loading and batch processing

**File Format Support:**
- **LArCV**: ROOT-based sparse tensor format for liquid argon detectors
- **HDF5**: Hierarchical data format with compression and metadata
- **ROOT**: High-energy physics standard format
- **NumPy**: Native Python array format

**Key Components:**
- **Parsers**: Extract and transform data from various file formats
- **Collaters**: Batch multiple samples together for ML training
- **Samplers**: Control data sampling strategies and ordering
- **Writers**: Output processed data and results

**Writing Custom I/O Functions:**
1. Add a parser in the appropriate parser module
2. Implement or modify collate functions for batching
3. Optionally create custom sampling functions

**Example HDF5 Configuration:**
```yaml
io:
  writer:
    name: HDF5Writer
    file_name: output.h5
    input_keys: ['input_data', 'truth_labels']
    result_keys: ['predictions', 'metrics']
```

**Features:**
- Lazy loading for memory efficiency
- Multi-file dataset handling
- Automatic data type conversions
- Metadata preservation
- Progress tracking and logging
- Error recovery and validation

**Conditional Imports:**
PyTorch-dependent functionality is conditionally imported and available
only when PyTorch is installed. Check `TORCH_IO_AVAILABLE` flag.
"""

# Always import PyTorch-independent core functionality
from .core import *

# Conditionally import PyTorch-dependent functionality
try:
    from .torch import *

    TORCH_IO_AVAILABLE = True
except ImportError:
    TORCH_IO_AVAILABLE = False
