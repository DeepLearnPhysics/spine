"""Data structures and containers for neutrino physics analysis.

This module defines all core data structures used throughout the SPINE package,
from low-level detector data to high-level physics objects.

**Core Data Structures (Always Available):**
- `out`: Generic output containers and base classes
- `particle`: Particle objects with kinematics and properties
- `neutrino`: Neutrino interaction data structures
- `optical`: Photon detector and light information
- `crt`: Cosmic Ray Tagger hit data
- `trigger`: Trigger and timing information
- `meta`: Metadata and run information containers
- `run_info`: Run-level data and configuration
- `list`: List-based data containers and utilities
- `batch`: Efficient batched data structures for ML training
  - `TensorBatch`: Sparse tensor batching with automatic collation
  - `EdgeIndexBatch`: Graph edge index batching
  - `IndexBatch`: General index-based batching utilities

**Data Hierarchy:**
```
Raw Detector Data → Reconstructed Objects → Physics Analysis
     ↓                      ↓                     ↓
  Voxels/Hits        Particles/Tracks      Interactions
  CRT Hits          Showers/Clusters       Event Info
  PMT/SiPM          Vertices              Neutrinos
```

**Key Features:**
- **Inheritance-based design** with shared base classes
- **Automatic docstring merging** for comprehensive help()
- **Lazy evaluation** for memory efficiency
- **Type safety** with proper annotations
- **Serialization support** for I/O operations
- **GPU compatibility** through PyTorch integration

**Base Class Pattern:**
All data structures inherit from base classes that provide:
- Common attributes (ID, size, coordinates, etc.)
- Truth matching capabilities
- Standardized string representations
- Property validation and type checking

**Conditional Availability:**
PyTorch-dependent batch structures are only available when PyTorch is installed.
Check the `BATCH_AVAILABLE` flag to determine availability.

**Example Usage:**
```python
from spine.data import Particle, TensorBatch

# Create physics objects
particle = Particle(id=0, pid=13, momentum=[1.2, 0.5, 2.1])

# Batch data for ML (torch handled conditionally internally)
batch = TensorBatch(data_list, batch_size=32)
```
"""

from .batch import *
from .crt import *
from .list import *
from .meta import *
from .neutrino import *
from .optical import *
from .out import *
from .particle import *
from .run_info import *
from .trigger import *
