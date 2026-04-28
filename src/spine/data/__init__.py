"""Data structures and containers.

This module defines all core data structures used throughout the SPINE package,
from low-level detector data to high-level physics objects.

**Core Data Structures (Always Available):**
- `particle`: Particle objects with kinematics and properties
- `neutrino`: Neutrino interaction data structures
- `optical`: Photon detector and light information
- `crt`: Cosmic Ray Tagger hit data
- `trigger`: Trigger and timing information
- `meta`: Metadata and run information containers
- `run_info`: Run-level data and configuration
- `list`: List-based data containers and utilities
- `out`: High-level representation output containers
  - `Reco/TruthFragment`: Particle fragment with kinematics and properties
  - `Reco/TruthParticle`: Particle with kinematics and properties
  - `Reco/TruthInteraction`: Interaction with associated particles
- `batch`: Efficient batched data structures for ML training
  - `TensorBatch`: Sparse tensor batching with automatic collation/splitting
  - `EdgeIndexBatch`: Graph edge index batching
  - `IndexBatch`: General index-based batching utilities

**Data Hierarchy:**
```
Raw Detector Data → Reconstructed Objects → Physics Analysis
     ↓                      ↓                     ↓
  Voxels/Hits       Particles/Tracks       Interactions
  CRT Hits          Showers/Clusters       Event Info
  PMT/SiPM          Vertices               Neutrinos
```

**Key Features:**
- **Inheritance-based design** with shared base classes
- **Automatic docstring merging** for comprehensive help()
- **Lazy evaluation** for memory efficiency
- **Type safety** with proper annotations
- **Serialization support** for I/O operations
- **GPU compatibility** through PyTorch integration

**Units Metadata Convention:**
Fields specify units in metadata using one of two approaches:

1. **Fixed Units** (e.g., 'MeV', 'ns', 'GeV/c'):
   For physical quantities independent of coordinate representation:
   - Energy: 'MeV', 'GeV'
   - Time: 'ns', 'us', 's'
   - Momentum: 'MeV/c', 'GeV/c'
   - dE/dx: 'MeV/cm'

2. **Instance Units** ('instance'):
   For spatial quantities that transform with to_cm()/to_px():
   - Position coordinates
   - Distances and lengths
   - Spatial extents

The `field_units` property dynamically resolves 'instance' to the current
coordinate system ('cm' or 'px'). See `spine.data.base` module docstring
for detailed convention and examples.

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
from .larcv import *
from .list import *
from .out import *
