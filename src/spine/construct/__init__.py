"""Data structure construction and analysis object building.

This module handles the construction of high-level analysis data classes from
ML reconstruction outputs and provides I/O capabilities for analysis workflows.

**Note**: The configuration block is still named `build:` for backward compatibility,
while the module is named `construct` to avoid git ignore conflicts.

**Core Construction Management:**
- `BuildManager`: Central factory for analysis object construction
  - Configuration-driven building pipeline
  - Truth-reconstruction matching
  - Quality validation and filtering
  - Batch processing capabilities

**Analysis Data Classes:**

**Fragment-Level Objects:**
- `RecoFragment`: Reconstructed cluster/fragment with ML predictions
- `TruthFragment`: Monte Carlo truth fragment with simulation info
  - Spatial coordinates and energy depositions
  - Particle association and containment
  - Reconstruction quality metrics
  - Truth matching and overlap calculations

**Particle-Level Objects:**
- `RecoParticle`: Reconstructed particle with kinematics and properties
- `TruthParticle`: Monte Carlo truth particle with full genealogy
  - 4-momentum and trajectory information
  - Particle identification and confidence
  - Parent-child relationships
  - Detector interaction points

**Interaction-Level Objects:**
- `RecoInteraction`: Complete reconstructed neutrino interaction
- `TruthInteraction`: Monte Carlo truth interaction with primaries
  - Primary vertex and topology
  - All associated particles and fragments
  - Event-level quantities and classifications
  - Cross-detector information (CRT, PMT)

**Core Functionality:**

**Object Construction:**
- **From ML Chain**: Build objects from raw model predictions
  - Segmentation and clustering results
  - Particle identification scores
  - Point proposal network outputs
  - Graph neural network predictions

**Data Loading:**
- **From HDF5**: Load pre-built objects from analysis files
  - Efficient hierarchical data access
  - Lazy loading for large datasets
  - Metadata preservation and validation
  - Cross-reference integrity checking

**Quality Control:**
- Object validation and consistency checks
- Truth matching quality assessment
- Reconstruction completeness metrics
- Error flagging and diagnostic info

**Data Hierarchy:**
```
ML Outputs → Build Process → Analysis Objects → Physics Analysis
    ↓            ↓              ↓                ↓
Predictions   Construction   Fragments         Performance
Clusters      Matching       Particles         Studies
Scores        Validation     Interactions      Results
```

**Object Relationships:**
- **Hierarchical Structure**: Interactions → Particles → Fragments
- **Truth Matching**: Reco objects linked to MC truth counterparts
- **Cross-References**: Bi-directional navigation between related objects
- **Inheritance**: Shared base classes with automatic docstring merging

**I/O Capabilities:**
- **HDF5 Export**: Structured storage with compression and metadata
- **Batch Processing**: Multi-file dataset construction
- **Memory Management**: Efficient loading for large-scale analysis
- **Format Validation**: Schema checking and version compatibility

**Key Features:**
- Type-safe object construction
- Automatic truth-reco association
- Comprehensive object validation
- Flexible configuration system
- Memory-efficient data structures
- Rich metadata preservation
- Cross-platform compatibility

**Example Usage:**
```python
from spine.construct import BuildManager

# Build objects from ML chain output
builder = BuildManager(config)
objects = builder.build_from_ml_chain(ml_results)

# Load objects from analysis file
objects = builder.load_from_file('analysis.h5')

# Access hierarchy
interaction = objects.interactions[0]
particles = interaction.particles  # List of RecoParticle
fragments = particles[0].fragments  # List of RecoFragment
```

This module provides the foundation for all downstream physics analysis
by creating well-structured, validated analysis objects with full provenance.
"""

from .manager import BuildManager
