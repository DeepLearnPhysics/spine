"""Post-processing tools for ML-based neutrino reconstruction.

This module transforms raw ML model outputs into high-level physics quantities,
serving as the bridge between deep learning predictions and physics analysis.

**Core Post-Processing Management:**
- `PostManager`: Central coordinator for post-processing workflows
  - Multi-stage processing pipeline
  - Configurable algorithm chains
  - Quality control and validation
  - Output formatting and export

**Physics Reconstruction Algorithms:**

**Particle Reconstruction:**
- **Direction Reconstruction**: Principal component analysis and track fitting
- **Energy Estimation**: Range-based (CSDA) and calorimetric methods
- **Particle ID Refinement**: Chi-squared fits and likelihood methods
- **Vertex Finding**: Track intersection and clustering algorithms
- **Momentum Reconstruction**: Multiple Coulomb scattering analysis

**Detector Integration:**
- **Optical Flash Matching**: Time and spatial correlation with PMT signals
- **Cosmic Ray Tagging**: CRT hit matching and cosmic rejection
- **Trigger Matching**: Event timing and readout window association
- **Calibration Application**: Energy scale and detector response corrections

**Event-Level Processing:**
- **Interaction Classification**: Primary particle topology identification
- **Containment Analysis**: Fiducial volume and boundary effect studies
- **Background Rejection**: Cosmic ray and noise event filtering
- **Quality Metrics**: Reconstruction confidence and uncertainty estimation

**Algorithm Categories:**

**Tracking Algorithms:**
- 3D track fitting and smoothing
- Kinematic endpoint determination
- Track-shower separation
- Multiple track reconstruction

**Calorimetry:**
- Energy deposition analysis
- dE/dx calculation and calibration
- Range-energy relationships
- Electromagnetic shower energy

**Matching and Association:**
- Truth-to-reco object matching
- Cross-detector hit correlation
- Time-of-flight calculations
- Multi-plane track linking

**Data Flow:**
```
ML Predictions → Post-Processing → Physics Objects → Analysis
      ↓                ↓                 ↓           ↓
 Raw Outputs    High-level Reco   Final Objects   Physics
 Segmentation   Particle Props     with Truth     Results
 Clustering     Energy/Direction   Matching       Plots
 Classification PID/Kinematics     Quality Flags  Tables
```

**Output Products:**
- Reconstructed particles with full kinematics
- Interaction vertices and topologies
- Energy measurements with uncertainties
- Particle identification scores
- Truth matching associations
- Quality and confidence metrics

**Key Features:**
- Modular algorithm framework
- Configurable processing chains
- Comprehensive error propagation
- Multi-threaded processing support
- Extensive validation and monitoring
- Integration with detector calibration

**Example Configuration:**
```yaml
post:
  chain:
    - name: ParticleBuilder
      track_min_length: 3.0
    - name: EnergyReconstructor
      method: 'csda'
    - name: FlashMatcher
      time_window: 10.0
```

This module is essential for converting raw ML outputs into analysis-ready
physics objects with proper uncertainties and quality assessments.
"""

from .manager import PostManager
