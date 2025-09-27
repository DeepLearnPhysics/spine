"""Analysis scripts and performance evaluation tools.

This module provides high-level analysis capabilities for neutrino physics
reconstruction results, building upon the output of the ML reconstruction
chain and post-processing modules.

**Core Analysis Management:**
- `AnaManager`: Central orchestrator for analysis workflows
  - Configuration-driven analysis pipeline
  - Multi-file dataset processing
  - Result aggregation and export
  - Performance metrics calculation

**Analysis Categories:**

**Reconstruction Performance:**
- Particle identification efficiency and purity
- Energy resolution and bias studies
- Spatial resolution metrics
- Angular reconstruction accuracy
- Vertex finding performance

**Physics Analysis:**
- Event topology classification
- Neutrino interaction type identification
- Cross-section measurements
- Background rejection studies
- Signal efficiency optimization

**Quality Assessment:**
- Truth matching quality metrics
- Reconstruction completeness
- Fragment and particle merging/splitting rates
- Detector systematic effects
- Data/MC comparison studies

**Output Products:**
- Statistical summaries and histograms
- Performance plots and visualizations
- Efficiency/purity curves
- Resolution studies
- Publication-ready figures
- Analysis reports and tables

**Workflow Integration:**
```
ML Chain → Post-Processing → Analysis → Physics Results
    ↓              ↓              ↓           ↓
 Raw Reco      High-level     Metrics     Publications
 Objects       Physics        Plots       Papers
              Quantities     Tables      Talks
```

**Key Features:**
- Automated truth matching and validation
- Configurable analysis cuts and selections
- Statistical uncertainty propagation
- Systematic error evaluation
- Batch processing capabilities
- Interactive analysis notebooks
- Custom metric definitions

**Example Usage:**
```python
from spine.ana import AnaManager

# Initialize analysis with configuration
ana = AnaManager(config_file='analysis.yaml')

# Run full analysis pipeline
results = ana.analyze(input_files, output_dir)

# Generate performance plots
ana.plot_performance(results)
```

This module serves as the final step in the SPINE reconstruction pipeline,
transforming raw ML outputs into physics-ready analysis results.
"""

from .manager import AnaManager
