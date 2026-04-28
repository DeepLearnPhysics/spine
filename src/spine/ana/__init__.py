"""Analysis scripts and performance evaluation tools.

This module provides high-level analysis capabilities for neutrino physics
reconstruction results, building upon the output of the ML reconstruction
chain and post-processing modules.

Core analysis management:

- ``AnaManager`` orchestrates configuration-driven analysis workflows.

Analysis categories:

- Reconstruction-performance studies for PID, energy, spatial, angular, and vertex metrics.
- Physics analyses for topology, interaction type, and selection optimization.
- Quality-assessment workflows for truth matching, completeness, and systematic studies.

Output products:

- Histograms, resolution studies, efficiency curves, and publication-ready figures.

Example
-------

.. code-block:: python

  from spine.ana import AnaManager

  ana = AnaManager(config_file="analysis.yaml")
  results = ana.analyze(input_files, output_dir)
  ana.plot_performance(results)

This module serves as the final step in the SPINE reconstruction pipeline,
transforming raw ML outputs into physics-ready analysis results.
"""

from .manager import AnaManager
