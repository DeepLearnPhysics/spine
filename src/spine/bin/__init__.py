"""SPINE Command Line Interface and Utility Scripts.

This module provides command-line tools for the SPINE machine learning framework,
including the main consolidated CLI and specialized data processing utilities.

Main Components
---------------

**Primary CLI**:
    run.py : Consolidated command-line interface for training, validation,
             inference, and analysis with conditional PyTorch imports

**Data Processing Utilities**:
    larcv_check_valid.py : Validate LArCV files for corruption
    larcv_count_entries.py : Count entries in LArCV files
    larcv_find_duplicates.py : Find duplicate events in datasets
    larcv_find_run.py : Find specific run numbers in datasets
    larcv_inject_run_number.py : Inject run numbers into LArCV files
    output_check_valid.py : Validate SPINE analysis output files

Architecture
------------
The CLI system uses conditional imports to gracefully handle optional dependencies
like PyTorch. The main `spine` command provides access to all functionality through
subcommands, while individual utility scripts can be run independently for specific
data processing tasks.

Usage Examples
--------------
Main CLI usage::

    spine train --config config/train_uresnet.cfg --source data.h5
    spine validate --config config/train_uresnet.cfg --weight-path weights.ckpt
    spine analyze --config config/analysis.cfg --source data.h5 --output results.h5

Utility script usage::

    python -m spine.bin.larcv_check_valid input.root
    python -m spine.bin.larcv_count_entries data/*.root
    python -m spine.bin.output_check_valid output.h5

Notes
-----
- All CLI tools handle missing PyTorch gracefully through conditional imports
- Data processing utilities work independently of ML dependencies
- Configuration files use YAML format with hierarchical structure
- Output validation ensures analysis results meet expected format requirements
"""
