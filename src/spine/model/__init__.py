"""Machine learning models for neutrino physics reconstruction.

This module handles the construction, training, and execution of deep learning models
for liquid argon time projection chamber (LArTPC) data analysis.

**Model Management:**
- `ModelManager`: Central class for model lifecycle management
  - Model instantiation and configuration
  - Training loop orchestration
  - Validation and testing
  - Checkpoint saving and loading
  - Multi-GPU support

**Supported Model Architectures:**
- **UResNet**: U-Net with ResNet blocks for semantic segmentation
- **PPN**: Point Proposal Network for particle endpoint detection
- **SPICE**: Spatial Point cloud Instance Clustering and Embedding
- **GrapPA**: Graph neural networks for particle analysis
- **Chain Models**: End-to-end reconstruction pipelines

**Key Features:**
- Modular architecture with configurable components
- Support for sparse and dense convolutions
- Graph neural network implementations
- Multi-task learning capabilities
- Automatic mixed precision training
- Distributed training support
- Comprehensive logging and monitoring

**Training Pipeline:**
1. Data loading and preprocessing
2. Model architecture selection
3. Loss function configuration
4. Optimizer and scheduler setup
5. Training loop with validation
6. Model evaluation and metrics
7. Result visualization and analysis

**Example Usage:**
```python
from spine.model import ModelManager

# Initialize model manager with configuration
manager = ModelManager(config)

# Train the model
manager.train(train_loader, val_loader)

# Run inference
results = manager.forward(data_batch)
```

The module integrates with the broader SPINE ecosystem for data I/O,
visualization, and post-processing workflows.
"""

from .manager import ModelManager
