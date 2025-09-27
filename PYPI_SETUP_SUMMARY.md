# SPINE PyPI Deployment - Complete Setup

This document summarizes the complete PyPI deployment setup for the SPINE package with multiple variants.

## 📦 Package Variants Created

### 1. `spine-ml` (Full Package)
- **Purpose**: Complete ML pipeline with neural networks
- **Dependencies**: numpy, scipy, pandas, PyYAML, h5py + torch, scikit-learn, numba
- **Target**: ML researchers, model developers
- **Install**: `pip install spine-ml[full,viz]`

### 2. `spine-ml-core` (Minimal Package)  
- **Purpose**: Data structures, utilities, visualization only
- **Dependencies**: numpy, scipy, pandas, PyYAML, h5py only
- **Target**: Data analysis, visualization, post-processing
- **Install**: `pip install spine-ml-core`

### 3. Optional Dependencies (Install Separately)
- **MinkowskiEngine**: Sparse convolutions (requires CUDA compilation)
- **LArCV**: LArTPC data I/O (requires ROOT, use conda-forge)
- **ROOT**: High-energy physics framework

## 🏗️ Files Created

### Core Configuration
- ✅ `pyproject.toml` - Full package configuration
- ✅ `pyproject-core.toml` - Core package configuration  
- ✅ `MANIFEST.in` - Package file inclusion rules
- ✅ `LICENSE` - MIT license file

### Scripts & Automation
- ✅ `build_packages.sh` - Manual build script
- ✅ `.github/workflows/publish.yml` - Automated CI/CD pipeline
- ✅ `test_installation.py` - Package validation script

### Documentation
- ✅ `README.md` - Updated with installation instructions
- ✅ `DEPLOYMENT.md` - Comprehensive deployment guide
- ✅ `requirements.txt` - Core dependencies
- ✅ `requirements-dev.txt` - Development dependencies

### Code Quality
- ✅ `.pre-commit-config.yaml` - Code formatting and linting
- ✅ Console script entry point: `spine` command

### Package Structure Updates
- ✅ Moved `bin/` scripts to `spine/bin/` for proper packaging
- ✅ Fixed import paths to work as installed package
- ✅ Updated console script configuration

## 🚀 Installation Options for Users

```bash
# Minimal core (data structures, utilities, basic I/O)
pip install spine-ml-core

# Core + visualization tools
pip install spine-ml-core[viz]

# Full ML package (recommended for most users)
pip install spine-ml[full,viz]

# Everything including development tools
pip install spine-ml[all]

# Special dependencies (install separately)
pip install MinkowskiEngine  # Requires CUDA
conda install -c conda-forge larcv  # Requires ROOT
```

## 🔧 Development Workflow

### Local Development
```bash
# Clone and install in development mode
git clone https://github.com/DeepLearnPhysics/SPINE.git
cd spine
pip install -e .[dev,all]

# Set up pre-commit hooks
pre-commit install
```

### Manual Build & Test
```bash
# Build both packages
./build_packages.sh

# Test installation
./test_installation.py

# Upload to PyPI
twine upload dist/*
```

### Automated Deployment
```bash
# Create version tag (triggers automated build & deployment)
git tag v0.6.2
git push origin v0.6.2

# Or create GitHub release for full deployment to production PyPI
```

## 📊 Deployment Pipeline

### GitHub Actions Workflow
1. **Trigger**: Version tags (`v*`) or GitHub releases
2. **Build**: Both package variants on multiple Python versions
3. **Test**: Import tests and basic functionality checks
4. **Deploy**: 
   - Tags → Test PyPI
   - Releases → Production PyPI

### Quality Checks
- ✅ Multi-Python version testing (3.8-3.11)
- ✅ Import validation for both package variants
- ✅ Code formatting with Black + isort
- ✅ Linting with flake8
- ✅ Package integrity with twine check

## 🎯 Usage Examples

### Command Line
```bash
# After installation, use the spine command
spine --config config/train_uresnet.cfg --source data.h5
```

### Python API
```python
import spine
from spine.driver import Driver

# Load and run configuration
cfg = {...}  # Load your config
driver = Driver(cfg)
driver.run()
```

### Visualization Only (Core Package)
```python
import spine.vis as vis
import spine.data as data

# Use visualization tools without ML dependencies
drawer = vis.Drawer()
# ... visualization code
```

## 🔍 Testing & Validation

### Package Validation
```bash
# Test core package
pip install spine-ml-core
python test_installation.py

# Test full package  
pip install spine-ml[full]
python test_installation.py
```

### Expected Outcomes
- **Core package**: Data structures, I/O, visualization work
- **Full package**: Additionally supports ML models and training
- **Console script**: `spine` command available after installation

## 📝 Next Steps

### For Deployment
1. **Set up PyPI accounts** and generate API tokens
2. **Add secrets** to GitHub repository:
   - `TEST_PYPI_API_TOKEN`
   - `PYPI_API_TOKEN`
3. **Create first release** or tag to trigger deployment
4. **Monitor** deployment and test installation

### For Users
- **Documentation**: Update project documentation with new install instructions
- **Examples**: Create example notebooks using `pip install spine-ml`
- **Migration**: Help existing users transition from manual installation

## 🛠️ Maintenance

### Regular Tasks
- Update dependency versions in `pyproject.toml`
- Monitor CI/CD pipeline for failures
- Test installation in clean environments
- Update documentation as needed

### Version Updates
1. Update `spine/version.py`
2. Test locally with build scripts
3. Create GitHub release to trigger deployment
4. Verify packages on PyPI

This setup provides a robust, maintainable deployment system that accommodates different user needs while maintaining code quality and reliable distribution.