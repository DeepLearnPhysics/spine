# SPINE PyPI Deployment Guide

This guide explains how to deploy SPINE to PyPI with multiple package variants.

## Package Variants

SPINE is distributed as multiple packages to accommodate different use cases:

### 1. `spine-ml` (Full Package)
- **Description**: Complete SPINE package with ML dependencies
- **Target Users**: ML researchers, neural network developers
- **Dependencies**: Core + PyTorch + scikit-learn + numba
- **Excludes**: MinkowskiEngine (requires special installation)

### 2. `spine-ml-core` (Minimal Package)
- **Description**: Core data structures and utilities only
- **Target Users**: Data processing, visualization, analysis
- **Dependencies**: Only numpy, scipy, pandas, PyYAML, h5py
- **Use Cases**: Visualization-only workflows, data preprocessing

## Installation Commands for Users

```bash
# Minimal core package
pip install spine-ml-core

# Full ML package (recommended for most users)
pip install spine-ml[full,viz]

# Core + visualization only
pip install spine-ml-core[viz]

# Everything (development)
pip install spine-ml[all]
```

## Build Process

### Manual Build

1. **Prepare environment:**
   ```bash
   pip install build twine wheel
   ```

2. **Run build script:**
   ```bash
   ./build_packages.sh
   ```

3. **Check packages:**
   ```bash
   twine check dist/*
   ```

### Automated Build (GitHub Actions)

The repository includes automated building and publishing via GitHub Actions:

- **Trigger**: Push to version tags (e.g., `v0.6.2`) or releases
- **Process**: 
  1. Build both packages
  2. Run tests
  3. Publish to Test PyPI (tags)
  4. Publish to Production PyPI (releases)

## PyPI Accounts Setup

### Required API Tokens

1. **Test PyPI**: `TEST_PYPI_API_TOKEN`
2. **Production PyPI**: `PYPI_API_TOKEN`

### Package Names Registered

- `spine-ml`: Main package
- `spine-ml-core`: Minimal package

## Deployment Steps

### 1. Version Update
Update version in `spine/version.py`:
```python
__version__ = '0.6.2'
```

### 2. Manual Deployment

```bash
# Build packages
./build_packages.sh

# Test upload (optional)
twine upload --repository testpypi dist/*

# Production upload
twine upload dist/*
```

### 3. Automated Deployment

```bash
# Create and push version tag
git tag v0.6.2
git push origin v0.6.2

# Or create a GitHub release
# This will trigger the full build and deployment pipeline
```

## Package Configuration

### pyproject.toml (Full Package)
- Name: `spine-ml`
- Dependencies: Core + ML libraries
- Excludes: Heavy dependencies requiring special installation

### pyproject-core.toml (Minimal Package)
- Name: `spine-ml-core`
- Dependencies: Only essential scientific libraries
- Excludes: All ML model code (`spine/model/` directory)

## Testing Deployment

### Test Installation

```bash
# Test core package
pip install spine-ml-core
python -c "import spine; print(spine.__version__)"

# Test full package
pip install spine-ml
python -c "from spine.driver import Driver; print('Success')"

# Test console script
spine --help
```

### Test in Clean Environment

```bash
# Create virtual environment
python -m venv test_env
source test_env/bin/activate  # Linux/Mac
# test_env\Scripts\activate  # Windows

# Test installation
pip install spine-ml
```

## Troubleshooting

### Common Issues

1. **Import errors**: Check dependencies in pyproject.toml match actual usage
2. **Console script not working**: Verify entry point configuration
3. **Missing files**: Check MANIFEST.in includes all necessary files
4. **Build failures**: Ensure build dependencies are installed

### Dependency Notes

**Special Dependencies** (not included in PyPI packages):
- **MinkowskiEngine**: Requires CUDA/compilation, install separately
- **LArCV**: Requires ROOT, install via conda-forge
- **ROOT**: System-dependent, install separately

## Maintenance

### Regular Tasks

1. **Version updates**: Keep dependencies current
2. **Testing**: Ensure all variants work in clean environments
3. **Documentation**: Update installation instructions
4. **CI/CD**: Monitor GitHub Actions for failures

### Package Updates

When updating packages:
1. Update version in `spine/version.py`
2. Update dependencies if needed
3. Test locally with `./build_packages.sh`
4. Create version tag or GitHub release
5. Monitor automated deployment

## References

- [Python Packaging User Guide](https://packaging.python.org/)
- [PyPI Upload Guide](https://packaging.python.org/tutorials/packaging-projects/)
- [setuptools Documentation](https://setuptools.pypa.io/)
- [GitHub Actions for Python](https://docs.github.com/en/actions/automating-builds-and-tests/building-and-testing-python)