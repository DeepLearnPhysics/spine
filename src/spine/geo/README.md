# Geometry File Generation Guide

This guide explains how to produce geometry files for different detector types to use with SPINE.

## Overview

SPINE uses YAML geometry files that describe:
- **TPC (Time Projection Chamber)**: Active detector volume boundaries and properties
- **CRT (Cosmic Ray Tagger)**: Optional cosmic ray veto system geometry
- **Optical System**: Optional light detection system configuration

Geometry files are located in `spine/utils/geo/source/` and can be loaded using:
```python
from spine.utils.geo import Geometry
geo = Geometry('detector_name')
```

---

## LArSoft Geometries (DUNE Wire TPCs + SBN)

For wire-based liquid argon TPCs (SBND, ICARUS, ProtoDUNE, etc.) that use LArSoft geometry services.

### Requirements
- LArSoft environment with the detector's geometry configuration
- Access to dump the LArSoft geometry to a text file

### Method 1: Automated Extraction (Recommended)

**This is the easiest and most reliable method for extracting TPC geometry from LArSoft.**

1. **Dump the LArSoft geometry to a text file:**
   
   In your LArSoft environment, use the geometry dump utility:
   ```bash
   source /path/to/larsoft/setup
   setup <detector>  # e.g., setup sbndcode, setup icaruscode
   
   # Dump the geometry (replace with your detector's dump FCL file)
   lar -c dump_<detector>_geometry.fcl > geometry_dump.txt
   
   # For example:
   # lar -c dump_sbnd_geometry.fcl
   # lar -c dump_icarus_geometry.fcl
   ```

2. **Run the SPINE geometry parser:**
   
   Use the `parse_larsoft_geometry.py` script to automatically extract TPC information:
   
   ```bash
   # Basic usage (works for ICARUS, ProtoDUNE, etc.)
   python bin/parse_larsoft_geometry.py --source geometry_dump.txt --output detector_geometry.yaml
   
   # For SBND (with 4mm cathode and 3mm pixels):
   python bin/parse_larsoft_geometry.py \
       --source SBND-geometry.txt \
       --output sbnd_geometry.yaml \
       --cathode-thickness 0.4 \
       --pixel-size 0.3
   
   # For ICARUS (no adjustments needed):
   python bin/parse_larsoft_geometry.py \
       --source ICARUS-geometry.txt \
       --output icarus_geometry.yaml
   ```
   
   **Command-line options:**
   - `--source, -s`: Path to the geometry dump text file (required)
   - `--output, -o`: Path for output YAML file (optional, defaults to `<source>_tpc.yaml`)
   - `--cathode-thickness, -c`: Cathode thickness in cm (default: 0.0)
     - For SBND with 4mm cathode, use `0.4`
     - Reduces drift distance and adjusts TPC centers
   - `--pixel-size, -p`: Pixel size in cm (default: 0.0)
     - For SBND with 3mm pixels, use `0.3`
     - Pads the outer X edge to account for pixel pitch

3. **The script automatically:**
   - Parses all TPCs from the geometry dump
   - Groups TPCs that share wire planes (e.g., ICARUS pairs T:0+T:1, T:2+T:3)
   - Extracts accurate dimensions from wire coverage and active volume boxes
   - Applies cathode thickness and pixel size corrections if specified
   - Generates a YAML file with the TPC section ready to use

4. **Example output:**
   ```yaml
   tpc:
     dimensions: [148.2, 316.82, 1789.902]
     module_ids: [0, 0, 1, 1]
     det_ids: [0, 0, 1, 1]
     positions:
     - [-284.39, -23.45, 0.0]
     - [-136.04, -23.45, 0.0]
     - [136.04, -23.45, 0.0]
     - [284.39, -23.45, 0.0]
   ```

5. **Move the file to the geometry source directory:**
   ```bash
   mv detector_geometry.yaml src/spine/utils/geo/source/
   ```

### Method 2: Manual Extraction (Legacy)

**Note:** The automated script (Method 1) is preferred. Use this method only if you need to extract CRT or optical system information not provided by the automated script.

1. **Set up the LArSoft environment:**
   ```bash
   source /path/to/larsoft/setup
   setup <detector>  # e.g., setup sbndcode, setup icaruscode
   ```

2. **Extract geometry information:**
   
   Create a simple LArSoft analyzer or use an interactive ROOT session to access the geometry service:
   
   ```cpp
   // In a LArSoft analyzer or ROOT macro
   art::ServiceHandle<geo::Geometry> geom;
   
   // Get TPC boundaries (active volume)
   auto const& tpc = geom->TPC(0, 0);  // TPC 0, Cryostat 0
   double x_min = tpc.MinX();
   double x_max = tpc.MaxX();
   double y_min = tpc.MinY();
   double y_max = tpc.MaxY();
   double z_min = tpc.MinZ();
   double z_max = tpc.MaxZ();
   
   // Get detector properties
   auto const detprop = art::ServiceHandle<detinfo::DetectorPropertiesService>()->provider();
   double drift_velocity = detprop->DriftVelocity();  // cm/μs
   ```

3. **Get CRT information (if available):**
   ```cpp
   // CRT geometry varies by detector
   // For SBND/ICARUS, consult the CRT geometry files
   // Extract positions, dimensions, and orientations of CRT modules
   ```

4. **Create the YAML file:**
   
   Use the template below and fill in the extracted values:
   
   ```yaml
   # Example: sbnd_geometry.yaml
   detector: sbnd
   
   tpc:
     # Active volume boundaries [cm]
     boundaries:
       - [-210.215, -0.0, 0.0]      # [x_min, y_min, z_min]
       - [210.215, 400.0, 500.0]     # [x_max, y_max, z_max]
     
     # Detector properties
     drift_velocity: 0.1114  # cm/μs
     time_sampling: 0.5      # μs (time tick)
     
   crt:  # Optional
     # CRT module positions and dimensions
     boundaries:
       - [-230.0, -50.0, -50.0]  # Extended volume including CRT
       - [230.0, 450.0, 550.0]
     
     modules:
       # List of CRT module specifications
       # Each module: position, dimensions, orientation
   ```

### Method 3: Using GDML Files (Advanced)

**Note:** Use the automated script (Method 1) for TPC geometry. This method is primarily for extracting CRT or other detector components.

1. **Locate the GDML file:**
   ```bash
   # GDML files are typically in the detector's geometry directory
   # e.g., $SBNDCODE_DIR/gdml/ or $ICARUSCODE_DIR/gdml/
   ```

2. **Parse the GDML:**
   - Use a GDML parser or ROOT's TGeoManager
   - Extract TPC volume definitions (look for "volTPC" or similar)
   - Extract detector hall and CRT volumes

3. **Extract key parameters from LArSoft fcl files:**
   ```bash
   # Look in standard_*.fcl files for detector properties
   # e.g., services.DetectorPropertiesService.DriftVelocity
   ```

### Example Detectors

#### SBND (Short-Baseline Near Detector)
```yaml
detector: sbnd
tpc:
  boundaries:
    - [-210.215, -0.0, 0.0]
    - [210.215, 400.0, 500.0]
  drift_velocity: 0.1114  # cm/μs
```

#### ICARUS
```yaml
detector: icarus
tpc:
  boundaries:
    - [-395.0, -181.86, -894.951]
    - [395.0, 134.96, 894.951]
  drift_velocity: 0.16   # cm/μs
```

---

## Pixel Detectors (ndlar-flow/FLOW files)

For pixel-based liquid argon TPCs (2x2, ND-LAr, FSD, etc.) that use charge-light reconstruction.

### Requirements
- Access to FLOW HDF5 files from ndlar-flow or larnd-sim
- The `parse_flow_geometry.py` script in `bin/`

### Method: Automated Extraction from FLOW Files (Recommended)

**This is the easiest and most reliable method for extracting geometry from pixel detectors.**

1. **Locate a FLOW data file:**
   
   FLOW files contain embedded geometry information and are typically named:
   ```
   <detector>_<run>.flow.<event>.FLOW.hdf5
   # Examples:
   # FSD_CosmicRun3.flow.0000000.FLOW.hdf5
   # 2x2_run1.flow.0000000.FLOW.hdf5
   # ndlar_simulation.flow.0000000.FLOW.hdf5
   ```

2. **Run the SPINE geometry parser:**
   
   Use the `parse_flow_geometry.py` script to automatically extract both TPC and optical geometry:
   
   ```bash
   # Basic usage
   python bin/parse_flow_geometry.py \
       --source path/to/detector.flow.hdf5 \
       --tag mr6 \
       --opdet-thickness 1.0 \
       --output detector_geometry.yaml
   
   # For FSD with optical detectors
   python bin/parse_flow_geometry.py \
       --source FSD_CosmicRun3.flow.0000000.FLOW.hdf5 \
       --tag cr3 \
       --opdet-thickness 1.0 \
       --output fsd_cr3_geometry.yaml
   ```
   
   **Command-line options:**
   - `--source, -s`: Path to the FLOW HDF5 file (required)
   - `--tag, -t`: Tag to identify geometry revision, e.g., 'mr5', 'mr6' (required)
   - `--output, -o`: Path for output YAML file (optional, defaults to `<source>_geometry.yaml`)
   - `--opdet-thickness`: Thickness for optical detectors in cm (always needed)
     - Use when optical detector bounds have zero thickness
     - Typical value: 1.0 cm for ACL/LCM

3. **The script automatically:**
   - Extracts TPC module boundaries and positions from `module_RO_bounds`
   - Accounts for cathode thickness in TPC center calculations
   - Extracts optical detector geometry from `det_id` and `det_bounds` datasets
   - Groups optical detectors by unique dimensions (shape types)
   - Creates per-channel detector ID mappings for light reconstruction
   - Detects detector name from embedded geometry file references
   - Generates complete YAML with TPC and optical sections

4. **Example output:**
   ```yaml
   name: FSD
   tag: cr3
   version: 6.0
   crs_files: [data/fsd_flow/multi_tile_layout-3.0.40_NDLArModule_v3.yaml]
   lrs_file: data/fsd_flow/light_module_desc-6.0.1.yaml
   tpc:
     dimensions: [46.788, 297.6, 95.232]
     module_ids: [0, 0]
     positions:
     - [23.394, 0.0, 0.0]
     - [-23.394, 0.0, 0.0]
   optical:
     volume: tpc
     shape: [box, box]
     mirror: true
     dimensions:
     - [48.0, 31.02, 0.5]  # Tall modules
     - [48.0, 10.34, 0.5]  # Short modules
     shape_ids: [1, 1, 1, 0, ...]  # Shape ID per unique detector
     det_ids: [20, 39, 20, 39, ...]  # Detector ID per channel
     positions:
     - [0, -149.93, -49.0]
     - [0, -139.59, -49.0]
     # ... one position per unique detector
   ```

5. **Understanding the output:**
   - **TPC section:**
     - `dimensions`: [x, y, z] size of each TPC module in cm
     - `positions`: Center position of each TPC (accounting for cathode)
     - `module_ids`: Module identifier for each TPC
   
   - **Optical section** (if available):
     - `dimensions`: List of unique optical detector shapes [x, y, z] in cm
     - `positions`: Position of each unique optical detector (projected to x=0)
     - `shape_ids`: Shape type index for each unique detector
     - `det_ids`: Detector ID for each channel (multiple channels per detector)
     - `mirror`: Whether to mirror detectors across cathode

6. **Move the file to the geometry source directory:**
   ```bash
   mv detector_geometry.yaml src/spine/utils/geo/source/
   ```

### Manual Extraction (Legacy)

**Note:** The automated script is strongly preferred. Use manual extraction only if you need to customize geometry not provided by the automated script.

1. **Locate the detector geometry configuration:**
   ```bash
   # ndlar-flow geometries are typically in:
   # - ndlar_flow/reco/config/
   # - ndlar_flow/detector_properties/
   ```

2. **Extract geometry information from configuration files:**
   
   From the ndlar-flow configuration:
   ```yaml
   # Look for these parameters:
   tpc_centers:        # Center positions of each TPC module [cm]
   tpc_borders:        # Boundaries of active volume [cm]
   pixel_pitch:        # Pixel size [cm]
   drift_length:       # Drift distance [cm]
   ```

3. **Create the SPINE geometry file:**
   
   ```yaml
   # Example: 2x2_geometry.yaml
   detector: 2x2
   
   tpc:
     boundaries:
       - [-67.5, -67.5, 0.0]      # [x_min, y_min, z_min]
       - [67.5, 67.5, 100.8]       # [x_max, y_max, z_max]
     
     # For multi-module detectors, specify module layouts
     modules:
       - id: 0
         center: [-33.75, -33.75, 50.4]  # Module center [cm]
         boundaries:
           - [-67.5, -67.5, 0.0]
           - [0.0, 0.0, 100.8]
       
       - id: 1
         center: [33.75, -33.75, 50.4]
         boundaries:
           - [0.0, -67.5, 0.0]
           - [67.5, 0.0, 100.8]
       # ... additional modules
     
     # Pixel detector properties
     pixel_pitch: 0.434   # cm
     drift_velocity: 0.16  # cm/μs
   
   # Light system (optional)
   optical:
     # Light detector positions and properties
     detectors:
       - id: 0
         position: [0.0, 0.0, 50.4]  # [x, y, z] in cm
         type: "sipm"
   ```

### Multi-Module Detectors

For detectors with multiple TPC modules (2x2, ND-LAr, etc.):

1. **Extract module layout:**
   - Number of modules and their arrangement
   - Individual module boundaries and centers
   - Gap/dead space between modules

2. **Define overall detector boundaries:**
   - Bounding box that contains all modules
   - Include space between modules if applicable

3. **List individual modules:**
   - Each module with its own boundaries and center
   - Module ID matching the simulation/data convention

### Example Detectors

#### 2x2 Demonstrator
```yaml
name: 2x2
tag: mr6
version: 6.0
tpc:
  dimensions: [30.6, 129.6, 64.0]
  module_ids: [0, 0, 1, 1, 2, 2, 3, 3]
  positions:
    # Extracted from FLOW file with cathode corrections
optical:
  # Automatically extracted if available in FLOW file
```

#### FSD (Final System Demonstrator)
```yaml
name: FSD
tag: cr3
version: 6.0
crs_files: [data/fsd_flow/multi_tile_layout-3.0.40_NDLArModule_v3.yaml]
lrs_file: data/fsd_flow/light_module_desc-6.0.1.yaml
tpc:
  dimensions: [46.788, 297.6, 95.232]
  module_ids: [0, 0]
  positions:
  - [23.394, 0.0, 0.0]
  - [-23.394, 0.0, 0.0]
optical:
  volume: tpc
  shape: [box, box]
  mirror: true
  dimensions:
  - [48.0, 31.02, 0.5]  # 120 channels across tall modules
  - [48.0, 10.34, 0.5]  # 120 channels across short modules
  shape_ids: [1, 1, 1, 0, ...]  # 40 unique detectors
  det_ids: [20, 39, 20, 39, ...]  # 120 total channels
  positions: # 40 unique detector positions
    - [0, -149.93, -49.0]
    # ...
```

#### ND-LAr (ArgonCube 2x2 modules)
```yaml
name: ND-LAr
tag: mr6
tpc:
  # Multi-module configuration extracted from FLOW file
  # Accounts for 35 modules in 5×7 grid
```

---

## Coordinate System Conventions

**SPINE uses the following coordinate conventions:**

- **X-axis**: Horizontal, perpendicular to beam (drift direction for most detectors)
- **Y-axis**: Vertical (up is positive)
- **Z-axis**: Along the beam direction (for beam detectors)

**Units:**
- All distances in **centimeters [cm]**
- Time in **microseconds [μs]**
- Drift velocity in **cm/μs**

### Optical Detector Conventions

For optical detector geometries extracted from FLOW files:

- **Positions**: Projected to x=0 plane (cathode position)
- **det_ids**: Each channel has a detector ID (multiple channels per physical detector)
- **shape_ids**: Index into the dimensions list for each unique detector shape
- **Mirroring**: When `mirror: true`, detectors are mirrored across the cathode to both TPCs

**Channel-to-Detector Mapping:**
- Physical detectors may have multiple readout channels (e.g., SiPM arrays)
- `det_ids` array contains one entry per channel
- `positions` and `shape_ids` arrays contain one entry per unique physical detector
- Use `det_ids` to map channels to their corresponding position/shape

---

## Validation

After creating a geometry file, validate it:

```python
from spine.utils.geo import Geometry

# Load your geometry
geo = Geometry('your_detector')

# Check TPC information
print(f"TPC dimensions: {geo.tpc.dimensions}")
print(f"TPC positions: {geo.tpc.positions}")
print(f"Number of TPCs: {len(geo.tpc.positions)}")

# Check optical detector information (if available)
if hasattr(geo, 'optical') and geo.optical is not None:
    print(f"\nOptical detector dimensions: {geo.optical.dimensions}")
    print(f"Number of unique detectors: {len(geo.optical.positions)}")
    print(f"Number of channels: {len(geo.optical.det_ids)}")
    print(f"Number of shape types: {len(geo.optical.dimensions)}")
    
    # Verify channel-to-detector mapping
    unique_det_ids = len(set(geo.optical.det_ids))
    print(f"Unique detector IDs in channels: {unique_det_ids}")

# Test point containment (if boundaries defined)
if hasattr(geo.tpc, 'contains'):
    test_point = [0.0, 0.0, 0.0]
    print(f"\nIs {test_point} in TPC? {geo.tpc.contains(test_point)}")
```

### Validation Checklist

For **TPC geometry**:
- [ ] Number of TPC positions matches expected detector configuration
- [ ] TPC dimensions are physically reasonable
- [ ] Module IDs correctly identify separate detector modules
- [ ] Cathode corrections properly account for dead space

For **Optical geometry**:
- [ ] Number of unique detectors matches physical detector count
- [ ] Number of channels equals sum of channels across all detectors
- [ ] Shape IDs correctly index into dimensions list
- [ ] Positions are projected to x=0 (cathode plane)
- [ ] All det_ids in channel list reference valid detectors
- [ ] Dimensions include proper thickness (not zero unless intended)

---

## Adding New Geometries

To add a new detector geometry:

1. Create a YAML file in `spine/utils/geo/source/`
2. Name it `<detector_name>_geometry.yaml`
3. Follow the structure from existing examples
4. Validate using the code above
5. Submit a pull request with:
   - The geometry file
   - Source/reference for the geometry values
   - Validation output

---

## Common Issues

**Issue**: Coordinates don't match between SPINE and LArSoft
- **Solution**: Check coordinate system conventions (LArSoft may use different axes)
- Some detectors rotate coordinates between detector and "standard" conventions

**Issue**: Drift direction confusion
- **Solution**: Ensure x-direction is drift in your geometry
- Verify drift velocity sign matches your convention

**Issue**: Module boundaries overlap or have gaps
- **Solution**: Double-check module positions and boundaries
- Account for dead spaces or gaps between modules explicitly

**Issue**: Optical detector z-extent is zero
- **Solution**: Use `--opdet-thickness` argument to specify physical thickness
- Typical value: 0.5 cm for SiPM arrays
- This is common when detector bounds only specify the active plane

**Issue**: Optical detector channel count doesn't match expectations
- **Solution**: Verify that `det_ids` array has one entry per readout channel
- Physical detectors may have multiple channels (e.g., 4-12 per module)
- Check that filtering correctly accounts for multi-TPC configurations

**Issue**: Cathode corrections not properly applied
- **Solution**: Ensure `parse_flow_geometry.py` is using cathode_thickness from HDF5
- TPC positions should account for cathode dead space between drift regions
- Verify TPC width = (module_width - cathode_thickness) / 2

**Issue**: Geometry version mismatch between data and YAML
- **Solution**: Use `--tag` argument to specify the correct geometry revision
- Tag should match the detector configuration used in simulation/reconstruction
- Version is automatically extracted from tag (e.g., 'mr6' → version 6.0)

---

## References

- **LArSoft Geometry**: https://larsoft.github.io/LArSoftWiki/Geometry
- **ndlar-flow**: https://github.com/DUNE/ndlar_flow
- **larnd-sim**: https://github.com/DUNE/larnd-sim
- **FLOW File Format**: HDF5 files with embedded geometry information
  - `geometry_info/module_RO_bounds`: TPC module boundaries
  - `geometry_info/det_id`: Optical detector channel IDs
  - `geometry_info/det_bounds`: Optical detector physical boundaries

### Recent Improvements

**December 2024:**
- Added `parse_flow_geometry.py` script for automated extraction from FLOW HDF5 files
- Implemented optical detector geometry extraction with per-channel detector ID mapping
- Added support for multi-shape optical detector configurations
- Improved cathode thickness corrections for accurate TPC center calculations
- Added automatic detector name detection from embedded geometry file references
- Implemented geometry version extraction from tags (e.g., 'mr6' → 6.0)

For questions or issues, please open an issue on the SPINE GitHub repository.
