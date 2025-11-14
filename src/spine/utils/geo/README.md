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

## Pixel Detectors (ndlar-flow files)

For pixel-based liquid argon TPCs (2x2, ND-LAr, Module-0, etc.) that use charge-light reconstruction.

### Requirements
- Access to ndlar-flow or larnd-sim geometry files
- Detector configuration YAML from simulation/reconstruction

### Method: Using ndlar-flow/larnd-sim Geometry

1. **Locate the detector geometry file:**
   ```bash
   # ndlar-flow geometries are typically in:
   # - ndlar_flow/reco/config/
   # - ndlar_flow/detector_properties/
   ```

2. **Key files to reference:**
   - `detector_properties.yaml` - Contains TPC dimensions, pixel pitch
   - `module_*.yaml` - Module-specific layouts
   - Simulation configuration files with detector dimensions

3. **Extract geometry information:**
   
   From the ndlar-flow configuration:
   ```yaml
   # Look for these parameters:
   tpc_centers:        # Center positions of each TPC module [cm]
   tpc_borders:        # Boundaries of active volume [cm]
   pixel_pitch:        # Pixel size [cm]
   drift_length:       # Drift distance [cm]
   ```

4. **Create the SPINE geometry file:**
   
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
detector: 2x2
tpc:
  boundaries:
    - [-67.5, -67.5, 0.0]
    - [67.5, 67.5, 100.8]
  pixel_pitch: 0.434
```

#### ND-LAr (ArgonCube 2x2 modules)
```yaml
detector: ndlar
tpc:
  boundaries:
    - [-304.4, -304.4, -635.95]
    - [304.4, 304.4, 635.95]
  modules:
    # 35 modules in a 5x7 grid
    # Each module: 101.6 cm × 101.6 cm × 101.6 cm
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

---

## Validation

After creating a geometry file, validate it:

```python
from spine.utils.geo import Geometry

# Load your geometry
geo = Geometry('your_detector')

# Check TPC boundaries
print(f"TPC boundaries: {geo.tpc.boundaries}")
print(f"TPC dimensions: {geo.tpc.dimensions}")
print(f"TPC center: {geo.tpc.center}")

# Visualize (if CRT is defined)
if geo.crt is not None:
    print(f"CRT boundaries: {geo.crt.boundaries}")

# Test point containment
test_point = [0.0, 0.0, 0.0]
print(f"Is {test_point} in TPC? {geo.tpc.contains(test_point)}")
```

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

---

## References

- **LArSoft Geometry**: https://larsoft.github.io/LArSoftWiki/Geometry
- **ndlar-flow**: https://github.com/DUNE/ndlar_flow
- **larnd-sim**: https://github.com/DUNE/larnd-sim

For questions or issues, please open an issue on the SPINE GitHub repository.
