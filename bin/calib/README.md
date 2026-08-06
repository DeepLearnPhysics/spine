# Calibration Scripts

Small diagnostic and validation scripts for calibration inputs and corrections.

## Inspect a space-charge field map

`sce_field_check.py` samples points in an ICARUS or SBND detector geometry,
applies a ROOT TH3 space-charge displacement map, and writes one PNG for each
of the x, y, and z displacement components.

```bash
python3 bin/calib/sce_field_check.py \
  --detector icarus \
  --map-file /path/to/SCEoffsets_ICARUS_E500_voxelTH3.root \
  --map-prefix TrueFwd_Displacement \
  --output-dir sce/plots
```

By default the script samples 50,000 reproducible points across the detector
TPC geometry. `--sample-volume` can instead select the map bounds or the
intersection of map and detector bounds. `--bounds` controls how the field map
handles points outside its domain. Plot density and output size can be adjusted
with `--num-points`, `--point-size`, and `--dpi`.

The script requires the optional ROOT and Matplotlib dependencies used by the
field-map reader and plotting backend.

Position-dependent maps can be selected with ordered suffixes and boundaries.
For example, this uses ``_E`` for x below zero and ``_W`` at or above zero:

```bash
python3 bin/calib/sce_field_check.py \
  --detector sbnd \
  --map-file /path/to/SCEoffsets_SBND_E500_dualmap_voxelTH3.root \
  --map-prefix TrueFwd_Displacement \
  --map-suffixes _E _W \
  --map-selection-axis 0 \
  --map-selection-boundaries 0
```

Use ``--map-suffix _E`` instead to apply the `_E` map everywhere.
