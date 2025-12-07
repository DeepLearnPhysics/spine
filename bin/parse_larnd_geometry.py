#!/usr/bin/env python3
"""
Parse LArND geometry HDF5 files and extract relevant detector information
in YAML format to be used by SPINE.

Example usage:
    python3 parse_larnd_geometry.py --source FSD_CosmicRun3.flow.0000000.FLOW.hdf5 --output larnd_geometry.yaml
"""

import argparse
from collections import defaultdict

import h5py
import numpy as np
import yaml


def main(source, output=None, opdet_thickness=None):
    """
    Main function for parsing LArND geometry HDF5 files.

    Parameters
    ----------
    source : str
        Path to the LArND geometry HDF5 file
    output : str, optional
        Path to output YAML file (if None, define it based on input file name)
    opdet_thickness : float, optional
        Thickness for optical detectors if calculated z extent is 0

    Returns
    -------
    None
    """

    # Read geometry bounds and optical info from HDF5 file
    with h5py.File(source, "r") as f:
        # Extract metadata
        class_version = f["geometry_info"].attrs["class_version"]
        runlist_file = f["run_info"].attrs["runlist_file"]

        # Parse detector name from runlist file path
        # e.g., 'data/proto_nd_flow/runlist-2x2-mcexample.txt' -> '2x2'
        # e.g., 'data/fsd_flow/runlist.fsd.txt' -> 'fsd'
        detector_name = None
        if "2x2" in runlist_file:
            detector_name = "2x2"
        elif "fsd" in runlist_file.lower():
            detector_name = "fsd"
        else:
            # Fallback: extract from directory name
            parts = runlist_file.split("/")
            if len(parts) > 1:
                dir_name = parts[1]  # e.g., 'proto_nd_flow' or 'fsd_flow'
                detector_name = dir_name.replace("_flow", "")

        bounds = np.array(f["geometry_info"].attrs["module_RO_bounds"])
        cathode_thickness = float(
            np.array(f["geometry_info"].attrs["cathode_thickness"])
        )

        # Extract TPC geometry information
        positions = []
        dimensions = None
        tpc_z_centers = []  # Track TPC Z centers for optical offset calculation
        for i in range(bounds.shape[0]):
            min_coords = bounds[i, 0].astype(float).tolist()
            max_coords = bounds[i, 1].astype(float).tolist()
            center_x = round((min_coords[0] + max_coords[0]) / 2, 4)
            half_width = round((max_coords[0] - min_coords[0]) / 2, 4)
            tpc_width = round(half_width - cathode_thickness / 2, 4)
            center_y = round((min_coords[1] + max_coords[1]) / 2, 4)
            center_z = round((min_coords[2] + max_coords[2]) / 2, 4)
            tpc_z_centers.append(center_z)
            # Flip order: TPC 2 (higher x) first, then TPC 1 (lower x)
            positions.append([round(center_x + tpc_width / 2, 4), center_y, center_z])
            positions.append([round(center_x - tpc_width / 2, 4), center_y, center_z])
            dims = [
                round(tpc_width, 4),
                round(max_coords[1] - min_coords[1], 4),
                round(max_coords[2] - min_coords[2], 4),
            ]
            if dimensions is None:
                dimensions = dims
        module_ids = [i for i in range(bounds.shape[0]) for _ in range(2)]

        # Extract optical detector information from HDF5 groups
        det_id_data = f["geometry_info"]["det_id"]["data"][:]
        sipm_abs_pos_data = f["geometry_info"]["sipm_abs_pos"]["data"][:]
        det_bounds_data = f["geometry_info"]["det_bounds"]["data"][:]

        # Filter to only include valid optical detectors (validity flag True and det_id >= 0)
        valid_indices = [i for i, x in enumerate(det_id_data) if x[1] and x[0] >= 0]
        # Further filter to only include those with valid bounds
        valid_indices = [
            i for i in valid_indices if det_bounds_data[i % len(det_bounds_data)][1]
        ]

        # Collect all valid detector info
        all_det_info = []
        for idx in valid_indices:
            det_id = int(det_id_data[idx][0])
            pos = sipm_abs_pos_data[idx][0]

            # Calculate dimensions from bounds
            bounds_idx = idx % len(det_bounds_data)
            min_pos, max_pos = det_bounds_data[bounds_idx][0]
            x_extent = float(max_pos[0] - min_pos[0])
            y_extent = float(max_pos[1] - min_pos[1])
            z_extent = float(max_pos[2] - min_pos[2])
            if z_extent == 0:
                if opdet_thickness is None:
                    raise ValueError(
                        "Calculated z extent is 0, please provide --opdet-thickness argument."
                    )
                z_extent = opdet_thickness

            all_det_info.append(
                {
                    "det_id": det_id,
                    "position": [float(pos[0]), float(pos[1]), float(pos[2])],
                    "dimensions": [x_extent, y_extent, z_extent],
                }
            )

        # Aggregate optical detectors: Group by det_id (each det_id is mirrored across TPCs)
        # Project to central plane (x=0) for position

        aggregated_positions = []
        aggregated_det_ids = []
        aggregated_shape_ids = []

        # Track unique dimension types
        unique_dimensions = {}

        # Get unique TPC Z centers to calculate optical offsets
        unique_tpc_z = sorted(set(tpc_z_centers))

        # Find the Z value closest to TPC center for each hemisphere
        # This represents where optical detectors are positioned relative to TPC
        z_positive = [
            info["position"][2] for info in all_det_info if info["position"][2] > 0
        ]
        z_negative = [
            info["position"][2] for info in all_det_info if info["position"][2] < 0
        ]

        unique_z_pos = sorted(set(z_positive))
        unique_z_neg = sorted(set(z_negative))

        # Find TPC centers for each hemisphere and calculate optical offset
        if len(unique_tpc_z) > 0:
            tpc_z_pos = (
                [z for z in unique_tpc_z if z > 0][0]
                if any(z > 0 for z in unique_tpc_z)
                else 0
            )
            tpc_z_neg = (
                [z for z in unique_tpc_z if z < 0][0]
                if any(z < 0 for z in unique_tpc_z)
                else 0
            )

            # Optical detectors are at the innermost Z position (smallest absolute value)
            optical_z_abs_pos = (
                min(unique_z_pos, key=abs) if unique_z_pos else tpc_z_pos
            )
            optical_z_abs_neg = (
                max(unique_z_neg, key=lambda z: -abs(z)) if unique_z_neg else tpc_z_neg
            )

            # Calculate offset from TPC center
            optical_z_pos = round(float(optical_z_abs_pos - tpc_z_pos), 2)
            optical_z_neg = round(float(optical_z_abs_neg - tpc_z_neg), 2)
        else:
            optical_z_pos = 0
            optical_z_neg = 0

        # Group detectors by det_id only
        det_id_groups = defaultdict(lambda: {"positions": [], "dims": None})

        for info in all_det_info:
            det_id = info["det_id"]
            det_id_groups[det_id]["positions"].append(info["position"])
            if det_id_groups[det_id]["dims"] is None:
                det_id_groups[det_id]["dims"] = info["dimensions"]

        # Sort det_ids for consistent ordering
        sorted_det_ids = sorted(det_id_groups.keys())

        for det_id in sorted_det_ids:
            group = det_id_groups[det_id]
            det_positions = group["positions"]

            # Calculate average Y position (project X to 0, average across all instances)
            avg_y = round(float(np.mean([p[1] for p in det_positions])), 2)

            # Determine which hemisphere this det_id primarily belongs to
            # Assign offset with matching sign to hemisphere
            z_values = [p[2] for p in det_positions]
            avg_z_raw = np.mean(z_values)
            if avg_z_raw > 0:
                avg_z = -optical_z_pos  # positive hemisphere gets negative offset
            else:
                avg_z = -optical_z_neg  # negative hemisphere gets positive offset

            # Use x=0 for all positions (central plane)
            aggregated_positions.append([0, avg_y, avg_z])

            # Determine shape ID based on y extent (height)
            dims = group["dims"]
            shape_id = 0 if dims[1] > 20 else 1
            aggregated_shape_ids.append(shape_id)

            # Determine mirror count based on det_id pattern
            # Det_ids 0, 4, 8, 12, ... (multiples of 4) appear 6 times (mirrored across 3 module pairs)
            # Other det_ids appear 2 times (mirrored across 1 module pair)
            if det_id % 4 == 0:
                mirror_count = 6
                # Update shape_id to 0 for these (they're the "tall" category)
                aggregated_shape_ids[-1] = 0
            else:
                mirror_count = 2
                # Update shape_id to 1 for these
                aggregated_shape_ids[-1] = 1

            aggregated_det_ids.extend([det_id] * mirror_count)

            # Track unique dimension type
            dim_key = (round(dims[0], 2), round(dims[1], 2), round(dims[2], 2))
            if dim_key not in unique_dimensions:
                unique_dimensions[dim_key] = list(dim_key)

        # Create final dimensions list (2 unique types sorted by shape_id: tall first, then short)
        optical_dimensions = sorted(
            unique_dimensions.values(), key=lambda d: d[1], reverse=True
        )

        # Compose the YAML structure
        geometry_yaml = {}

        # Add metadata fields
        if detector_name:
            geometry_yaml["name"] = detector_name.upper()
            geometry_yaml["tag"] = detector_name.lower()
        geometry_yaml["version"] = class_version

        # Add geometry sections
        geometry_yaml["tpc"] = {
            "dimensions": dimensions,
            "module_ids": module_ids,
            "positions": positions,
        }
        geometry_yaml["optical"] = {
            "volume": "tpc",
            "shape": ["box", "box"],
            "mirror": True,
            "dimensions": optical_dimensions,
            "shape_ids": aggregated_shape_ids,
            "det_ids": aggregated_det_ids,
            "positions": aggregated_positions,
        }

        # Print the YAML structure to console
        print("TPC section for YAML file:")
        print("=" * 60)
        print(yaml.dump(geometry_yaml, default_flow_style=None, sort_keys=False))
        print("=" * 60)

        # Write the YAML to file
        output_path = (
            output if output is not None else source.replace(".hdf5", "_geometry.yaml")
        )
        with open(output_path, "w", encoding="utf-8") as fout:
            yaml.dump(geometry_yaml, fout, default_flow_style=None, sort_keys=False)
        print(f"\nYAML file written to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse LArND geometry HDF5 files")

    parser.add_argument(
        "--source",
        "-s",
        help="Path to the LArND geometry HDF5 file",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--output",
        "-o",
        help="Path to output YAML file (optional)",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--opdet-thickness",
        help="Thickness for optical detectors if z extent is 0",
        type=float,
    )

    args = parser.parse_args()

    main(source=args.source, output=args.output, opdet_thickness=args.opdet_thickness)
