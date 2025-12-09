#!/usr/bin/env python3
"""
Parse LArND geometry HDF5 files and extract relevant detector information
in YAML format to be used by SPINE.

Example usage:
    python3 parse_larnd_geometry.py --source FSD_CosmicRun3.flow.0000000.FLOW.hdf5
    --output larnd_geometry.yaml --opdet-thickness 1.0 --tag cr3
"""

import argparse
import re

import h5py
import numpy as np
import yaml


def extract_tpc_geometry(f):
    """
    Extract TPC geometry information from HDF5 file.

    Parameters
    ----------
    f : h5py.File
        Open HDF5 file handle

    Returns
    -------
    dict
        Dictionary containing:
        - positions: list of TPC center positions
        - dimensions: TPC dimensions [x, y, z]
        - module_ids: list of module IDs
    """
    bounds = np.array(f["geometry_info"].attrs["module_RO_bounds"])
    cathode_thickness = float(np.array(f["geometry_info"].attrs["cathode_thickness"]))

    positions = []
    dimensions = None

    for i in range(bounds.shape[0]):
        min_coords = bounds[i, 0].astype(float).tolist()
        max_coords = bounds[i, 1].astype(float).tolist()
        center_x = round((min_coords[0] + max_coords[0]) / 2, 4)
        half_width = round((max_coords[0] - min_coords[0]) / 2, 4)
        tpc_width = round(half_width - cathode_thickness / 2, 4)
        center_y = round((min_coords[1] + max_coords[1]) / 2, 4)
        center_z = round((min_coords[2] + max_coords[2]) / 2, 4)

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

    return {
        "positions": positions,
        "dimensions": dimensions,
        "module_ids": module_ids,
    }


def extract_optical_geometry(f, tpc_positions, opdet_thickness=None):
    """
    Extract optical detector geometry information from HDF5 file.

    Parameters
    ----------
    f : h5py.File
        Open HDF5 file handle
    tpc_positions : list
        List of TPC center positions
    opdet_thickness : float, optional
        Thickness for optical detectors if calculated z extent is 0

    Returns
    -------
    dict or None
        Dictionary containing optical geometry info, or None if not available:
        - dimensions: list of unique dimension types
        - positions: list of detector positions (one per channel)
        - det_ids: list of detector IDs (one per channel)
        - shape_ids: list of shape IDs (one per channel)
    """
    if "det_id" not in f["geometry_info"]:
        return None

    det_id_data = f["geometry_info"]["det_id"]["data"][:]
    det_bounds_data = f["geometry_info"]["det_bounds"]["data"][:]

    num_tpcs = len(tpc_positions)
    first_tpc_center = tpc_positions[0]

    # Build mapping: unique_det_id -> geometry info (from TPC 0)
    det_id_geometry = {}
    det_ids = np.array([x for x, valid in det_id_data if valid and x >= 0])
    unique_det_ids = sorted(set(det_ids))
    channel_det_ids = []

    for det_id in unique_det_ids:
        # For TPC 0: bounds_idx = 1 + det_id * num_tpcs
        bounds_idx = 1 + det_id * num_tpcs

        if bounds_idx >= len(det_bounds_data):
            continue

        bounds_entry = det_bounds_data[bounds_idx]
        if not bounds_entry[1]:  # Check validity
            continue

        min_pos, max_pos = bounds_entry[0]

        # Calculate center position
        center_x = float((min_pos[0] + max_pos[0]) / 2)
        center_y = float((min_pos[1] + max_pos[1]) / 2)
        center_z = float((min_pos[2] + max_pos[2]) / 2)

        # Calculate dimensions
        x_extent = float(max_pos[0] - min_pos[0])
        y_extent = float(max_pos[1] - min_pos[1])
        z_extent = float(max_pos[2] - min_pos[2])
        if z_extent == 0:
            if opdet_thickness is None:
                raise ValueError(
                    "Calculated z extent is 0, please provide --opdet-thickness argument."
                )
            z_extent = opdet_thickness

        # Offset position relative to first TPC center
        rel_x = center_x - first_tpc_center[0]
        rel_y = center_y - first_tpc_center[1]
        rel_z = center_z - first_tpc_center[2]

        det_id_geometry[det_id] = {
            "position": [round(rel_x, 2), round(rel_y, 2), round(rel_z, 2)],
            "dimensions": [
                round(x_extent, 2),
                round(y_extent, 2),
                round(z_extent, 2),
            ],
        }

        # Extract the det ID of each channel based on the det_id multiplicity
        # associated with each detector ID, assuming channels are in order
        # TODO: This could be extracted from `det_id_data` more robustrly,
        # but I currently do not understand the data structure well enough.
        channel_det_ids.extend([int(det_id)] * (np.sum(det_ids == det_id) // num_tpcs))

    # Build aggregated positions and shape_ids (one per unique det_id)
    aggregated_positions = []
    aggregated_shape_ids = []
    unique_dimensions = {}

    for det_id in sorted(det_id_geometry.keys()):
        geom = det_id_geometry[det_id]

        # Project to x=0, keep Y and Z as-is from bounds
        pos_y = geom["position"][1]
        pos_z = geom["position"][2]

        aggregated_positions.append([0, pos_y, pos_z])

        # Track unique dimensions
        dim_key = tuple(geom["dimensions"])
        if dim_key not in unique_dimensions:
            unique_dimensions[dim_key] = geom["dimensions"]

        aggregated_shape_ids.append(dim_key)

    # Create final dimensions list sorted by Y extent (tall first)
    optical_dimensions = sorted(
        unique_dimensions.values(), key=lambda d: d[1], reverse=True
    )

    # Create mapping from dimension key to shape_id based on sorted order
    dim_to_shape_id = {}
    for idx, dim in enumerate(optical_dimensions):
        dim_key = tuple(dim)
        dim_to_shape_id[dim_key] = idx

    # Convert stored dimension keys to shape_ids
    aggregated_shape_ids = [
        dim_to_shape_id[dim_key] for dim_key in aggregated_shape_ids
    ]

    return {
        "dimensions": optical_dimensions,
        "positions": aggregated_positions,
        "det_ids": channel_det_ids,
        "shape_ids": aggregated_shape_ids,
    }


def main(source, tag, output=None, opdet_thickness=None):
    """
    Main function for parsing LArND geometry HDF5 files.

    Parameters
    ----------
    source : str
        Path to the LArND geometry HDF5 file
    tag : str
        Tag to identify the geometry revision (e.g., 'mr5', 'mr6').
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

        # Extract geometry file references if available
        if "crs_geometry_file" in f["geometry_info"].attrs:
            crs_file_raw = f["geometry_info"].attrs["crs_geometry_file"]
            assert isinstance(
                crs_file_raw, str
            ), "CRS geometry file attribute is not a string."
            crs_geometry_files = [str(crs_file_raw)]
        elif "crs_geometry_files" in f["geometry_info"].attrs:
            # Convert numpy array to list of strings
            crs_files_raw = f["geometry_info"].attrs["crs_geometry_files"]
            assert isinstance(
                crs_files_raw, np.ndarray
            ), "CRS geometry files attribute is not a numpy array."
            crs_geometry_files = [str(f) for f in crs_files_raw]
        else:
            raise ValueError("CRS geometry files not found in HDF5 attributes.")

        lrs_geometry_file = None
        if "lrs_geometry_file" in f["geometry_info"].attrs:
            lrs_geometry_file = str(f["geometry_info"].attrs["lrs_geometry_file"])

        # Parse detector name from the geometry name
        detector_match = re.search(r"(\w+)_flow", crs_geometry_files[0])
        backup_match = re.search(r"(\w+)_layout", crs_geometry_files[0])
        if detector_match:
            detector_name = detector_match.group(1)
            if detector_name == "proto_nd":
                detector_name = "2x2"
            elif detector_name == "fsd":
                detector_name = "FSD"
            elif detector_name == "ndlar":
                detector_name = "ND-LAr"
            else:
                raise ValueError(
                    f"Unknown detector name parsed from CRS geometry file: {detector_name}"
                )
        elif backup_match:
            detector_name = backup_match.group(1)
            if detector_name.startswith("module"):
                detector_name = "2x2-Single"
            else:
                raise ValueError(
                    f"Unknown detector name parsed from CRS geometry file: {detector_name}"
                )
        else:
            raise ValueError("Could not parse detector name from CRS geometry file.")

        # Extract TPC geometry
        tpc_geometry = extract_tpc_geometry(f)

        # Extract optical geometry (if available)
        optical_geometry = extract_optical_geometry(
            f, tpc_geometry["positions"], opdet_thickness
        )

        # Compose the YAML structure
        geometry_yaml = {}

        # Add metadata fields
        assert (
            detector_name is not None
        ), "Could not determine detector name from runlist file."

        # Determine tag and extract version from it
        geometry_yaml["name"] = detector_name
        geometry_yaml["tag"] = tag

        # Match patterns like 'mr5', 'mr6.5', 'mr6-5', etc.
        match = re.search(r"(\d+)([.\-](\d+))?", tag)
        if match:
            major = match.group(1)
            minor = match.group(3) if match.group(3) else None
            if minor:
                version = float(f"{major}.{minor}")
            else:
                version = int(major)
        else:
            # Fallback to class_version if no version can be extracted from tag
            version = class_version

        geometry_yaml["version"] = version

        # Add geometry file references if available
        if crs_geometry_files is not None:
            geometry_yaml["crs_files"] = crs_geometry_files
        if lrs_geometry_file is not None:
            geometry_yaml["lrs_file"] = lrs_geometry_file

        # Add geometry sections
        geometry_yaml["tpc"] = {
            "dimensions": tpc_geometry["dimensions"],
            "module_ids": tpc_geometry["module_ids"],
            "positions": tpc_geometry["positions"],
        }

        # Only add optical section if detector info was available
        if optical_geometry is not None:
            geometry_yaml["optical"] = {
                "volume": "tpc",
                "shape": ["box", "box"],
                "mirror": True,
                "dimensions": optical_geometry["dimensions"],
                "shape_ids": optical_geometry["shape_ids"],
                "det_ids": optical_geometry["det_ids"],
                "positions": optical_geometry["positions"],
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

    parser.add_argument(
        "--tag",
        "-t",
        help="Tag to identify geometry revision (e.g., 'mr5', 'mr6').",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    main(
        source=args.source,
        tag=args.tag,
        output=args.output,
        opdet_thickness=args.opdet_thickness,
    )
