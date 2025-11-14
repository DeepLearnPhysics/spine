#!/usr/bin/env python3
"""Parse LArSoft geometry files and extract relevant detector information
in YAML format to be used by SPINE.

First dump the geometry from LArSoft using:
.. code-block:: bash

    lar -c dump_{detector}_geometry.fcl

Then run this script on the dumped text file.
"""

import argparse
import os
import re

import yaml


def parse_vector(text):
    """Parse a vector from text like '(x,y,z)'.

    Parameters
    ----------
    text : str
        Text containing a vector in format (x,y,z)

    Returns
    -------
    list
        List of three float values [x, y, z]
    """
    # Support scientific notation like 2.84217e-14
    match = re.search(r"\(([-\d.eE+]+),([-\d.eE+]+),([-\d.eE+]+)\)", text)
    if match:
        return [float(match.group(1)), float(match.group(2)), float(match.group(3))]
    return None


def parse_tpc_block(lines, start_idx, cathode_thickness=0.0, pixel_size=0.0):
    """Parse a single TPC block from the geometry file.

    Parameters
    ----------
    lines : list
        List of text lines from the geometry file
    start_idx : int
        Index where the TPC block starts
    cathode_thickness : float, optional
        Cathode thickness in cm (default: 0.0). Applied as offset to front face position.
    pixel_size : float, optional
        Pixel size in cm (default: 0.0). Pads the outer X edge.

    Returns
    -------
    dict
        Dictionary containing TPC information:
        - cryostat: Cryostat number
        - tpc: TPC number within cryostat
        - dimensions: Active volume dimensions [width, height, length]
        - position: Front face position [x, y, z] (adjusted for cathode thickness and pixel size)
        - drift_direction: Drift direction vector
        - plane_angles: List of plane angles in radians
    """
    tpc_info = {}

    # Parse TPC header: "TPC C:X T:Y (dimensions) at (position)"
    header = lines[start_idx]

    # Extract cryostat and TPC numbers
    cryo_match = re.search(r"TPC C:(\d+) T:(\d+)", header)
    if cryo_match:
        tpc_info["cryostat"] = int(cryo_match.group(1))
        tpc_info["tpc"] = int(cryo_match.group(2))

    # Extract dimensions and position from header
    dim_match = re.search(r"\(([-\d.eE+]+) x ([-\d.eE+]+) x ([-\d.eE+]+)\)", header)
    if dim_match:
        tpc_info["header_dimensions"] = [
            float(dim_match.group(1)),
            float(dim_match.group(2)),
            float(dim_match.group(3)),
        ]

    header_pos = parse_vector(header.split(" at ")[-1] if " at " in header else "")
    if header_pos:
        tpc_info["header_position"] = header_pos

    # Parse subsequent lines for detailed info
    # Look further ahead to capture all plane angles (typically 3 planes per TPC)
    # In full geometry files, there may be thousands of wire detail lines between
    # the plane summary lines, so we need to search until the next TPC
    for i in range(start_idx + 1, len(lines)):
        line = lines[i]

        # Stop if we hit the next TPC or cryostat (but not before we finish parsing planes)
        if (
            line.strip().startswith("TPC C:") or line.strip().startswith("Cryostat C:")
        ) and i > start_idx + 8:
            break

        # Parse drift direction and cathode position
        if "drift direction" in line:
            drift_dir = parse_vector(line)
            if drift_dir:
                tpc_info["drift_direction"] = drift_dir

            # Also extract cathode position and drift distance
            # Format: "drift direction (x,y,z) from cathode around (x,y,z) through XXX cm"
            cathode_match = re.search(
                r"from cathode around\s*\(([-\d.eE+]+),([-\d.eE+]+),([-\d.eE+]+)\)",
                line,
            )
            if cathode_match:
                tpc_info["cathode_x"] = float(cathode_match.group(1))

            drift_dist_match = re.search(r"through\s+([-\d.eE+]+)\s+cm", line)
            if drift_dist_match:
                tpc_info["drift_distance"] = float(drift_dist_match.group(1))

        # Parse active volume dimensions and front face position
        elif "active volume" in line and "front face at" in line:
            # Extract dimensions
            dim_match = re.search(
                r"\(([-\d.eE+]+) x ([-\d.eE+]+) x ([-\d.eE+]+)\)", line
            )
            if dim_match:
                active_dims = [
                    float(dim_match.group(1)),
                    float(dim_match.group(2)),
                    float(dim_match.group(3)),
                ]
                tpc_info["dimensions"] = active_dims

            # Extract front face position and calculate center
            # In SPINE YAML format, the position represents the CENTER of the TPC
            front_face = parse_vector(line.split("front face at")[1])
            if front_face and "dimensions" in tpc_info:
                # Calculate center from front face and dimensions
                position = [
                    front_face[0]
                    + tpc_info["dimensions"][0]
                    / 2,  # X: center = front + half dimension
                    front_face[1],  # Y: already at center
                    front_face[2]
                    + tpc_info["dimensions"][2]
                    / 2,  # Z: center = front + half dimension
                ]
                tpc_info["position"] = position

        # Parse active volume box to get actual edges
        elif "active volume box:" in line:
            # Format: "active volume box: (x1,y1,z1) -- (x2,y2,z2)"
            box_match = re.search(
                r"\(([-\d.eE+]+),([-\d.eE+]+),([-\d.eE+]+)\)\s*--\s*\(([-\d.eE+]+),([-\d.eE+]+),([-\d.eE+]+)\)",
                line,
            )
            if box_match:
                min_coords = [float(box_match.group(i)) for i in range(1, 4)]
                max_coords = [float(box_match.group(i)) for i in range(4, 7)]
                tpc_info["box_min"] = min_coords
                tpc_info["box_max"] = max_coords
                # Calculate dimension and center from box
                tpc_info["dimensions"] = [
                    max_coords[0] - min_coords[0],
                    max_coords[1] - min_coords[1],
                    max_coords[2] - min_coords[2],
                ]
                tpc_info["position"] = [
                    (min_coords[0] + max_coords[0]) / 2,
                    (min_coords[1] + max_coords[1]) / 2,
                    (min_coords[2] + max_coords[2]) / 2,
                ]

        # Parse plane angles and wire normals
        elif "plane C:" in line and "theta:" in line:
            if "plane_angles" not in tpc_info:
                tpc_info["plane_angles"] = []
            if "wire_normals" not in tpc_info:
                tpc_info["wire_normals"] = []
            if "plane_positions" not in tpc_info:
                tpc_info["plane_positions"] = []

            # Extract plane angle
            theta_match = re.search(r"theta: ([-\d.]+)", line)
            if theta_match:
                angle = float(theta_match.group(1))
                tpc_info["plane_angles"].append(angle)

            # Extract plane position
            plane_pos = parse_vector(line.split(" at ")[1].split(" cm,")[0])
            if plane_pos:
                tpc_info["plane_positions"].append(plane_pos)

            # Look ahead 2 lines for the "direction of increasing wire number"
            # which gives us the normal to the wire
            if i + 2 < len(lines):
                next_line = lines[i + 2]
                if "direction of increasing wire number:" in next_line:
                    # Extract the vector after "direction of increasing wire number:"
                    # Format: "... direction of increasing wire number: (0,0.5,0.866025) ..."
                    match = re.search(
                        r"direction of increasing wire number:\s*\(([-\d.eE+]+),([-\d.eE+]+),([-\d.eE+]+)\)",
                        next_line,
                    )
                    if match:
                        wire_normal = [
                            float(match.group(1)),
                            float(match.group(2)),
                            float(match.group(3)),
                        ]
                        # Store only YZ components for ICARUS (x is drift direction)
                        tpc_info["wire_normals"].append(
                            [0.0, wire_normal[1], wire_normal[2]]
                        )
                    else:
                        # If parsing failed, add placeholder
                        tpc_info["wire_normals"].append([0.0, 0.0, 0.0])
                else:
                    # If line not found, add placeholder
                    tpc_info["wire_normals"].append([0.0, 0.0, 0.0])

                # Look ahead 3 lines for wire coverage dimensions
                # Format: "wire direction: ...; width XXX cm in direction: ..., depth YYY cm in direction: ..."
                if i + 3 < len(lines):
                    wire_line = lines[i + 3]
                    if (
                        "wire direction:" in wire_line
                        and "width" in wire_line
                        and "depth" in wire_line
                    ):
                        # Extract width (corresponds to length/z dimension)
                        width_match = re.search(r"width\s+([-\d.eE+]+)\s+cm", wire_line)
                        # Extract depth (corresponds to height/y dimension)
                        depth_match = re.search(r"depth\s+([-\d.eE+]+)\s+cm", wire_line)

                        if "wire_coverage" not in tpc_info:
                            tpc_info["wire_coverage"] = {"width": None, "depth": None}

                        if width_match:
                            tpc_info["wire_coverage"]["width"] = float(
                                width_match.group(1)
                            )
                        if depth_match:
                            tpc_info["wire_coverage"]["depth"] = float(
                                depth_match.group(1)
                            )

    # Start with active volume if available (most accurate for actual sensitive volume)
    # The parse already puts active volume in 'dimensions' and 'position'

    # If no active volume was found, fall back to header values
    if "dimensions" not in tpc_info and "header_dimensions" in tpc_info:
        tpc_info["dimensions"] = tpc_info["header_dimensions"]
    if "position" not in tpc_info and "header_position" in tpc_info:
        tpc_info["position"] = tpc_info["header_position"]

    # Apply cathode thickness adjustment
    # The cathode at x=0 reduces the available drift distance for each TPC
    # For SBND: cathode is 0.4 cm thick, so each TPC loses 0.2 cm of drift space on the side nearest x=0
    if cathode_thickness > 0 and "box_min" in tpc_info and "box_max" in tpc_info:
        box_min = tpc_info["box_min"]
        box_max = tpc_info["box_max"]

        # Adjust the edge closest to x=0 by half the cathode thickness
        # For negative X TPCs: right edge (max) moves left
        # For positive X TPCs: left edge (min) moves right
        if box_max[0] <= 0:  # TPC is entirely on negative X side
            box_max[0] -= cathode_thickness / 2
        elif box_min[0] >= 0:  # TPC is entirely on positive X side
            box_min[0] += cathode_thickness / 2

        # Recalculate dimension and center from adjusted box
        tpc_info["dimensions"] = [
            box_max[0] - box_min[0],
            box_max[1] - box_min[1],
            box_max[2] - box_min[2],
        ]
        tpc_info["position"] = [
            (box_min[0] + box_max[0]) / 2,
            (box_min[1] + box_max[1]) / 2,
            (box_min[2] + box_max[2]) / 2,
        ]
    elif cathode_thickness > 0 and "position" in tpc_info and "dimensions" in tpc_info:
        # Fallback if box parsing failed
        pos = tpc_info["position"]
        dims = tpc_info["dimensions"]

        # Reduce X dimension by half the cathode thickness
        old_dim_x = dims[0]
        dims[0] = dims[0] - cathode_thickness / 2

        # Adjust center position: when dimension shrinks, center shifts away from x=0
        dim_change = old_dim_x - dims[0]
        if pos[0] < 0:
            pos[0] -= dim_change / 2
        elif pos[0] > 0:
            pos[0] += dim_change / 2

    # Override Y and Z with wire coverage dimensions if available (most accurate for wire coverage)
    # This is done AFTER cathode adjustment to preserve the accurate wire coverage values
    if "wire_coverage" in tpc_info and "dimensions" in tpc_info:
        wire_cov = tpc_info["wire_coverage"]
        dims = tpc_info["dimensions"]

        # Y dimension from wire depth
        if wire_cov.get("depth") is not None:
            dims[1] = wire_cov["depth"]
        # Z dimension from wire width
        if wire_cov.get("width") is not None:
            dims[2] = wire_cov["width"]

    # Apply pixel size padding to outer X edge
    # This pads the edge away from x=0, shifting centers and increasing X dimension
    # For SBND: 0.3 cm (3mm) pixel size
    if pixel_size > 0 and "box_min" in tpc_info and "box_max" in tpc_info:
        box_min = tpc_info["box_min"]
        box_max = tpc_info["box_max"]

        # Pad the edge away from x=0
        # For negative X TPCs: left edge (min) moves more negative
        # For positive X TPCs: right edge (max) moves more positive
        if box_max[0] <= 0:  # TPC is entirely on negative X side
            box_min[0] -= pixel_size
        elif box_min[0] >= 0:  # TPC is entirely on positive X side
            box_max[0] += pixel_size

        # Recalculate dimension and center from padded box
        tpc_info["dimensions"][0] = box_max[0] - box_min[0]
        tpc_info["position"][0] = (box_min[0] + box_max[0]) / 2
    elif pixel_size > 0 and "position" in tpc_info and "dimensions" in tpc_info:
        # Fallback if box parsing failed
        pos = tpc_info["position"]
        dims = tpc_info["dimensions"]

        # Increase X dimension by pixel size
        dims[0] = dims[0] + pixel_size

        # Shift center away from x=0 by half the pixel size
        if pos[0] < 0:
            pos[0] -= pixel_size / 2
        elif pos[0] > 0:
            pos[0] += pixel_size / 2

    return tpc_info


def main(source, output=None, cathode_thickness=0.0, pixel_size=0.0):
    """Main function for parsing LArSoft geometry files.

    Parameters
    ----------
    source : str
        Path to the dumped LArSoft geometry text file
    output : str, optional
        Path to output YAML file (if None, define it based on input file name)
    cathode_thickness : float, optional
        Cathode thickness in cm (default: 0.0). For SBND, use 0.4 cm to account
        for the cathode plane at x=0 that creates an artificial offset in the
        front face positions.
    pixel_size : float, optional
        Pixel size in cm (default: 0.0). Pads the outer X edge by this amount.
        For SBND, use 0.3 cm to account for 3mm pixel size.
    """
    # Read the dumped geometry file
    with open(source, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Parse all TPC blocks
    tpc_list = []
    for i, line in enumerate(lines):
        if line.strip().startswith("TPC C:"):
            tpc_info = parse_tpc_block(
                lines, i, cathode_thickness=cathode_thickness, pixel_size=pixel_size
            )
            if tpc_info and "position" in tpc_info and "dimensions" in tpc_info:
                tpc_list.append(tpc_info)

    print(f"Found {len(tpc_list)} TPCs\n")

    # Sort TPCs by cryostat and TPC number for consistent ordering
    tpc_list.sort(key=lambda x: (x["cryostat"], x["tpc"]))

    # Keep track of all TPCs for det_ids mapping
    all_tpcs = tpc_list.copy()

    # Group TPCs by shared cathode position (physical drift volumes)
    # TPCs that drift from the same cathode are part of the same physical TPC
    # However, TPCs drifting in opposite directions from the same cathode are separate
    # This handles:
    # - ICARUS: TPCs 0/1 share wire planes and cathode → same drift volume
    # - SBND: same cathode but opposite drift directions → separate drift volumes
    # - ProtoDUNE: multiple readout TPCs drift same direction from same cathode → same drift volume

    # For each TPC, create a "signature" based on cathode X position and drift direction
    # If no cathode info, fall back to wire plane X positions and X dimension
    tpc_signatures = []
    for tpc in tpc_list:
        if "cathode_x" in tpc and "drift_direction" in tpc:
            # Use cathode X position and drift direction sign as the signature
            cathode_x = round(tpc["cathode_x"], 1)
            drift_sign = 1 if tpc["drift_direction"][0] > 0 else -1
            tpc_signatures.append((cathode_x, drift_sign))
        elif "plane_positions" in tpc and len(tpc["plane_positions"]) > 0:
            # Fallback: Use wire plane X positions and X dimension
            plane_x_positions = tuple(
                round(pos[0], 1) for pos in tpc["plane_positions"]
            )
            x_dimension = round(tpc["dimensions"][0], 1)
            tpc_signatures.append((plane_x_positions, x_dimension))
        else:
            # Last fallback: use TPC center X position and X dimension
            x_dimension = round(tpc["dimensions"][0], 1)
            tpc_signatures.append(((round(tpc["position"][0], 1),), x_dimension))

    # Group TPCs with matching signatures
    signature_to_group = {}
    group_counter = 0
    tpc_groups = {}

    for tpc, signature in zip(tpc_list, tpc_signatures):
        if signature not in signature_to_group:
            signature_to_group[signature] = group_counter
            tpc_groups[group_counter] = []
            group_counter += 1

        group_id = signature_to_group[signature]
        tpc_groups[group_id].append(tpc)

    # Sort groups by their ID (which maintains TPC order)
    sorted_groups = [tpc_groups[gid] for gid in sorted(tpc_groups.keys())]

    # Compute combined dimensions and positions for TPC groups
    combined_tpcs = []
    for group_tpcs in sorted_groups:

        # Sort by z position, then by y position
        group_tpcs.sort(key=lambda t: (t["position"][2], t["position"][1]))

        # Get dimensions from first TPC
        dims = group_tpcs[0]["dimensions"].copy()

        # Check if TPCs are arranged in a multi-dimensional grid
        # by looking at unique Y and Z positions
        unique_y = sorted(set(round(tpc["position"][1], 1) for tpc in group_tpcs))
        unique_z = sorted(set(round(tpc["position"][2], 1) for tpc in group_tpcs))

        if len(unique_y) > 1 or len(unique_z) > 1:
            # Multi-dimensional arrangement - compute bounding box
            # Get min/max positions for each axis
            min_y = min(
                tpc["position"][1] - tpc["dimensions"][1] / 2 for tpc in group_tpcs
            )
            max_y = max(
                tpc["position"][1] + tpc["dimensions"][1] / 2 for tpc in group_tpcs
            )
            min_z = min(
                tpc["position"][2] - tpc["dimensions"][2] / 2 for tpc in group_tpcs
            )
            max_z = max(
                tpc["position"][2] + tpc["dimensions"][2] / 2 for tpc in group_tpcs
            )

            # Set combined dimensions
            dims[1] = max_y - min_y
            dims[2] = max_z - min_z

            # Position is at the center of the bounding box
            pos = group_tpcs[0]["position"].copy()
            pos[1] = (min_y + max_y) / 2
            pos[2] = (min_z + max_z) / 2
        else:
            # Single-dimensional arrangement along z-axis (original logic)
            # Compute combined z-dimension (sum of all TPCs in group along z)
            total_z_length = sum(tpc["dimensions"][2] for tpc in group_tpcs)
            dims[2] = total_z_length

            # Position: for single TPC, keep original z; for multiple TPCs combined along z, set z=0.0 (center)
            pos = group_tpcs[0]["position"].copy()
            if len(group_tpcs) > 1:
                # Multiple TPCs combined along z-axis - center at z=0
                pos[2] = 0.0

        combined_tpcs.append(
            {
                "dimensions": dims,
                "position": pos,
                "cryostat": group_tpcs[0]["cryostat"],
                "tpc_group": group_tpcs,
            }
        )

    # Build YAML structure for TPC section
    tpc_yaml = {
        "dimensions": None,  # Will be set from first combined TPC
        "module_ids": [],
        "positions": [],
    }

    # Build det_ids: map ALL logical TPC IDs (within first cryostat) to physical TPC indices
    # This must include filtered TPCs, which map to -1
    # This describes the TPC structure per module (same for all modules)
    first_cryo = combined_tpcs[0]["cryostat"]
    first_cryo_combined = [
        tpc for tpc in combined_tpcs if tpc["cryostat"] == first_cryo
    ]
    first_cryo_all = [tpc for tpc in all_tpcs if tpc["cryostat"] == first_cryo]

    # Map logical TPC IDs to physical indices (or -1 if filtered)
    logical_to_physical = {}
    for phys_idx, combined_tpc in enumerate(first_cryo_combined):
        for tpc_info in combined_tpc["tpc_group"]:
            logical_to_physical[tpc_info["tpc"]] = phys_idx

    # Build det_ids for ALL logical TPCs (including filtered ones)
    det_ids = []
    for tpc in first_cryo_all:
        logical_id = tpc["tpc"]
        det_ids.append(logical_to_physical.get(logical_id, -1))

    # Only include det_ids if not trivial (not [0, 1, 2, ...])
    if det_ids != list(range(len(det_ids))):
        tpc_yaml["det_ids"] = det_ids

    # Collect all dimensions for each physical TPC
    all_dimensions = [
        [round(d, 4) for d in combined_tpc["dimensions"]]
        for combined_tpc in combined_tpcs
    ]

    # Check if all dimensions are the same
    if all(dim == all_dimensions[0] for dim in all_dimensions):
        tpc_yaml["dimensions"] = all_dimensions[0]
    else:
        tpc_yaml["dimensions"] = all_dimensions

    # Extract module IDs, positions, and drift directions
    drift_dirs = []

    def clean_zero(val):
        # Convert -0.0 to 0.0
        return 0.0 if val == 0 or val == -0.0 else val

    for combined_tpc in combined_tpcs:
        tpc_yaml["module_ids"].append(combined_tpc["cryostat"])
        tpc_yaml["positions"].append([round(p, 4) for p in combined_tpc["position"]])
        # Collect drift direction from first TPC in group
        drift_dir = combined_tpc["tpc_group"][0].get("drift_direction")
        if drift_dir:
            drift_dirs.append([clean_zero(round(x, 4)) for x in drift_dir])

    # Store drift_dirs only if the number of combined TPCs per module is not 2
    num_modules = len(set(tpc_yaml["module_ids"]))
    tpcs_per_module = (
        len(combined_tpcs) // num_modules if num_modules > 0 else len(combined_tpcs)
    )
    if tpcs_per_module != 2:
        tpc_yaml["drift_dirs"] = drift_dirs

    # Build top-level YAML structure
    yaml_data = {}

    # Extract detector from file name (e.g. ProtoDUNE-SP-geometry.txt -> ProtoDUNE-SP)
    base = os.path.basename(source)
    detector_match = re.match(r"([A-Za-z0-9_-]+)-geometry", base)
    if not detector_match:
        raise ValueError("Could not extract detector name from file name.")
    yaml_data["detector"] = detector_match.group(1)

    # Extract tag and version from the second line
    assert len(lines) > 1, "Geometry text file is too short."
    tag_match = re.search(r"Detector\s+([a-zA-Z0-9_-]+)", lines[1])
    if not tag_match:
        raise ValueError("Could not extract tag from geometry file.")
    tag = tag_match.group(1)
    yaml_data["tag"] = tag

    # Version: last number in tag
    version_match = re.search(r"(\d+)$", tag)
    if not version_match:
        raise ValueError("Could not extract version from tag.")
    yaml_data["version"] = int(version_match.group(1))

    # Add the TPC geometry information
    yaml_data["tpc"] = tpc_yaml

    # Print the YAML structure
    print("TPC section for YAML file:")
    print("=" * 60)
    print(yaml.dump(yaml_data, default_flow_style=None, sort_keys=False))
    print("=" * 60)

    # Also print a summary
    print(
        f"\nParsed {len(tpc_list)} individual TPCs into {len(combined_tpcs)} combined TPC volumes:"
    )
    for i, combined_tpc in enumerate(combined_tpcs):
        group = combined_tpc["tpc_group"]
        tpc_ids = ", ".join([f"T:{t['tpc']}" for t in group])
        angles_str = f", plane angles: {group[0].get('plane_angles', 'N/A')}"
        print(
            f"  Combined TPC {i}: C:{combined_tpc['cryostat']} ({tpc_ids}) "
            f"at {combined_tpc['position']} with dims {combined_tpc['dimensions']}{angles_str}"
        )

    # Write the final YAML to file
    output_path = output if output is not None else source.replace(".txt", "_tpc.yaml")
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_data, f, default_flow_style=None, sort_keys=False)
    print(f"\nYAML file written to: {output_path}")


if __name__ == "__main__":
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description="Parse dumped LArSoft geometry files")

    parser.add_argument(
        "--source",
        "-s",
        help="Path to the dumped LArSoft geometry text file",
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
        "--cathode-thickness",
        "-c",
        help="Cathode thickness in cm (default: 0.0). The cathode at x=0 reduces the available "
        "drift distance for each TPC. For SBND with a 4mm cathode, use 0.4 cm.",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--pixel-size",
        "-p",
        help="Pixel size in cm (default: 0.0). Pads the outer X edge by this amount, shifting "
        "TPC centers away from x=0 and increasing X dimension. For SBND with 3mm pixels, use 0.3 cm.",
        type=float,
        default=0.0,
    )

    args = parser.parse_args()

    # Execute the main function
    main(
        source=args.source,
        output=args.output,
        cathode_thickness=args.cathode_thickness,
        pixel_size=args.pixel_size,
    )
