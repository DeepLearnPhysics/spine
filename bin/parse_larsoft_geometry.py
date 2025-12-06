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
from warnings import warn

import numpy as np
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
    match = re.search(r"\(([-\d.eE+]+),([-\d.eE+]+),([-\d.eE+]+)\)", text)
    if match:
        return [float(match.group(i)) for i in range(1, 4)]
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
        Cathode thickness in cm (default: 0.0)
    pixel_size : float, optional
        Pixel size in cm (default: 0.0)

    Returns
    -------
    dict
        Dictionary containing TPC information with keys:
        - cryostat, tpc: Cryostat and TPC numbers
        - dimensions: Active volume dimensions [width, height, length]
        - position: Center position [x, y, z]
        - drift_direction: Drift direction vector
        - plane_angles: List of plane angles in radians
    """
    tpc_info = {}
    header = lines[start_idx]

    # Extract cryostat and TPC numbers
    cryo_match = re.search(r"TPC C:(\d+) T:(\d+)", header)
    if cryo_match:
        tpc_info["cryostat"] = int(cryo_match.group(1))
        tpc_info["tpc"] = int(cryo_match.group(2))

    # Parse subsequent lines for detailed info
    for i in range(start_idx + 1, len(lines)):
        line = lines[i]

        # Stop if we hit the next TPC or cryostat
        if (
            line.strip().startswith("TPC C:") or line.strip().startswith("Cryostat C:")
        ) and i > start_idx + 8:
            break

        # Parse drift direction and cathode position
        if "drift direction" in line:
            drift_dir = parse_vector(line)
            if drift_dir:
                tpc_info["drift_direction"] = drift_dir

            cathode_match = re.search(
                r"from cathode around\s*\(([-\d.eE+]+),([-\d.eE+]+),([-\d.eE+]+)\)",
                line,
            )
            if cathode_match:
                tpc_info["cathode_x"] = float(cathode_match.group(1))

        # Parse active volume box (most accurate)
        elif "active volume box:" in line:
            box_match = re.search(
                r"\(([-\d.eE+]+),([-\d.eE+]+),([-\d.eE+]+)\)\s*--\s*\(([-\d.eE+]+),([-\d.eE+]+),([-\d.eE+]+)\)",
                line,
            )
            if box_match:
                min_coords = [float(box_match.group(i)) for i in range(1, 4)]
                max_coords = [float(box_match.group(i)) for i in range(4, 7)]
                tpc_info["box_min"] = min_coords
                tpc_info["box_max"] = max_coords
                tpc_info["dimensions"] = [
                    max_coords[i] - min_coords[i] for i in range(3)
                ]
                tpc_info["position"] = [
                    (min_coords[i] + max_coords[i]) / 2 for i in range(3)
                ]

        # Parse plane angles and positions
        elif "plane C:" in line and "theta:" in line:
            if "plane_angles" not in tpc_info:
                tpc_info["plane_angles"] = []
                tpc_info["plane_positions"] = []

            theta_match = re.search(r"theta: ([-\d.]+)", line)
            if theta_match:
                tpc_info["plane_angles"].append(float(theta_match.group(1)))

            plane_pos = parse_vector(line.split(" at ")[1].split(" cm,")[0])
            if plane_pos:
                tpc_info["plane_positions"].append(plane_pos)

            # Look ahead for wire coverage dimensions
            if i + 3 < len(lines):
                wire_line = lines[i + 3]
                if "width" in wire_line and "depth" in wire_line:
                    width_match = re.search(r"width\s+([-\d.eE+]+)\s+cm", wire_line)
                    depth_match = re.search(r"depth\s+([-\d.eE+]+)\s+cm", wire_line)

                    if "wire_coverage" not in tpc_info:
                        tpc_info["wire_coverage"] = {}

                    if width_match:
                        tpc_info["wire_coverage"]["width"] = float(width_match.group(1))
                    if depth_match:
                        tpc_info["wire_coverage"]["depth"] = float(depth_match.group(1))

    # If no box was parsed, return incomplete info
    if "dimensions" not in tpc_info or "position" not in tpc_info:
        return tpc_info

    # Apply cathode thickness adjustment
    if cathode_thickness > 0 and "box_min" in tpc_info:
        box_min = tpc_info["box_min"]
        box_max = tpc_info["box_max"]

        # Adjust the edge closest to x=0 by half the cathode thickness
        if box_max[0] <= 0:  # Negative X side
            box_max[0] -= cathode_thickness / 2
        elif box_min[0] >= 0:  # Positive X side
            box_min[0] += cathode_thickness / 2

        # Recalculate dimension and center
        tpc_info["dimensions"][0] = box_max[0] - box_min[0]
        tpc_info["position"][0] = (box_min[0] + box_max[0]) / 2

    # Override Y and Z with wire coverage dimensions if available
    if "wire_coverage" in tpc_info:
        wire_cov = tpc_info["wire_coverage"]
        if "depth" in wire_cov:
            tpc_info["dimensions"][1] = wire_cov["depth"]
        if "width" in wire_cov:
            tpc_info["dimensions"][2] = wire_cov["width"]

    # Apply pixel size padding to outer X edge
    if pixel_size > 0 and "box_min" in tpc_info:
        box_min = tpc_info["box_min"]
        box_max = tpc_info["box_max"]

        # Pad the edge away from x=0
        if box_max[0] <= 0:
            box_min[0] -= pixel_size
        elif box_min[0] >= 0:
            box_max[0] += pixel_size

        # Recalculate dimension and center
        tpc_info["dimensions"][0] = box_max[0] - box_min[0]
        tpc_info["position"][0] = (box_min[0] + box_max[0]) / 2

    return tpc_info


def parse_optical_block(lines, start_idx):
    """Parse a single optical block from the geometry file.

    Parameters
    ----------
    lines : list
        List of text lines from the geometry file
    start_idx : int
        Index where the optical block starts

    Returns
    -------
    dict
        Dictionary with keys: shape, dimensions, shape_ids, positions
    """
    shapes = []
    dimensions = []
    positions = []

    for i in range(start_idx, len(lines)):
        line = lines[i]
        if not line.strip().startswith("[OpDet #"):
            break

        # Extract shape
        type_match = re.search(r",\s*([a-zA-Z]+)", line)
        if type_match:
            shape = type_match.group(1).lower()
            if shape in ("hemispherical", "spherical"):
                shape = "ellipsoid"
            elif shape == "bar":
                shape = "box"
            elif shape == "radius":
                shape = "disk"
            else:
                raise ValueError(f"Unknown optical detector shape: {shape}")
            shapes.append(shape)
        else:
            raise ValueError("Could not parse optical detector shape")

        # Extract dimensions based on shape
        if shape == "ellipsoid":
            radius_match = re.search(r"external radius\s+([-\d.eE+]+)\s+cm", line)
            if radius_match:
                radius = float(radius_match.group(1))
                dimensions.append([2 * radius, radius, 2 * radius])
            else:
                raise ValueError("Could not parse ellipsoid radius")

        elif shape == "box":
            size_match = re.search(
                r"bar size\s+([-\d.eE+]+)\s+x\s+([-\d.eE+]+)\s+x\s+([-\d.eE+]+)\s+cm",
                line,
            )
            if size_match:
                dimensions.append([float(size_match.group(i)) for i in range(1, 4)])
            else:
                raise ValueError("Could not parse box size")

        elif shape == "disk":
            radius_match = re.search(r"radius:\s+([-\d.eE+]+)\s+cm", line)
            length_match = re.search(r"length:\s+([-\d.eE+]+)\s+cm", line)
            if radius_match and length_match:
                radius = float(radius_match.group(1))
                thickness = float(length_match.group(1))
                dimensions.append([2 * radius, thickness, 2 * radius])
            else:
                raise ValueError("Could not parse disk dimensions")

        # Handle rotation angle
        angle_match = re.search(r"theta\(z\):\s+([-\d.eE+]+)\s+rad", line)
        if angle_match:
            z_angle = float(angle_match.group(1))
            rot_matrix = np.array(
                [
                    [np.cos(z_angle), -np.sin(z_angle), 0],
                    [np.sin(z_angle), np.cos(z_angle), 0],
                    [0, 0, 1],
                ]
            )
            dimensions[-1] = np.abs(rot_matrix.dot(np.array(dimensions[-1]))).tolist()
        elif shape != "box":
            # If no angle, normal is along Z (swap Y and Z)
            dims = dimensions[-1]
            dimensions[-1] = [dims[0], dims[2], dims[1]]

        # Extract position
        pos = parse_vector(line.split("centered at")[1].split(" cm,")[0])
        if pos:
            positions.append(pos)
        else:
            raise ValueError("Could not parse optical detector position")

    # Build optical info structure
    op_info = {
        "shape": [],
        "dimensions": [],
        "positions": np.array(positions),
        "shape_ids": np.zeros(len(positions), dtype=int),
    }

    # Group by unique shape and dimension
    idx = 0
    for shape in np.unique(shapes):
        shape_indices = np.where(np.array(shapes) == shape)[0]
        for dim in np.unique(np.array(dimensions)[shape_indices], axis=0):
            dim_indices = shape_indices[
                (np.array(dimensions)[shape_indices] == dim).all(axis=1)
            ]
            op_info["shape"].append(shape)
            op_info["dimensions"].append(dim.tolist())
            op_info["shape_ids"][dim_indices] = idx
            idx += 1

    # Simplify if only one type
    if idx < 2:
        op_info["shape"] = op_info["shape"][0]
        op_info["dimensions"] = op_info["dimensions"][0]
        del op_info["shape_ids"]

    return op_info


def parse_crt_block(lines, start_idx):
    """Parse a single CRT block from the geometry file.

    Parameters
    ----------
    lines : list
        List of text lines from the geometry file
    start_idx : int
        Index where the CRT block starts

    Returns
    -------
    dict
        Dictionary with keys: name, dimensions, position, normal
    """
    header = lines[start_idx]
    name_match = re.search(r"\"volAuxDet.*\"", header)
    if name_match:
        name = name_match.group(0).replace('"', "")
    else:
        raise ValueError("Could not parse CRT module name")

    # Parse module-level position and size from header
    module_position = parse_vector(header.split("centered at")[1].split(" cm,")[0])
    size_match = re.search(
        r"size\s+\(\s+([-\d.eE+]+)\s+x\s+([-\d.eE+]+)\s+x\s+([-\d.eE+]+)\s+\)\s+cm",
        header,
    )
    if not size_match:
        raise ValueError("Could not parse module size")
    module_size = np.array([float(size_match.group(i)) for i in range(1, 4)])

    # Parse number of sensitive volumes
    if start_idx + 1 >= len(lines):
        raise ValueError("CRT block is incomplete")

    second_line = lines[start_idx + 1]
    sensitive_match = re.search(r"with\s+(\d+)\s+sensitive volumes", second_line)
    if not sensitive_match:
        raise ValueError("Could not parse number of sensitive volumes")

    num_sensitive = int(sensitive_match.group(1))

    # Parse all sensitive volumes
    sensitive_volumes = []
    for i in range(num_sensitive):
        line_idx = start_idx + 2 + i
        if line_idx >= len(lines):
            break

        volume_line = lines[line_idx]

        # Parse position
        vol_pos = parse_vector(volume_line.split("centered at")[1].split(" cm,")[0])
        if vol_pos is None:
            continue

        # Parse size
        size_match = re.search(
            r"size\s+\(\s+([-\d.eE+]+)\s+x\s+([-\d.eE+]+)\s+x\s+([-\d.eE+]+)\s+\)\s+cm",
            volume_line,
        )
        if not size_match:
            continue

        vol_size = [float(size_match.group(i)) for i in range(1, 4)]

        # Parse normal vector
        vol_normal = parse_vector(volume_line.split("normal facing")[1])
        if vol_normal is None:
            continue

        sensitive_volumes.append(
            {
                "position": vol_pos,
                "size": vol_size,
                "normal": vol_normal,
            }
        )

    if not sensitive_volumes:
        raise ValueError("No valid sensitive volumes found in CRT block")

    # Check for mixed orientations first, before aggregation
    normals = np.array([v["normal"] for v in sensitive_volumes])

    # Round normals to identify unique orientations (accounting for tiny floating point errors)
    unique_normals = np.unique(np.round(normals, decimals=6), axis=0)

    # Aggregate: find bounding box of all sensitive volumes
    positions = np.array([v["position"] for v in sensitive_volumes])
    sizes = np.array([v["size"] for v in sensitive_volumes])

    if len(unique_normals) > 1:
        # Mixed orientations detected - these sub-planes are stacked along one axis
        # That stacking axis is the true normal direction for the combined detector plane
        normal_groups = {}
        normal_groups_indices = {}
        for i, sv in enumerate(sensitive_volumes):
            normal_key = tuple(np.round(sv["normal"], decimals=6))
            if normal_key not in normal_groups:
                normal_groups[normal_key] = []
                normal_groups_indices[normal_key] = []
            normal_groups[normal_key].append(sv["position"])
            normal_groups_indices[normal_key].append(i)

        # Calculate mean position for each orientation group
        group_means = np.array(
            [np.mean(positions, axis=0) for positions in normal_groups.values()]
        )

        # Find which axis has the SMALLEST non-zero separation (the stacking axis)
        # The two orientation groups are stacked with a tiny offset along the normal direction
        separations = np.max(group_means, axis=0) - np.min(group_means, axis=0)

        # Filter out near-zero separations (< 0.1 cm) and find minimum
        non_zero_separations = np.where(separations > 0.1, separations, np.inf)
        stacking_axis = int(np.argmin(non_zero_separations))

        # The stacking axis is the true normal direction
        normal_axis = stacking_axis

        # For mixed-orientation modules, we still need to apply coordinate transformation
        # based on the dominant normal vector, then ensure thin dimension matches stacking axis

        # Get the dominant normal from the first sensitive volume
        normal_vector = np.array(sensitive_volumes[0]["normal"])
        abs_normal = np.abs(normal_vector)
        dominant_axis = int(np.argmax(abs_normal))

        # Apply coordinate transformation to module dimensions
        transformed_size = module_size.copy()

        if dominant_axis == 0:  # X-axis dominant: LOCAL-X is normal
            # LOCAL [X, Y, Z] → GLOBAL [Z, Y, X]
            transformed_size = np.array(
                [module_size[2], module_size[1], module_size[0]]
            )
        elif dominant_axis == 1:  # Y-axis dominant: LOCAL-Y is normal
            # LOCAL [X, Y, Z] → GLOBAL [Y, Z, X]
            transformed_size = np.array(
                [module_size[1], module_size[2], module_size[0]]
            )
        # If dominant_axis == 2, no transformation needed

        # Now ensure the thin dimension corresponds to the stacking axis
        thin_idx = int(np.argmin(transformed_size))
        if thin_idx != stacking_axis:
            # Swap the thin dimension with the stacking axis dimension
            transformed_size[thin_idx], transformed_size[stacking_axis] = (
                transformed_size[stacking_axis],
                transformed_size[thin_idx],
            )

        transformed_size = transformed_size.tolist()
        aggregated_position = module_position

    else:
        # Single orientation:
        # 1. Find stacking axis from position variance (in GLOBAL coords)
        # 2. Replace that dimension with stacking_span (this is 'b', second thinnest)
        # 3. Apply normal_facing swap logic based on which axis is the normal:
        #    - normal_axis_local = 2 (Z-normal): no swap
        #    - normal_axis_local = 1 (Y-normal): swap Y↔Z only if stacking is NOT on Z
        #    - normal_axis_local = 0 (X-normal): always swap the two non-stacking dims

        normal_vector = np.array(sensitive_volumes[0]["normal"])
        abs_normal = np.abs(normal_vector)
        normal_axis_local = int(np.argmax(abs_normal))

        # Get the size of a single sensitive volume (LOCAL coordinates)
        single_size_local = np.array(sizes[0])

        # Step 1: Find stacking axis (positions are in global coords)
        position_variance = np.var(positions, axis=0)
        stacking_axis = int(np.argmax(position_variance))

        # Step 2: Sort the LOCAL dimensions, find the bar width (second thinnest)
        size_perm = np.argsort(single_size_local)
        width = single_size_local[size_perm[1]]

        # Step 3: Calculate stacking span (including half strip width on each end)
        stacking_span = (
            positions[:, stacking_axis].max() - positions[:, stacking_axis].min()
        )
        stacking_span += width  # Add one strip width

        # Step 4: Build dimensions with stacking_span on the global stacking axis
        # If it is different from its original position, rearrange
        transformed_size = single_size_local.copy()
        transformed_size[size_perm[1]] = single_size_local[stacking_axis]
        transformed_size[stacking_axis] = stacking_span

        # Step 5: Apply the coordinate swap based on normal_axis_local
        # Rule: Don't apply a swap that involves the stacking axis
        if normal_axis_local == 2:  # Z-normal: no swap needed
            final_size = transformed_size
        elif normal_axis_local == 1:  # Y-normal: wants to swap Y↔Z
            if stacking_axis == 1 or stacking_axis == 2:
                # Swap involves stacking axis, don't do it
                final_size = transformed_size
            else:
                # Stacking is on X, safe to swap Y↔Z
                final_size = transformed_size[[0, 2, 1]]
        else:  # normal_axis_local == 0 (X-normal): wants to swap X↔Z
            if stacking_axis == 0:
                # Stacking on X, swap the other two (Y↔Z)
                final_size = transformed_size[[0, 2, 1]]
            elif stacking_axis == 2:
                # Stacking on Z, X↔Z would move stacking
                # But we may need to swap X↔Y depending on which local dim is thinnest
                # If Z is thinnest in local coords, swap X↔Y so X becomes thinnest
                # If Y is thinnest in local coords, don't swap to keep Y thinnest
                if np.argmin(single_size_local) == 2:  # Z thinnest in local
                    final_size = transformed_size[[1, 0, 2]]  # Swap X↔Y
                else:
                    final_size = transformed_size  # Keep as is
            else:
                # Stacking is on Y, safe to swap X↔Z
                final_size = transformed_size[[2, 1, 0]]

        # Step 6: Determine final normal axis (thinnest dimension)
        normal_axis = int(np.argmin(final_size))

        # Calculate aggregated position (average - positions are already global)
        aggregated_position = np.mean(positions, axis=0).tolist()
        transformed_size = final_size.tolist()

    return {
        "name": name,
        "dimensions": (
            transformed_size.tolist()
            if isinstance(transformed_size, np.ndarray)
            else transformed_size
        ),
        "position": aggregated_position,  # Position stays in original global coordinates
        "normal": normal_axis,
    }


def group_tpcs_by_cathode(tpc_list):
    """Group TPCs by shared cathode and drift direction.

    Parameters
    ----------
    tpc_list : list
        List of TPC info dictionaries

    Returns
    -------
    list
        List of TPC groups (each group is a list of TPC dicts)
    """
    # Create signatures for each TPC
    tpc_signatures = []
    for tpc in tpc_list:
        if "cathode_x" in tpc and "drift_direction" in tpc:
            cathode_x = round(tpc["cathode_x"], 1)
            drift_sign = 1 if tpc["drift_direction"][0] > 0 else -1
            tpc_signatures.append((cathode_x, drift_sign))
        elif "plane_positions" in tpc and len(tpc["plane_positions"]) > 0:
            plane_x_positions = tuple(
                round(pos[0], 1) for pos in tpc["plane_positions"]
            )
            x_dimension = round(tpc["dimensions"][0], 1)
            tpc_signatures.append((plane_x_positions, x_dimension))
        else:
            x_dimension = round(tpc["dimensions"][0], 1)
            tpc_signatures.append(((round(tpc["position"][0], 1),), x_dimension))

    # Group TPCs with matching signatures
    signature_to_group = {}
    tpc_groups = {}
    group_counter = 0

    for tpc, signature in zip(tpc_list, tpc_signatures):
        if signature not in signature_to_group:
            signature_to_group[signature] = group_counter
            tpc_groups[group_counter] = []
            group_counter += 1

        group_id = signature_to_group[signature]
        tpc_groups[group_id].append(tpc)

    return [tpc_groups[gid] for gid in sorted(tpc_groups.keys())]


def combine_tpc_group(group_tpcs):
    """Combine a group of TPCs into a single combined TPC.

    Parameters
    ----------
    group_tpcs : list
        List of TPC info dictionaries to combine

    Returns
    -------
    dict
        Combined TPC info
    """
    # Sort by z, then y position
    group_tpcs.sort(key=lambda t: (t["position"][2], t["position"][1]))

    dims = group_tpcs[0]["dimensions"].copy()

    # Check for multi-dimensional grid arrangement
    unique_y = sorted(set(round(tpc["position"][1], 1) for tpc in group_tpcs))
    unique_z = sorted(set(round(tpc["position"][2], 1) for tpc in group_tpcs))

    if len(unique_y) > 1 or len(unique_z) > 1:
        # Multi-dimensional arrangement - compute bounding box
        min_y = min(tpc["position"][1] - tpc["dimensions"][1] / 2 for tpc in group_tpcs)
        max_y = max(tpc["position"][1] + tpc["dimensions"][1] / 2 for tpc in group_tpcs)
        min_z = min(tpc["position"][2] - tpc["dimensions"][2] / 2 for tpc in group_tpcs)
        max_z = max(tpc["position"][2] + tpc["dimensions"][2] / 2 for tpc in group_tpcs)

        dims[1] = max_y - min_y
        dims[2] = max_z - min_z

        pos = group_tpcs[0]["position"].copy()
        pos[1] = (min_y + max_y) / 2
        pos[2] = (min_z + max_z) / 2
    else:
        # Single-dimensional arrangement along z-axis
        total_z_length = sum(tpc["dimensions"][2] for tpc in group_tpcs)
        dims[2] = total_z_length

        pos = group_tpcs[0]["position"].copy()
        if len(group_tpcs) > 1:
            pos[2] = 0.0  # Center at z=0 for multiple TPCs

    return {
        "dimensions": dims,
        "position": pos,
        "cryostat": group_tpcs[0]["cryostat"],
        "tpc_group": group_tpcs,
    }


def build_tpc_yaml(combined_tpcs, all_tpcs):
    """Build YAML structure for TPC section.

    Parameters
    ----------
    combined_tpcs : list
        List of combined TPC info dictionaries
    all_tpcs : list
        List of all original TPC info dictionaries

    Returns
    -------
    dict
        TPC YAML structure
    """
    tpc_yaml = {
        "dimensions": None,
        "module_ids": [],
        "positions": [],
    }

    # Build det_ids mapping
    first_cryo = combined_tpcs[0]["cryostat"]
    first_cryo_combined = [
        tpc for tpc in combined_tpcs if tpc["cryostat"] == first_cryo
    ]
    first_cryo_all = [tpc for tpc in all_tpcs if tpc["cryostat"] == first_cryo]

    logical_to_physical = {}
    for phys_idx, combined_tpc in enumerate(first_cryo_combined):
        for tpc_info in combined_tpc["tpc_group"]:
            logical_to_physical[tpc_info["tpc"]] = phys_idx

    det_ids = [logical_to_physical.get(tpc["tpc"], -1) for tpc in first_cryo_all]

    # Only include det_ids if not trivial
    if det_ids != list(range(len(det_ids))):
        tpc_yaml["det_ids"] = det_ids

    # Collect dimensions
    all_dimensions = [[round(d, 4) for d in tpc["dimensions"]] for tpc in combined_tpcs]

    if all(dim == all_dimensions[0] for dim in all_dimensions):
        tpc_yaml["dimensions"] = all_dimensions[0]
    else:
        tpc_yaml["dimensions"] = all_dimensions

    # Collect module IDs, positions, and drift directions
    drift_dirs = []
    for combined_tpc in combined_tpcs:
        tpc_yaml["module_ids"].append(combined_tpc["cryostat"])
        tpc_yaml["positions"].append([round(p, 4) for p in combined_tpc["position"]])

        drift_dir = combined_tpc["tpc_group"][0].get("drift_direction")
        if drift_dir:
            drift_dirs.append(
                [0.0 if x == 0 or x == -0.0 else round(x, 4) for x in drift_dir]
            )

    # Store drift_dirs only if number of TPCs per module is not 2
    num_modules = len(set(tpc_yaml["module_ids"]))
    tpcs_per_module = (
        len(combined_tpcs) // num_modules if num_modules > 0 else len(combined_tpcs)
    )
    if tpcs_per_module != 2:
        tpc_yaml["drift_dirs"] = drift_dirs

    return tpc_yaml


def build_optical_yaml(op_info_list, tpc_yaml):
    """Build YAML structure for optical section.

    Parameters
    ----------
    op_info_list : list
        List of optical detector info dictionaries
    tpc_yaml : dict
        TPC YAML structure (for offset calculation)

    Returns
    -------
    dict
        Optical YAML structure
    """
    if not op_info_list:
        return {}

    if len(op_info_list) > 1:
        warn(
            f"Multiple optical detector blocks found ({len(op_info_list)}). Only using first block."
        )

    op_info = op_info_list[0]

    # Calculate module center offset
    num_modules = len(set(tpc_yaml["module_ids"]))
    tpcs_per_module = (
        len(tpc_yaml["positions"]) // num_modules
        if num_modules > 0
        else len(tpc_yaml["positions"])
    )

    offset = np.zeros(3, dtype=float)
    for i, positions in enumerate(tpc_yaml["positions"]):
        if tpc_yaml["module_ids"][i] == 0:
            offset += np.array(positions) / tpcs_per_module

    # Offset positions to be relative to module center
    op_info["positions"] = op_info["positions"] - offset

    op_yaml = {
        "volume": "module",
        "global_index": len(op_info_list) < 2,
        "shape": (
            str(op_info["shape"])
            if isinstance(op_info["shape"], str)
            else [str(s) for s in op_info["shape"]]
        ),
        "dimensions": (
            [float(round(d, 4)) for d in op_info["dimensions"]]
            if isinstance(op_info["dimensions"][0], (int, float))
            else [[float(round(d, 4)) for d in dim] for dim in op_info["dimensions"]]
        ),
    }

    # Add shape_ids before positions to match original ordering
    if "shape_ids" in op_info:
        op_yaml["shape_ids"] = op_info["shape_ids"].tolist()

    op_yaml["positions"] = [
        [float(round(p, 4)) for p in pos] for pos in op_info["positions"]
    ]

    return op_yaml


def build_crt_yaml(crt_info_list):
    """Build YAML structure for CRT section.

    Parameters
    ----------
    crt_info_list : list
        List of CRT info dictionaries

    Returns
    -------
    dict
        CRT YAML structure
    """
    if not crt_info_list:
        return {}

    # Extract base names by removing ALL numbers
    # e.g., "volAuxDetMINOSmodule104cut400South" -> "volAuxDetMINOSmodulecutSouth"
    # e.g., "volAuxDetCRTStripArray5" -> "volAuxDetCRTStripArray"
    base_names = []
    for crt in crt_info_list:
        name = crt["name"]
        # Remove the volAuxDet prefix temporarily for easier pattern matching
        if name.startswith("volAuxDet"):
            name = name[9:]  # Remove "volAuxDet"

        # Remove ALL digit sequences
        base = re.sub(r"\d+", "", name)
        base = "volAuxDet" + base
        base_names.append(base)

    # Preserve order by using dict to track first occurrence
    seen_base_names = {}
    for i, base_name in enumerate(base_names):
        if base_name not in seen_base_names:
            seen_base_names[base_name] = i

    combined_crt_info = []
    # Iterate in order of first appearance
    for base_name in sorted(seen_base_names.keys(), key=lambda x: seen_base_names[x]):
        matching_modules = [
            crt for i, crt in enumerate(crt_info_list) if base_names[i] == base_name
        ]

        if len(matching_modules) > 1:
            # Group modules by normal direction first, then by coplanarity
            # This handles cases like SBND where all modules have the same name pattern
            # but belong to different physical planes

            # First, group by normal
            normal_groups = {}
            for mod in matching_modules:
                normal_axis = mod["normal"]
                if normal_axis not in normal_groups:
                    normal_groups[normal_axis] = []
                normal_groups[normal_axis].append(mod)

            # Then, within each normal group, subdivide by coplanarity
            coplanar_groups = []
            for normal_axis, modules_with_normal in normal_groups.items():
                # Create sub-groups for this normal direction
                normal_coplanar_groups = []

                # Group by position along the normal axis
                for mod in modules_with_normal:
                    pos_along_normal = mod["position"][normal_axis]

                    # Find existing coplanar group with similar position along normal
                    found_group = False
                    for group in normal_coplanar_groups:
                        # Check all modules in group to prevent chain drift
                        # Module must be within 25cm of ALL existing modules in group
                        all_close = all(
                            abs(pos_along_normal - m["position"][normal_axis]) < 25.0
                            for m in group
                        )

                        if all_close:
                            group.append(mod)
                            found_group = True
                            break

                    if not found_group:
                        normal_coplanar_groups.append([mod])

                # Add this normal's groups to the overall list
                coplanar_groups.extend(normal_coplanar_groups)

            # Now process each coplanar group separately
            for group in coplanar_groups:
                if len(group) > 1:
                    # Combine into bounding box
                    min_pos = np.min(
                        [
                            np.array(mod["position"]) - np.array(mod["dimensions"]) / 2
                            for mod in group
                        ],
                        axis=0,
                    )
                    max_pos = np.max(
                        [
                            np.array(mod["position"]) + np.array(mod["dimensions"]) / 2
                            for mod in group
                        ],
                        axis=0,
                    )
                    combined_dimensions = (max_pos - min_pos).tolist()
                    combined_position = ((min_pos + max_pos) / 2).tolist()

                    # Determine normal: prefer consensus from individual modules
                    individual_normals = [mod["normal"] for mod in group]
                    if len(set(individual_normals)) == 1:
                        # All modules agree on normal direction
                        combined_normal = individual_normals[0]
                    else:
                        # Disagreement: use thinnest dimension of aggregated result
                        combined_normal = int(np.argmin(combined_dimensions))

                    combined_crt_info.append(
                        {
                            "normal": combined_normal,
                            "dimensions": combined_dimensions,
                            "position": combined_position,
                        }
                    )
                else:
                    combined_crt_info.append(group[0])
        else:
            combined_crt_info.append(matching_modules[0])

    return {
        "normals": [crt["normal"] for crt in combined_crt_info],
        "dimensions": [
            [round(d, 4) for d in crt["dimensions"]] for crt in combined_crt_info
        ],
        "positions": [
            [round(p, 4) for p in crt["position"]] for crt in combined_crt_info
        ],
    }


def main(source, output=None, cathode_thickness=0.0, pixel_size=0.0):
    """Main function for parsing LArSoft geometry files.

    Parameters
    ----------
    source : str
        Path to the dumped LArSoft geometry text file
    output : str, optional
        Path to output YAML file
    cathode_thickness : float, optional
        Cathode thickness in cm (default: 0.0)
    pixel_size : float, optional
        Pixel size in cm (default: 0.0)
    """
    # Read geometry file
    with open(source, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Extract detector name from file name
    base = os.path.basename(source)
    detector_match = re.match(r"([A-Za-z0-9_-]+)-geometry", base)
    if not detector_match:
        raise ValueError("Could not extract detector name from file name")
    detector = detector_match.group(1)

    # Extract tag and version
    assert len(lines) > 1, "Geometry text file is too short"
    tag_match = re.search(r"Detector\s+([a-zA-Z0-9_-]+)", lines[1])
    if not tag_match:
        raise ValueError("Could not extract tag from geometry file")
    tag = tag_match.group(1)

    version_match = re.search(r"(\d+)$", tag)
    if not version_match:
        raise ValueError("Could not extract version from tag")
    version = int(version_match.group(1))

    # Extract GDML file name
    gdml = None
    if lines and ".gdml" in lines[0]:
        match = re.search(r"'([^']+\.gdml)'", lines[0])
        if match:
            gdml = os.path.basename(match.group(1))

    # Parse all detector blocks
    tpc_list = []
    op_info_list = []
    crt_info_list = []

    for i, line in enumerate(lines):
        if line.strip().startswith("TPC C:"):
            tpc_info = parse_tpc_block(lines, i, cathode_thickness, pixel_size)
            if "position" in tpc_info and "dimensions" in tpc_info:
                tpc_list.append(tpc_info)
        elif line.strip().startswith("[OpDet #0]"):
            op_info = parse_optical_block(lines, i)
            if "positions" in op_info and "dimensions" in op_info:
                op_info_list.append(op_info)
        elif line.strip().split("]")[-1].startswith(' "volAuxDet'):
            crt_info = parse_crt_block(lines, i)
            if "position" in crt_info and "dimensions" in crt_info:
                crt_info_list.append(crt_info)

    print(f"Found {len(tpc_list)} logical TPCs")
    print(f"Found {len(op_info_list)} optical detector blocks")
    print(f"Found {len(crt_info_list)} CRT detector blocks\n")

    # Sort and process TPCs
    tpc_list.sort(key=lambda x: (x["cryostat"], x["tpc"]))
    all_tpcs = tpc_list.copy()

    # Group and combine TPCs
    tpc_groups = group_tpcs_by_cathode(tpc_list)
    combined_tpcs = [combine_tpc_group(group) for group in tpc_groups]

    # Build YAML structures
    tpc_yaml = build_tpc_yaml(combined_tpcs, all_tpcs)
    op_yaml = build_optical_yaml(op_info_list, tpc_yaml)
    crt_yaml = build_crt_yaml(crt_info_list)

    # Build top-level YAML
    yaml_data = {
        "name": detector,
        "tag": tag,
        "version": version,
        "gdml": gdml,
        "tpc": tpc_yaml,
    }

    if op_yaml:
        yaml_data["optical"] = op_yaml
    if crt_yaml:
        yaml_data["crt"] = crt_yaml

    # Print YAML
    print("YAML file:")
    print("=" * 60)
    print(yaml.dump(yaml_data, default_flow_style=None, sort_keys=False))
    print("=" * 60)

    # Print summary
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

    # Write YAML file
    output_path = output if output else source.replace(".txt", "_tpc.yaml")
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_data, f, default_flow_style=None, sort_keys=False)
    print(f"\nYAML file written to: {output_path}")


if __name__ == "__main__":
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
        help="Cathode thickness in cm (default: 0.0)",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--pixel-size",
        "-p",
        help="Pixel size in cm (default: 0.0)",
        type=float,
        default=0.0,
    )

    args = parser.parse_args()

    main(
        source=args.source,
        output=args.output,
        cathode_thickness=args.cathode_thickness,
        pixel_size=args.pixel_size,
    )
