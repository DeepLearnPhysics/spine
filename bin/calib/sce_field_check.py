#!/usr/bin/env python3
"""Visual sanity check for ROOT TH3 space-charge displacement maps."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from spine.calib.field import FieldCalibrator
from spine.geo import GeoManager

DEFAULT_MAPS = {
    "icarus": "sce/SCEoffsets_ICARUS_E500_voxelTH3.root",
    "sbnd": "sce/SCEoffsets_SBND_E500_voxelTH3.root",
}

AXES = ("x", "y", "z")
PROJECTIONS = ((0, 1), (0, 2), (1, 2))


def main() -> None:
    args = parse_args()

    import matplotlib.pyplot as plt

    detector = args.detector.lower()
    map_file = Path(args.map_file or DEFAULT_MAPS[detector])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    geo = GeoManager.initialize_or_get(
        args.detector, tag=args.tag, version=args.version
    )
    calibrator = FieldCalibrator(
        map_file=map_file,
        map_prefix=args.map_prefix,
        scale=args.scale,
        bounds=args.bounds,
    )

    points = sample_points(
        geo,
        calibrator.field_map.range,
        num_points=args.num_points,
        mode=args.sample_volume,
        seed=args.seed,
    )
    offsets = apply_field(calibrator, geo, points) - points

    for component, name in enumerate(AXES):
        fig = plot_component(
            points,
            offsets[:, component],
            component_name=f"d{name}",
            detector=geo.name,
            map_file=map_file,
            sample_volume=args.sample_volume,
            point_size=args.point_size,
        )
        out_path = output_dir / f"{detector}_{args.map_prefix}_d{name}.png"
        fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample points in an SBND or ICARUS geometry, apply a TH3 SCE "
            "displacement map, and save dx/dy/dz projection plots."
        )
    )
    parser.add_argument(
        "--detector",
        choices=sorted(DEFAULT_MAPS),
        default="sbnd",
        help="Detector geometry and default SCE map to use.",
    )
    parser.add_argument("--tag", help="Optional geometry tag.")
    parser.add_argument("--version", help="Optional geometry version.")
    parser.add_argument(
        "--map-file",
        help="ROOT SCE map file. Defaults to the matching file under sce/.",
    )
    parser.add_argument(
        "--map-prefix",
        default="TrueFwd_Displacement",
        help=(
            "TH3 component prefix. The script reads {prefix}_X/Y/Z. "
            "Use TrueBkwd_Displacement for backward maps."
        ),
    )
    parser.add_argument(
        "--sample-volume",
        choices=("intersection", "geometry", "map"),
        default="geometry",
        help=(
            "Volume to sample: detector TPC geometry, SCE map bounds, or their "
            "intersection."
        ),
    )
    parser.add_argument(
        "--bounds",
        choices=("clip", "zero", "raise"),
        default="zero",
        help="Out-of-map behavior used by FieldMap.",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=50000,
        help="Number of random points to sample.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed for reproducible point throws.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Displacement scale passed to FieldCalibrator.",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=1.0,
        help="Scatter marker size.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Output image DPI.",
    )
    parser.add_argument(
        "--output-dir",
        default="sce/plots",
        help="Directory where dx/dy/dz PNG files are written.",
    )
    return parser.parse_args()


def sample_points(
    geo,
    map_range: np.ndarray,
    num_points: int,
    mode: str,
    seed: int,
) -> np.ndarray:
    """Sample points from the requested detector/map volume."""
    rng = np.random.default_rng(seed)
    volumes = sample_volumes(geo, map_range, mode)
    if not volumes:
        raise ValueError(
            "No non-empty sample volume found. Try --sample-volume geometry or map."
        )

    volumes_arr = np.asarray(volumes, dtype=float)
    extents = volumes_arr[:, :, 1] - volumes_arr[:, :, 0]
    weights = np.prod(extents, axis=1)
    weights = weights / np.sum(weights)

    choices = rng.choice(len(volumes), size=num_points, p=weights)
    points = np.empty((num_points, 3), dtype=float)
    for volume_id, volume in enumerate(volumes_arr):
        mask = choices == volume_id
        count = int(np.count_nonzero(mask))
        if count == 0:
            continue
        low = volume[:, 0]
        high = volume[:, 1]
        points[mask] = rng.uniform(low, high, size=(count, 3))

    return points


def apply_field(calibrator: FieldCalibrator, geo, points: np.ndarray) -> np.ndarray:
    """Apply a field calibrator to a mixed-TPC point cloud."""
    corrected = np.asarray(points, dtype=float).copy()
    tpc_ids = geo.get_closest_tpc(points)
    for tpc_id in range(geo.tpc.num_chambers):
        index = np.where(tpc_ids == tpc_id)[0]
        if len(index) == 0:
            continue
        corrected[index] = calibrator.process(points[index], tpc_id)

    return corrected


def sample_volumes(geo, map_range: np.ndarray, mode: str) -> list[np.ndarray]:
    """Build a list of box volumes to sample."""
    if mode == "map":
        return [np.asarray(map_range, dtype=float)]

    volumes = []
    for chamber in geo.tpc.chambers:
        volume = np.asarray(chamber.boundaries, dtype=float)
        if mode == "intersection":
            volume = np.stack(
                (
                    np.maximum(volume[:, 0], map_range[:, 0]),
                    np.minimum(volume[:, 1], map_range[:, 1]),
                ),
                axis=1,
            )
        if np.all(volume[:, 1] > volume[:, 0]):
            volumes.append(volume)

    return volumes


def plot_component(
    points: np.ndarray,
    values: np.ndarray,
    component_name: str,
    detector: str,
    map_file: Path,
    sample_volume: str,
    point_size: float,
):
    """Draw one offset component in the three 2D coordinate projections."""
    import matplotlib.pyplot as plt

    vmax = np.max(np.abs(values))
    if vmax == 0.0:
        vmax = 1.0

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    scatter = None
    for ax, projection in zip(axes, PROJECTIONS):
        xaxis, yaxis = projection
        scatter = ax.scatter(
            points[:, xaxis],
            points[:, yaxis],
            c=values,
            s=point_size,
            cmap="coolwarm",
            vmin=-vmax,
            vmax=vmax,
            linewidths=0,
            rasterized=True,
        )
        ax.set_xlabel(f"{AXES[xaxis]} [cm]")
        ax.set_ylabel(f"{AXES[yaxis]} [cm]")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linewidth=0.3, alpha=0.35)
        ax.set_title(f"{AXES[xaxis]}-{AXES[yaxis]}")

    fig.suptitle(
        f"{detector} {component_name} offset [cm] | {sample_volume} | {map_file.name}"
    )
    assert scatter is not None
    colorbar = fig.colorbar(scatter, ax=axes, shrink=0.9)
    colorbar.set_label(f"{component_name} [cm]")
    return fig


if __name__ == "__main__":
    main()
