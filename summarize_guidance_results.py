import argparse
import csv
import json
from pathlib import Path

import numpy as np
import trimesh

from evaluate_generation import evaluate as evaluate_metrics


def load_mesh_summary(mesh_path: Path) -> dict:
    asset = trimesh.load(mesh_path, force="scene")
    if isinstance(asset, trimesh.Scene):
        geometries = tuple(asset.geometry.values())
        mesh = trimesh.util.concatenate(geometries)
    else:
        mesh = asset

    bounds = np.asarray(mesh.bounds)
    extents = bounds[1] - bounds[0]
    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "extent_x": float(extents[0]),
        "extent_y": float(extents[1]),
        "extent_z": float(extents[2]),
    }


def load_guidance_summary(json_path: Path) -> dict:
    if not json_path.exists():
        return {}

    with open(json_path) as f:
        payload = json.load(f)

    steps = payload.get("guidance_steps", [])
    if not steps:
        return {
            "guidance_type": payload.get("guidance_type"),
            "guidance_strength": payload.get("guidance_strength"),
            "export_mode": payload.get("export_mode", "unknown"),
        }

    first = steps[0]
    last = steps[-1]
    losses = [step.get("guidance_loss") for step in steps if "guidance_loss" in step]
    ious = [step.get("occupancy_iou") for step in steps if "occupancy_iou" in step]
    return {
        "guidance_type": payload.get("guidance_type"),
        "guidance_strength": payload.get("guidance_strength"),
        "export_mode": payload.get("export_mode", "textured_glb"),
        "normalize_padding": payload.get("normalize_padding"),
        "normalize_padding_xyz": payload.get("normalize_padding_xyz"),
        "drop_components_below_y": payload.get("drop_components_below_y"),
        "drop_components_below_area": payload.get("drop_components_below_area"),
        "guided_steps": len(steps),
        "loss_start": float(first.get("guidance_loss")) if "guidance_loss" in first else None,
        "loss_end": float(last.get("guidance_loss")) if "guidance_loss" in last else None,
        "loss_min": float(min(losses)) if losses else None,
        "loss_max": float(max(losses)) if losses else None,
        "iou_end": float(last.get("occupancy_iou")) if "occupancy_iou" in last else None,
        "iou_max": float(max(ious)) if ious else None,
    }


def summarize_entry(mesh_path: Path, image_path: Path | None = None) -> dict:
    row = {
        "name": mesh_path.stem,
        "path": str(mesh_path),
        "kind": "guidance" if mesh_path.parent.name.startswith("guidance_") else mesh_path.parent.name,
    }
    row.update(load_mesh_summary(mesh_path))
    row.update(load_guidance_summary(mesh_path.with_suffix(".json")))
    if image_path is not None:
        row.update(evaluate_metrics(mesh_path, image_path))
    return row


def format_value(value):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def print_table(rows: list[dict]) -> None:
    headers = [
        "name",
        "kind",
        "guidance_type",
        "guidance_strength",
        "silhouette_iou",
        "bottom_fill_ratio",
        "containment_recall",
        "outside_fraction",
        "largest_component_share",
        "vertices",
        "faces",
        "drop_components_below_area",
        "normalize_padding_xyz",
    ]
    widths = {
        header: max(len(header), *(len(format_value(row.get(header))) for row in rows)) for header in headers
    }
    print(" | ".join(header.ljust(widths[header]) for header in headers))
    print("-+-".join("-" * widths[header] for header in headers))
    for row in rows:
        print(" | ".join(format_value(row.get(header)).ljust(widths[header]) for header in headers))


def main():
    parser = argparse.ArgumentParser(description="Summarize baseline and geometry-guidance outputs")
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[
            "results/spacecontrol_sneaker_padx02_best",
            "results/guidance_containment_sneaker_padx02_best",
        ],
        help="Directories to scan for GLB results",
    )
    parser.add_argument("--image", type=str, default="assets/shoe4_rembg.png", help="Reference image for evaluation")
    parser.add_argument("--csv_out", type=str, default=None, help="Optional CSV output path")
    args = parser.parse_args()

    mesh_paths = []
    for input_path in args.inputs:
        mesh_paths.extend(sorted(Path(input_path).glob("*.glb")))

    image_path = Path(args.image) if args.image else None
    rows = [summarize_entry(mesh_path, image_path) for mesh_path in mesh_paths]
    print_table(rows)

    if args.csv_out:
        with open(args.csv_out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sorted({key for row in rows for key in row.keys()}))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
