import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image


def load_mesh(mesh_path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(mesh_path, force="scene")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    return mesh


def load_image_mask(image_path: Path, threshold: int = 245) -> np.ndarray:
    image = np.array(Image.open(image_path).convert("RGB"))
    mask = ((image < threshold).any(axis=-1)).astype(np.uint8)
    ys, xs = np.where(mask)
    return mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]


def best_silhouette_iou(mesh: trimesh.Trimesh, image_mask: np.ndarray, pitch: float = 0.008) -> dict:
    vox = mesh.voxelized(pitch=pitch)
    matrix = vox.matrix
    best = None
    for proj_axis in [0, 1, 2]:
        silhouette = matrix.any(axis=proj_axis).astype(np.uint8)
        ys, xs = np.where(silhouette)
        silhouette = silhouette[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
        for transpose in [False, True]:
            arr = silhouette.T if transpose else silhouette
            for flip_x in [False, True]:
                arr_x = np.fliplr(arr) if flip_x else arr
                for flip_y in [False, True]:
                    arr_xy = np.flipud(arr_x) if flip_y else arr_x
                    resized = Image.fromarray(arr_xy * 255).resize(
                        (image_mask.shape[1], image_mask.shape[0]), Image.Resampling.NEAREST
                    )
                    pred_mask = (np.array(resized) > 127).astype(np.uint8)
                    inter = (pred_mask & image_mask).sum()
                    union = ((pred_mask | image_mask) > 0).sum()
                    iou = float(inter / union)
                    candidate = {
                        "silhouette_iou": iou,
                        "proj_axis": proj_axis,
                        "transpose": transpose,
                        "flip_x": flip_x,
                        "flip_y": flip_y,
                    }
                    if best is None or candidate["silhouette_iou"] > best["silhouette_iou"]:
                        best = candidate
    return best


def fragmentation_metrics(mesh: trimesh.Trimesh) -> dict:
    components = mesh.split(only_watertight=False)
    areas = np.array([component.area for component in components], dtype=np.float64)
    total_area = float(areas.sum())
    return {
        "component_count": int(len(components)),
        "largest_component_share": float(areas.max() / total_area) if total_area > 0 else 0.0,
        "major_component_count": int((areas > total_area * 0.01).sum()) if total_area > 0 else 0,
    }


def bottom_fill_metrics(mesh: trimesh.Trimesh, pitch: float = 0.008, band_ratio: float = 0.08) -> dict:
    vox = mesh.voxelized(pitch=pitch)
    matrix = vox.matrix
    if matrix.ndim != 3 or matrix.shape[1] == 0:
        return {
            "bottom_fill_ratio": 0.0,
            "bottom_band_layers": 0,
        }

    footprint = matrix.any(axis=1)
    footprint_area = int(footprint.sum())
    if footprint_area == 0:
        return {
            "bottom_fill_ratio": 0.0,
            "bottom_band_layers": 0,
        }

    band_layers = max(2, int(np.ceil(matrix.shape[1] * band_ratio)))
    band_layers = min(band_layers, matrix.shape[1])
    bottom_band = matrix[:, :band_layers, :].any(axis=1)
    bottom_fill_ratio = float(bottom_band.sum() / footprint_area)
    return {
        "bottom_fill_ratio": bottom_fill_ratio,
        "bottom_band_layers": int(band_layers),
    }


def load_guidance_metrics(stats_path: Path) -> dict:
    if not stats_path.exists():
        return {}
    with open(stats_path) as f:
        payload = json.load(f)
    steps = payload.get("guidance_steps", [])
    if not steps:
        return {}
    last = steps[-1]
    keys = [
        "containment_recall",
        "outside_fraction",
        "shell_fill",
        "envelope_iou",
        "guidance_loss",
    ]
    return {key: float(last[key]) for key in keys if key in last}


def evaluate(mesh_path: Path, image_path: Path) -> dict:
    mesh = load_mesh(mesh_path)
    image_mask = load_image_mask(image_path)
    metrics = {}
    metrics.update(best_silhouette_iou(mesh, image_mask))
    metrics.update(fragmentation_metrics(mesh))
    metrics.update(bottom_fill_metrics(mesh))
    metrics.update(load_guidance_metrics(mesh_path.with_suffix(".json")))
    return metrics


def judge(metrics: dict, thresholds: dict) -> dict:
    checks = {
        "silhouette_iou": metrics.get("silhouette_iou", 0.0) >= thresholds["silhouette_iou"],
        "containment_recall": metrics.get("containment_recall", 0.0) >= thresholds["containment_recall"],
        "outside_fraction": metrics.get("outside_fraction", 1.0) <= thresholds["outside_fraction"],
        "largest_component_share": metrics.get("largest_component_share", 0.0) >= thresholds["largest_component_share"],
    }
    checks["passed"] = all(checks.values())
    return checks


def main():
    parser = argparse.ArgumentParser(description="Evaluate generated shoe meshes against fixed success metrics")
    parser.add_argument("--mesh", type=str, required=True)
    parser.add_argument("--image", type=str, default="assets/shoe4_rembg.png")
    parser.add_argument("--silhouette_iou", type=float, default=0.80)
    parser.add_argument("--containment_recall", type=float, default=0.85)
    parser.add_argument("--outside_fraction", type=float, default=0.45)
    parser.add_argument("--largest_component_share", type=float, default=0.15)
    args = parser.parse_args()

    thresholds = {
        "silhouette_iou": args.silhouette_iou,
        "containment_recall": args.containment_recall,
        "outside_fraction": args.outside_fraction,
        "largest_component_share": args.largest_component_share,
    }
    metrics = evaluate(Path(args.mesh), Path(args.image))
    checks = judge(metrics, thresholds)
    print(json.dumps({"metrics": metrics, "thresholds": thresholds, "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
