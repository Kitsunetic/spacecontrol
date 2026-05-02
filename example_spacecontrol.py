import os
import gc
import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
import torch as th
import trimesh
from PIL import Image

# Trellis v1 Imports
from trellis.pipelines import TrellisTextTo3DPipeline
from trellis.utils import postprocessing_utils


def resolve_padding(args) -> np.ndarray:
    padding = np.array([args.normalize_padding, args.normalize_padding, args.normalize_padding], dtype=np.float64)
    if args.normalize_padding_x is not None:
        padding[0] = args.normalize_padding_x
    if args.normalize_padding_y is not None:
        padding[1] = args.normalize_padding_y
    if args.normalize_padding_z is not None:
        padding[2] = args.normalize_padding_z
    fill_ratio = 1.0 - 2.0 * padding
    if np.any(fill_ratio <= 0.0) or np.any(fill_ratio > 1.0):
        raise ValueError(f"normalize padding must keep per-axis fill ratios in (0, 1], got padding={padding.tolist()}")
    return padding


def apply_denormalization_transform(glb, center: np.ndarray, scales: np.ndarray) -> None:
    inv_scale = np.eye(4)
    inv_scale[0, 0] = 1.0 / scales[0]
    inv_scale[1, 1] = 1.0 / scales[1]
    inv_scale[2, 2] = 1.0 / scales[2]
    rot_matrix = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    glb.apply_transform(inv_scale)
    glb.apply_translation(center)
    glb.apply_transform(rot_matrix)


def filter_components(
    glb: trimesh.Trimesh,
    max_y_threshold: float | None = None,
    min_area: float | None = None,
) -> trimesh.Trimesh:
    components = glb.split(only_watertight=False)
    keep = []
    for component in components:
        if max_y_threshold is not None and component.bounds[1][1] < max_y_threshold:
            continue
        if min_area is not None and component.area < min_area:
            continue
        keep.append(component)
    if not keep:
        return glb
    return trimesh.util.concatenate(keep)


def main(args):
    # 출력 디렉토리 생성
    os.makedirs(args.out_dir, exist_ok=True)

    # 입력 이미지 경로에서 파일명만 추출
    image_basename = os.path.splitext(os.path.basename(args.image))[0]
    mesh_out = os.path.join(args.out_dir, f"{image_basename}-tau{args.tau}.glb")

    # 1. Load Pipeline (v1)
    print("Loading Trellis v1 Pipeline...")
    # 노트북 환경과 동일하게 "gui" 폴더 또는 해당되는 모델 경로 사용
    pipeline = TrellisTextTo3DPipeline.from_pretrained("gui")
    pipeline.cuda()

    # 2. Load Image & Spatial Control Shape
    print(f"Loading image from: {args.image}")
    image_prompt = Image.open(args.image)

    print(f"Using spatial control mesh from: {args.control}")
    last_path = Path(args.control)
    last_normalized_path = last_path.parent / f"{last_path.stem}_normalized{last_path.suffix}"

    # 3. Normalize Mesh (v1 특화 로직)
    print("Normalizing spatial control mesh...")
    mesh = o3d.io.read_triangle_mesh(str(last_path))
    aabb = mesh.get_axis_aligned_bounding_box()
    min_bound = aabb.get_min_bound()
    max_bound = aabb.get_max_bound()
    center = (min_bound + max_bound) / 2
    max_extent = (max_bound - min_bound).max()
    padding = resolve_padding(args)
    scales = (1.0 - 2.0 * padding) / max_extent
    vertices = np.asarray(mesh.vertices)
    vertices = (vertices - center) * scales[None, :]
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    o3d.io.write_triangle_mesh(str(last_normalized_path), mesh)

    # 4. Run Pipeline with SpaceControl
    # CLI의 tau 값이 노트북의 t0_idx_value 로 매핑됩니다.
    print(f"Generating 3D model with tau (t0_idx)={args.tau}...")
    outputs = pipeline.run(
        args.prompt,
        image_prompt,
        seed=1,
        sparse_structure_sampler_params={
            "steps": 12,
            "cfg_strength": 7.5,
            "t0_idx_value": args.tau,  # tau 값을 적용
            "spatial_control_mesh_path": str(last_normalized_path),
        },
    )

    # 5. Convert to GLB
    print("Postprocessing to GLB...")
    glb = postprocessing_utils.to_glb(
        outputs["gaussian"][0],
        outputs["mesh"][0],
        simplify=0.95,
        texture_size=1024,
    )

    # 6. Denormalize & Rotate (원래 크기와 위치, 회전값 복구)
    print("Applying denormalization and rotations...")
    apply_denormalization_transform(glb, center, scales)
    if args.drop_components_below_y is not None or args.drop_components_below_area is not None:
        glb = filter_components(
            glb,
            max_y_threshold=args.drop_components_below_y,
            min_area=args.drop_components_below_area,
        )
    # glb.apply_scale(1.0) # 필요시 rescale 조절

    # 7. Export to GLB
    print(f"Exporting model to {mesh_out}...")
    glb.export(str(mesh_out))

    # Clean up 메모리 누수 방지
    gc.collect()
    th.cuda.empty_cache()
    print("✨ Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SpaceControl Trellis v1 Pipeline")

    # v2의 example_spacecontrol.py와 완전히 동일한 인자 구성
    parser.add_argument("--image", type=str, required=True, help="Path to the input image")
    parser.add_argument("--control", type=str, required=True, help="Path to the spatial control mesh")
    parser.add_argument("--prompt", type=str, default="a low-top sneaker", help="Sparse-stage text prompt")
    parser.add_argument("--tau", type=int, default=6, help="Strength of spatial control (maps to t0_idx, default 6)")
    parser.add_argument("--normalize_padding", type=float, default=0.0)
    parser.add_argument("--normalize_padding_x", type=float, default=0.02)
    parser.add_argument("--normalize_padding_y", type=float, default=0.0)
    parser.add_argument("--normalize_padding_z", type=float, default=0.0)
    parser.add_argument("--drop_components_below_y", type=float, default=None)
    parser.add_argument("--drop_components_below_area", type=float, default=None)
    parser.add_argument("--out_dir", type=str, default="outputs", help="Directory to save the generated files")

    args = parser.parse_args()
    main(args)
