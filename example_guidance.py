import argparse
import gc
import json
import os
from pathlib import Path

import numpy as np
import open3d as o3d
import torch as th
import torch.nn.functional as F
import trimesh
from PIL import Image

from trellis.pipelines import TrellisTextTo3DPipeline
from trellis.pipelines.guidance import build_geometry_guidance
from trellis.pipelines.samplers import FlowEulerGeometryGuidanceSampler
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


def apply_denormalization_transform(glb: trimesh.Trimesh, center: np.ndarray, scales: np.ndarray) -> None:
    inv_scale = np.eye(4)
    inv_scale[0, 0] = 1.0 / scales[0]
    inv_scale[1, 1] = 1.0 / scales[1]
    inv_scale[2, 2] = 1.0 / scales[2]
    rot_matrix = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )
    glb.apply_transform(inv_scale)
    glb.apply_translation(center)
    glb.apply_transform(rot_matrix)


def normalize_mesh(mesh_path: str, padding: np.ndarray) -> tuple[Path, np.ndarray, np.ndarray]:
    source_path = Path(mesh_path)
    normalized_path = source_path.parent / f"{source_path.stem}_guidance_normalized{source_path.suffix}"

    mesh = o3d.io.read_triangle_mesh(str(source_path))
    aabb = mesh.get_axis_aligned_bounding_box()
    min_bound = aabb.get_min_bound()
    max_bound = aabb.get_max_bound()
    center = (min_bound + max_bound) / 2
    max_extent = (max_bound - min_bound).max()
    scales = (1.0 - 2.0 * padding) / max_extent
    vertices = np.asarray(mesh.vertices)
    vertices = (vertices - center) * scales[None, :]
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    o3d.io.write_triangle_mesh(str(normalized_path), mesh)
    return normalized_path, center, scales


def move_models(pipeline, model_names, device: str):
    for name in model_names:
        model = pipeline.models.get(name)
        if model is not None:
            model.to(th.device(device))


def get_image_cond(pipeline, image: Image.Image) -> dict:
    image_tensor = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    image_tensor = th.from_numpy(image_tensor).permute(2, 0, 1).float()[None]
    image_device = next(pipeline.models["image_cond_model"].parameters()).device
    image_tensor = pipeline.image_cond_model_transform(image_tensor).to(image_device)
    features = pipeline.models["image_cond_model"](image_tensor, is_training=True)["x_prenorm"]
    patchtokens = F.layer_norm(features, features.shape[-1:])
    neg_cond = th.zeros_like(patchtokens)
    return {
        "cond": patchtokens,
        "neg_cond": neg_cond,
    }


def export_mesh_only_glb(mesh, mesh_out: str, center: np.ndarray, scales: np.ndarray) -> None:
    vertices = mesh.vertices.detach().cpu().numpy()
    faces = mesh.faces.detach().cpu().numpy()

    vertex_colors = None
    if getattr(mesh, "vertex_attrs", None) is not None:
        attrs = mesh.vertex_attrs.detach().cpu().numpy()
        if attrs.ndim == 2 and attrs.shape[1] >= 3:
            vertex_colors = np.clip(attrs[:, :3], 0.0, 1.0)
            vertex_colors = (vertex_colors * 255).astype(np.uint8)

    glb = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=vertex_colors, process=False)
    apply_denormalization_transform(glb, center, scales)
    glb.export(str(mesh_out))


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
    cleaned = trimesh.util.concatenate(keep)
    return cleaned


def run_guidance(args):
    os.makedirs(args.out_dir, exist_ok=True)

    image_basename = os.path.splitext(os.path.basename(args.image))[0]
    mesh_out = os.path.join(args.out_dir, f"{image_basename}-{args.guidance_type}-g{args.guidance_strength}.glb")
    stats_out = os.path.join(args.out_dir, f"{image_basename}-{args.guidance_type}-g{args.guidance_strength}.json")

    pipeline = TrellisTextTo3DPipeline.from_pretrained("gui")
    pipeline.cuda()

    padding = resolve_padding(args)
    normalized_mesh_path, center, scales = normalize_mesh(args.control, padding)
    image_prompt = Image.open(args.image)
    if args.preprocess_image:
        image_prompt = pipeline.preprocess_image(image_prompt)

    guidance = build_geometry_guidance(
        args.guidance_type,
        pipeline,
        str(normalized_mesh_path),
        bce_weight=args.bce_weight,
        dice_weight=args.dice_weight,
        envelope_radius=args.envelope_radius,
        contain_weight=args.contain_weight,
        outside_weight=args.outside_weight,
        shell_weight=args.shell_weight,
    )
    guidance_sampler = FlowEulerGeometryGuidanceSampler(sigma_min=pipeline.sparse_structure_sampler.sigma_min)

    cond_text = pipeline.get_cond_text([args.sparse_prompt])
    move_models(
        pipeline,
        [
            "slat_decoder_gs",
            "slat_decoder_rf",
            "slat_decoder_mesh",
            "slat_flow_model_text",
            "slat_flow_model_image",
            "image_cond_model",
        ],
        "cpu",
    )
    pipeline.text_cond_model["model"].cpu()
    if args.guidance_type == "latent":
        pipeline.models["sparse_structure_encoder"].cpu()
    th.cuda.empty_cache()

    th.manual_seed(args.seed)
    flow_model = pipeline.models["sparse_structure_flow_model"]
    reso = flow_model.resolution
    noise = th.randn(1, flow_model.in_channels, reso, reso, reso, device=th.device("cuda"))
    sampler_output = guidance_sampler.sample(
        flow_model,
        noise,
        **cond_text,
        steps=args.steps,
        cfg_strength=args.cfg_strength,
        cfg_interval=(args.cfg_interval_start, args.cfg_interval_end),
        rescale_t=args.rescale_t,
        geometry_guidance=guidance,
        guidance_strength=args.guidance_strength,
        guidance_interval=(args.guidance_interval_start, args.guidance_interval_end),
        guidance_schedule=args.guidance_schedule,
        guidance_rescale=args.guidance_rescale,
        grad_clip=args.grad_clip,
        verbose=True,
    )
    guidance_steps = list(sampler_output.guidance)
    z_s = sampler_output.samples
    decoder = pipeline.models["sparse_structure_decoder"]
    coords = th.argwhere(decoder(z_s) > 0)[:, [0, 2, 3, 4]].int()
    del sampler_output
    del z_s
    th.cuda.empty_cache()

    move_models(
        pipeline,
        [
            "slat_decoder_gs",
            "slat_decoder_mesh",
            "slat_flow_model_image",
            "image_cond_model",
        ],
        "cuda",
    )
    pipeline.models["sparse_structure_decoder"].cuda()
    pipeline.models["sparse_structure_flow_model"].cpu()
    th.cuda.empty_cache()

    with th.no_grad():
        cond_image = get_image_cond(pipeline, image_prompt)
        slat = pipeline.sample_slat(cond_image, coords, {})
    del cond_image
    del coords
    th.cuda.empty_cache()

    move_models(
        pipeline,
        [
            "slat_flow_model_image",
            "image_cond_model",
            "sparse_structure_decoder",
            "slat_decoder_rf",
        ],
        "cpu",
    )
    th.cuda.empty_cache()

    export_mode = "textured_glb"
    try:
        with th.no_grad():
            outputs = pipeline.decode_slat(slat, ["mesh", "gaussian"])
        glb = postprocessing_utils.to_glb(
            outputs["gaussian"][0],
            outputs["mesh"][0],
            simplify=0.95,
            texture_size=1024,
        )

        apply_denormalization_transform(glb, center, scales)
        if args.drop_components_below_y is not None or args.drop_components_below_area is not None:
            glb = filter_components(
                glb,
                max_y_threshold=args.drop_components_below_y,
                min_area=args.drop_components_below_area,
            )
        glb.export(str(mesh_out))
        del outputs
        del glb
    except RuntimeError as err:
        if "out of memory" not in str(err).lower():
            raise
        export_mode = "mesh_only_fallback"
        gc.collect()
        th.cuda.empty_cache()
        mesh_decoder = pipeline.models["slat_decoder_mesh"]
        if hasattr(mesh_decoder, "convert_to_fp16"):
            mesh_decoder.convert_to_fp16()
        with th.no_grad():
            mesh = mesh_decoder(slat.half())[0]
        export_mesh_only_glb(mesh, mesh_out, center, scales)
        del mesh

    del slat
    th.cuda.empty_cache()

    with open(stats_out, "w") as f:
        json.dump(
            {
                "image": args.image,
                "control": args.control,
                "guidance_type": args.guidance_type,
                "guidance_strength": args.guidance_strength,
                "guidance_interval": [args.guidance_interval_start, args.guidance_interval_end],
                "guidance_schedule": args.guidance_schedule,
                "guidance_steps": guidance_steps,
                "export_mode": export_mode,
                "drop_components_below_y": args.drop_components_below_y,
                "drop_components_below_area": args.drop_components_below_area,
                "normalize_padding": args.normalize_padding,
                "normalize_padding_xyz": padding.tolist(),
            },
            f,
            indent=2,
        )

    gc.collect()
    th.cuda.empty_cache()
    return mesh_out, stats_out


def main():
    parser = argparse.ArgumentParser(description="Run geometry guidance variants for Trellis sparse-structure sampling")
    parser.add_argument("--image", type=str, default="assets/shoe4_rembg.png", help="Path to the input image")
    parser.add_argument("--control", type=str, default="assets/last_normalized.ply", help="Path to the control mesh")
    parser.add_argument("--guidance_type", type=str, choices=["latent", "occupancy", "containment"], default="containment")
    parser.add_argument(
        "--sparse_prompt",
        type=str,
        default="a low-top sneaker",
        help="Text prompt used for sparse-structure sampling",
    )
    parser.add_argument("--guidance_strength", type=float, default=1.0)
    parser.add_argument("--guidance_interval_start", type=float, default=0.5)
    parser.add_argument("--guidance_interval_end", type=float, default=0.95)
    parser.add_argument("--guidance_schedule", type=str, choices=["constant", "linear_decay", "linear_rise"], default="constant")
    parser.add_argument("--guidance_rescale", action="store_true")
    parser.add_argument("--no_guidance_rescale", action="store_false", dest="guidance_rescale")
    parser.set_defaults(guidance_rescale=True)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--cfg_strength", type=float, default=7.5)
    parser.add_argument("--cfg_interval_start", type=float, default=0.5)
    parser.add_argument("--cfg_interval_end", type=float, default=0.95)
    parser.add_argument("--rescale_t", type=float, default=3.0)
    parser.add_argument("--bce_weight", type=float, default=1.0)
    parser.add_argument("--dice_weight", type=float, default=1.0)
    parser.add_argument("--envelope_radius", type=int, default=2)
    parser.add_argument("--contain_weight", type=float, default=2.0)
    parser.add_argument("--outside_weight", type=float, default=2.5)
    parser.add_argument("--shell_weight", type=float, default=0.05)
    parser.add_argument("--drop_components_below_y", type=float, default=None)
    parser.add_argument("--drop_components_below_area", type=float, default=None)
    parser.add_argument("--normalize_padding", type=float, default=0.0)
    parser.add_argument("--normalize_padding_x", type=float, default=0.02)
    parser.add_argument("--normalize_padding_y", type=float, default=0.0)
    parser.add_argument("--normalize_padding_z", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out_dir", type=str, default="outputs_guidance")
    parser.add_argument("--preprocess_image", action="store_true")
    parser.add_argument("--no_preprocess_image", action="store_false", dest="preprocess_image")
    parser.set_defaults(preprocess_image=True)
    args = parser.parse_args()

    mesh_out, stats_out = run_guidance(args)
    print(f"Saved guided mesh to {mesh_out}")
    print(f"Saved run metadata to {stats_out}")


if __name__ == "__main__":
    main()
