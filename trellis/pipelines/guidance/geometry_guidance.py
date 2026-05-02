from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from gui import utils


class BaseGeometryGuidance(ABC):
    """
    Base class for geometry guidance strategies that operate on sparse-structure
    predictions during sampling.
    """

    def __init__(self, pipeline, spatial_control_path: str):
        self.pipeline = pipeline
        self.spatial_control_path = spatial_control_path

    def _match_batch(self, tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
        if tensor.shape[0] == batch_size:
            return tensor
        if tensor.shape[0] != 1:
            raise ValueError(f"Cannot broadcast tensor with batch size {tensor.shape[0]} to {batch_size}")
        return tensor.repeat(batch_size, *([1] * (tensor.ndim - 1)))

    @abstractmethod
    def compute_loss(self, pred_x0: torch.Tensor, t: float) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the guidance loss for the predicted x0 at timestep t.
        """


class LatentGeometryGuidance(BaseGeometryGuidance):
    """
    Guidance that directly aligns the predicted sparse-structure latent with the
    encoded control latent.
    """

    def __init__(self, pipeline, spatial_control_path: str):
        super().__init__(pipeline, spatial_control_path)
        with torch.no_grad():
            self.target_latent = pipeline.encode_spatial_control(spatial_control_path).detach()

    def compute_loss(self, pred_x0: torch.Tensor, t: float) -> Tuple[torch.Tensor, Dict[str, float]]:
        target = self._match_batch(self.target_latent, pred_x0.shape[0]).to(device=pred_x0.device, dtype=pred_x0.dtype)
        loss = F.mse_loss(pred_x0, target)
        return loss, {
            "guidance_loss": float(loss.detach().item()),
            "latent_mse": float(loss.detach().item()),
            "t": float(t),
        }


class OccupancyGeometryGuidance(BaseGeometryGuidance):
    """
    Guidance that decodes the predicted sparse-structure latent and compares it
    against the target occupancy grid.
    """

    def __init__(
        self,
        pipeline,
        spatial_control_path: str,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
    ):
        super().__init__(pipeline, spatial_control_path)
        self.decoder = pipeline.models["sparse_structure_decoder"]
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        with torch.no_grad():
            self.target_occupancy = utils.voxelize_sq_francis(spatial_control_path).to(device=pipeline.device).detach()

    def compute_loss(self, pred_x0: torch.Tensor, t: float) -> Tuple[torch.Tensor, Dict[str, float]]:
        logits = self.decoder(pred_x0)
        target = self._match_batch(self.target_occupancy, logits.shape[0]).to(device=logits.device, dtype=logits.dtype)
        bce = F.binary_cross_entropy_with_logits(logits, target)

        probs = torch.sigmoid(logits)
        dims = tuple(range(1, probs.ndim))
        intersection = (probs * target).sum(dim=dims)
        denom = probs.sum(dim=dims) + target.sum(dim=dims)
        dice = 1.0 - ((2.0 * intersection + 1e-6) / (denom + 1e-6))
        dice_loss = dice.mean()
        loss = self.bce_weight * bce + self.dice_weight * dice_loss

        pred_binary = (probs > 0.5).float()
        union = ((pred_binary + target) > 0).float().sum(dim=dims)
        iou = ((pred_binary * target).sum(dim=dims) / union.clamp_min(1.0)).mean()

        return loss, {
            "guidance_loss": float(loss.detach().item()),
            "occupancy_bce": float(bce.detach().item()),
            "occupancy_dice": float(dice_loss.detach().item()),
            "occupancy_iou": float(iou.detach().item()),
            "t": float(t),
        }


class ContainmentGeometryGuidance(BaseGeometryGuidance):
    """
    Guidance that treats the control mesh as an internal support shape.
    The generated coarse occupancy should contain the last while staying inside
    a dilated envelope around it.
    """

    def __init__(
        self,
        pipeline,
        spatial_control_path: str,
        envelope_radius: int = 4,
        contain_weight: float = 2.0,
        outside_weight: float = 1.0,
        shell_weight: float = 0.35,
    ):
        super().__init__(pipeline, spatial_control_path)
        self.decoder = pipeline.models["sparse_structure_decoder"]
        self.envelope_radius = envelope_radius
        self.contain_weight = contain_weight
        self.outside_weight = outside_weight
        self.shell_weight = shell_weight
        with torch.no_grad():
            last = utils.voxelize_sq_francis(spatial_control_path).to(device=pipeline.device).detach()
            envelope = self._dilate(last, envelope_radius)
            shell = torch.clamp(envelope - last, min=0.0, max=1.0)
            self.last_occupancy = last
            self.envelope_occupancy = envelope
            self.shell_occupancy = shell

    def _dilate(self, occupancy: torch.Tensor, radius: int) -> torch.Tensor:
        if radius <= 0:
            return occupancy
        kernel = 2 * radius + 1
        return F.max_pool3d(occupancy, kernel_size=kernel, stride=1, padding=radius)

    def compute_loss(self, pred_x0: torch.Tensor, t: float) -> Tuple[torch.Tensor, Dict[str, float]]:
        logits = self.decoder(pred_x0)
        probs = torch.sigmoid(logits)

        last = self._match_batch(self.last_occupancy, logits.shape[0]).to(device=logits.device, dtype=logits.dtype)
        envelope = self._match_batch(self.envelope_occupancy, logits.shape[0]).to(device=logits.device, dtype=logits.dtype)
        shell = self._match_batch(self.shell_occupancy, logits.shape[0]).to(device=logits.device, dtype=logits.dtype)

        inside_mask = last > 0.5
        outside_mask = envelope < 0.5
        shell_mask = shell > 0.5

        contain_loss = F.softplus(-logits[inside_mask]).mean()
        outside_loss = F.softplus(logits[outside_mask]).mean()
        if shell_mask.any():
            shell_loss = F.softplus(-logits[shell_mask]).mean()
        else:
            shell_loss = torch.zeros((), device=logits.device, dtype=logits.dtype)

        loss = (
            self.contain_weight * contain_loss
            + self.outside_weight * outside_loss
            + self.shell_weight * shell_loss
        )

        pred_binary = (probs > 0.5).float()
        dims = tuple(range(1, probs.ndim))
        containment = ((pred_binary * last).sum(dim=dims) / last.sum(dim=dims).clamp_min(1.0)).mean()
        outside_fraction = ((pred_binary * (1.0 - envelope)).sum(dim=dims) / pred_binary.sum(dim=dims).clamp_min(1.0)).mean()
        shell_fill = ((pred_binary * shell).sum(dim=dims) / shell.sum(dim=dims).clamp_min(1.0)).mean()
        envelope_iou = (
            (pred_binary * envelope).sum(dim=dims)
            / ((pred_binary + envelope) > 0).float().sum(dim=dims).clamp_min(1.0)
        ).mean()

        return loss, {
            "guidance_loss": float(loss.detach().item()),
            "contain_loss": float(contain_loss.detach().item()),
            "outside_loss": float(outside_loss.detach().item()),
            "shell_loss": float(shell_loss.detach().item()),
            "containment_recall": float(containment.detach().item()),
            "outside_fraction": float(outside_fraction.detach().item()),
            "shell_fill": float(shell_fill.detach().item()),
            "envelope_iou": float(envelope_iou.detach().item()),
            "t": float(t),
        }


def build_geometry_guidance(guidance_type: str, pipeline, spatial_control_path: str, **kwargs) -> BaseGeometryGuidance:
    guidance_type = guidance_type.lower()
    if guidance_type == "latent":
        return LatentGeometryGuidance(pipeline, spatial_control_path)
    if guidance_type == "occupancy":
        return OccupancyGeometryGuidance(
            pipeline,
            spatial_control_path,
            bce_weight=kwargs.get("bce_weight", 1.0),
            dice_weight=kwargs.get("dice_weight", 1.0),
        )
    if guidance_type == "containment":
        return ContainmentGeometryGuidance(
            pipeline,
            spatial_control_path,
            envelope_radius=kwargs.get("envelope_radius", 4),
            contain_weight=kwargs.get("contain_weight", 2.0),
            outside_weight=kwargs.get("outside_weight", 1.0),
            shell_weight=kwargs.get("shell_weight", 0.35),
        )
    raise ValueError(f"Unsupported guidance type: {guidance_type}")
