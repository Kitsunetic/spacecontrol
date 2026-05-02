from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from easydict import EasyDict as edict
from tqdm import tqdm

from .flow_euler import FlowEulerSampler
from .guidance_interval_mixin import GuidanceIntervalSamplerMixin


class FlowEulerGeometryGuidanceSampler(GuidanceIntervalSamplerMixin, FlowEulerSampler):
    """
    Euler sampler with classifier-free guidance and an additional geometry
    guidance term driven by a user-provided loss over predicted x0.
    """

    def _guidance_active(self, t: float, guidance_interval: Tuple[float, float]) -> bool:
        return guidance_interval[0] <= t <= guidance_interval[1]

    def _guidance_scale(
        self,
        t: float,
        guidance_strength: float,
        guidance_interval: Tuple[float, float],
        guidance_schedule: str,
    ) -> float:
        if guidance_schedule == "constant":
            return guidance_strength

        denom = max(guidance_interval[1] - guidance_interval[0], 1e-6)
        normalized_t = (t - guidance_interval[0]) / denom
        normalized_t = float(np.clip(normalized_t, 0.0, 1.0))

        if guidance_schedule == "linear_decay":
            return guidance_strength * normalized_t
        if guidance_schedule == "linear_rise":
            return guidance_strength * (1.0 - normalized_t)

        raise ValueError(f"Unsupported guidance schedule: {guidance_schedule}")

    def _clip_grad(self, grad: torch.Tensor, grad_clip: Optional[float]) -> torch.Tensor:
        if grad_clip is None:
            return grad
        flat = grad.reshape(grad.shape[0], -1)
        norms = flat.norm(dim=1, keepdim=True).clamp_min(1e-8)
        factors = torch.clamp(grad_clip / norms, max=1.0)
        return grad * factors.view(-1, *([1] * (grad.ndim - 1)))

    def _sample_once_with_geometry_guidance(
        self,
        model,
        x_t,
        t: float,
        t_prev: float,
        cond,
        *,
        neg_cond,
        cfg_strength: float,
        cfg_interval: Tuple[float, float],
        geometry_guidance,
        guidance_strength: float,
        guidance_interval: Tuple[float, float],
        guidance_schedule: str,
        guidance_rescale: bool,
        grad_clip: Optional[float],
        guidance_cfg_mode: str,
        **kwargs,
    ):
        x_t = x_t.detach().requires_grad_(True)
        if guidance_cfg_mode == "with_cfg":
            pred_x_0, _, pred_v_for_guidance = self._get_model_prediction(
                model,
                x_t,
                t,
                cond,
                neg_cond=neg_cond,
                cfg_strength=cfg_strength,
                cfg_interval=cfg_interval,
                **kwargs,
            )
        elif guidance_cfg_mode == "cond_only":
            pred_v_for_guidance = FlowEulerSampler._inference_model(self, model, x_t, t, cond, **kwargs)
            pred_x_0, _ = self._v_to_xstart_eps(x_t=x_t, t=t, v=pred_v_for_guidance)
        else:
            raise ValueError(f"Unsupported guidance CFG mode: {guidance_cfg_mode}")

        loss, metrics = geometry_guidance.compute_loss(pred_x_0, t)
        grad = torch.autograd.grad(loss, x_t, only_inputs=True)[0]
        grad = self._clip_grad(grad, grad_clip)

        with torch.no_grad():
            pred_x_0_sample, _, pred_v = self._get_model_prediction(
                model,
                x_t.detach(),
                t,
                cond,
                neg_cond=neg_cond,
                cfg_strength=cfg_strength,
                cfg_interval=cfg_interval,
                **kwargs,
            )

            if guidance_rescale:
                grad_rms = grad.square().mean().sqrt().clamp_min(1e-8)
                pred_rms = pred_v.detach().square().mean().sqrt().clamp_min(1e-8)
                grad = grad * (pred_rms / grad_rms)

        scale = self._guidance_scale(t, guidance_strength, guidance_interval, guidance_schedule)
        pred_v = pred_v + scale * grad
        pred_x_prev = x_t - (t - t_prev) * pred_v

        metrics = dict(metrics)
        metrics["guidance_grad_rms"] = float(grad.detach().square().mean().sqrt().item())
        metrics["guidance_scale"] = float(scale)

        return edict({"pred_x_prev": pred_x_prev.detach(), "pred_x_0": pred_x_0_sample.detach(), "guidance": metrics})

    def sample(
        self,
        model,
        noise,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        cfg_strength: float = 3.0,
        cfg_interval: Tuple[float, float] = (0.0, 1.0),
        geometry_guidance=None,
        guidance_strength: float = 1.0,
        guidance_interval: Tuple[float, float] = (0.5, 0.95),
        guidance_schedule: str = "constant",
        guidance_rescale: bool = True,
        grad_clip: Optional[float] = 5.0,
        guidance_cfg_mode: str = "cond_only",
        verbose: bool = True,
        **kwargs,
    ):
        sample = noise
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))

        ret = edict({"samples": None, "pred_x_t": [], "pred_x_0": [], "guidance": []})
        base_args: Dict[str, Any] = {
            "neg_cond": neg_cond,
            "cfg_strength": cfg_strength,
            "cfg_interval": cfg_interval,
        }
        base_args.update(kwargs)

        for t, t_prev in tqdm(t_pairs, desc="Sampling", disable=not verbose):
            guidance_active = geometry_guidance is not None and self._guidance_active(t, guidance_interval)
            if guidance_active:
                out = self._sample_once_with_geometry_guidance(
                    model,
                    sample,
                    t,
                    t_prev,
                    cond,
                    geometry_guidance=geometry_guidance,
                    guidance_strength=guidance_strength,
                    guidance_interval=guidance_interval,
                    guidance_schedule=guidance_schedule,
                    guidance_rescale=guidance_rescale,
                    grad_clip=grad_clip,
                    guidance_cfg_mode=guidance_cfg_mode,
                    **base_args,
                )
                ret.guidance.append(out.guidance)
            else:
                with torch.no_grad():
                    out = super().sample_once(model, sample, t, t_prev, cond, **base_args)
            sample = out.pred_x_prev.detach()
            ret.pred_x_t.append(out.pred_x_prev)
            ret.pred_x_0.append(out.pred_x_0)

        ret.samples = sample
        return ret
