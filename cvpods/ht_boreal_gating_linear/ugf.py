"""Parameter-free uncertainty-guided fusion for FPN features.

This module intentionally contains the complete UGF implementation and its
configuration adapter so the existing FCOS/HT training code only needs to
consume the fused feature list.
"""

import math
from typing import Optional, Sequence, Union

import torch
import torch.nn.functional as F
from torch import nn


UNCERTAINTY_METHODS = ("normalized_l2", "raw_variance", "cosine")


UGF_CONFIG_DEFAULTS = {
    "ENABLED": True,
    "BETA": 2.0,
    "BETA_END": None,  # None => defaults to BETA (no decay)
    "BETA_SCHEDULE_STEPS": 0,  # 0 => disabled, beta stays constant at BETA
    "BETA_SCHEDULE_TYPE": "linear",  # one of SCHEDULE_FNS: linear/cosine/exponential/sigmoid
    "BETA_SCHEDULE_RATE": None,  # only used by exponential/sigmoid; None => that schedule's own default
    "EPS": 1.0e-6,
    "REFERENCE_LEVEL": "P4",
    # one of UNCERTAINTY_METHODS:
    #   normalized_l2 (default) -- ||F_l/||F_l|| - mean_k(F_k/||F_k||)||^2, averaged over levels
    #   raw_variance            -- same but without the per-level L2 normalization (magnitude-sensitive)
    #   cosine                  -- mean_l(1 - cos(F_l, mean_k F_k)), on raw (unnormalized) features
    "UNCERTAINTY_METHOD": "normalized_l2",
    "RELIABILITY_ENABLED": False,
    "RELIABILITY_ALPHA": 1.0,
    "UNCERTAINTY_LOSS_ENABLED": False,
    "UNCERTAINTY_LOSS_WEIGHT": 0.2,
    "UNCERTAINTY_LOSS_WEIGHT_START": None,  # None => defaults to UNCERTAINTY_LOSS_WEIGHT (no ramp)
    "UNCERTAINTY_LOSS_SCHEDULE_STEPS": 0,  # 0 => disabled, weight stays constant
    "UNCERTAINTY_LOSS_SCHEDULE_TYPE": "linear",
    "UNCERTAINTY_LOSS_SCHEDULE_RATE": None,
}


def _progress(step: int, total_steps: int) -> float:
    """Fractional progress of ``step`` within ``[0, total_steps]``, clamped to [0, 1]."""
    return min(max(step, 0), total_steps) / float(total_steps)


def linear_schedule(step: int, start: float, end: float, total_steps: int, rate: Optional[float] = None) -> float:
    """Linearly interpolate from ``start`` (at ``step<=0``) to ``end`` (at
    ``step>=total_steps``). ``total_steps<=0`` disables the schedule and
    always returns ``start``. ``rate`` is accepted for signature parity with
    the other schedules but has no effect here."""
    if total_steps <= 0:
        return float(start)
    t = _progress(step, total_steps)
    return float(start) + (float(end) - float(start)) * t


def cosine_schedule(step: int, start: float, end: float, total_steps: int, rate: Optional[float] = None) -> float:
    """Cosine-annealed interpolation from ``start`` to ``end``: slow at both
    ends, fastest around the midpoint (the usual "cosine decay" shape).
    ``rate`` is accepted for signature parity but has no effect here."""
    if total_steps <= 0:
        return float(start)
    t = _progress(step, total_steps)
    start, end = float(start), float(end)
    return end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * t))


def exponential_schedule(step: int, start: float, end: float, total_steps: int, rate: Optional[float] = None) -> float:
    """Exponential-eased interpolation from ``start`` to ``end``: most of the
    change happens early (or late, for negative ``rate``). Endpoints are
    matched exactly at ``step=0``/``step>=total_steps`` even when ``end==0``,
    unlike a raw geometric interpolation. ``rate`` controls the curvature
    (default 5.0); ``rate=0`` degenerates to :func:`linear_schedule`."""
    if total_steps <= 0:
        return float(start)
    if rate is None:
        rate = 5.0
    if rate == 0:
        return linear_schedule(step, start, end, total_steps)
    t = _progress(step, total_steps)
    start, end = float(start), float(end)
    norm = (1.0 - math.exp(-rate * t)) / (1.0 - math.exp(-rate))
    return start + (end - start) * norm


def sigmoid_schedule(step: int, start: float, end: float, total_steps: int, rate: Optional[float] = None) -> float:
    """Sigmoid-eased (S-curve) interpolation from ``start`` to ``end``: slow
    start, fast middle, slow finish. Endpoints are normalized so ``step=0``
    gives exactly ``start`` and ``step>=total_steps`` gives exactly ``end``.
    ``rate`` controls the steepness of the middle transition (default 10.0)."""
    if total_steps <= 0:
        return float(start)
    if rate is None:
        rate = 10.0
    t = _progress(step, total_steps)
    start, end = float(start), float(end)

    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-rate * (x - 0.5)))

    raw0, raw1, raw_t = _sigmoid(0.0), _sigmoid(1.0), _sigmoid(t)
    norm = (raw_t - raw0) / (raw1 - raw0)
    return start + (end - start) * norm


def gaussian_rampup_schedule(
    step: int, start: float, end: float, total_steps: int, rate: Optional[float] = None
) -> float:
    """Gaussian ramp-up from ``start`` toward ``end``, using the semi-supervised
    ramp-up curve from Laine & Aila (2016) / Tarvainen & Valpola (2017, Mean
    Teacher): ``exp(-5 * (1 - t)^2)``. Reaches exactly ``end`` at
    ``step>=total_steps``; unlike the other schedules here this does *not*
    force an exact value at ``step=0`` -- that is how the paper's curve
    itself behaves (at ``t=0`` it is ``exp(-5) ~= 0.0067`` of the way from
    ``start`` to ``end``, not exactly ``start``). ``rate`` overrides the
    paper's fixed steepness constant (5.0) if given."""
    if total_steps <= 0:
        return float(start)
    if rate is None:
        rate = 5.0
    t = _progress(step, total_steps)
    start, end = float(start), float(end)
    f = math.exp(-rate * (1.0 - t) ** 2)
    return start + (end - start) * f


SCHEDULE_FNS = {
    "linear": linear_schedule,
    "cosine": cosine_schedule,
    "exponential": exponential_schedule,
    "sigmoid": sigmoid_schedule,
    "gaussian_rampup": gaussian_rampup_schedule,
}


def scheduled_value(
    schedule_type: str, step: int, start: float, end: float, total_steps: int, rate: Optional[float] = None
) -> float:
    """Dispatch to one of :data:`SCHEDULE_FNS` by name. Swapping the schedule
    "linear"/"cosine"/"exponential"/"sigmoid" is purely a config change; the
    call sites (UGF's ``beta`` property, FCOS's ``uncertainty_loss_weight``
    property) never need to change."""
    try:
        fn = SCHEDULE_FNS[schedule_type]
    except KeyError:
        raise ValueError(
            "unknown schedule type '{}', expected one of {}".format(
                schedule_type, sorted(SCHEDULE_FNS)
            )
        )
    return fn(step, start, end, total_steps, rate=rate)


class UncertaintyGuidedInterLevelFusion(nn.Module):
    """Fuse raw FPN features using inter-level directional agreement.

    The module has no trainable parameters. A fixed binomial low-pass filter is
    registered as a buffer and is used before every finer-to-coarser subsample.
    """

    def __init__(
        self,
        beta: float = 2.0,
        beta_end: Optional[float] = None,
        beta_schedule_steps: int = 0,
        beta_schedule_type: str = "linear",
        beta_schedule_rate: Optional[float] = None,
        reference_level: Union[int, str] = 1,
        eps: float = 1.0e-6,
        level_names: Optional[Sequence[str]] = None,
        reliability_enabled: bool = False,
        reliability_alpha: float = 1.0,
        uncertainty_method: str = "normalized_l2",
    ):
        super().__init__()
        if uncertainty_method not in UNCERTAINTY_METHODS:
            raise ValueError(
                "uncertainty_method must be one of {}, got '{}'".format(
                    UNCERTAINTY_METHODS, uncertainty_method
                )
            )
        if not math.isfinite(beta) or beta < 0:
            raise ValueError("beta must be a finite non-negative value")
        beta_end = beta if beta_end is None else beta_end
        if not math.isfinite(beta_end) or beta_end < 0:
            raise ValueError("beta_end must be a finite non-negative value")
        if beta_schedule_steps < 0:
            raise ValueError("beta_schedule_steps must be non-negative")
        if beta_schedule_type not in SCHEDULE_FNS:
            raise ValueError(
                "beta_schedule_type must be one of {}, got '{}'".format(
                    sorted(SCHEDULE_FNS), beta_schedule_type
                )
            )
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be a finite positive value")
        if not math.isfinite(reliability_alpha) or reliability_alpha < 0:
            raise ValueError("reliability_alpha must be a finite non-negative value")

        # beta follows beta_start -> beta_end over beta_schedule_steps using
        # beta_schedule_type's curve (see SCHEDULE_FNS), driven by set_step()
        # (called externally, e.g. once per training iteration).
        # beta_schedule_steps=0 keeps beta constant at beta_start, matching
        # the original fixed-beta behavior exactly regardless of the type.
        self.beta_start = float(beta)
        self.beta_end = float(beta_end)
        self.beta_schedule_steps = int(beta_schedule_steps)
        self.beta_schedule_type = beta_schedule_type
        self.beta_schedule_rate = beta_schedule_rate
        self._step = 0
        self.eps = float(eps)
        self.reference_level = reference_level
        self.level_names = tuple(level_names) if level_names is not None else None
        self.reliability_enabled = bool(reliability_enabled)
        self.reliability_alpha = float(reliability_alpha)
        self.uncertainty_method = uncertainty_method

        kernel_1d = torch.tensor([1.0, 2.0, 1.0])
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        self.register_buffer("blur_kernel", (kernel_2d / kernel_2d.sum())[None, None])

    @property
    def beta(self) -> float:
        return scheduled_value(
            self.beta_schedule_type, self._step, self.beta_start, self.beta_end,
            self.beta_schedule_steps, rate=self.beta_schedule_rate,
        )

    def set_step(self, step: int) -> None:
        """Advance the step counter driving the beta linear schedule. No-op
        numerically when beta_schedule_steps=0 (beta stays at beta_start)."""
        self._step = int(step)

    def _reference_index(self, num_levels: int) -> int:
        reference = self.reference_level
        if isinstance(reference, str):
            normalized = reference.lower()
            if self.level_names is not None:
                names = [str(name).lower() for name in self.level_names]
                if normalized in names:
                    reference = names.index(normalized)
            if isinstance(reference, str) and normalized.startswith("p"):
                suffix = normalized[1:]
                if suffix.isdigit():
                    reference = int(suffix) - 3

        if not isinstance(reference, int):
            raise ValueError("reference_level must be an index or an FPN name such as 'P4'")
        if reference < 0 or reference >= num_levels:
            raise ValueError(
                "reference_level index {} is invalid for {} levels".format(
                    reference, num_levels
                )
            )
        return reference

    def _blur_downsample(self, feature: torch.Tensor) -> torch.Tensor:
        channels = feature.shape[1]
        kernel = self.blur_kernel.to(dtype=feature.dtype).expand(channels, 1, 3, 3)
        padded = F.pad(feature, (1, 1, 1, 1), mode="replicate")
        filtered = F.conv2d(padded, kernel, groups=channels)
        return filtered[:, :, ::2, ::2]

    def _align_features(self, features, reference_index):
        target_size = features[reference_index].shape[-2:]
        aligned = []
        for level_index, feature in enumerate(features):
            if level_index < reference_index:
                aligned_feature = feature
                for _ in range(reference_index - level_index):
                    aligned_feature = self._blur_downsample(aligned_feature)
                # Handles uncommon odd-size pyramids while preserving the fixed
                # low-pass + stride-2 path as the primary downsampling operation.
                if aligned_feature.shape[-2:] != target_size:
                    aligned_feature = F.interpolate(
                        aligned_feature,
                        size=target_size,
                        mode="bilinear",
                        align_corners=False,
                    )
            elif level_index == reference_index:
                aligned_feature = feature
            else:
                aligned_feature = F.interpolate(
                    feature,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
            aligned.append(aligned_feature)
        return aligned

    @staticmethod
    def _validate_features(features):
        if not isinstance(features, (list, tuple)):
            raise TypeError("features must be a list or tuple of FPN tensors")
        if len(features) < 2:
            raise ValueError("UGF requires at least two FPN levels")
        first = features[0]
        if first.ndim != 4:
            raise ValueError("each FPN feature must have shape [B, C, H, W]")
        batch_size, channels = first.shape[:2]
        for feature in features[1:]:
            if feature.ndim != 4:
                raise ValueError("each FPN feature must have shape [B, C, H, W]")
            if feature.shape[0] != batch_size or feature.shape[1] != channels:
                raise ValueError("all FPN levels must share batch and channel dimensions")

    def forward(self, features, return_aux: bool = False):
        self._validate_features(features)
        reference_index = self._reference_index(len(features))
        aligned_list = self._align_features(features, reference_index)

        # Raw aligned values are retained for fusion. Only this separate tensor
        # is normalized for directional disagreement estimation.
        aligned = torch.stack(aligned_list, dim=1)  # [B, L, C, Href, Wref]
        normalized = F.normalize(aligned, p=2, dim=2, eps=self.eps)
        mean_normalized = normalized.mean(dim=1, keepdim=True)

        if self.uncertainty_method == "normalized_l2":
            disagreement = (normalized - mean_normalized).pow(2).sum(
                dim=2, keepdim=True
            )
            uncertainty = disagreement.mean(dim=1)  # [B, 1, Href, Wref]
        elif self.uncertainty_method == "raw_variance":
            mean_aligned = aligned.mean(dim=1, keepdim=True)
            disagreement = (aligned - mean_aligned).pow(2).sum(dim=2, keepdim=True)
            uncertainty = disagreement.mean(dim=1)  # [B, 1, Href, Wref]
        elif self.uncertainty_method == "cosine":
            mean_aligned = aligned.mean(dim=1, keepdim=True).expand_as(aligned)
            cos_sim = F.cosine_similarity(aligned, mean_aligned, dim=2, eps=self.eps)  # [B, L, Href, Wref]
            uncertainty = (1.0 - cos_sim).mean(dim=1, keepdim=True)  # [B, 1, Href, Wref]
        else:
            raise ValueError(
                "unknown uncertainty_method '{}', expected one of {}".format(
                    self.uncertainty_method, UNCERTAINTY_METHODS
                )
            )

        gate = torch.exp(-self.beta * uncertainty)

        num_levels = aligned.shape[1]
        sum_all = aligned.sum(dim=1, keepdim=True)

        if self.reliability_enabled:
            # Leave-one-out consensus disagreement per level k, using the same
            # normalized features and channel-wise squared-L2-sum convention as
            # the uncertainty computation above (no 1/C or channel mean).
            sum_normalized = normalized.sum(dim=1, keepdim=True)
            loo_mean_normalized = (sum_normalized - normalized) / float(num_levels - 1)
            reliability_disagreement = (normalized - loo_mean_normalized).pow(2).sum(
                dim=2, keepdim=True
            )  # [B, L, 1, Href, Wref]

            # w_{k->l} = softmax_k(-alpha * reliability_disagreement_k), k != l.
            # For each target l, source k's weight is e_k normalized over k != l:
            #   w_{k->l} = e_k / (sum_j e_j - e_l)
            # so F_cross_l = sum_{k!=l} w_{k->l} * F_aligned_k reduces to the
            # leave-one-out ratio below (same algebraic trick as the uniform
            # cross_level: exclude index l from both the numerator sum and the
            # normalizing sum).
            reliability_score = torch.exp(-self.reliability_alpha * reliability_disagreement)
            score_sum_all = reliability_score.sum(dim=1, keepdim=True)

            weighted_sum_all = (reliability_score * aligned).sum(dim=1, keepdim=True)
            cross_level = (weighted_sum_all - reliability_score * aligned) / (
                score_sum_all - reliability_score
            )
        else:
            cross_level = (sum_all - aligned) / float(num_levels - 1)

        gated_residual = cross_level * gate.unsqueeze(1)

        outputs = []
        for level_index, feature in enumerate(features):
            residual = gated_residual[:, level_index]
            if residual.shape[-2:] != feature.shape[-2:]:
                residual = F.interpolate(
                    residual,
                    size=feature.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            outputs.append(feature + residual)

        fused_features = tuple(outputs) if isinstance(features, tuple) else outputs
        if return_aux:
            return fused_features, {
                "uncertainty": uncertainty,
                "gate": gate,
            }
        return fused_features


def build_ugf_from_config(cfg, level_names=None):
    """Build UGF from ``MODEL.UGF`` or return ``None`` when disabled."""
    ugf_cfg = getattr(cfg.MODEL, "UGF", None)
    if ugf_cfg is None:
        return None
    if not bool(getattr(ugf_cfg, "ENABLED", UGF_CONFIG_DEFAULTS["ENABLED"])):
        return None
    return UncertaintyGuidedInterLevelFusion(
        beta=getattr(ugf_cfg, "BETA", UGF_CONFIG_DEFAULTS["BETA"]),
        beta_end=getattr(ugf_cfg, "BETA_END", UGF_CONFIG_DEFAULTS["BETA_END"]),
        beta_schedule_steps=getattr(
            ugf_cfg, "BETA_SCHEDULE_STEPS", UGF_CONFIG_DEFAULTS["BETA_SCHEDULE_STEPS"]
        ),
        beta_schedule_type=getattr(
            ugf_cfg, "BETA_SCHEDULE_TYPE", UGF_CONFIG_DEFAULTS["BETA_SCHEDULE_TYPE"]
        ),
        beta_schedule_rate=getattr(
            ugf_cfg, "BETA_SCHEDULE_RATE", UGF_CONFIG_DEFAULTS["BETA_SCHEDULE_RATE"]
        ),
        eps=getattr(ugf_cfg, "EPS", UGF_CONFIG_DEFAULTS["EPS"]),
        reference_level=getattr(
            ugf_cfg,
            "REFERENCE_LEVEL",
            UGF_CONFIG_DEFAULTS["REFERENCE_LEVEL"],
        ),
        level_names=level_names,
        reliability_enabled=bool(getattr(
            ugf_cfg,
            "RELIABILITY_ENABLED",
            UGF_CONFIG_DEFAULTS["RELIABILITY_ENABLED"],
        )),
        reliability_alpha=getattr(
            ugf_cfg,
            "RELIABILITY_ALPHA",
            UGF_CONFIG_DEFAULTS["RELIABILITY_ALPHA"],
        ),
        uncertainty_method=getattr(
            ugf_cfg,
            "UNCERTAINTY_METHOD",
            UGF_CONFIG_DEFAULTS["UNCERTAINTY_METHOD"],
        ),
    )


def uncertainty_frobenius_loss(uncertainty: torch.Tensor) -> torch.Tensor:
    """Global normalized Frobenius-norm loss over a UGF uncertainty map.

    ``uncertainty`` is the ``aux["uncertainty"]`` tensor returned by
    :meth:`UncertaintyGuidedInterLevelFusion.forward` (``return_aux=True``),
    shape ``[B, 1, H, W]``. Per image: ``||U_b||_F / sqrt(H * W)``, then
    averaged over the batch. Callers must not detach ``uncertainty`` before
    passing it in, so gradients keep flowing U -> normalized features -> FPN.
    """
    height, width = uncertainty.shape[-2:]
    fro_per_image = uncertainty.flatten(1).norm(p=2, dim=1) / math.sqrt(height * width)
    return fro_per_image.mean()
