import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    'BCEDiceLoss',
    'DiceBCELoss',
    'SobelBoundaryLoss',
    'BoundaryAwareSegLoss',
    'HSPMLoss',
    'EdgeSupervisionLoss',
    'FBDMLoss',
    'HSPMFBDMLoss',
    'mask_to_edge',
]


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice


class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        prob = torch.sigmoid(input)
        num = target.size(0)
        prob = prob.view(num, -1)
        target = target.view(num, -1)
        intersection = prob * target
        dice = (2. * intersection.sum(1) + self.smooth) / (
            prob.sum(1) + target.sum(1) + self.smooth
        )
        dice = 1 - dice.sum() / num
        return bce + dice


class SobelBoundaryLoss(nn.Module):
    def __init__(self, smooth=1e-5, eps=1e-6):
        super().__init__()
        self.smooth = smooth
        self.eps = eps
        sobel_x = torch.tensor(
            [[[-1., 0., 1.],
              [-2., 0., 2.],
              [-1., 0., 1.]]]
        ).unsqueeze(0)
        sobel_y = torch.tensor(
            [[[-1., -2., -1.],
              [0., 0., 0.],
              [1., 2., 1.]]]
        ).unsqueeze(0)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def _sobel_edges(self, mask):
        channels = mask.shape[1]
        sobel_x = self.sobel_x.to(dtype=mask.dtype, device=mask.device).repeat(channels, 1, 1, 1)
        sobel_y = self.sobel_y.to(dtype=mask.dtype, device=mask.device).repeat(channels, 1, 1, 1)
        grad_x = F.conv2d(mask, sobel_x, padding=1, groups=channels)
        grad_y = F.conv2d(mask, sobel_y, padding=1, groups=channels)
        edge = torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + self.eps)
        return torch.clamp(edge / 4.0, min=0.0, max=1.0)

    def forward(self, input, target):
        prob = torch.sigmoid(input)
        target = target.float().clamp(0.0, 1.0)
        pred_edge = self._sobel_edges(prob).clamp(self.eps, 1.0 - self.eps)
        target_edge = self._sobel_edges(target).detach()

        bce = F.binary_cross_entropy(pred_edge, target_edge)
        num = target.size(0)
        pred_edge = pred_edge.view(num, -1)
        target_edge = target_edge.view(num, -1)
        intersection = pred_edge * target_edge
        dice = (2. * intersection.sum(1) + self.smooth) / (
            pred_edge.sum(1) + target_edge.sum(1) + self.smooth
        )
        dice = 1 - dice.sum() / num
        return bce + dice


class BoundaryAwareSegLoss(nn.Module):
    def __init__(self, lambda_b=0.3):
        super().__init__()
        self.lambda_b = float(lambda_b)
        self.seg_loss = DiceBCELoss()
        self.boundary_loss = SobelBoundaryLoss()

    def forward(self, input, target):
        loss_seg = self.seg_loss(input, target)
        loss_bnd = self.boundary_loss(input, target)
        return loss_seg + self.lambda_b * loss_bnd


class HSPMLoss(nn.Module):
    def __init__(self, coarse_weight=0.3):
        super().__init__()
        if coarse_weight < 0:
            raise ValueError("coarse_weight must be non-negative.")
        self.coarse_weight = float(coarse_weight)
        self.seg_loss = BCEDiceLoss()

    def forward(self, outputs, target, coarse_weight=None):
        if not isinstance(outputs, dict):
            raise TypeError("HSPMLoss expects model outputs to be a dictionary.")
        if "seg" not in outputs or "coarse" not in outputs:
            raise KeyError("HSPMLoss requires 'seg' and 'coarse' output keys.")

        current_coarse_weight = self.coarse_weight if coarse_weight is None else float(coarse_weight)
        if current_coarse_weight < 0:
            raise ValueError("coarse_weight must be non-negative.")
        final_loss = self.seg_loss(outputs["seg"], target)
        if current_coarse_weight == 0:
            return final_loss
        coarse_target = F.interpolate(target, size=outputs["coarse"].shape[-2:], mode="nearest")
        coarse_loss = self.seg_loss(outputs["coarse"], coarse_target)
        return final_loss + current_coarse_weight * coarse_loss


def mask_to_edge(mask, kernel_size=3):
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer.")
    mask = mask.float().clamp(0.0, 1.0)
    padding = kernel_size // 2
    dilation = F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=padding)
    erosion = 1.0 - F.max_pool2d(1.0 - mask, kernel_size=kernel_size, stride=1, padding=padding)
    return (dilation - erosion).clamp(0.0, 1.0)


class EdgeSupervisionLoss(nn.Module):
    LOSS_TYPES = {"legacy", "balanced_bce_dice", "focal_dice"}

    def __init__(
        self,
        loss_type="legacy",
        pos_weight=20.0,
        focal_alpha=0.95,
        focal_gamma=2.0,
        smooth=1e-5,
    ):
        super().__init__()
        if loss_type not in self.LOSS_TYPES:
            raise ValueError(
                "loss_type must be one of: legacy, balanced_bce_dice, focal_dice."
            )
        if pos_weight <= 0:
            raise ValueError("pos_weight must be positive.")
        if not 0.0 < focal_alpha < 1.0:
            raise ValueError("focal_alpha must be in (0, 1).")
        if focal_gamma < 0:
            raise ValueError("focal_gamma must be non-negative.")

        self.loss_type = loss_type
        self.focal_alpha = float(focal_alpha)
        self.focal_gamma = float(focal_gamma)
        self.smooth = float(smooth)
        self.legacy_loss = BCEDiceLoss()
        self.register_buffer("pos_weight", torch.tensor([float(pos_weight)]))

    def _soft_dice_loss(self, logits, target):
        probability = torch.sigmoid(logits)
        batch_size = target.size(0)
        probability = probability.reshape(batch_size, -1)
        target = target.reshape(batch_size, -1)
        intersection = (probability * target).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (
            probability.sum(dim=1) + target.sum(dim=1) + self.smooth
        )
        return 1.0 - dice.mean()

    def forward(self, logits, target):
        target = target.to(device=logits.device, dtype=logits.dtype)
        if self.loss_type == "legacy":
            return self.legacy_loss(logits, target)

        dice = self._soft_dice_loss(logits, target)
        if self.loss_type == "balanced_bce_dice":
            classification = F.binary_cross_entropy_with_logits(
                logits,
                target,
                pos_weight=self.pos_weight.to(dtype=logits.dtype),
            )
        else:
            elementwise_bce = F.binary_cross_entropy_with_logits(
                logits,
                target,
                reduction="none",
            )
            probability = torch.sigmoid(logits)
            p_t = target * probability + (1.0 - target) * (1.0 - probability)
            alpha_t = (
                target * self.focal_alpha
                + (1.0 - target) * (1.0 - self.focal_alpha)
            )
            classification = (
                alpha_t * (1.0 - p_t).pow(self.focal_gamma) * elementwise_bce
            ).mean()
        return 0.5 * classification + 0.5 * dice


def _edge_target_at_size(target, size, kernel_size):
    if target.shape[-2:] != size:
        target = F.interpolate(target, size=size, mode="nearest")
    return mask_to_edge(target, kernel_size=kernel_size)


def _edge_diagnostics(logits, target):
    with torch.no_grad():
        probability = torch.sigmoid(logits.detach())
        return {
            "edge_target_ratio": target.detach().float().mean(),
            "edge_prob_mean": probability.float().mean(),
            "edge_pred_positive_ratio": (probability > 0.5).float().mean(),
        }


def _weighted_edge_diagnostics(weighted_diagnostics):
    return {
        name: sum(weight * diagnostics[name] for weight, diagnostics in weighted_diagnostics)
        for name in (
            "edge_target_ratio",
            "edge_prob_mean",
            "edge_pred_positive_ratio",
        )
    }


class HSPMFBDMLoss(nn.Module):
    def __init__(
        self,
        coarse_weight=0.3,
        edge_weight=0.05,
        edge_kernel_size=3,
        boundary_band_weight=0.0,
        boundary_band_kernel_size=7,
        edge_loss_type="legacy",
        edge_pos_weight=20.0,
        edge_focal_alpha=0.95,
        edge_focal_gamma=2.0,
        protected_refinement=False,
        refine_weight=0.0,
        preserve_weight=0.0,
    ):
        super().__init__()
        if coarse_weight < 0:
            raise ValueError("coarse_weight must be non-negative.")
        if edge_weight < 0:
            raise ValueError("edge_weight must be non-negative.")
        if boundary_band_weight < 0:
            raise ValueError("boundary_band_weight must be non-negative.")
        if refine_weight < 0 or preserve_weight < 0:
            raise ValueError("refine_weight and preserve_weight must be non-negative.")
        if boundary_band_kernel_size <= 0 or boundary_band_kernel_size % 2 == 0:
            raise ValueError("boundary_band_kernel_size must be a positive odd integer.")
        self.coarse_weight = float(coarse_weight)
        self.edge_weight = float(edge_weight)
        self.edge_kernel_size = int(edge_kernel_size)
        self.boundary_band_weight = float(boundary_band_weight)
        self.boundary_band_kernel_size = int(boundary_band_kernel_size)
        self.protected_refinement = bool(protected_refinement)
        self.refine_weight = float(refine_weight)
        self.preserve_weight = float(preserve_weight)
        self.seg_loss = BCEDiceLoss()
        self.edge_loss = EdgeSupervisionLoss(
            loss_type=edge_loss_type,
            pos_weight=edge_pos_weight,
            focal_alpha=edge_focal_alpha,
            focal_gamma=edge_focal_gamma,
        )
        self.last_edge_diagnostics = None

    def get_edge_diagnostics(self):
        return self.last_edge_diagnostics

    def forward(
        self,
        outputs,
        target,
        coarse_weight=None,
        edge_weight=None,
        boundary_band_weight=None,
        refine_weight=None,
        preserve_weight=None,
        return_components=False,
    ):
        if not isinstance(outputs, dict):
            raise TypeError("HSPMFBDMLoss expects model outputs to be a dictionary.")
        if "seg" not in outputs or "coarse" not in outputs:
            raise KeyError("HSPMFBDMLoss requires 'seg' and 'coarse' output keys.")

        current_coarse_weight = self.coarse_weight if coarse_weight is None else float(coarse_weight)
        current_edge_weight = self.edge_weight if edge_weight is None else float(edge_weight)
        current_boundary_band_weight = (
            self.boundary_band_weight
            if boundary_band_weight is None
            else float(boundary_band_weight)
        )
        current_refine_weight = (
            self.refine_weight if refine_weight is None else float(refine_weight)
        )
        current_preserve_weight = (
            self.preserve_weight if preserve_weight is None else float(preserve_weight)
        )
        if min(
            current_coarse_weight,
            current_edge_weight,
            current_boundary_band_weight,
            current_refine_weight,
            current_preserve_weight,
        ) < 0:
            raise ValueError("dynamic HSPM-FBDM loss weights must be non-negative.")

        if self.protected_refinement:
            required_keys = {"base_seg", "logit_correction", "boundary_gate"}
            missing_keys = required_keys.difference(outputs)
            if missing_keys:
                raise KeyError(
                    "Protected HSPM-FBDM refinement requires output keys: "
                    f"{sorted(required_keys)}."
                )
            seg = self.seg_loss(outputs["base_seg"], target)
        else:
            seg = self.seg_loss(outputs["seg"], target)
        coarse_weighted = seg.new_zeros(())
        edge_raw = seg.new_zeros(())
        edge_weighted = seg.new_zeros(())
        boundary_band_weighted = seg.new_zeros(())
        refine_weighted = seg.new_zeros(())
        preserve_weighted = seg.new_zeros(())
        self.last_edge_diagnostics = None
        if current_coarse_weight > 0:
            coarse_target = F.interpolate(target, size=outputs["coarse"].shape[-2:], mode="nearest")
            coarse_weighted = current_coarse_weight * self.seg_loss(outputs["coarse"], coarse_target)
        if current_edge_weight > 0:
            if "edge" not in outputs:
                raise KeyError("HSPMFBDMLoss requires 'edge' output when edge_weight > 0.")
            edge_target = _edge_target_at_size(
                target,
                outputs["edge"].shape[-2:],
                self.edge_kernel_size,
            )
            edge_raw = self.edge_loss(outputs["edge"], edge_target)
            edge_weighted = current_edge_weight * edge_raw
            self.last_edge_diagnostics = _edge_diagnostics(outputs["edge"], edge_target)
        if self.protected_refinement and current_refine_weight > 0:
            refined_logits = outputs["base_seg"].detach() + outputs["logit_correction"]
            refine_weighted = current_refine_weight * self.seg_loss(refined_logits, target)
        if self.protected_refinement and current_preserve_weight > 0:
            boundary_gate = outputs["boundary_gate"].detach().to(
                device=outputs["logit_correction"].device,
                dtype=outputs["logit_correction"].dtype,
            )
            preserve = (
                (1.0 - boundary_gate) * outputs["logit_correction"].square()
            ).mean()
            preserve_weighted = current_preserve_weight * preserve
        if current_boundary_band_weight > 0:
            required_keys = {"base_seg", "logit_correction"}
            missing_keys = required_keys.difference(outputs)
            if missing_keys:
                raise KeyError(
                    "HSPMFBDMLoss boundary-band loss requires output keys: "
                    f"{sorted(required_keys)}."
                )
            band_logits = outputs["base_seg"].detach() + outputs["logit_correction"]
            band_target = target
            if band_target.shape[-2:] != band_logits.shape[-2:]:
                band_target = F.interpolate(
                    band_target,
                    size=band_logits.shape[-2:],
                    mode="nearest",
                )
            band = mask_to_edge(target, kernel_size=self.boundary_band_kernel_size)
            if band.shape[-2:] != band_logits.shape[-2:]:
                band = F.interpolate(band, size=band_logits.shape[-2:], mode="nearest")
            band = band.to(device=band_logits.device, dtype=band_logits.dtype)
            band_target = band_target.to(device=band_logits.device, dtype=band_logits.dtype)
            band_elementwise = F.binary_cross_entropy_with_logits(
                band_logits,
                band_target,
                reduction="none",
            )
            band_loss = (band * band_elementwise).sum() / band.sum().clamp_min(1.0)
            boundary_band_weighted = current_boundary_band_weight * band_loss

        total = (
            seg
            + coarse_weighted
            + edge_weighted
            + boundary_band_weighted
            + refine_weighted
            + preserve_weighted
        )
        if not return_components:
            return total
        return total, {
            "seg": seg,
            "coarse_weighted": coarse_weighted,
            "edge_raw": edge_raw,
            "edge_weighted": edge_weighted,
            "boundary_band_weighted": boundary_band_weighted,
            "refine_weighted": refine_weighted,
            "preserve_weighted": preserve_weighted,
            "total": total,
        }


class FBDMLoss(nn.Module):
    def __init__(
        self,
        edge_weight=0.05,
        edge_kernel_size=3,
        x2_edge_ratio=0.3,
        edge_loss_type="legacy",
        edge_pos_weight=20.0,
        edge_focal_alpha=0.95,
        edge_focal_gamma=2.0,
    ):
        super().__init__()
        if edge_weight < 0:
            raise ValueError("edge_weight must be non-negative.")
        if not 0.0 < x2_edge_ratio <= 1.0:
            raise ValueError("x2_edge_ratio must be in (0, 1].")
        self.edge_weight = float(edge_weight)
        self.edge_kernel_size = int(edge_kernel_size)
        self.x2_edge_ratio = float(x2_edge_ratio)
        self.seg_loss = BCEDiceLoss()
        self.edge_loss = EdgeSupervisionLoss(
            loss_type=edge_loss_type,
            pos_weight=edge_pos_weight,
            focal_alpha=edge_focal_alpha,
            focal_gamma=edge_focal_gamma,
        )
        self.last_edge_diagnostics = None

    def _edge_target(self, target, size):
        return _edge_target_at_size(target, size, self.edge_kernel_size)

    def get_edge_diagnostics(self):
        return self.last_edge_diagnostics

    def forward(self, outputs, target, edge_weight=None, return_components=False):
        if not isinstance(outputs, dict):
            raise TypeError("FBDMLoss expects model outputs to be a dictionary.")
        if "seg" not in outputs:
            raise KeyError("FBDMLoss requires a 'seg' output key.")

        current_edge_weight = self.edge_weight if edge_weight is None else float(edge_weight)
        if current_edge_weight < 0:
            raise ValueError("edge_weight must be non-negative.")

        seg = self.seg_loss(outputs["seg"], target)
        edge_raw = seg.new_zeros(())
        edge_weighted = seg.new_zeros(())
        self.last_edge_diagnostics = None
        if current_edge_weight > 0:
            if "edge" not in outputs:
                raise KeyError("FBDMLoss requires an 'edge' output key when edge_weight > 0.")
            edge_target = self._edge_target(target, outputs["edge"].shape[-2:])
            primary_edge_loss = self.edge_loss(outputs["edge"], edge_target)
            primary_diagnostics = _edge_diagnostics(outputs["edge"], edge_target)
            if "edge_x2" in outputs:
                x2_target = self._edge_target(target, outputs["edge_x2"].shape[-2:])
                x2_edge_loss = self.edge_loss(outputs["edge_x2"], x2_target)
                x2_diagnostics = _edge_diagnostics(outputs["edge_x2"], x2_target)
                x1_ratio = 1.0 / (1.0 + self.x2_edge_ratio)
                x2_ratio = self.x2_edge_ratio / (1.0 + self.x2_edge_ratio)
                edge_raw = x1_ratio * primary_edge_loss + x2_ratio * x2_edge_loss
                self.last_edge_diagnostics = _weighted_edge_diagnostics(
                    (
                        (x1_ratio, primary_diagnostics),
                        (x2_ratio, x2_diagnostics),
                    )
                )
            else:
                edge_raw = primary_edge_loss
                self.last_edge_diagnostics = primary_diagnostics
            edge_weighted = current_edge_weight * edge_raw

        total = seg + edge_weighted
        if not return_components:
            return total
        return total, {
            "seg": seg,
            "edge_raw": edge_raw,
            "edge_weighted": edge_weighted,
            "total": total,
        }


def compute_kl_loss(p, q):
    p_loss = F.kl_div(F.log_softmax(p, dim=-1),
                      F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1),
                      F.softmax(p, dim=-1), reduction='none')

    p_loss = p_loss.mean()
    q_loss = q_loss.mean()

    loss = (p_loss + q_loss) / 2
    return loss
