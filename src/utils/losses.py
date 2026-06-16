import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    'BCEDiceLoss',
    'DiceBCELoss',
    'SobelBoundaryLoss',
    'BoundaryAwareSegLoss',
    'HSPMLoss',
    'FBDMLoss',
    'HSPMFBDMLoss',
    'mask_to_edge',
    'UBRDLoss',
    'APBRLoss',
    'SDFRLoss',
    'SDFRV2Loss',
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


class HSPMFBDMLoss(nn.Module):
    def __init__(
        self,
        coarse_weight=0.3,
        edge_weight=0.05,
        edge_kernel_size=3,
        boundary_band_weight=0.0,
        boundary_band_kernel_size=7,
    ):
        super().__init__()
        if coarse_weight < 0:
            raise ValueError("coarse_weight must be non-negative.")
        if edge_weight < 0:
            raise ValueError("edge_weight must be non-negative.")
        if boundary_band_weight < 0:
            raise ValueError("boundary_band_weight must be non-negative.")
        if boundary_band_kernel_size <= 0 or boundary_band_kernel_size % 2 == 0:
            raise ValueError("boundary_band_kernel_size must be a positive odd integer.")
        self.coarse_weight = float(coarse_weight)
        self.edge_weight = float(edge_weight)
        self.edge_kernel_size = int(edge_kernel_size)
        self.boundary_band_weight = float(boundary_band_weight)
        self.boundary_band_kernel_size = int(boundary_band_kernel_size)
        self.seg_loss = BCEDiceLoss()

    def forward(
        self,
        outputs,
        target,
        coarse_weight=None,
        edge_weight=None,
        boundary_band_weight=None,
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
        if min(current_coarse_weight, current_edge_weight, current_boundary_band_weight) < 0:
            raise ValueError("dynamic HSPM-FBDM loss weights must be non-negative.")

        seg = self.seg_loss(outputs["seg"], target)
        coarse_weighted = seg.new_zeros(())
        edge_weighted = seg.new_zeros(())
        boundary_band_weighted = seg.new_zeros(())
        if current_coarse_weight > 0:
            coarse_target = F.interpolate(target, size=outputs["coarse"].shape[-2:], mode="nearest")
            coarse_weighted = current_coarse_weight * self.seg_loss(outputs["coarse"], coarse_target)
        if current_edge_weight > 0:
            if "edge" not in outputs:
                raise KeyError("HSPMFBDMLoss requires 'edge' output when edge_weight > 0.")
            edge_target = mask_to_edge(target, kernel_size=self.edge_kernel_size)
            if edge_target.shape[-2:] != outputs["edge"].shape[-2:]:
                edge_target = F.interpolate(edge_target, size=outputs["edge"].shape[-2:], mode="nearest")
            edge_weighted = current_edge_weight * self.seg_loss(outputs["edge"], edge_target)
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

        total = seg + coarse_weighted + edge_weighted + boundary_band_weighted
        if not return_components:
            return total
        return total, {
            "seg": seg,
            "coarse_weighted": coarse_weighted,
            "edge_weighted": edge_weighted,
            "boundary_band_weighted": boundary_band_weighted,
            "total": total,
        }


class FBDMLoss(nn.Module):
    def __init__(self, edge_weight=0.05, edge_kernel_size=3):
        super().__init__()
        if edge_weight < 0:
            raise ValueError("edge_weight must be non-negative.")
        self.edge_weight = float(edge_weight)
        self.edge_kernel_size = int(edge_kernel_size)
        self.seg_loss = BCEDiceLoss()

    def forward(self, outputs, target, edge_weight=None, return_components=False):
        if not isinstance(outputs, dict):
            raise TypeError("FBDMLoss expects model outputs to be a dictionary.")
        if "seg" not in outputs:
            raise KeyError("FBDMLoss requires a 'seg' output key.")

        current_edge_weight = self.edge_weight if edge_weight is None else float(edge_weight)
        if current_edge_weight < 0:
            raise ValueError("edge_weight must be non-negative.")

        seg = self.seg_loss(outputs["seg"], target)
        edge_weighted = seg.new_zeros(())
        if current_edge_weight > 0:
            if "edge" not in outputs:
                raise KeyError("FBDMLoss requires an 'edge' output key when edge_weight > 0.")
            edge_target = mask_to_edge(target, kernel_size=self.edge_kernel_size)
            if edge_target.shape[-2:] != outputs["edge"].shape[-2:]:
                edge_target = F.interpolate(edge_target, size=outputs["edge"].shape[-2:], mode="nearest")
            edge_weighted = current_edge_weight * self.seg_loss(outputs["edge"], edge_target)

        total = seg + edge_weighted
        if not return_components:
            return total
        return total, {
            "seg": seg,
            "edge_weighted": edge_weighted,
            "total": total,
        }


class UBRDLoss(nn.Module):
    def __init__(self, coarse_weight=0.3, boundary_weight=0.0):
        super().__init__()
        if coarse_weight < 0:
            raise ValueError("coarse_weight must be non-negative.")
        if boundary_weight < 0:
            raise ValueError("boundary_weight must be non-negative.")
        self.coarse_weight = float(coarse_weight)
        self.boundary_weight = float(boundary_weight)
        self.seg_loss = BCEDiceLoss()
        self.boundary_loss = SobelBoundaryLoss()

    def forward(self, outputs, target):
        if not isinstance(outputs, dict):
            raise TypeError("UBRDLoss expects model outputs to be a dictionary.")
        if "seg" not in outputs or "coarse" not in outputs:
            raise KeyError("UBRDLoss requires 'seg' and 'coarse' output keys.")

        loss = self.seg_loss(outputs["seg"], target)
        if self.coarse_weight > 0:
            coarse_target = F.interpolate(target, size=outputs["coarse"].shape[-2:], mode="nearest")
            coarse_loss = self.seg_loss(outputs["coarse"], coarse_target)
            loss = loss + self.coarse_weight * coarse_loss
        if self.boundary_weight > 0:
            loss = loss + self.boundary_weight * self.boundary_loss(outputs["seg"], target)
        return loss


class APBRLoss(nn.Module):
    def __init__(
        self,
        coarse_weight=0.1,
        intermediate_weight=0.15,
        boundary_weight=0.1,
    ):
        super().__init__()
        for name, value in (
            ("coarse_weight", coarse_weight),
            ("intermediate_weight", intermediate_weight),
            ("boundary_weight", boundary_weight),
        ):
            if value < 0:
                raise ValueError(f"{name} must be non-negative.")
        self.coarse_weight = float(coarse_weight)
        self.intermediate_weight = float(intermediate_weight)
        self.boundary_weight = float(boundary_weight)
        self.seg_loss = BCEDiceLoss()
        self.boundary_loss = SobelBoundaryLoss()

    def forward(self, outputs, target, coarse_weight=None, return_components=False):
        if not isinstance(outputs, dict):
            raise TypeError("APBRLoss expects model outputs to be a dictionary.")
        required_keys = {"seg", "coarse", "refine_half"}
        missing_keys = required_keys.difference(outputs)
        if missing_keys:
            raise KeyError(f"APBRLoss requires output keys: {sorted(required_keys)}.")

        current_coarse_weight = self.coarse_weight if coarse_weight is None else float(coarse_weight)
        if current_coarse_weight < 0:
            raise ValueError("coarse_weight must be non-negative.")

        seg = self.seg_loss(outputs["seg"], target)
        coarse_weighted = seg.new_zeros(())
        intermediate_weighted = seg.new_zeros(())
        boundary_weighted = seg.new_zeros(())
        if current_coarse_weight > 0:
            coarse_target = F.interpolate(target, size=outputs["coarse"].shape[-2:], mode="nearest")
            coarse_weighted = current_coarse_weight * self.seg_loss(outputs["coarse"], coarse_target)
        if self.intermediate_weight > 0:
            refine_target = F.interpolate(target, size=outputs["refine_half"].shape[-2:], mode="nearest")
            intermediate_weighted = self.intermediate_weight * self.seg_loss(outputs["refine_half"], refine_target)
        if self.boundary_weight > 0:
            boundary_weighted = self.boundary_weight * self.boundary_loss(outputs["seg"], target)
        total = seg + coarse_weighted + intermediate_weighted + boundary_weighted
        if not return_components:
            return total
        return total, {
            "seg": seg,
            "coarse_weighted": coarse_weighted,
            "intermediate_weighted": intermediate_weighted,
            "boundary_weighted": boundary_weighted,
            "total": total,
        }


class SDFRLoss(nn.Module):
    def __init__(
        self,
        coarse_weight=0.1,
        sdf_weight=0.2,
        boundary_temperature=0.2,
        boundary_emphasis=4.0,
    ):
        super().__init__()
        for name, value in (
            ("coarse_weight", coarse_weight),
            ("sdf_weight", sdf_weight),
            ("boundary_emphasis", boundary_emphasis),
        ):
            if value < 0:
                raise ValueError(f"{name} must be non-negative.")
        if boundary_temperature <= 0:
            raise ValueError("boundary_temperature must be positive.")
        self.coarse_weight = float(coarse_weight)
        self.sdf_weight = float(sdf_weight)
        self.boundary_temperature = float(boundary_temperature)
        self.boundary_emphasis = float(boundary_emphasis)
        self.seg_loss = BCEDiceLoss()

    def forward(
        self,
        outputs,
        target,
        target_sdf,
        coarse_weight=None,
        sdf_weight=None,
        return_components=False,
    ):
        if not isinstance(outputs, dict):
            raise TypeError("SDFRLoss expects model outputs to be a dictionary.")
        required_keys = {"seg", "coarse", "sdf"}
        missing_keys = required_keys.difference(outputs)
        if missing_keys:
            raise KeyError(f"SDFRLoss requires output keys: {sorted(required_keys)}.")
        if target_sdf is None:
            raise ValueError("target_sdf is required for SDFRLoss.")

        current_coarse_weight = (
            self.coarse_weight if coarse_weight is None else float(coarse_weight)
        )
        current_sdf_weight = (
            self.sdf_weight if sdf_weight is None else float(sdf_weight)
        )
        if current_coarse_weight < 0 or current_sdf_weight < 0:
            raise ValueError("Dynamic SDFR loss weights must be non-negative.")

        seg = self.seg_loss(outputs["seg"], target)
        coarse_weighted = seg.new_zeros(())
        sdf_weighted = seg.new_zeros(())
        if current_coarse_weight > 0:
            coarse_target = F.interpolate(
                target,
                size=outputs["coarse"].shape[-2:],
                mode="nearest",
            )
            coarse_weighted = current_coarse_weight * self.seg_loss(
                outputs["coarse"],
                coarse_target,
            )
        if current_sdf_weight > 0:
            target_sdf = target_sdf.to(
                device=outputs["sdf"].device,
                dtype=outputs["sdf"].dtype,
            )
            if target_sdf.shape != outputs["sdf"].shape:
                target_sdf = F.interpolate(
                    target_sdf,
                    size=outputs["sdf"].shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            weights = 1.0 + self.boundary_emphasis * torch.exp(
                -target_sdf.abs() / self.boundary_temperature
            )
            sdf_elementwise = F.smooth_l1_loss(
                outputs["sdf"],
                target_sdf,
                reduction="none",
            )
            sdf_loss = (weights * sdf_elementwise).sum() / weights.sum().clamp_min(1e-6)
            sdf_weighted = current_sdf_weight * sdf_loss

        total = seg + coarse_weighted + sdf_weighted
        if not return_components:
            return total
        return total, {
            "seg": seg,
            "coarse_weighted": coarse_weighted,
            "sdf_weighted": sdf_weighted,
            "total": total,
        }


class SDFRV2Loss(SDFRLoss):
    def __init__(
        self,
        coarse_weight=0.1,
        sdf_weight=0.2,
        boundary_temperature=0.2,
        boundary_emphasis=4.0,
        base_weight=0.0,
        band_width=0.2,
        band_weight=0.1,
    ):
        super().__init__(
            coarse_weight=coarse_weight,
            sdf_weight=sdf_weight,
            boundary_temperature=boundary_temperature,
            boundary_emphasis=boundary_emphasis,
        )
        for name, value in (
            ("base_weight", base_weight),
            ("band_weight", band_weight),
        ):
            if value < 0:
                raise ValueError(f"{name} must be non-negative.")
        if not 0.0 < band_width <= 1.0:
            raise ValueError("band_width must be in (0, 1].")
        self.base_weight = float(base_weight)
        self.band_width = float(band_width)
        self.band_weight = float(band_weight)

    def forward(
        self,
        outputs,
        target,
        target_sdf,
        coarse_weight=None,
        sdf_weight=None,
        band_weight=None,
        return_components=False,
    ):
        if not isinstance(outputs, dict):
            raise TypeError("SDFRV2Loss expects model outputs to be a dictionary.")
        required_keys = {"seg", "base_seg", "coarse", "sdf", "logit_correction"}
        missing_keys = required_keys.difference(outputs)
        if missing_keys:
            raise KeyError(f"SDFRV2Loss requires output keys: {sorted(required_keys)}.")
        if target_sdf is None:
            raise ValueError("target_sdf is required for SDFRV2Loss.")

        current_coarse_weight = (
            self.coarse_weight if coarse_weight is None else float(coarse_weight)
        )
        current_sdf_weight = (
            self.sdf_weight if sdf_weight is None else float(sdf_weight)
        )
        current_band_weight = (
            self.band_weight if band_weight is None else float(band_weight)
        )
        if min(current_coarse_weight, current_sdf_weight, current_band_weight) < 0:
            raise ValueError("Dynamic SDFR V2 loss weights must be non-negative.")

        seg = self.seg_loss(outputs["seg"], target)
        base_weighted = seg.new_zeros(())
        coarse_weighted = seg.new_zeros(())
        sdf_weighted = seg.new_zeros(())
        band_weighted = seg.new_zeros(())

        if self.base_weight > 0:
            base_weighted = self.base_weight * self.seg_loss(
                outputs["base_seg"],
                target,
            )

        if current_coarse_weight > 0:
            coarse_target = F.interpolate(
                target,
                size=outputs["coarse"].shape[-2:],
                mode="nearest",
            )
            coarse_weighted = current_coarse_weight * self.seg_loss(
                outputs["coarse"],
                coarse_target,
            )

        target_sdf = target_sdf.to(
            device=outputs["sdf"].device,
            dtype=outputs["sdf"].dtype,
        )
        if target_sdf.shape != outputs["sdf"].shape:
            target_sdf = F.interpolate(
                target_sdf,
                size=outputs["sdf"].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        if current_sdf_weight > 0:
            weights = 1.0 + self.boundary_emphasis * torch.exp(
                -target_sdf.abs() / self.boundary_temperature
            )
            sdf_elementwise = F.smooth_l1_loss(
                outputs["sdf"],
                target_sdf,
                reduction="none",
            )
            sdf_loss = (weights * sdf_elementwise).sum() / weights.sum().clamp_min(1e-6)
            sdf_weighted = current_sdf_weight * sdf_loss

        if current_band_weight > 0:
            band = (target_sdf.abs() <= self.band_width).to(outputs["seg"].dtype)
            if band.shape != outputs["seg"].shape:
                band = F.interpolate(
                    band,
                    size=outputs["seg"].shape[-2:],
                    mode="nearest",
                )
            band_logits = outputs["base_seg"].detach() + outputs["logit_correction"]
            band_elementwise = F.binary_cross_entropy_with_logits(
                band_logits,
                target,
                reduction="none",
            )
            band_loss = (band * band_elementwise).sum() / band.sum().clamp_min(1.0)
            band_weighted = current_band_weight * band_loss

        total = seg + base_weighted + coarse_weighted + sdf_weighted + band_weighted
        if not return_components:
            return total
        return total, {
            "seg": seg,
            "base_weighted": base_weighted,
            "coarse_weighted": coarse_weighted,
            "sdf_weighted": sdf_weighted,
            "band_weighted": band_weighted,
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
