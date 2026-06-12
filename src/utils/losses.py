import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    'BCEDiceLoss',
    'DiceBCELoss',
    'SobelBoundaryLoss',
    'BoundaryAwareSegLoss',
    'HSPMLoss',
    'UBRDLoss',
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


def compute_kl_loss(p, q):
    p_loss = F.kl_div(F.log_softmax(p, dim=-1),
                      F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1),
                      F.softmax(p, dim=-1), reduction='none')

    p_loss = p_loss.mean()
    q_loss = q_loss.mean()

    loss = (p_loss + q_loss) / 2
    return loss
