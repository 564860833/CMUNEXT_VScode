import argparse
import csv
import os

import cv2
import numpy as np
import torch
from albumentations import Compose, Normalize, Resize
from torch.utils.data import DataLoader

import src.utils.losses as losses
from src.dataloader.dataset import MedicalDataSets
from src.utils.metrics import iou_score, boundary_scores, find_best_threshold
from src.network.conv_based.AttU_Net import AttU_Net
from src.network.conv_based.CMUNet import CMUNet
from src.network.conv_based.CMUNeXt import cmunext
from src.network.conv_based.CMUNeXt_FBDM import cmunext_fbdm
from src.network.conv_based.CMUNeXt_FBDM_Best0616 import cmunext_fbdm_best0616
from src.network.conv_based.CMUNeXt_USLGSF import cmunext_uslgsf
from src.network.conv_based.CMUNeXt_USLGSF_V2 import cmunext_uslgsf_v2
from src.network.conv_based.CMUNeXt_USLGSF_V3 import cmunext_uslgsf_v3
from src.network.conv_based.CMUNeXt_HSPM import cmunext_hspm
from src.network.conv_based.CMUNeXt_HSPM_Best0616 import cmunext_hspm_best0616
from src.network.conv_based.CMUNeXt_HSPM_FBDM import cmunext_hspm_fbdm, cmunext_hspm_fbdm_v2
from src.network.conv_based.CMUNeXt_BA_DualGAG import cmunext_ba_dualgag
from src.network.conv_based.CMUNeXt_BA_DualGAG_SpeckleEnhance import cmunext_ba_dualgag_speckleenhance
from src.network.conv_based.CMUNeXt_DualGAG import cmunext_dualgag
from src.network.conv_based.CMUNeXt_DualGAG_SpeckleEnhance import cmunext_dualgag_speckleenhance
from src.network.conv_based.CMUNeXt_SpeckleEnhance import cmunext_speckle
from src.network.conv_based.MK_UNet import MK_UNet
from src.network.conv_based.U_Net import U_Net
from src.network.conv_based.UNet3plus import UNet3plus
from src.network.conv_based.UNetplus import ResNet34UnetPlus
from src.network.conv_based.UNeXt import UNext
from src.network.hybrid_based.Mobile_U_ViT import mobileuvit
from src.network.transfomer_based.transformer_based_network import get_transformer_based_model


HSPM_FBDM_V2_MODELS = {"CMUNeXt_HSPM_FBDM_V2"}
HSPM_FBDM_MODELS = {"CMUNeXt_HSPM_FBDM", *HSPM_FBDM_V2_MODELS}
FBDM_BEST0616_MODEL = "CMUNeXt_FBDM_Best0616"
HSPM_BEST0616_MODEL = "CMUNeXt_HSPM_Best0616"
FBDM_ONLY_MODELS = {"CMUNeXt_FBDM", FBDM_BEST0616_MODEL}
HSPM_ONLY_MODELS = {"CMUNeXt_HSPM", HSPM_BEST0616_MODEL}
HSPM_MODELS = {*HSPM_ONLY_MODELS, *HSPM_FBDM_MODELS}


def parse_gag_stages(value):
    if isinstance(value, (tuple, list)):
        return tuple(value)

    stages = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            raise argparse.ArgumentTypeError("gag_stages must be a comma-separated list, e.g. 1,3.")
        try:
            stage = int(item)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("gag_stages values must be integers in [0, 1, 2, 3].") from exc
        if stage not in {0, 1, 2, 3}:
            raise argparse.ArgumentTypeError("gag_stages values must be in [0, 1, 2, 3].")
        if stage not in stages:
            stages.append(stage)

    if not stages:
        raise argparse.ArgumentTypeError("gag_stages must include at least one stage.")
    return tuple(stages)


def parse_ddsr_stages(value):
    if isinstance(value, (tuple, list)):
        return tuple(value)

    stages = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            raise argparse.ArgumentTypeError("ddsr_stages must be a comma-separated list, e.g. 0,1.")
        try:
            stage = int(item)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("ddsr_stages values must be integers in [0, 1, 2, 3].") from exc
        if stage not in {0, 1, 2, 3}:
            raise argparse.ArgumentTypeError("ddsr_stages values must be in [0, 1, 2, 3].")
        if stage not in stages:
            stages.append(stage)

    if not stages:
        raise argparse.ArgumentTypeError("ddsr_stages must include at least one stage.")
    return tuple(stages)


def parse_uslgsf_stages(value):
    if isinstance(value, (tuple, list)):
        return tuple(value)

    stages = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            raise argparse.ArgumentTypeError("uslgsf_stages must be a comma-separated list, e.g. 0,1.")
        try:
            stage = int(item)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "uslgsf_stages values must be integers in [0, 1, 2, 3]."
            ) from exc
        if stage not in {0, 1, 2, 3}:
            raise argparse.ArgumentTypeError("uslgsf_stages values must be in [0, 1, 2, 3].")
        if stage not in stages:
            stages.append(stage)

    if not stages:
        raise argparse.ArgumentTypeError("uslgsf_stages must include at least one stage.")
    return tuple(stages)


def parse_uslgsf_smooth_kernels(value):
    if isinstance(value, (tuple, list)):
        kernels = tuple(int(kernel) for kernel in value)
    else:
        try:
            kernels = tuple(int(item.strip()) for item in str(value).split(","))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "uslgsf_smooth_kernels must contain two odd integers, e.g. 3,7."
            ) from exc
    if len(kernels) != 2 or any(kernel <= 0 or kernel % 2 == 0 for kernel in kernels):
        raise argparse.ArgumentTypeError(
            "uslgsf_smooth_kernels must contain two positive odd integers."
        )
    if kernels[0] >= kernels[1]:
        raise argparse.ArgumentTypeError(
            "uslgsf_smooth_kernels must be ordered from small to large."
        )
    return kernels


def apply_best0616_presets(args):
    if args.model == FBDM_BEST0616_MODEL:
        args.fbdm_edge_aux_only = True
        args.fbdm_gate_init = 0.01
        args.fbdm_gate_max = 0.06
        args.fbdm_edge_loss_weight = 0.03
        args.fbdm_edge_loss_final_weight = 0.003
        args.fbdm_edge_loss_decay_epochs = 150
        args.fbdm_residual_warmup_epochs = 40
    elif args.model == HSPM_BEST0616_MODEL:
        args.hspm_mode = "full"
        args.hspm_backbone_mode = "dual_path"
        args.hspm_fusion_mode = "global"
        args.hspm_mixer_mode = "legacy"
        args.hspm_gamma_init = 0.1
        args.hspm_gamma_max = 0.3
        args.hspm_temperature = 0.1
        args.hspm_prototype_dropout = 0.0
        args.hspm_fusion_gate_init = 0.05
        args.hspm_fusion_gate_max = 0.3
        args.hspm_coarse_loss_weight = 0.1
        args.hspm_coarse_loss_final_weight = 0.02
        args.hspm_coarse_loss_decay_epochs = 150
    return args


def build_model(args, parser):
    if args.model == "CMUNet":
        model = CMUNet(output_ch=args.num_classes)
    elif args.model == "MK_UNet":
        model = MK_UNet(num_classes=args.num_classes, in_channels=3)
    elif args.model == "CMUNeXt":
        model = cmunext(num_classes=args.num_classes)
    elif args.model == "CMUNeXt_FBDM":
        model = cmunext_fbdm(
            num_classes=args.num_classes,
            fbdm_gate_init=args.fbdm_gate_init,
            fbdm_gate_max=args.fbdm_gate_max,
            fbdm_edge_aux_only=args.fbdm_edge_aux_only,
        )
    elif args.model == FBDM_BEST0616_MODEL:
        model = cmunext_fbdm_best0616(num_classes=args.num_classes)
    elif args.model == "CMUNeXt_USLGSF":
        model = cmunext_uslgsf(
            num_classes=args.num_classes,
            uslgsf_stages=args.uslgsf_stages,
            uslgsf_smooth_kernels=args.uslgsf_smooth_kernels,
            uslgsf_context_downsample=args.uslgsf_context_downsample,
            uslgsf_alpha_init=args.uslgsf_alpha_init,
            uslgsf_alpha_max=args.uslgsf_alpha_max,
            uslgsf_mode=args.uslgsf_mode,
        )
    elif args.model == "CMUNeXt_USLGSF_V2":
        model = cmunext_uslgsf_v2(
            num_classes=args.num_classes,
            uslgsf_stages=args.uslgsf_stages,
            uslgsf_smooth_kernels=args.uslgsf_smooth_kernels,
            uslgsf_context_downsample=args.uslgsf_context_downsample,
            uslgsf_alpha_init=args.uslgsf_alpha_init,
            uslgsf_alpha_max=args.uslgsf_alpha_max,
            uslgsf_mode=args.uslgsf_mode,
        )
    elif args.model == "CMUNeXt_USLGSF_V3":
        model = cmunext_uslgsf_v3(
            num_classes=args.num_classes,
            uslgsf_stages=args.uslgsf_stages,
            uslgsf_smooth_kernels=args.uslgsf_smooth_kernels,
            uslgsf_context_downsample=args.uslgsf_context_downsample,
            uslgsf_alpha_init=args.uslgsf_alpha_init,
            uslgsf_alpha_max=args.uslgsf_alpha_max,
            uslgsf_mode=args.uslgsf_mode,
            uslgsf_residual_init_scale=args.uslgsf_residual_init_scale,
        )
    elif args.model == "CMUNeXt_HSPM":
        model = cmunext_hspm(
            num_classes=args.num_classes,
            hspm_mode=args.hspm_mode,
            hspm_mixer_mode=args.hspm_mixer_mode,
            hspm_gamma_init=args.hspm_gamma_init,
            hspm_gamma_max=args.hspm_gamma_max,
            hspm_temperature=args.hspm_temperature,
            hspm_prototype_dropout=args.hspm_prototype_dropout,
            hspm_backbone_mode=args.hspm_backbone_mode,
            hspm_fusion_gate_init=args.hspm_fusion_gate_init,
            hspm_fusion_gate_max=args.hspm_fusion_gate_max,
            hspm_fusion_mode=args.hspm_fusion_mode,
            hspm_small_area_threshold=args.hspm_small_area_threshold,
            hspm_small_area_temperature=args.hspm_small_area_temperature,
        )
    elif args.model == HSPM_BEST0616_MODEL:
        model = cmunext_hspm_best0616(num_classes=args.num_classes)
    elif args.model == "CMUNeXt_HSPM_FBDM":
        model = cmunext_hspm_fbdm(
            num_classes=args.num_classes,
            hspm_mode=args.hspm_mode,
            hspm_mixer_mode=args.hspm_mixer_mode,
            hspm_gamma_init=args.hspm_gamma_init,
            hspm_gamma_max=args.hspm_gamma_max,
            hspm_temperature=args.hspm_temperature,
            hspm_prototype_dropout=args.hspm_prototype_dropout,
            hspm_backbone_mode=args.hspm_backbone_mode,
            hspm_fusion_gate_init=args.hspm_fusion_gate_init,
            hspm_fusion_gate_max=args.hspm_fusion_gate_max,
            hspm_fusion_mode=args.hspm_fusion_mode,
            hspm_small_area_threshold=args.hspm_small_area_threshold,
            hspm_small_area_temperature=args.hspm_small_area_temperature,
            fbdm_use_hspm_prior=not args.fbdm_no_hspm_prior,
            fbdm_detach_hspm_prior=not args.fbdm_no_detach_hspm_prior,
            fbdm_semantic_uncertainty_weight=args.fbdm_semantic_uncertainty_weight,
            fbdm_semantic_coarse_weight=args.fbdm_semantic_coarse_weight,
            fbdm_semantic_gate_base=args.fbdm_semantic_gate_base,
            fbdm_gate_init=args.fbdm_gate_init,
            fbdm_gate_max=args.fbdm_gate_max,
            fbdm_edge_aux_only=args.fbdm_edge_aux_only,
        )
    elif args.model == "CMUNeXt_HSPM_FBDM_V2":
        model = cmunext_hspm_fbdm_v2(
            num_classes=args.num_classes,
            hspm_mode=args.hspm_mode,
            hspm_mixer_mode=args.hspm_mixer_mode,
            hspm_gamma_init=args.hspm_gamma_init,
            hspm_gamma_max=args.hspm_gamma_max,
            hspm_temperature=args.hspm_temperature,
            hspm_prototype_dropout=args.hspm_prototype_dropout,
            hspm_backbone_mode=args.hspm_backbone_mode,
            hspm_fusion_gate_init=args.hspm_fusion_gate_init,
            hspm_fusion_gate_max=args.hspm_fusion_gate_max,
            hspm_fusion_mode=args.hspm_fusion_mode,
            hspm_small_area_threshold=args.hspm_small_area_threshold,
            hspm_small_area_temperature=args.hspm_small_area_temperature,
            fbdm_use_hspm_prior=not args.fbdm_no_hspm_prior,
            fbdm_detach_hspm_prior=not args.fbdm_no_detach_hspm_prior,
            fbdm_semantic_uncertainty_weight=args.fbdm_semantic_uncertainty_weight,
            fbdm_semantic_coarse_weight=args.fbdm_semantic_coarse_weight,
            fbdm_semantic_gate_base=args.fbdm_semantic_gate_base,
            fbdm_gate_init=args.fbdm_gate_init,
            fbdm_gate_max=args.fbdm_gate_max,
            fbdm_correction_scale_init=args.fbdm_correction_scale_init,
            fbdm_correction_scale_max=args.fbdm_correction_scale_max,
        )
    elif args.model == "CMUNeXt_DualGAG":
        model = cmunext_dualgag(num_classes=args.num_classes, gag_stages=args.gag_stages)
    elif args.model == "CMUNeXt_BA_DualGAG":
        model = cmunext_ba_dualgag(num_classes=args.num_classes, gag_stages=args.gag_stages)
    elif args.model == "CMUNeXt_BA_DualGAG_SpeckleEnhance":
        model = cmunext_ba_dualgag_speckleenhance(
            num_classes=args.num_classes,
            ddsr_stages=args.ddsr_stages,
            gag_stages=args.gag_stages,
            ddsr_smooth_k=args.ddsr_smooth_k,
            ddsr_max_scale=args.ddsr_max_scale,
            ddsr_skip_only=args.ddsr_mode == "skip_only",
            ddsr_aux_init=args.ddsr_aux_init,
        )
    elif args.model == "CMUNeXt_SpeckleEnhance":
        model = cmunext_speckle(
            num_classes=args.num_classes,
            ddsr_stages=args.ddsr_stages,
            ddsr_smooth_k=args.ddsr_smooth_k,
            ddsr_max_scale=args.ddsr_max_scale,
            ddsr_skip_only=args.ddsr_mode == "skip_only",
        )
    elif args.model == "CMUNeXt_DualGAG_SpeckleEnhance":
        model = cmunext_dualgag_speckleenhance(
            num_classes=args.num_classes,
            ddsr_stages=args.ddsr_stages,
            gag_stages=args.gag_stages,
            ddsr_smooth_k=args.ddsr_smooth_k,
            ddsr_max_scale=args.ddsr_max_scale,
            ddsr_skip_only=args.ddsr_mode == "skip_only",
        )
    elif args.model == "U_Net":
        model = U_Net(output_ch=args.num_classes)
    elif args.model == "AttU_Net":
        model = AttU_Net(output_ch=args.num_classes)
    elif args.model == "UNext":
        model = UNext(num_classes=args.num_classes)
    elif args.model == "UNetplus":
        model = ResNet34UnetPlus(num_class=args.num_classes)
    elif args.model == "UNet3plus":
        model = UNet3plus(n_classes=args.num_classes)
    elif args.model == "Mobile_U_ViT":
        model = mobileuvit(out_channel=args.num_classes)
    else:
        print(f"Attempting to load transformer-based model: {args.model}")
        model = get_transformer_based_model(
            parser=parser,
            model_name=args.model,
            img_size=args.img_size,
            num_classes=args.num_classes,
            in_ch=3,
        )

    return model


def load_model(model_path, args, device, parser):
    model = build_model(args, parser)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    if args.model in HSPM_MODELS:
        effective_fusion_gate = model.effective_fusion_gate()
        if effective_fusion_gate is not None:
            print(f"HSPM effective fusion gate: {effective_fusion_gate.detach().cpu().item():.4f}")
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)
    model.eval()
    return model


def forward_with_model(model, model_name, x):
    return model(x)


def get_seg_logits(outputs):
    if isinstance(outputs, dict):
        if "seg" in outputs:
            return outputs["seg"]
        if "pred_refined" in outputs:
            return outputs["pred_refined"]
        if "pred_main" in outputs:
            return outputs["pred_main"]
        raise KeyError(f"Unsupported output keys: {sorted(outputs.keys())}")
    return outputs


def get_val_transform(img_size):
    return Compose([
        Resize(img_size, img_size),
        Normalize(),
    ])


def build_validation_thresholds(args):
    if args.val_threshold_mode == "fixed":
        thresholds = [args.val_threshold]
    else:
        if args.val_threshold_step <= 0:
            raise ValueError("val_threshold_step must be positive.")
        if args.val_threshold_start > args.val_threshold_end:
            raise ValueError("val_threshold_start must be <= val_threshold_end.")
        thresholds = np.arange(
            args.val_threshold_start,
            args.val_threshold_end + args.val_threshold_step * 0.5,
            args.val_threshold_step,
        )

    thresholds = [
        round(float(np.clip(threshold, 1e-4, 1.0 - 1e-4)), 4)
        for threshold in thresholds
    ]
    return sorted(set(thresholds))


def get_batch_strings(sampled_batch, key, batch_size, fallback_prefix=None):
    values = sampled_batch.get(key)
    if values is None:
        if fallback_prefix is None:
            raise KeyError(f"Missing required batch field: {key}")
        return [f"{fallback_prefix}_{idx}" for idx in range(batch_size)]
    if isinstance(values, str):
        return [values]
    return [str(value) for value in values]


def prepare_case_name(case_name):
    case_name = os.path.splitext(os.path.basename(case_name))[0]
    for character in '<>:"/\\|?*':
        case_name = case_name.replace(character, "_")
    return case_name


def draw_contour_overlay(image, target_mask, predicted_mask):
    overlay = image.copy()
    target_contours, _ = cv2.findContours(target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    predicted_contours, _ = cv2.findContours(predicted_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, target_contours, -1, (0, 255, 0), 2)
    cv2.drawContours(overlay, predicted_contours, -1, (0, 0, 255), 2)
    return overlay


def build_error_map(target_mask, predicted_mask):
    target = target_mask > 0
    predicted = predicted_mask > 0
    error_map = np.zeros((*target.shape, 3), dtype=np.uint8)
    error_map[target & predicted] = (0, 255, 0)
    error_map[~target & predicted] = (0, 0, 255)
    error_map[target & ~predicted] = (255, 0, 0)
    return error_map


def compute_case_metrics(probability, target, threshold):
    probability = probability.unsqueeze(0)
    target = target.unsqueeze(0)
    iou, dice, se, pc, f1, sp, acc = iou_score(
        probability,
        target,
        threshold=threshold,
        from_logits=False,
    )
    hd95, assd = boundary_scores(
        probability,
        target,
        threshold=threshold,
        from_logits=False,
    )
    return {
        "iou": iou,
        "dice": dice,
        "se": se,
        "pc": pc,
        "f1": f1,
        "sp": sp,
        "hd95": hd95,
        "assd": assd,
        "acc": acc,
    }


def should_save_visual(case_name, visual_mode, visual_case):
    if visual_mode == "all":
        return True
    if visual_mode == "selected":
        return prepare_case_name(case_name) == prepare_case_name(visual_case)
    return False


def save_case_outputs(
        prob_batches, target_batches, case_name_batches, image_path_batches, mask_path_batches,
        threshold, save_dir, visual_mode="none", visual_case=None):
    output_dirs = None
    if visual_mode != "none":
        output_dirs = {
            "mask": os.path.join(save_dir, "masks"),
            "heatmap": os.path.join(save_dir, "heatmaps"),
            "contour": os.path.join(save_dir, "contours"),
            "error_map": os.path.join(save_dir, "error_maps"),
        }
        for output_dir in output_dirs.values():
            os.makedirs(output_dir, exist_ok=True)

    metric_rows = []
    saved_visual_count = 0
    for probabilities, targets, case_names, image_paths, mask_paths in zip(
            prob_batches, target_batches, case_name_batches, image_path_batches, mask_path_batches):
        for probability, target, case_name, image_path, mask_path in zip(
                probabilities, targets, case_names, image_paths, mask_paths):
            case_name = prepare_case_name(case_name)
            case_metrics = compute_case_metrics(probability, target, threshold)
            metric_rows.append({
                "case_name": case_name,
                "threshold": threshold,
                **{key: round(value, 6) for key, value in case_metrics.items()},
            })

            if not should_save_visual(case_name, visual_mode, visual_case):
                continue

            display_image = cv2.imread(image_path)
            target_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if display_image is None:
                raise FileNotFoundError(f"Unable to read original image: {image_path}")
            if target_mask is None:
                raise FileNotFoundError(f"Unable to read original mask: {mask_path}")

            original_height, original_width = display_image.shape[:2]
            if target_mask.shape != (original_height, original_width):
                target_mask = cv2.resize(
                    target_mask,
                    (original_width, original_height),
                    interpolation=cv2.INTER_NEAREST,
                )
            probability_map = probability.squeeze(0).numpy()
            probability_map = cv2.resize(
                probability_map,
                (original_width, original_height),
                interpolation=cv2.INTER_LINEAR,
            )
            target_mask = ((target_mask > 127) * 255).astype(np.uint8)
            predicted_mask = ((probability_map > threshold) * 255).astype(np.uint8)
            output_name = f"{case_name}.png"

            cv2.imwrite(os.path.join(output_dirs["mask"], output_name), predicted_mask)

            heatmap = cv2.applyColorMap(
                np.clip(probability_map * 255.0, 0, 255).astype(np.uint8),
                cv2.COLORMAP_JET,
            )
            cv2.imwrite(os.path.join(output_dirs["heatmap"], output_name), heatmap)

            contour_overlay = draw_contour_overlay(display_image, target_mask, predicted_mask)
            cv2.imwrite(os.path.join(output_dirs["contour"], output_name), contour_overlay)

            error_map = build_error_map(target_mask, predicted_mask)
            cv2.imwrite(os.path.join(output_dirs["error_map"], output_name), error_map)
            saved_visual_count += 1

    metric_path = os.path.join(save_dir, "per_case_metrics.csv")
    fieldnames = ["case_name", "threshold", "iou", "dice", "se", "pc", "f1", "sp", "hd95", "assd", "acc"]
    with open(metric_path, "w", newline="", encoding="utf-8") as metric_file:
        writer = csv.DictWriter(metric_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metric_rows)

    if visual_mode == "selected" and saved_visual_count == 0:
        raise ValueError(f"visual_case '{visual_case}' was not found in the validation set.")
    print(f"Saved per-case metrics to: {metric_path}")
    if visual_mode != "none":
        print(f"Saved visualizations for {saved_visual_count} case(s) to: {save_dir}")


def validate(model, val_loader, criterion, device, args, save_dir="validation_results"):
    model.eval()
    val_loss = 0.0
    prob_batches = []
    target_batches = []
    case_name_batches = []
    image_path_batches = []
    mask_path_batches = []
    metrics = None
    threshold = args.val_threshold
    totals = {
        "iou": 0.0,
        "dice": 0.0,
        "se": 0.0,
        "pc": 0.0,
        "f1": 0.0,
        "sp": 0.0,
        "hd95": 0.0,
        "assd": 0.0,
        "acc": 0.0,
    }
    total_count = 0
    visual_mode = getattr(args, "visual_mode", "none")
    visual_case = getattr(args, "visual_case", None)
    if visual_mode not in {"none", "selected", "all"}:
        raise ValueError(f"Unsupported visual_mode: {visual_mode}")
    if visual_mode == "selected" and not visual_case:
        raise ValueError("visual_case is required when visual_mode='selected'.")
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for i_batch, sampled_batch in enumerate(val_loader):
            img_batch, label_batch = sampled_batch["image"], sampled_batch["label"]
            img_batch, label_batch = img_batch.to(device), label_batch.to(device)
            outputs = forward_with_model(model, args.model, img_batch)
            seg_logits = get_seg_logits(outputs)
            if args.model in HSPM_MODELS:
                loss = criterion(outputs, label_batch)
            elif args.model in FBDM_ONLY_MODELS:
                loss = criterion(outputs, label_batch)
            else:
                loss = criterion(seg_logits, label_batch)
            probabilities = torch.sigmoid(seg_logits).detach().cpu()
            targets = label_batch.detach().cpu()

            prob_batches.append(probabilities)
            target_batches.append(targets)
            batch_size = int(img_batch.shape[0])
            case_name_batches.append(get_batch_strings(
                sampled_batch,
                "case_name",
                batch_size,
                fallback_prefix=f"batch_{i_batch}_img",
            ))
            if visual_mode == "none":
                image_path_batches.append([None] * batch_size)
                mask_path_batches.append([None] * batch_size)
            else:
                image_path_batches.append(get_batch_strings(sampled_batch, "image_path", batch_size))
                mask_path_batches.append(get_batch_strings(sampled_batch, "mask_path", batch_size))

            val_loss += loss.item()
            if args.val_threshold_mode == "scan":
                continue

            batch_count = int(img_batch.shape[0])
            iou, dice, se, pc, f1, sp, acc = iou_score(seg_logits, label_batch, threshold=threshold)
            hd95, assd = boundary_scores(seg_logits, label_batch, threshold=threshold)
            totals["iou"] += iou * batch_count
            totals["dice"] += dice * batch_count
            totals["se"] += se * batch_count
            totals["pc"] += pc * batch_count
            totals["f1"] += f1 * batch_count
            totals["sp"] += sp * batch_count
            totals["hd95"] += hd95 * batch_count
            totals["assd"] += assd * batch_count
            totals["acc"] += acc * batch_count
            total_count += batch_count

    val_loss /= len(val_loader)
    if args.val_threshold_mode == "scan":
        thresholds = build_validation_thresholds(args)
        metrics = find_best_threshold(
            prob_batches,
            target_batches,
            thresholds,
            select_metric=args.val_threshold_metric,
            from_logits=False,
        )
        threshold = metrics["threshold"]

    else:
        metrics = {
            "threshold": threshold,
            "iou": totals["iou"] / max(total_count, 1),
            "dice": totals["dice"] / max(total_count, 1),
            "se": totals["se"] / max(total_count, 1),
            "pc": totals["pc"] / max(total_count, 1),
            "f1": totals["f1"] / max(total_count, 1),
            "sp": totals["sp"] / max(total_count, 1),
            "hd95": totals["hd95"] / max(total_count, 1),
            "assd": totals["assd"] / max(total_count, 1),
            "acc": totals["acc"] / max(total_count, 1),
        }

    save_case_outputs(
        prob_batches,
        target_batches,
        case_name_batches,
        image_path_batches,
        mask_path_batches,
        threshold,
        save_dir,
        visual_mode=visual_mode,
        visual_case=visual_case,
    )

    print(
        f"Validation loss: {val_loss:.4f}, threshold: {metrics['threshold']:.4f}, "
        f"IoU: {metrics['iou']:.4f}, Dice: {metrics['dice']:.4f}, F1: {metrics['f1']:.4f}, "
        f"SE: {metrics['se']:.4f}, PC: {metrics['pc']:.4f}, SP: {metrics['sp']:.4f}, "
        f"HD95: {metrics['hd95']:.4f}, ASSD: {metrics['assd']:.4f}, ACC: {metrics['acc']:.4f}"
    )
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validation script for medical image segmentation")

    model_choices = [
        "CMUNet", "CMUNeXt", "CMUNeXt_FBDM", FBDM_BEST0616_MODEL, "CMUNeXt_USLGSF", "CMUNeXt_USLGSF_V2", "CMUNeXt_USLGSF_V3", "CMUNeXt_HSPM", HSPM_BEST0616_MODEL, "CMUNeXt_HSPM_FBDM", "CMUNeXt_HSPM_FBDM_V2",
        "CMUNeXt_DualGAG", "CMUNeXt_BA_DualGAG",
        "CMUNeXt_SpeckleEnhance", "CMUNeXt_DualGAG_SpeckleEnhance",
        "CMUNeXt_BA_DualGAG_SpeckleEnhance",
        "U_Net", "MK_UNet", "AttU_Net", "UNext", "UNetplus", "UNet3plus",
        "TransUnet", "SwinUnet", "MedT", "Mobile_U_ViT",
    ]
    parser.add_argument("--model", type=str, default="U_Net", choices=model_choices, help="model type")
    parser.add_argument("--model_path", type=str, default="./checkpoint/U_Net_model.pth",
                        help="Path to the trained model")
    parser.add_argument("--base_dir", type=str, default="./data/test", help="base directory of dataset")
    parser.add_argument("--train_file_dir", type=str, default="train.txt",
                        help="(Required by MedicalDataSets) train file directory")
    parser.add_argument("--val_file_dir", type=str, default="test_val.txt", help="validation file list")
    parser.add_argument("--img_size", type=int, default=256, help="image size")
    parser.add_argument("--num_classes", type=int, default=1, help="number of classes")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--ddsr_stages", type=parse_ddsr_stages, default=(0, 1),
                        help="Comma-separated DDSR stages, e.g. 0,1 or 2,3 or 0,1,2,3")
    parser.add_argument("--ddsr_smooth_k", type=int, default=5,
                        help="DDSR average-pooling kernel size")
    parser.add_argument("--ddsr_max_scale", type=float, default=0.05,
                        help="Upper bound for DDSR residual scale")
    parser.add_argument("--ddsr_mode", type=str, default="skip_only", choices=["skip_only", "propagate"],
                        help="Use DDSR only for decoder skips or propagate it through the encoder")
    parser.add_argument("--ddsr_aux_init", type=float, default=0.1,
                        help="Initial DDSR auxiliary residual blend")
    parser.add_argument("--uslgsf_stages", type=parse_uslgsf_stages, default=(0, 1),
                        help="Comma-separated US-LGSF skip stages, e.g. 0,1 or 0,1,2")
    parser.add_argument("--uslgsf_smooth_kernels", type=parse_uslgsf_smooth_kernels, default=(3, 7),
                        help="Small and large odd smoothing kernels for US-LGSF")
    parser.add_argument("--uslgsf_context_downsample", type=int, default=2,
                        help="Downsampling factor for the US-LGSF low-frequency context path")
    parser.add_argument("--uslgsf_alpha_init", type=float, default=0.05,
                        help="Initial effective US-LGSF residual blend")
    parser.add_argument("--uslgsf_alpha_max", type=float, default=0.5,
                        help="Maximum effective US-LGSF residual blend")
    parser.add_argument("--uslgsf_mode", type=str, default="full",
                        choices=["full", "context_only", "structure_only", "relevance_only"],
                        help="US-LGSF full model or a core ablation mode")
    parser.add_argument("--uslgsf_residual_init_scale", type=float, default=0.05,
                        help="Residual projection initialization scale for CMUNeXt_USLGSF_V3")
    parser.add_argument("--gag_stages", type=parse_gag_stages, default=(2, 3),
                        help="Comma-separated DualGAG stages, e.g. 0,1 or 1,3 or 0,1,2,3")
    parser.add_argument("--boundary_loss_weight", type=float, default=0.3,
                        help="Boundary loss weight for CMUNeXt_BA_DualGAG")
    parser.add_argument("--hspm_mode", type=str, default="full", choices=["full", "context_only"],
                        help="Enable the full HSPM or keep only its high-resolution context bottleneck")
    parser.add_argument("--hspm_coarse_loss_weight", type=float, default=0.3,
                        help="Auxiliary coarse segmentation loss weight for CMUNeXt_HSPM")
    parser.add_argument("--hspm_mixer_mode", type=str, default="legacy",
                        choices=["legacy", "bounded", "stable"],
                        help="Prototype mixer behavior for CMUNeXt_HSPM")
    parser.add_argument("--hspm_gamma_init", type=float, default=0.1,
                        help="Initial prototype residual strength")
    parser.add_argument("--hspm_gamma_max", type=float, default=0.3,
                        help="Maximum effective prototype residual strength in bounded/stable modes")
    parser.add_argument("--hspm_temperature", type=float, default=0.1,
                        help="Prototype assignment temperature")
    parser.add_argument("--hspm_prototype_dropout", type=float, default=0.0,
                        help="Dropout2d probability on stable prototype residuals")
    parser.add_argument("--hspm_backbone_mode", type=str, default="highres_only",
                        choices=["highres_only", "dual_path"],
                        help="Use the legacy high-resolution-only bottleneck or the dual-path CMUNeXt bottleneck")
    parser.add_argument("--hspm_fusion_gate_init", type=float, default=0.05,
                        help="Initial effective HSPM residual gate in dual-path mode")
    parser.add_argument("--hspm_fusion_gate_max", type=float, default=0.3,
                        help="Maximum effective HSPM residual gate in dual-path mode")
    parser.add_argument("--hspm_fusion_mode", type=str, default="global", choices=["global", "size_aware"],
                        help="Use global or size-aware spatial HSPM residual fusion in dual-path mode")
    parser.add_argument("--hspm_small_area_threshold", type=float, default=0.05,
                        help="Predicted coarse area threshold used by size-aware HSPM fusion")
    parser.add_argument("--hspm_small_area_temperature", type=float, default=0.02,
                        help="Smoothness of the small-lesion decision in size-aware HSPM fusion")
    parser.add_argument("--hspm_prototype_warmup_epochs", type=int, default=0,
                        help="Training-only compatibility option; inference always uses full prototype injection")
    parser.add_argument("--hspm_coarse_loss_final_weight", type=float, default=None,
                        help="Training-only compatibility option")
    parser.add_argument("--hspm_coarse_loss_decay_epochs", type=int, default=0,
                        help="Training-only compatibility option")
    parser.add_argument("--fbdm_no_hspm_prior", action="store_true",
                        help="Disable HSPM coarse/uncertainty priors inside CMUNeXt_HSPM_FBDM")
    parser.add_argument("--fbdm_no_detach_hspm_prior", action="store_true",
                        help="Allow FBDM gradients to flow into HSPM priors; default keeps priors detached")
    parser.add_argument("--fbdm_semantic_uncertainty_weight", type=float, default=0.7,
                        help="Uncertainty weight in the FBDM semantic boundary prior")
    parser.add_argument("--fbdm_semantic_coarse_weight", type=float, default=0.3,
                        help="Coarse probability weight in the FBDM semantic boundary prior")
    parser.add_argument("--fbdm_semantic_gate_base", type=float, default=0.7,
                        help="Conservative base term in the FBDM boundary gate")
    parser.add_argument("--fbdm_gate_init", type=float, default=0.03,
                        help="Initial effective FBDM residual strength")
    parser.add_argument("--fbdm_gate_max", type=float, default=0.2,
                        help="Maximum effective FBDM residual strength")
    parser.add_argument("--fbdm_edge_aux_only", action="store_true",
                        help="Use FBDM only as an edge auxiliary branch without residual injection")
    parser.add_argument("--fbdm_edge_loss_weight", type=float, default=0.05,
                        help="Auxiliary edge loss weight for FBDM models")
    parser.add_argument("--fbdm_edge_kernel_size", type=int, default=3,
                        help="Odd kernel size used to build edge supervision masks for FBDM models")
    parser.add_argument("--fbdm_correction_scale_init", type=float, default=0.05,
                        help="Initial effective bounded logit-correction scale for FBDM V2")
    parser.add_argument("--fbdm_correction_scale_max", type=float, default=0.3,
                        help="Maximum effective bounded logit-correction scale for FBDM V2")
    parser.add_argument("--fbdm_correction_warmup_epochs", type=int, default=40,
                        help="Training-only compatibility option; inference uses full correction scale")
    parser.add_argument("--fbdm_boundary_band_loss_weight", type=float, default=0.0,
                        help="Boundary-band loss weight for FBDM V2 evaluation")
    parser.add_argument("--fbdm_boundary_band_loss_final_weight", type=float, default=None,
                        help="Training-only compatibility option")
    parser.add_argument("--fbdm_boundary_band_loss_decay_epochs", type=int, default=0,
                        help="Training-only compatibility option")
    parser.add_argument("--fbdm_boundary_band_kernel_size", type=int, default=7,
                        help="Odd kernel size used to build GT boundary bands for FBDM V2 evaluation")
    parser.add_argument("--early_stop_patience", type=int, default=0,
                        help="Training-only compatibility option")
    parser.add_argument("--early_stop_min_delta", type=float, default=0.001,
                        help="Training-only compatibility option")
    parser.add_argument("--val_threshold_mode", type=str, default="fixed", choices=["fixed", "scan"],
                        help="Use a fixed validation threshold or scan a threshold range")
    parser.add_argument("--val_threshold", type=float, default=0.5,
                        help="Validation threshold when val_threshold_mode=fixed")
    parser.add_argument("--val_threshold_start", type=float, default=0.30,
                        help="Threshold scan start when val_threshold_mode=scan")
    parser.add_argument("--val_threshold_end", type=float, default=0.70,
                        help="Threshold scan end when val_threshold_mode=scan")
    parser.add_argument("--val_threshold_step", type=float, default=0.02,
                        help="Threshold scan step when val_threshold_mode=scan")
    parser.add_argument("--val_threshold_metric", type=str, default="iou", choices=["iou", "f1"],
                        help="Metric used to pick the best validation threshold")
    parser.add_argument("--visual_mode", type=str, default="none", choices=["none", "selected", "all"],
                        help="Save no visualizations, one selected case, or all cases")
    parser.add_argument("--visual_case", type=str, default=None,
                        help="Case name to visualize when visual_mode=selected")
    args = parser.parse_args()
    apply_best0616_presets(args)

    if args.fbdm_semantic_uncertainty_weight < 0 or args.fbdm_semantic_coarse_weight < 0:
        parser.error("FBDM semantic prior weights must be non-negative.")
    if not 0.0 <= args.fbdm_semantic_gate_base <= 1.0:
        parser.error("--fbdm_semantic_gate_base must be in [0, 1].")
    if not 0.0 < args.fbdm_gate_init < args.fbdm_gate_max:
        parser.error("--fbdm_gate_init must be in (0, --fbdm_gate_max).")
    if args.fbdm_edge_loss_weight < 0:
        parser.error("--fbdm_edge_loss_weight must be non-negative.")
    if args.fbdm_edge_kernel_size <= 0 or args.fbdm_edge_kernel_size % 2 == 0:
        parser.error("--fbdm_edge_kernel_size must be a positive odd integer.")
    if not 0.0 < args.fbdm_correction_scale_init < args.fbdm_correction_scale_max:
        parser.error("--fbdm_correction_scale_init must be in (0, --fbdm_correction_scale_max).")
    if args.fbdm_correction_warmup_epochs < 0:
        parser.error("--fbdm_correction_warmup_epochs must be non-negative.")
    if args.fbdm_boundary_band_loss_weight < 0:
        parser.error("--fbdm_boundary_band_loss_weight must be non-negative.")
    if (
        args.fbdm_boundary_band_loss_final_weight is not None
        and args.fbdm_boundary_band_loss_final_weight < 0
    ):
        parser.error("--fbdm_boundary_band_loss_final_weight must be non-negative.")
    if args.fbdm_boundary_band_loss_decay_epochs < 0:
        parser.error("--fbdm_boundary_band_loss_decay_epochs must be non-negative.")
    if args.fbdm_boundary_band_kernel_size <= 0 or args.fbdm_boundary_band_kernel_size % 2 == 0:
        parser.error("--fbdm_boundary_band_kernel_size must be a positive odd integer.")
    boundary_band_enabled = args.fbdm_boundary_band_loss_weight > 0 or (
        args.fbdm_boundary_band_loss_final_weight is not None
        and args.fbdm_boundary_band_loss_final_weight > 0
    )
    if boundary_band_enabled and args.model not in HSPM_FBDM_V2_MODELS:
        parser.error("--fbdm_boundary_band_loss_weight is only supported by CMUNeXt_HSPM_FBDM_V2.")
    if args.visual_mode == "selected" and not args.visual_case:
        parser.error("--visual_case is required when --visual_mode selected")

    if args.model == "SwinUnet" and args.img_size == 256:
        print("SwinUnet requires 224x224 input, adjusting img_size.")
        args.img_size = 224

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model_path, args, device, parser)

    val_transform = get_val_transform(args.img_size)
    db_val = MedicalDataSets(
        base_dir=args.base_dir,
        split="val",
        transform=val_transform,
        train_file_dir=args.train_file_dir,
        val_file_dir=args.val_file_dir,
        divide_image_by_255=args.model == "CMUNeXt",
    )
    val_loader = DataLoader(db_val, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    if args.model in HSPM_FBDM_MODELS:
        criterion = losses.__dict__["HSPMFBDMLoss"](
            coarse_weight=args.hspm_coarse_loss_weight,
            edge_weight=args.fbdm_edge_loss_weight,
            edge_kernel_size=args.fbdm_edge_kernel_size,
            boundary_band_weight=args.fbdm_boundary_band_loss_weight,
            boundary_band_kernel_size=args.fbdm_boundary_band_kernel_size,
        ).to(device)
    elif args.model in FBDM_ONLY_MODELS:
        criterion = losses.__dict__["FBDMLoss"](
            edge_weight=args.fbdm_edge_loss_weight,
            edge_kernel_size=args.fbdm_edge_kernel_size,
        ).to(device)
    elif args.model in HSPM_ONLY_MODELS:
        criterion = losses.__dict__["HSPMLoss"](coarse_weight=args.hspm_coarse_loss_weight).to(device)
    elif args.model in {"CMUNeXt_BA_DualGAG", "CMUNeXt_BA_DualGAG_SpeckleEnhance"}:
        criterion = losses.__dict__["BoundaryAwareSegLoss"](lambda_b=args.boundary_loss_weight).to(device)
    else:
        criterion = losses.__dict__["BCEDiceLoss"]().to(device)
    save_directory = os.path.join(os.path.dirname(args.model_path), f"predictions_{args.model}")
    print(f"Saving predictions to: {save_directory}")
    validate(model, val_loader, criterion, device, args=args, save_dir=save_directory)
