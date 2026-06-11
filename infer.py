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


def build_model(args, parser):
    if args.model == "CMUNet":
        model = CMUNet(output_ch=args.num_classes)
    elif args.model == "MK_UNet":
        model = MK_UNet(num_classes=args.num_classes, in_channels=3)
    elif args.model == "CMUNeXt":
        model = cmunext(num_classes=args.num_classes)
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
        "CMUNet", "CMUNeXt", "CMUNeXt_DualGAG", "CMUNeXt_BA_DualGAG",
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
    parser.add_argument("--gag_stages", type=parse_gag_stages, default=(2, 3),
                        help="Comma-separated DualGAG stages, e.g. 0,1 or 1,3 or 0,1,2,3")
    parser.add_argument("--boundary_loss_weight", type=float, default=0.3,
                        help="Boundary loss weight for CMUNeXt_BA_DualGAG")
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
    )
    val_loader = DataLoader(db_val, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    if args.model in {"CMUNeXt_BA_DualGAG", "CMUNeXt_BA_DualGAG_SpeckleEnhance"}:
        criterion = losses.__dict__["BoundaryAwareSegLoss"](lambda_b=args.boundary_loss_weight).to(device)
    else:
        criterion = losses.__dict__["BCEDiceLoss"]().to(device)
    save_directory = os.path.join(os.path.dirname(args.model_path), f"predictions_{args.model}")
    print(f"Saving predictions to: {save_directory}")
    validate(model, val_loader, criterion, device, args=args, save_dir=save_directory)
