import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from albumentations import Compose, Normalize, Resize
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataloader.dataset import MedicalDataSets
from src.network.conv_based.CMUNeXt_HSPM_APBR import cmunext_hspm_apbr
from src.network.conv_based.CMUNeXt_HSPM_APBR_V2 import cmunext_hspm_apbr_v2
from src.utils.metrics import boundary_scores, iou_score


MODES = {
    "baseline": frozenset(),
    "oracle_half": frozenset({"half"}),
    "oracle_full": frozenset({"full"}),
    "oracle_half_full": frozenset({"half", "full"}),
}
COMMON_DIAGNOSTIC_NAMES = (
    "ambiguity_gate_mean",
    "oracle_recovery_target_mean",
    "oracle_recovery_gate_mean",
    "combined_gate_mean",
    "recovery_added_mean",
    "recovery_dominant_ratio",
)
V2_DIAGNOSTIC_NAMES = (
    "base_probability_mean",
    "feature_logit_delta_abs_mean",
    "correction_logit_abs_mean",
    "correction_logit_abs_p99",
    "correction_logit_abs_max",
    "total_logit_delta_abs_mean",
    "effective_feature_scale",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate frozen APBR v1/v2 models with Oracle Recovery routing."
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def resolve_path(path):
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def load_config(model_path):
    config_path = model_path.parent / "training_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Training config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as config_file:
        return json.load(config_file)


def build_model(config):
    model_name = config.get("model", "CMUNeXt_HSPM_APBR")
    factories = {
        "CMUNeXt_HSPM_APBR": cmunext_hspm_apbr,
        "CMUNeXt_HSPM_APBR_V2": cmunext_hspm_apbr_v2,
    }
    if model_name not in factories:
        raise ValueError(f"Unsupported Oracle Recovery model: {model_name}")
    return factories[model_name](
        num_classes=config.get("num_classes", 1),
        dims=tuple(config.get("hspm_dims", (16, 32, 128, 160, 256))),
        hspm_mode=config.get("hspm_mode", "full"),
        hspm_mixer_mode=config.get("hspm_mixer_mode", "legacy"),
        hspm_gamma_init=config.get("hspm_gamma_init", 0.1),
        hspm_gamma_max=config.get("hspm_gamma_max", 0.3),
        hspm_temperature=config.get("hspm_temperature", 0.1),
        hspm_prototype_dropout=config.get("hspm_prototype_dropout", 0.0),
        hspm_fusion_gate_init=config.get("hspm_fusion_gate_init", 0.05),
        hspm_fusion_gate_max=config.get("hspm_fusion_gate_max", 0.3),
        hspm_fusion_mode=config.get("hspm_fusion_mode", "global"),
        hspm_small_area_threshold=config.get("hspm_small_area_threshold", 0.05),
        hspm_small_area_temperature=config.get("hspm_small_area_temperature", 0.02),
        apbr_mode=config.get("apbr_mode", "full"),
    )


def select_device(requested_device):
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(requested_device)


def load_checkpoint(model, model_path, device):
    state_dict = torch.load(model_path, map_location=device)
    if state_dict and all(name.startswith("module.") for name in state_dict):
        state_dict = {
            name[len("module."):]: value
            for name, value in state_dict.items()
        }
    model.load_state_dict(state_dict, strict=True)


def build_loader(config, batch_size, num_workers):
    base_dir = resolve_path(config["base_dir"])
    image_size = int(config.get("img_size", 256))
    dataset = MedicalDataSets(
        base_dir=str(base_dir),
        split="val",
        transform=Compose([Resize(image_size, image_size), Normalize()]),
        train_file_dir=config.get("train_file_dir", "train.txt"),
        val_file_dir=config.get("val_file_dir", "val.txt"),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )


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


def update_diagnostics(
    diagnostic_totals,
    model,
    mode_name,
    batch_size,
    diagnostic_names,
):
    diagnostics = model.get_apbr_diagnostics()
    for stage_name in ("half", "full"):
        for diagnostic_name in diagnostic_names:
            key = f"{stage_name}_{diagnostic_name}"
            diagnostic_totals[(mode_name, stage_name, diagnostic_name)] += (
                float(diagnostics[key].detach().cpu()) * batch_size
            )


def evaluate(model, loader, device, threshold, diagnostic_names):
    per_mode_rows = {mode_name: [] for mode_name in MODES}
    diagnostic_totals = defaultdict(float)
    diagnostic_count = 0

    model.eval()
    model.set_apbr_route_scale(1.0)
    with torch.no_grad():
        for sampled_batch in loader:
            images = sampled_batch["image"].to(device)
            targets = sampled_batch["label"].to(device)
            case_names = sampled_batch["case_name"]
            batch_size = int(images.shape[0])
            diagnostic_count += batch_size

            for mode_name, oracle_stages in MODES.items():
                if oracle_stages:
                    outputs = model.forward_oracle_recovery(
                        images,
                        targets,
                        oracle_stages=oracle_stages,
                    )
                else:
                    outputs = model(images)
                probabilities = torch.sigmoid(outputs["seg"]).detach().cpu()
                cpu_targets = targets.detach().cpu()
                update_diagnostics(
                    diagnostic_totals,
                    model,
                    mode_name,
                    batch_size,
                    diagnostic_names,
                )

                for probability, target, case_name in zip(
                    probabilities,
                    cpu_targets,
                    case_names,
                ):
                    per_mode_rows[mode_name].append(
                        {
                            "case_name": case_name,
                            **compute_case_metrics(
                                probability,
                                target,
                                threshold,
                            ),
                        }
                    )

    return per_mode_rows, diagnostic_totals, diagnostic_count


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_summary(per_mode_rows, threshold):
    summary_rows = []
    for mode_name, rows in per_mode_rows.items():
        count = max(len(rows), 1)
        summary_rows.append(
            {
                "mode": mode_name,
                "threshold": threshold,
                "case_count": len(rows),
                "mean_iou": sum(row["iou"] for row in rows) / count,
                "mean_hd95": sum(row["hd95"] for row in rows) / count,
                "mean_assd": sum(row["assd"] for row in rows) / count,
                "iou_below_0_1_count": sum(row["iou"] < 0.1 for row in rows),
                "iou_zero_count": sum(round(row["iou"], 6) == 0.0 for row in rows),
            }
        )
    return summary_rows


def build_per_case_comparison(per_mode_rows):
    indexed_rows = {
        mode_name: {row["case_name"]: row for row in rows}
        for mode_name, rows in per_mode_rows.items()
    }
    baseline_rows = per_mode_rows["baseline"]
    comparison_rows = []
    for baseline_row in baseline_rows:
        case_name = baseline_row["case_name"]
        output_row = {"case_name": case_name}
        for mode_name in MODES:
            mode_iou = indexed_rows[mode_name][case_name]["iou"]
            output_row[f"{mode_name}_iou"] = mode_iou
            if mode_name != "baseline":
                output_row[f"{mode_name}_iou_delta"] = (
                    mode_iou - baseline_row["iou"]
                )
        comparison_rows.append(output_row)
    return comparison_rows


def build_route_diagnostics(
    diagnostic_totals,
    diagnostic_count,
    diagnostic_names,
):
    rows = []
    for mode_name in MODES:
        for stage_name in ("half", "full"):
            row = {"mode": mode_name, "stage": stage_name}
            for diagnostic_name in diagnostic_names:
                row[diagnostic_name] = (
                    diagnostic_totals[(mode_name, stage_name, diagnostic_name)]
                    / max(diagnostic_count, 1)
                )
            rows.append(row)
    return rows


def print_summary(summary_rows):
    print("\nOracle Recovery summary")
    for row in summary_rows:
        print(
            f"{row['mode']}: mean_iou={row['mean_iou']:.4f}, "
            f"low_iou={row['iou_below_0_1_count']}, "
            f"zero_iou={row['iou_zero_count']}, "
            f"hd95={row['mean_hd95']:.4f}, assd={row['mean_assd']:.4f}"
        )


def main():
    args = parse_args()
    model_path = resolve_path(args.model_path)
    config = load_config(model_path)
    output_dir = (
        resolve_path(args.output_dir)
        if args.output_dir is not None
        else model_path.parent / "oracle_recovery_eval"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    threshold = (
        float(args.threshold)
        if args.threshold is not None
        else float(config.get("val_threshold", 0.5))
    )
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be in (0, 1).")
    batch_size = (
        int(args.batch_size)
        if args.batch_size is not None
        else int(config.get("batch_size", 8))
    )
    if batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative.")

    device = select_device(args.device)
    model = build_model(config).to(device)
    diagnostic_names = COMMON_DIAGNOSTIC_NAMES
    if config.get("model") == "CMUNeXt_HSPM_APBR_V2":
        diagnostic_names = (*diagnostic_names, *V2_DIAGNOSTIC_NAMES)
    load_checkpoint(model, model_path, device)
    loader = build_loader(config, batch_size, args.num_workers)
    per_mode_rows, diagnostic_totals, diagnostic_count = evaluate(
        model,
        loader,
        device,
        threshold,
        diagnostic_names,
    )

    summary_rows = build_summary(per_mode_rows, threshold)
    comparison_rows = build_per_case_comparison(per_mode_rows)
    diagnostic_rows = build_route_diagnostics(
        diagnostic_totals,
        diagnostic_count,
        diagnostic_names,
    )

    write_csv(
        output_dir / "summary.csv",
        summary_rows,
        (
            "mode",
            "threshold",
            "case_count",
            "mean_iou",
            "mean_hd95",
            "mean_assd",
            "iou_below_0_1_count",
            "iou_zero_count",
        ),
    )
    comparison_fields = ["case_name", "baseline_iou"]
    for mode_name in ("oracle_half", "oracle_full", "oracle_half_full"):
        comparison_fields.extend(
            [f"{mode_name}_iou", f"{mode_name}_iou_delta"]
        )
    write_csv(
        output_dir / "per_case_comparison.csv",
        comparison_rows,
        comparison_fields,
    )
    write_csv(
        output_dir / "route_diagnostics.csv",
        diagnostic_rows,
        ("mode", "stage", *diagnostic_names),
    )

    print_summary(summary_rows)
    print(f"\nSaved Oracle Recovery evaluation to: {output_dir}")


if __name__ == "__main__":
    main()
