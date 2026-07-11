"""
推理性能基准测试脚本
用于测量模型的推理时延、FPS、显存占用等指标，补充论文中"轻量化"主张所需的实测数据。

使用方法：
1. 快速测试（合成数据）：
   python benchmark_inference.py --mode synthetic --num_samples 100

2. 真实数据测试：
   python benchmark_inference.py --mode real --base_dir ./data/busi --val_file_dir busi_val.txt --num_samples 100

3. 批量测试多个模型：
   python benchmark_inference.py --mode synthetic --model_dir ./checkpoint --models CMUNeXt,CMUNeXt_BARM,CMUNeXt_HSPM_BARM
"""

import argparse
import os
import time
import json
import csv
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from albumentations import Compose, Normalize, Resize

from src.dataloader.dataset import MedicalDataSets
from src.network.conv_based.U_Net import U_Net
from src.network.conv_based.AttU_Net import AttU_Net
from src.network.conv_based.CMUNet import CMUNet
from src.network.conv_based.CMUNeXt import cmunext
from src.network.conv_based.CMUNeXt_BARM import cmunext_barm
from src.network.conv_based.CMUNeXt_HSPM import cmunext_hspm
from src.network.conv_based.CMUNeXt_HSPM_BARM import cmunext_hspm_barm, cmunext_hspm_barm_hfbypass
from src.network.conv_based.CMUNeXt_HSPM_Best0616 import cmunext_hspm_best0616
from src.network.conv_based.CMUNeXt_HSPM_Best0619 import cmunext_hspm_best0619
from src.network.conv_based.UNeXt import UNext
from src.network.conv_based.UNetplus import ResNet34UnetPlus
from src.network.conv_based.UNet3plus import UNet3plus
from src.network.conv_based.MA_UNet import ma_unet
from src.network.conv_based.MK_UNet import MK_UNet
from src.network.conv_based.TM_UNet import tm_unet
from src.network.hybrid_based.Mobile_U_ViT import mobileuvit
from src.network.transfomer_based.transUnet.transunet import TransUnet
from src.network.transfomer_based.swinUnet.config import get_config as get_swin_config
from src.network.transfomer_based.swinUnet.vision_transformer import SwinUnet


HSPM_BEST0616_MODEL = "CMUNeXt_HSPM_Best0616"
HSPM_BEST0619_MODEL = "CMUNeXt_HSPM_Best0619"
HSPM_BARM_MODEL = "CMUNeXt_HSPM_BARM"
HSPM_BARM_HFBYPASS_MODEL = "CMUNeXt_HSPM_BARM_HFBypass"
TRANSUNET_MODEL = "TransUnet"
SWIN_UNET_MODEL = "SwinUnet"
TRANSUNET_INPUT_SIZE = 256
SWIN_UNET_INPUT_SIZE = 224


def get_model_img_size(model_name, default_img_size):
    """返回固定输入架构所需的图像尺寸。"""
    if model_name == TRANSUNET_MODEL:
        return TRANSUNET_INPUT_SIZE
    if model_name == SWIN_UNET_MODEL:
        return SWIN_UNET_INPUT_SIZE
    return default_img_size


def build_swin_unet(num_classes=1):
    """使用仓库内的 224x224 配置构建 Swin-Unet。"""
    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "src", "network", "transfomer_based", "swinUnet",
        "swin_tiny_patch4_window7_224_lite.yaml",
    )
    config_args = argparse.Namespace(
        cfg=config_path,
        opts=None,
        batch_size=None,
        zip=False,
        cache_mode="part",
        resume=None,
        accumulation_steps=None,
        use_checkpoint=False,
        amp_opt_level="O1",
        tag=None,
        eval=False,
        throughput=False,
    )
    config = get_swin_config(config_args)
    return SwinUnet(config, img_size=SWIN_UNET_INPUT_SIZE, num_classes=num_classes)


def build_model(model_name, num_classes=1, img_size=256):
    """根据模型名称构建模型实例"""
    if model_name == "CMUNet":
        return CMUNet(output_ch=num_classes)
    elif model_name == "CMUNeXt":
        return cmunext(num_classes=num_classes)
    elif model_name == "CMUNeXt_BARM":
        return cmunext_barm(num_classes=num_classes)
    elif model_name == HSPM_BARM_MODEL:
        return cmunext_hspm_barm(num_classes=num_classes)
    elif model_name == HSPM_BARM_HFBYPASS_MODEL:
        return cmunext_hspm_barm_hfbypass(num_classes=num_classes)
    elif model_name == "CMUNeXt_HSPM":
        return cmunext_hspm(num_classes=num_classes)
    elif model_name == HSPM_BEST0616_MODEL:
        return cmunext_hspm_best0616(num_classes=num_classes)
    elif model_name == HSPM_BEST0619_MODEL:
        return cmunext_hspm_best0619(num_classes=num_classes)
    elif model_name == "U_Net":
        return U_Net(output_ch=num_classes)
    elif model_name == "AttU_Net":
        return AttU_Net(output_ch=num_classes)
    elif model_name == "UNext":
        return UNext(num_classes=num_classes)
    elif model_name == "UNetplus":
        return ResNet34UnetPlus(num_class=num_classes)
    elif model_name == "UNet3plus":
        return UNet3plus(n_classes=num_classes)
    elif model_name == "MA_UNet":
        return ma_unet(input_channel=3, num_classes=num_classes)
    elif model_name == "MK_UNet":
        return MK_UNet(num_classes=num_classes, in_channels=3)
    elif model_name == "TM_UNet":
        return tm_unet(input_channel=3, num_classes=num_classes, img_size=img_size, model_size="b")
    elif model_name == "Mobile_U_ViT":
        return mobileuvit(out_channel=num_classes)
    elif model_name == TRANSUNET_MODEL:
        return TransUnet(img_ch=3, output_ch=num_classes)
    elif model_name == SWIN_UNET_MODEL:
        return build_swin_unet(num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def count_parameters(model):
    """统计模型参数量（百万）"""
    return sum(p.numel() for p in model.parameters()) / 1e6


def measure_memory(model, input_tensor, device):
    """测量GPU峰值显存占用（MB）"""
    if device.type == "cpu":
        return 0.0

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()

    with torch.no_grad():
        _ = model(input_tensor)

    torch.cuda.synchronize()
    peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # MB
    return peak_memory


def benchmark_model(model, dataloader, device, warmup_iters=10, test_iters=100):
    """
    对单个模型进行性能基准测试

    Args:
        model: 待测试模型
        dataloader: 数据加载器
        device: 运行设备
        warmup_iters: 预热迭代次数
        test_iters: 测试迭代次数

    Returns:
        dict: 包含推理时延、FPS、显存等指标的字典
    """
    model.eval()

    # 预热阶段
    print(f"  Warming up ({warmup_iters} iterations)...")
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= warmup_iters:
                break
            if isinstance(batch, dict):
                img_batch = batch["image"].to(device)
            else:
                img_batch = batch[0].to(device)
            _ = model(img_batch)
            if device.type == "cuda":
                torch.cuda.synchronize()

    # 正式测试阶段
    print(f"  Testing ({test_iters} iterations)...")
    inference_times = []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, total=test_iters, desc="  Benchmarking")):
            if i >= test_iters:
                break

            if isinstance(batch, dict):
                img_batch = batch["image"].to(device)
            else:
                img_batch = batch[0].to(device)

            batch_size = img_batch.size(0)

            # 测量推理时间
            if device.type == "cuda":
                torch.cuda.synchronize()
            start_time = time.time()

            _ = model(img_batch)

            if device.type == "cuda":
                torch.cuda.synchronize()
            end_time = time.time()

            # 记录单张图片的推理时间
            per_image_time = (end_time - start_time) / batch_size
            inference_times.append(per_image_time)

    # 测量显存占用
    if isinstance(next(iter(dataloader)), dict):
        sample_input = next(iter(dataloader))["image"][:1].to(device)
    else:
        sample_input = next(iter(dataloader))[0][:1].to(device)
    peak_memory = measure_memory(model, sample_input, device)

    # 统计指标
    inference_times = np.array(inference_times)
    results = {
        "mean_latency_ms": float(np.mean(inference_times) * 1000),
        "std_latency_ms": float(np.std(inference_times) * 1000),
        "min_latency_ms": float(np.min(inference_times) * 1000),
        "max_latency_ms": float(np.max(inference_times) * 1000),
        "median_latency_ms": float(np.median(inference_times) * 1000),
        "p95_latency_ms": float(np.percentile(inference_times, 95) * 1000),
        "p99_latency_ms": float(np.percentile(inference_times, 99) * 1000),
        "fps": float(1.0 / np.mean(inference_times)),
        "peak_memory_mb": float(peak_memory),
    }

    return results


def create_synthetic_dataloader(batch_size, img_size, num_samples, device):
    """创建合成数据加载器（用于快速测试）"""
    images = torch.randn(num_samples, 3, img_size, img_size)
    dataset = TensorDataset(images)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    return dataloader


def create_real_dataloader(base_dir, val_file_dir, batch_size, img_size, num_samples):
    """创建真实数据加载器"""
    transform = Compose([
        Resize(img_size, img_size),
        Normalize(),
    ])

    dataset = MedicalDataSets(
        base_dir=base_dir,
        split="val",
        transform=transform,
        train_file_dir="train.txt",  # 必需参数，即使不使用
        val_file_dir=val_file_dir,
        divide_image_by_255=False,
    )

    # 限制样本数量
    if num_samples > 0 and num_samples < len(dataset):
        indices = np.random.choice(len(dataset), num_samples, replace=False)
        dataset = torch.utils.data.Subset(dataset, indices)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return dataloader


def find_model_checkpoint(model_name, model_dir):
    """在指定目录中查找模型权重文件"""
    # 优先查找 best 模型
    best_path = os.path.join(model_dir, f"{model_name}_model.pth")
    if os.path.exists(best_path):
        return best_path

    # 查找 last 模型
    last_path = os.path.join(model_dir, f"{model_name}_model_last.pth")
    if os.path.exists(last_path):
        return last_path

    # 递归查找子目录
    for root, dirs, files in os.walk(model_dir):
        for file in files:
            if file == f"{model_name}_model.pth" or file == f"{model_name}_model_last.pth":
                return os.path.join(root, file)

    return None


def save_results(results, output_dir):
    """保存测试结果为CSV和JSON"""
    os.makedirs(output_dir, exist_ok=True)

    # 保存为 CSV
    csv_path = os.path.join(output_dir, "benchmark_results.csv")
    fieldnames = [
        "model", "device", "batch_size", "img_size", "params_M",
        "mean_latency_ms", "std_latency_ms", "fps", "peak_memory_mb"
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = {k: result[k] for k in fieldnames if k in result}
            writer.writerow(row)

    print(f"\n=> Saved CSV results to: {csv_path}")

    # 保存为 JSON
    json_path = os.path.join(output_dir, "benchmark_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"=> Saved JSON results to: {json_path}")


def plot_results(results, output_dir):
    """绘制性能对比图表"""
    os.makedirs(output_dir, exist_ok=True)

    models = [r["model"] for r in results]
    fps = [r["fps"] for r in results]
    latency = [r["mean_latency_ms"] for r in results]
    memory = [r["peak_memory_mb"] for r in results]
    params = [r["params_M"] for r in results]

    # 图1: FPS 对比
    plt.figure(figsize=(12, 6))
    plt.bar(models, fps, color="steelblue")
    plt.xlabel("Model")
    plt.ylabel("FPS (Frames Per Second)")
    plt.title("Inference Throughput Comparison")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fps_comparison.png"), dpi=300)
    plt.close()

    # 图2: 推理时延对比
    plt.figure(figsize=(12, 6))
    plt.bar(models, latency, color="coral")
    plt.xlabel("Model")
    plt.ylabel("Latency (ms)")
    plt.title("Inference Latency Comparison")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "latency_comparison.png"), dpi=300)
    plt.close()

    # 图3: 显存占用对比
    plt.figure(figsize=(12, 6))
    plt.bar(models, memory, color="mediumseagreen")
    plt.xlabel("Model")
    plt.ylabel("Peak Memory (MB)")
    plt.title("GPU Memory Usage Comparison")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "memory_comparison.png"), dpi=300)
    plt.close()

    # 图4: 参数量 vs FPS 散点图
    plt.figure(figsize=(10, 6))
    plt.scatter(params, fps, s=100, alpha=0.6, c=memory, cmap="viridis")
    for i, model in enumerate(models):
        plt.annotate(model, (params[i], fps[i]), fontsize=8, alpha=0.7)
    plt.xlabel("Parameters (M)")
    plt.ylabel("FPS")
    plt.title("Parameters vs Throughput")
    plt.colorbar(label="Peak Memory (MB)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "params_vs_fps.png"), dpi=300)
    plt.close()

    print(f"=> Saved plots to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="推理性能基准测试")

    # 测试模式
    parser.add_argument("--mode", type=str, default="synthetic", choices=["synthetic", "real"],
                        help="测试模式：synthetic(合成数据，快速) 或 real(真实数据)")

    # 模型相关
    parser.add_argument("--models", type=str, default="CMUNeXt",
                        help="要测试的模型列表，逗号分隔，例如: CMUNeXt,CMUNeXt_BARM,U_Net")
    parser.add_argument("--model_dir", type=str, default="./checkpoint",
                        help="模型权重文件所在目录")
    parser.add_argument("--no_pretrained", action="store_true",
                        help="不加载预训练权重，仅测试随机初始化模型的推理速度")

    # 数据相关
    parser.add_argument("--base_dir", type=str, default="./data/busi",
                        help="真实数据集根目录（仅 mode=real 时需要）")
    parser.add_argument("--val_file_dir", type=str, default="busi_val.txt",
                        help="验证集文件列表（仅 mode=real 时需要）")
    parser.add_argument("--img_size", type=int, default=256,
                        help="默认输入图像尺寸（TransUnet 固定为 256，SwinUnet 固定为 224）")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="测试样本数量")

    # 测试配置
    parser.add_argument("--batch_size", type=int, default=1,
                        help="批量大小（论文通常报告 batch_size=1 的推理时延）")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                        help="运行设备")
    parser.add_argument("--warmup_iters", type=int, default=10,
                        help="预热迭代次数")
    parser.add_argument("--test_iters", type=int, default=100,
                        help="测试迭代次数")

    # 输出
    parser.add_argument("--output_dir", type=str, default="./.codex-scratch",
                        help="结果输出目录")

    args = parser.parse_args()

    # 解析模型列表
    model_names = [m.strip() for m in args.models.split(",")]

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"\n{'='*60}")
    print(f"Inference Benchmark Test")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Mode: {args.mode}")
    print(f"Batch size: {args.batch_size}")
    print(f"Default image size: {args.img_size}x{args.img_size}")
    print(f"Test samples: {args.num_samples}")
    print(f"Models: {', '.join(model_names)}")
    print(f"{'='*60}\n")

    # 测试每个模型
    all_results = []
    dataloaders = {}

    for model_name in model_names:
        model_img_size = get_model_img_size(model_name, args.img_size)
        print(f"\n{'='*60}")
        print(f"Testing model: {model_name}")
        print(f"Input size: {model_img_size}x{model_img_size}")
        print(f"{'='*60}")

        try:
            # 按输入尺寸复用数据加载器；SwinUnet 使用独立的 224x224 输入。
            if model_img_size not in dataloaders:
                print(f"Creating {args.mode} dataloader for {model_img_size}x{model_img_size} inputs...")
                if args.mode == "synthetic":
                    dataloaders[model_img_size] = create_synthetic_dataloader(
                        args.batch_size, model_img_size, args.num_samples, device
                    )
                else:
                    dataloaders[model_img_size] = create_real_dataloader(
                        args.base_dir,
                        args.val_file_dir,
                        args.batch_size,
                        model_img_size,
                        args.num_samples,
                    )
            dataloader = dataloaders[model_img_size]

            # 构建模型
            model = build_model(model_name, num_classes=1, img_size=model_img_size)

            # 加载预训练权重
            if not args.no_pretrained:
                checkpoint_path = find_model_checkpoint(model_name, args.model_dir)
                if checkpoint_path:
                    print(f"Loading checkpoint: {checkpoint_path}")
                    # 先加载到 CPU，避免临时 state_dict 占用 GPU 并污染峰值显存统计。
                    state_dict = torch.load(checkpoint_path, map_location="cpu")
                    model.load_state_dict(state_dict)
                    del state_dict
                else:
                    print(f"Warning: No checkpoint found for {model_name}, using random weights")
            else:
                print("Using random weights (no pretrained)")

            model.to(device)
            model.eval()

            # 统计参数量
            params = count_parameters(model)
            print(f"Parameters: {params:.2f}M")

            # 运行基准测试
            results = benchmark_model(
                model, dataloader, device,
                warmup_iters=args.warmup_iters,
                test_iters=args.test_iters
            )

            # 添加模型信息
            results["model"] = model_name
            results["device"] = str(device)
            results["batch_size"] = args.batch_size
            results["img_size"] = model_img_size
            results["params_M"] = params

            # 打印结果
            print(f"\nResults for {model_name}:")
            print(f"  Mean Latency: {results['mean_latency_ms']:.2f} ± {results['std_latency_ms']:.2f} ms")
            print(f"  Median Latency: {results['median_latency_ms']:.2f} ms")
            print(f"  P95 Latency: {results['p95_latency_ms']:.2f} ms")
            print(f"  FPS: {results['fps']:.2f}")
            print(f"  Peak Memory: {results['peak_memory_mb']:.2f} MB")

            all_results.append(results)

            # 清理显存
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error testing {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 保存结果
    if all_results:
        # 创建带时间戳的输出目录
        timestamp = time.strftime("%Y%m%d-%H%M")
        output_dir = os.path.join(args.output_dir, f"{timestamp}-benchmark")

        save_results(all_results, output_dir)
        plot_results(all_results, output_dir)

        # 打印汇总表格
        print(f"\n{'='*60}")
        print("Summary Table")
        print(f"{'='*60}")
        print(f"{'Model':<30} {'Input':>9} {'FPS':>8} {'Latency(ms)':>12} {'Memory(MB)':>12}")
        print(f"{'-'*71}")
        for result in all_results:
            input_size = f"{result['img_size']}x{result['img_size']}"
            print(f"{result['model']:<30} {input_size:>9} {result['fps']:>8.2f} "
                  f"{result['mean_latency_ms']:>12.2f} {result['peak_memory_mb']:>12.2f}")
        print(f"{'='*71}\n")

        print(f"[SUCCESS] All results saved to: {output_dir}")
    else:
        print("\n[ERROR] No successful results to save.")


if __name__ == "__main__":
    main()
