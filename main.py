import os
import argparse
import json
import random
import cv2
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
import time
import logging  # <=== 新增 1: 导入 logging 模块
import sys  # <=== 新增 2: 导入 sys 模块
import matplotlib.pyplot as plt  # <=== 新增 3: 导入 matplotlib

from torch.utils.data import DataLoader
from src.dataloader.dataset import (
    MedicalDataSets,
    TARGETED_SAMPLING_GROUPS,
    TrackingWeightedRandomSampler,
    build_targeted_sampling_metadata,
)
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from albumentations import HorizontalFlip, RandomRotate90, Resize, RandomBrightnessContrast, \
    OneOf, RandomGamma, GaussianBlur, MultiplicativeNoise, ShiftScaleRotate

import src.utils.losses as losses
from src.utils.util import AverageMeter
from src.utils.metrics import iou_score, boundary_scores, find_best_threshold

from src.network.conv_based.CMUNet import CMUNet
from src.network.conv_based.U_Net import U_Net
from src.network.conv_based.AttU_Net import AttU_Net
from src.network.conv_based.UNeXt import UNext
from src.network.conv_based.UNetplus import ResNet34UnetPlus
from src.network.conv_based.UNet3plus import UNet3plus
from src.network.conv_based.CMUNeXt import cmunext
from src.network.conv_based.CMUNeXt_BARM import cmunext_barm
from src.network.conv_based.CMUNeXt_FBDM import cmunext_fbdm
from src.network.conv_based.CMUNeXt_FBDM_Best0616 import cmunext_fbdm_best0616
from src.network.conv_based.CMUNeXt_USLGSF import cmunext_uslgsf
from src.network.conv_based.CMUNeXt_USLGSF_V2 import cmunext_uslgsf_v2
from src.network.conv_based.CMUNeXt_USLGSF_V3 import cmunext_uslgsf_v3
from src.network.conv_based.CMUNeXt_HSPM import cmunext_hspm
from src.network.conv_based.CMUNeXt_HSPM_Best0616 import cmunext_hspm_best0616
from src.network.conv_based.CMUNeXt_HSPM_Best0619 import cmunext_hspm_best0619
from src.network.conv_based.CMUNeXt_HSPM_FBDM import cmunext_hspm_fbdm, cmunext_hspm_fbdm_v2
from src.network.conv_based.CMUNeXt_HSPM_FBDM_Best0616 import cmunext_hspm_fbdm_best0616
from src.network.conv_based.CMUNeXt_HSPM_FBDM_0619 import cmunext_hspm_fbdm_0619
from src.network.conv_based.CMUNeXt_BA_DualGAG import cmunext_ba_dualgag
from src.network.conv_based.CMUNeXt_BA_DualGAG_SpeckleEnhance import cmunext_ba_dualgag_speckleenhance
from src.network.conv_based.CMUNeXt_DualGAG import cmunext_dualgag
from src.network.conv_based.CMUNeXt_SpeckleEnhance import cmunext_speckle
from src.network.conv_based.CMUNeXt_DualGAG_SpeckleEnhance import cmunext_dualgag_speckleenhance
from src.network.conv_based.MK_UNet import MK_UNet



from src.network.transfomer_based.transformer_based_network import get_transformer_based_model

from src.network.hybrid_based.Mobile_U_ViT import mobileuvit, mobileuvit_l


HSPM_FBDM_V2_MODELS = {"CMUNeXt_HSPM_FBDM_V2"}
HSPM_FBDM_BEST0616_MODEL = "CMUNeXt_HSPM_FBDM_Best0616"
HSPM_FBDM_0619_MODEL = "CMUNeXt_HSPM_FBDM_0619"
HSPM_FBDM_CORRECTION_MODELS = {
    *HSPM_FBDM_V2_MODELS,
    HSPM_FBDM_BEST0616_MODEL,
}
HSPM_FBDM_MODELS = {
    "CMUNeXt_HSPM_FBDM",
    HSPM_FBDM_BEST0616_MODEL,
    HSPM_FBDM_0619_MODEL,
    *HSPM_FBDM_V2_MODELS,
}
FBDM_BEST0616_MODEL = "CMUNeXt_FBDM_Best0616"
HSPM_BEST0616_MODEL = "CMUNeXt_HSPM_Best0616"
HSPM_BEST0619_MODEL = "CMUNeXt_HSPM_Best0619"
FBDM_ONLY_MODELS = {"CMUNeXt_FBDM", FBDM_BEST0616_MODEL}
FBDM_MODELS = {*HSPM_FBDM_MODELS, *FBDM_ONLY_MODELS}
HSPM_ONLY_MODELS = {"CMUNeXt_HSPM", HSPM_BEST0616_MODEL, HSPM_BEST0619_MODEL}
HSPM_MODELS = {*HSPM_ONLY_MODELS, *HSPM_FBDM_MODELS}
BARM_MODELS = {"CMUNeXt_BARM"}
BARM_DIAGNOSTIC_NAMES = (
    "effective_gamma",
    "band_mean",
    "hf_energy_mean",
    "logit_correction_abs_mean",
    "logit_correction_abs_max",
    "prediction_flip_ratio",
)
USLGSF_V3_MODELS = {"CMUNeXt_USLGSF_V3"}
USLGSF_V3_DIAGNOSTIC_NAMES = (
    "structure_reliability_mean",
    "decoder_relevance_mean",
    "structure_weight_mean",
    "relevance_weight_mean",
    "active_gate_mean",
    "route_scale",
    "effective_alpha",
    "residual_delta_abs_mean",
    "injection_encoder_rms_ratio",
)
FBDM_V2_DIAGNOSTIC_NAMES = (
    "correction_schedule_scale",
    "effective_correction_scale",
    "boundary_gate_mean",
    "boundary_gate_over_05",
    "safe_boundary_gate_mean",
    "boundary_feature_rms",
    "raw_correction_abs_mean",
    "raw_correction_abs_max",
    "logit_correction_abs_mean",
    "logit_correction_abs_max",
    "prediction_flip_ratio",
)
FBDM_EDGE_DIAGNOSTIC_NAMES = (
    "edge_target_ratio",
    "edge_prob_mean",
    "edge_pred_positive_ratio",
)


def seed_torch(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def parse_ddsr_stages(value):
    if isinstance(value, (tuple, list)):
        return tuple(value)

    stages = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            raise argparse.ArgumentTypeError("ddsr_stages must be a comma-separated list, e.g. 2,3.")
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


def parse_fbdm_stages(value):
    if isinstance(value, (tuple, list)):
        items = value
    else:
        items = str(value).split(",")

    stages = []
    for item in items:
        item = str(item).strip()
        if not item:
            raise argparse.ArgumentTypeError(
                "fbdm_stages must be 0, 1, or a comma-separated pair 0,1."
            )
        try:
            stage = int(item)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "fbdm_stages values must be integers 0 and/or 1."
            ) from exc
        if stage not in {0, 1}:
            raise argparse.ArgumentTypeError("fbdm_stages values must be 0 and/or 1.")
        if stage not in stages:
            stages.append(stage)

    normalized = tuple(sorted(stages))
    if normalized not in {(0,), (1,), (0, 1)}:
        raise argparse.ArgumentTypeError("fbdm_stages must be one of: 0, 1, or 0,1.")
    return normalized


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


def apply_best0616_presets(args):
    if args.model in {
        FBDM_BEST0616_MODEL,
        HSPM_FBDM_BEST0616_MODEL,
        HSPM_FBDM_0619_MODEL,
    }:
        args.fbdm_edge_aux_only = True
        args.fbdm_gate_init = 0.01
        args.fbdm_gate_max = 0.06
        args.fbdm_residual_warmup_epochs = 40
        args.fbdm_edge_loss_weight = 0.03
        args.fbdm_edge_loss_final_weight = 0.003
        args.fbdm_edge_loss_decay_epochs = 150
    if args.model == HSPM_FBDM_BEST0616_MODEL:
        args.fbdm_edge_loss_type = "balanced_bce_dice"
        args.fbdm_edge_kernel_size = 5
        args.fbdm_correction_scale_init = 0.05
        args.fbdm_correction_scale_max = 0.20
        args.fbdm_boundary_gate_floor = 0.20
        args.fbdm_correction_start_epoch = 40
        args.fbdm_correction_warmup_epochs = 40
        args.fbdm_refine_loss_weight = 1.0
        args.fbdm_preserve_loss_weight = 1.0
        args.fbdm_boundary_band_loss_weight = 0.10
        args.fbdm_boundary_band_loss_final_weight = 0.02
        args.fbdm_boundary_band_loss_decay_epochs = 150
        args.fbdm_lr_multiplier = 2.0
        args.fbdm_correction_lr_multiplier = 5.0
    if args.model == HSPM_BEST0616_MODEL:
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
    if args.model == HSPM_BEST0619_MODEL:
        args.hspm_mode = "full"
        args.hspm_backbone_mode = "dual_path"
        args.hspm_fusion_mode = "global"
        args.hspm_mixer_mode = "bounded"
        args.hspm_gamma_init = 0.1
        args.hspm_gamma_max = 0.35
        args.hspm_temperature = 0.1
        args.hspm_prototype_dropout = 0.0
        args.hspm_fusion_gate_init = 0.05
        args.hspm_fusion_gate_max = 0.3
        args.hspm_coarse_loss_weight = 0.1
        args.hspm_coarse_loss_final_weight = 0.02
        args.hspm_coarse_loss_decay_epochs = 150
    if args.model in {HSPM_FBDM_BEST0616_MODEL, HSPM_FBDM_0619_MODEL}:
        args.hspm_mode = "full"
        args.hspm_backbone_mode = "dual_path"
        args.hspm_fusion_mode = "global"
        args.hspm_mixer_mode = "bounded"
        args.hspm_gamma_init = 0.1
        args.hspm_gamma_max = 0.35
        args.hspm_temperature = 0.1
        args.hspm_prototype_dropout = 0.0
        args.hspm_fusion_gate_init = 0.05
        args.hspm_fusion_gate_max = 0.3
        args.hspm_coarse_loss_weight = 0.1
        args.hspm_coarse_loss_final_weight = 0.02
        args.hspm_coarse_loss_decay_epochs = 150
    if args.model == HSPM_FBDM_0619_MODEL:
        args.base_dir = "./data/busi"
        args.train_file_dir = "busi_train3.txt"
        args.val_file_dir = "busi_val3.txt"
        args.base_lr = 0.01
        args.batch_size = 8
        args.epoch = 300
        args.img_size = 256
        args.num_classes = 1
        args.seed = 41
        args.use_extra_aug = False
        args.sampling_mode = "uniform"
        args.val_threshold_mode = "fixed"
        args.val_threshold = 0.5
        args.val_threshold_metric = "iou"
        args.early_stop_patience = 0
        args.fbdm_stages = (0, 1)
        args.fbdm_x2_edge_ratio = 0.3
        args.fbdm_edge_loss_type = "balanced_bce_dice"
        args.fbdm_edge_pos_weight = 20.0
        args.fbdm_edge_kernel_size = 3
        args.fbdm_boundary_band_loss_weight = 0.0
        args.fbdm_boundary_band_loss_final_weight = None
        args.fbdm_boundary_band_loss_decay_epochs = 0
        args.fbdm_refine_loss_weight = 0.0
        args.fbdm_preserve_loss_weight = 0.0
        args.fbdm_lr_multiplier = 1.0
        args.fbdm_correction_lr_multiplier = 1.0
    return args


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="CMUNeXt",
                    choices=["Mobile_U_ViT", "CMUNeXt", "CMUNeXt_BARM", "CMUNeXt_FBDM", FBDM_BEST0616_MODEL, "CMUNeXt_USLGSF", "CMUNeXt_USLGSF_V2", "CMUNeXt_USLGSF_V3", "CMUNeXt_HSPM", HSPM_BEST0616_MODEL, HSPM_BEST0619_MODEL, "CMUNeXt_HSPM_FBDM", HSPM_FBDM_BEST0616_MODEL, HSPM_FBDM_0619_MODEL, "CMUNeXt_HSPM_FBDM_V2",
                             "CMUNeXt_DualGAG", "CMUNeXt_BA_DualGAG",
                             "CMUNeXt_SpeckleEnhance", "CMUNeXt_DualGAG_SpeckleEnhance",
                             "CMUNeXt_BA_DualGAG_SpeckleEnhance",
                             "CMUNet", "MK_UNet", "AttU_Net", "TransUnet", "R2U_Net", "U_Net",
                             "UNext", "UNetplus", "UNet3plus", "SwinUnet", "MedT", "TransUnet"], help='model')
parser.add_argument('--base_dir', type=str, default="./data/busi", help='dir')
parser.add_argument('--train_file_dir', type=str, default="busi_train.txt", help='dir')
parser.add_argument('--val_file_dir', type=str, default="busi_val.txt", help='dir')
parser.add_argument('--base_lr', type=float, default=0.01, help='segmentation network learning rate')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--epoch', type=int, default=300, help='train epoch')
parser.add_argument('--img_size', type=int, default=256, help='img size of per batch')
parser.add_argument('--num_classes', type=int, default=1, help='seg num_classes')
parser.add_argument('--seed', type=int, default=41, help='random seed')
parser.add_argument('--save_dir', type=str, default="./checkpoint", help='directory to save the best model')
# <=== 新增：是否开启额外数据增强的指令
parser.add_argument('--use_extra_aug', action='store_true', help='Whether to use conservative extra data augmentations')
parser.add_argument('--sampling_mode', type=str, default='uniform',
                    choices=['uniform', 'targeted'],
                    help='Uniform shuffling or BUSI pathology-and-size targeted sampling')
parser.add_argument('--targeted_small_area_threshold', type=float, default=0.05,
                    help='Upper-exclusive lesion area ratio for the targeted small bucket')
parser.add_argument('--targeted_large_area_threshold', type=float, default=0.20,
                    help='Upper-inclusive lesion area ratio for the targeted medium bucket')
parser.add_argument('--targeted_malignant_small_weight', type=float, default=1.2,
                    help='Sampling weight for small malignant BUSI cases')
parser.add_argument('--targeted_malignant_medium_weight', type=float, default=1.8,
                    help='Sampling weight for medium malignant BUSI cases')
parser.add_argument('--targeted_malignant_large_weight', type=float, default=1.3,
                    help='Sampling weight for large malignant BUSI cases')
parser.add_argument('--ddsr_stages', type=parse_ddsr_stages, default=(0, 1),
                    help='Comma-separated DDSR stages for CMUNeXt_SpeckleEnhance, e.g. 0,1 or 2,3 or 0,1,2,3')
parser.add_argument('--ddsr_smooth_k', type=int, default=5,
                    help='DDSR average-pooling kernel size for CMUNeXt_SpeckleEnhance')
parser.add_argument('--ddsr_max_scale', type=float, default=0.05,
                    help='Upper bound for DDSR residual scale in CMUNeXt_SpeckleEnhance')
parser.add_argument('--ddsr_mode', type=str, default="skip_only", choices=["skip_only", "propagate"],
                    help='Use DDSR only for decoder skips or propagate it through the encoder')
parser.add_argument('--ddsr_aux_init', type=float, default=0.1,
                    help='Initial DDSR auxiliary residual blend for DualGAG_SpeckleEnhance')
parser.add_argument('--uslgsf_stages', type=parse_uslgsf_stages, default=(0, 1),
                    help='Comma-separated US-LGSF skip stages, e.g. 0,1 or 0,1,2')
parser.add_argument('--uslgsf_smooth_kernels', type=parse_uslgsf_smooth_kernels, default=(3, 7),
                    help='Small and large odd smoothing kernels for US-LGSF')
parser.add_argument('--uslgsf_context_downsample', type=int, default=2,
                    help='Downsampling factor for the US-LGSF low-frequency context path')
parser.add_argument('--uslgsf_alpha_init', type=float, default=0.05,
                    help='Initial effective US-LGSF residual blend')
parser.add_argument('--uslgsf_alpha_max', type=float, default=0.5,
                    help='Maximum effective US-LGSF residual blend')
parser.add_argument('--uslgsf_mode', type=str, default="full",
                    choices=["full", "context_only", "structure_only", "relevance_only"],
                    help='US-LGSF full model or a core ablation mode')
parser.add_argument('--uslgsf_residual_init_scale', type=float, default=0.05,
                    help='Kaiming residual projection initialization scale for CMUNeXt_USLGSF_V3')
parser.add_argument('--gag_stages', type=parse_gag_stages, default=(2, 3),
                    help='Comma-separated DualGAG stages, e.g. 0,1 or 1,3 or 0,1,2,3')
parser.add_argument('--boundary_loss_weight', type=float, default=0.3,
                    help='Boundary loss weight for CMUNeXt_BA_DualGAG')
parser.add_argument('--barm_gate_init', type=float, default=0.02,
                    help='Initial effective BARM boundary-refinement strength')
parser.add_argument('--barm_gate_max', type=float, default=0.2,
                    help='Maximum effective BARM boundary-refinement strength')
parser.add_argument('--barm_hf_keep_init', type=float, default=0.3,
                    help='Initial high-frequency keep ratio for BARM wavelet energy')
parser.add_argument('--barm_warmup_epochs', type=int, default=60,
                    help='Epochs used to linearly warm up BARM logit refinement')
parser.add_argument('--barm_coarse_loss_weight', type=float, default=0.3,
                    help='Auxiliary pre-refinement segmentation loss weight for CMUNeXt_BARM')
parser.add_argument('--barm_boundary_loss_weight', type=float, default=0.2,
                    help='Boundary-band Dice loss weight for CMUNeXt_BARM')
parser.add_argument('--barm_edge_loss_weight', type=float, default=0.1,
                    help='Boundary-band edge supervision loss weight for CMUNeXt_BARM')
parser.add_argument('--barm_edge_band_width', type=int, default=2,
                    help='GT boundary band half-width for CMUNeXt_BARM edge supervision')
parser.add_argument('--barm_edge_pos_weight', type=float, default=10.0,
                    help='Positive-class weight for CMUNeXt_BARM edge BCE')
parser.add_argument('--hspm_mode', type=str, default="full", choices=["full", "context_only"],
                    help='Enable the full HSPM or keep only its high-resolution context bottleneck')
parser.add_argument('--hspm_coarse_loss_weight', type=float, default=0.3,
                    help='Auxiliary coarse segmentation loss weight for CMUNeXt_HSPM')
parser.add_argument('--hspm_mixer_mode', type=str, default="legacy", choices=["legacy", "bounded", "stable"],
                    help='Prototype mixer behavior for CMUNeXt_HSPM')
parser.add_argument('--hspm_gamma_init', type=float, default=0.1,
                    help='Initial prototype residual strength')
parser.add_argument('--hspm_gamma_max', type=float, default=0.3,
                    help='Maximum effective prototype residual strength in bounded/stable modes')
parser.add_argument('--hspm_temperature', type=float, default=0.1,
                    help='Prototype assignment temperature')
parser.add_argument('--hspm_prototype_dropout', type=float, default=0.0,
                    help='Dropout2d probability on stable prototype residuals')
parser.add_argument('--hspm_backbone_mode', type=str, default="highres_only",
                    choices=["highres_only", "dual_path"],
                    help='Use the legacy high-resolution-only bottleneck or the dual-path CMUNeXt bottleneck')
parser.add_argument('--hspm_fusion_gate_init', type=float, default=0.05,
                    help='Initial effective HSPM residual gate in dual-path mode')
parser.add_argument('--hspm_fusion_gate_max', type=float, default=0.3,
                    help='Maximum effective HSPM residual gate in dual-path mode')
parser.add_argument('--hspm_fusion_mode', type=str, default="global", choices=["global", "size_aware"],
                    help='Use global or size-aware spatial HSPM residual fusion in dual-path mode')
parser.add_argument('--hspm_small_area_threshold', type=float, default=0.05,
                    help='Predicted coarse area threshold used by size-aware HSPM fusion')
parser.add_argument('--hspm_small_area_temperature', type=float, default=0.02,
                    help='Smoothness of the small-lesion decision in size-aware HSPM fusion')
parser.add_argument('--hspm_prototype_warmup_epochs', type=int, default=0,
                    help='Epochs used to linearly warm up stable prototype injection')
parser.add_argument('--hspm_coarse_loss_final_weight', type=float, default=None,
                    help='Final HSPM coarse loss weight; defaults to the initial weight')
parser.add_argument('--hspm_coarse_loss_decay_epochs', type=int, default=0,
                    help='Epochs used to linearly decay the HSPM coarse loss weight')
parser.add_argument('--fbdm_no_hspm_prior', action='store_true',
                    help='Disable HSPM coarse/uncertainty priors inside CMUNeXt_HSPM_FBDM')
parser.add_argument('--fbdm_no_detach_hspm_prior', action='store_true',
                    help='Allow FBDM gradients to flow into HSPM priors; default keeps priors detached')
parser.add_argument('--fbdm_semantic_uncertainty_weight', type=float, default=0.7,
                    help='Uncertainty weight in the FBDM semantic boundary prior')
parser.add_argument('--fbdm_semantic_coarse_weight', type=float, default=0.3,
                    help='Coarse probability weight in the FBDM semantic boundary prior')
parser.add_argument('--fbdm_semantic_coarse_mode', type=str, default="prob", choices=["prob", "edge"],
                    help='Use coarse probability or Sobel edge of coarse probability in the FBDM semantic prior')
parser.add_argument('--fbdm_semantic_gate_base', type=float, default=0.7,
                    help='Conservative base term in the FBDM boundary gate')
parser.add_argument('--fbdm_gate_init', type=float, default=0.03,
                    help='Initial effective FBDM residual strength')
parser.add_argument('--fbdm_gate_max', type=float, default=0.2,
                    help='Maximum effective FBDM residual strength')
parser.add_argument('--fbdm_edge_aux_only', action='store_true',
                    help='Train FBDM edge auxiliary branch without injecting its residual into the segmentation path')
parser.add_argument('--fbdm_stages', type=parse_fbdm_stages, default=(0,),
                    help='Best0616 edge supervision stages: 0, 1, or 0,1')
parser.add_argument('--fbdm_x2_edge_ratio', type=float, default=0.30,
                    help='Relative x2/x1 edge-loss weight when Best0616 uses stages 0,1')
parser.add_argument('--fbdm_edge_loss_type', type=str, default='legacy',
                    choices=['legacy', 'balanced_bce_dice', 'focal_dice'],
                    help='Edge supervision loss used by FBDM models')
parser.add_argument('--fbdm_edge_pos_weight', type=float, default=20.0,
                    help='Positive-class weight for balanced FBDM edge BCE')
parser.add_argument('--fbdm_edge_focal_alpha', type=float, default=0.95,
                    help='Positive-class alpha for focal FBDM edge loss')
parser.add_argument('--fbdm_edge_focal_gamma', type=float, default=2.0,
                    help='Focusing gamma for focal FBDM edge loss')
parser.add_argument('--fbdm_edge_loss_weight', type=float, default=0.05,
                    help='Auxiliary edge loss weight for FBDM models')
parser.add_argument('--fbdm_edge_loss_final_weight', type=float, default=None,
                    help='Final FBDM edge loss weight; defaults to the initial weight')
parser.add_argument('--fbdm_edge_loss_decay_epochs', type=int, default=0,
                    help='Epochs used to linearly decay the FBDM edge loss weight')
parser.add_argument('--fbdm_edge_kernel_size', type=int, default=3,
                    help='Odd kernel size used to build edge supervision masks for FBDM models')
parser.add_argument('--fbdm_residual_warmup_epochs', type=int, default=0,
                    help='Epochs used to linearly warm up FBDM residual injection')
parser.add_argument('--fbdm_correction_scale_init', type=float, default=0.05,
                    help='Initial effective bounded logit-correction scale for FBDM V2')
parser.add_argument('--fbdm_correction_scale_max', type=float, default=0.3,
                    help='Maximum effective bounded logit-correction scale for FBDM V2')
parser.add_argument('--fbdm_boundary_gate_floor', type=float, default=0.0,
                    help='Minimum boundary gate used by protected FBDM correction')
parser.add_argument('--fbdm_lr_multiplier', type=float, default=1.0,
                    help='Learning-rate multiplier for the protected FBDM feature branch')
parser.add_argument('--fbdm_correction_lr_multiplier', type=float, default=1.0,
                    help='Learning-rate multiplier for the protected correction branch')
parser.add_argument('--fbdm_correction_warmup_epochs', type=int, default=40,
                    help='Epochs used to linearly warm up FBDM V2 logit correction')
parser.add_argument('--fbdm_correction_start_epoch', type=int, default=0,
                    help='Epoch at which protected FBDM logit correction starts')
parser.add_argument('--fbdm_refine_loss_weight', type=float, default=0.0,
                    help='Maximum protected FBDM refinement loss weight')
parser.add_argument('--fbdm_preserve_loss_weight', type=float, default=0.0,
                    help='Non-boundary logit-preservation loss weight')
parser.add_argument('--fbdm_boundary_band_loss_weight', type=float, default=0.0,
                    help='Boundary-band loss weight for FBDM V2 logit correction')
parser.add_argument('--fbdm_boundary_band_loss_final_weight', type=float, default=None,
                    help='Final FBDM V2 boundary-band loss weight; defaults to the initial weight')
parser.add_argument('--fbdm_boundary_band_loss_decay_epochs', type=int, default=0,
                    help='Epochs used to linearly decay the FBDM V2 boundary-band loss weight')
parser.add_argument('--fbdm_boundary_band_kernel_size', type=int, default=7,
                    help='Odd kernel size used to build GT boundary bands for FBDM V2 correction')
parser.add_argument('--early_stop_patience', type=int, default=0,
                    help='Stop after this many epochs without a significant validation IoU improvement; 0 disables')
parser.add_argument('--early_stop_min_delta', type=float, default=0.001,
                    help='Minimum validation IoU gain that resets early stopping patience')
parser.add_argument('--val_threshold_mode', type=str, default="fixed", choices=["fixed", "scan"],
                    help='Use a fixed validation threshold or scan a threshold range')
parser.add_argument('--val_threshold', type=float, default=0.5,
                    help='Validation threshold when val_threshold_mode=fixed')
parser.add_argument('--val_threshold_start', type=float, default=0.30,
                    help='Threshold scan start when val_threshold_mode=scan')
parser.add_argument('--val_threshold_end', type=float, default=0.70,
                    help='Threshold scan end when val_threshold_mode=scan')
parser.add_argument('--val_threshold_step', type=float, default=0.02,
                    help='Threshold scan step when val_threshold_mode=scan')
parser.add_argument('--val_threshold_metric', type=str, default="iou", choices=["iou", "f1"],
                    help='Metric used to pick the best validation threshold')
args = parser.parse_args()
apply_best0616_presets(args)
seed_torch(args.seed)


def get_model(args):
    if args.model == "CMUNet":
        model = CMUNet(output_ch=args.num_classes).cuda()
    elif args.model == "MK_UNet":
        model = MK_UNet(num_classes=args.num_classes, in_channels=3).cuda()
    elif args.model == "CMUNeXt":
        model = cmunext(num_classes=args.num_classes).cuda()
    elif args.model == "CMUNeXt_BARM":
        model = cmunext_barm(
            num_classes=args.num_classes,
            barm_gate_init=args.barm_gate_init,
            barm_gate_max=args.barm_gate_max,
            barm_hf_keep_init=args.barm_hf_keep_init,
        ).cuda()
    elif args.model == "CMUNeXt_FBDM":
        model = cmunext_fbdm(
            num_classes=args.num_classes,
            fbdm_gate_init=args.fbdm_gate_init,
            fbdm_gate_max=args.fbdm_gate_max,
            fbdm_edge_aux_only=args.fbdm_edge_aux_only,
        ).cuda()
    elif args.model == FBDM_BEST0616_MODEL:
        model = cmunext_fbdm_best0616(
            num_classes=args.num_classes,
            fbdm_stages=args.fbdm_stages,
        ).cuda()
    elif args.model == "CMUNeXt_USLGSF":
        model = cmunext_uslgsf(
            num_classes=args.num_classes,
            uslgsf_stages=args.uslgsf_stages,
            uslgsf_smooth_kernels=args.uslgsf_smooth_kernels,
            uslgsf_context_downsample=args.uslgsf_context_downsample,
            uslgsf_alpha_init=args.uslgsf_alpha_init,
            uslgsf_alpha_max=args.uslgsf_alpha_max,
            uslgsf_mode=args.uslgsf_mode,
        ).cuda()
    elif args.model == "CMUNeXt_USLGSF_V2":
        model = cmunext_uslgsf_v2(
            num_classes=args.num_classes,
            uslgsf_stages=args.uslgsf_stages,
            uslgsf_smooth_kernels=args.uslgsf_smooth_kernels,
            uslgsf_context_downsample=args.uslgsf_context_downsample,
            uslgsf_alpha_init=args.uslgsf_alpha_init,
            uslgsf_alpha_max=args.uslgsf_alpha_max,
            uslgsf_mode=args.uslgsf_mode,
        ).cuda()
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
        ).cuda()
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
        ).cuda()
    elif args.model == HSPM_BEST0616_MODEL:
        model = cmunext_hspm_best0616(num_classes=args.num_classes).cuda()
    elif args.model == HSPM_BEST0619_MODEL:
        model = cmunext_hspm_best0619(num_classes=args.num_classes).cuda()
    elif args.model == HSPM_FBDM_BEST0616_MODEL:
        model = cmunext_hspm_fbdm_best0616(
            num_classes=args.num_classes,
            hspm_gamma_init=args.hspm_gamma_init,
            hspm_gamma_max=args.hspm_gamma_max,
            hspm_fusion_gate_init=args.hspm_fusion_gate_init,
            hspm_fusion_gate_max=args.hspm_fusion_gate_max,
            fbdm_semantic_uncertainty_weight=args.fbdm_semantic_uncertainty_weight,
            fbdm_semantic_coarse_weight=args.fbdm_semantic_coarse_weight,
            fbdm_semantic_gate_base=args.fbdm_semantic_gate_base,
            fbdm_gate_init=args.fbdm_gate_init,
            fbdm_gate_max=args.fbdm_gate_max,
            fbdm_correction_scale_init=args.fbdm_correction_scale_init,
            fbdm_correction_scale_max=args.fbdm_correction_scale_max,
            fbdm_boundary_gate_floor=args.fbdm_boundary_gate_floor,
        ).cuda()
    elif args.model == HSPM_FBDM_0619_MODEL:
        model = cmunext_hspm_fbdm_0619(num_classes=args.num_classes).cuda()
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
            fbdm_semantic_coarse_mode=args.fbdm_semantic_coarse_mode,
            fbdm_semantic_gate_base=args.fbdm_semantic_gate_base,
            fbdm_gate_init=args.fbdm_gate_init,
            fbdm_gate_max=args.fbdm_gate_max,
            fbdm_edge_aux_only=args.fbdm_edge_aux_only,
        ).cuda()
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
        ).cuda()
    elif args.model == "CMUNeXt_DualGAG":
        model = cmunext_dualgag(num_classes=args.num_classes, gag_stages=args.gag_stages).cuda()
    elif args.model == "CMUNeXt_BA_DualGAG":
        model = cmunext_ba_dualgag(num_classes=args.num_classes, gag_stages=args.gag_stages).cuda()
    elif args.model == "CMUNeXt_BA_DualGAG_SpeckleEnhance":
        model = cmunext_ba_dualgag_speckleenhance(
            num_classes=args.num_classes,
            ddsr_stages=args.ddsr_stages,
            gag_stages=args.gag_stages,
            ddsr_smooth_k=args.ddsr_smooth_k,
            ddsr_max_scale=args.ddsr_max_scale,
            ddsr_skip_only=args.ddsr_mode == "skip_only",
            ddsr_aux_init=args.ddsr_aux_init,
        ).cuda()
    elif args.model == "CMUNeXt_SpeckleEnhance":
        model = cmunext_speckle(
            num_classes=args.num_classes,
            ddsr_stages=args.ddsr_stages,
            ddsr_smooth_k=args.ddsr_smooth_k,
            ddsr_max_scale=args.ddsr_max_scale,
            ddsr_skip_only=args.ddsr_mode == "skip_only",
        ).cuda()
    elif args.model == "CMUNeXt_DualGAG_SpeckleEnhance":
        model = cmunext_dualgag_speckleenhance(
            num_classes=args.num_classes,
            ddsr_stages=args.ddsr_stages,
            gag_stages=args.gag_stages,
            ddsr_smooth_k=args.ddsr_smooth_k,
            ddsr_max_scale=args.ddsr_max_scale,
            ddsr_skip_only=args.ddsr_mode == "skip_only",
            ddsr_aux_init=args.ddsr_aux_init,
        ).cuda()
    elif args.model == "U_Net":
        model = U_Net(output_ch=args.num_classes).cuda()
    elif args.model == "AttU_Net":
        model = AttU_Net(output_ch=args.num_classes).cuda()
    elif args.model == "UNext":
        model = UNext(num_classes=args.num_classes).cuda()
    elif args.model == "UNetplus":
        model = ResNet34UnetPlus(num_class=args.num_classes).cuda()
    elif args.model == "UNet3plus":
        model = UNet3plus(n_classes=args.num_classes).cuda()
    elif args.model == "Mobile_U_ViT":
        model = mobileuvit(out_channel=args.num_classes).cuda()
    else:
        model = get_transformer_based_model(parser=parser, model_name=args.model, img_size=args.img_size,
                                            num_classes=args.num_classes, in_ch=3).cuda()
    return model


def get_criterion(args):
    if args.model in BARM_MODELS:
        return losses.__dict__['CMUNeXtBARMLoss'](
            coarse_weight=args.barm_coarse_loss_weight,
            boundary_weight=args.barm_boundary_loss_weight,
            edge_weight=args.barm_edge_loss_weight,
            edge_band_width=args.barm_edge_band_width,
            edge_pos_weight=args.barm_edge_pos_weight,
        ).cuda()
    if args.model in HSPM_FBDM_MODELS:
        return losses.__dict__['HSPMFBDMLoss'](
            coarse_weight=args.hspm_coarse_loss_weight,
            edge_weight=args.fbdm_edge_loss_weight,
            edge_kernel_size=args.fbdm_edge_kernel_size,
            boundary_band_weight=args.fbdm_boundary_band_loss_weight,
            boundary_band_kernel_size=args.fbdm_boundary_band_kernel_size,
            edge_loss_type=args.fbdm_edge_loss_type,
            edge_pos_weight=args.fbdm_edge_pos_weight,
            edge_focal_alpha=args.fbdm_edge_focal_alpha,
            edge_focal_gamma=args.fbdm_edge_focal_gamma,
            x2_edge_ratio=args.fbdm_x2_edge_ratio,
            protected_refinement=args.model == HSPM_FBDM_BEST0616_MODEL,
            refine_weight=args.fbdm_refine_loss_weight,
            preserve_weight=args.fbdm_preserve_loss_weight,
        ).cuda()
    if args.model in FBDM_ONLY_MODELS:
        return losses.__dict__['FBDMLoss'](
            edge_weight=args.fbdm_edge_loss_weight,
            edge_kernel_size=args.fbdm_edge_kernel_size,
            x2_edge_ratio=args.fbdm_x2_edge_ratio,
            edge_loss_type=args.fbdm_edge_loss_type,
            edge_pos_weight=args.fbdm_edge_pos_weight,
            edge_focal_alpha=args.fbdm_edge_focal_alpha,
            edge_focal_gamma=args.fbdm_edge_focal_gamma,
        ).cuda()
    if args.model in HSPM_ONLY_MODELS:
        return losses.__dict__['HSPMLoss'](coarse_weight=args.hspm_coarse_loss_weight).cuda()
    if args.model in {"CMUNeXt_BA_DualGAG", "CMUNeXt_BA_DualGAG_SpeckleEnhance"}:
        return losses.__dict__['BoundaryAwareSegLoss'](lambda_b=args.boundary_loss_weight).cuda()
    return losses.__dict__['BCEDiceLoss']().cuda()


def forward_with_model(args, model, x, return_aux=True):
    return model(x)


def get_seg_logits(outputs):
    if isinstance(outputs, dict):
        if 'seg' in outputs:
            return outputs['seg']
        if 'pred_refined' in outputs:
            return outputs['pred_refined']
        if 'pred_main' in outputs:
            return outputs['pred_main']
        raise KeyError(f"Unsupported output keys: {sorted(outputs.keys())}")
    return outputs


def get_loss_tensor(loss_output):
    if isinstance(loss_output, tuple):
        return loss_output[0]
    return loss_output


def get_loss_components(loss_output):
    if not isinstance(loss_output, tuple) or len(loss_output) < 2:
        return None
    return loss_output[1]


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


def compute_loss(
    args,
    criterion,
    outputs,
    label_batch,
    sampled_batch=None,
    aux_weight=None,
    edge_weight=None,
    band_weight=None,
    refine_weight=None,
    preserve_weight=None,
):
    if args.model in BARM_MODELS:
        return criterion(outputs, label_batch, return_components=True)
    if args.model in HSPM_FBDM_MODELS:
        return criterion(
            outputs,
            label_batch,
            coarse_weight=aux_weight,
            edge_weight=edge_weight,
            boundary_band_weight=band_weight,
            refine_weight=refine_weight,
            preserve_weight=preserve_weight,
            return_components=True,
        )
    if args.model in FBDM_ONLY_MODELS:
        return criterion(outputs, label_batch, edge_weight=edge_weight, return_components=True)
    if args.model in HSPM_ONLY_MODELS:
        return criterion(outputs, label_batch, coarse_weight=aux_weight)
    return criterion(outputs, label_batch)


def get_hspm_coarse_weight(args, epoch_num):
    initial_weight = float(args.hspm_coarse_loss_weight)
    final_weight = args.hspm_coarse_loss_final_weight
    if final_weight is None:
        final_weight = initial_weight
    final_weight = float(final_weight)
    decay_epochs = int(args.hspm_coarse_loss_decay_epochs)
    if decay_epochs <= 0:
        return initial_weight
    progress = min(max(float(epoch_num) / decay_epochs, 0.0), 1.0)
    return initial_weight + progress * (final_weight - initial_weight)


def get_fbdm_edge_weight(args, epoch_num):
    initial_weight = float(args.fbdm_edge_loss_weight)
    final_weight = args.fbdm_edge_loss_final_weight
    if final_weight is None:
        final_weight = initial_weight
    final_weight = float(final_weight)
    decay_epochs = int(args.fbdm_edge_loss_decay_epochs)
    if decay_epochs <= 0:
        return initial_weight
    progress = min(max(float(epoch_num) / decay_epochs, 0.0), 1.0)
    return initial_weight + progress * (final_weight - initial_weight)


def get_fbdm_residual_scale(args, epoch_num):
    if args.model not in FBDM_MODELS:
        return 1.0
    warmup_epochs = int(args.fbdm_residual_warmup_epochs)
    if warmup_epochs <= 0:
        return 1.0
    return min(max(float(epoch_num) / warmup_epochs, 0.0), 1.0)


def get_fbdm_correction_schedule_scale(args, epoch_num):
    if args.model not in HSPM_FBDM_CORRECTION_MODELS:
        return 1.0
    start_epoch = (
        int(args.fbdm_correction_start_epoch)
        if args.model == HSPM_FBDM_BEST0616_MODEL
        else 0
    )
    if epoch_num < start_epoch:
        return 0.0
    warmup_epochs = int(args.fbdm_correction_warmup_epochs)
    if warmup_epochs <= 0:
        return 1.0
    return min(max(float(epoch_num - start_epoch) / warmup_epochs, 0.0), 1.0)


def get_fbdm_boundary_band_weight(args, epoch_num):
    initial_weight = float(args.fbdm_boundary_band_loss_weight)
    final_weight = args.fbdm_boundary_band_loss_final_weight
    if final_weight is None:
        final_weight = initial_weight
    final_weight = float(final_weight)
    decay_epochs = int(args.fbdm_boundary_band_loss_decay_epochs)
    if decay_epochs <= 0:
        return initial_weight
    progress = min(max(float(epoch_num) / decay_epochs, 0.0), 1.0)
    return initial_weight + progress * (final_weight - initial_weight)


def get_hspm_prototype_scale(args, epoch_num):
    if args.model not in HSPM_MODELS or args.hspm_mixer_mode != "stable":
        return 1.0
    warmup_epochs = int(args.hspm_prototype_warmup_epochs)
    if warmup_epochs <= 0:
        return 1.0
    return min(max(float(epoch_num) / warmup_epochs, 0.0), 1.0)


def configure_hspm_epoch(args, model, epoch_num):
    coarse_weight = get_hspm_coarse_weight(args, epoch_num)
    prototype_scale = get_hspm_prototype_scale(args, epoch_num)
    effective_gamma = None
    if args.model in HSPM_MODELS:
        hspm_model = model.module if isinstance(model, torch.nn.DataParallel) else model
        hspm_model.prototype_mixer.set_prototype_scale(prototype_scale)
        effective_gamma = float(hspm_model.prototype_mixer.effective_gamma().detach().cpu())
        if args.hspm_mixer_mode == "stable":
            effective_gamma *= prototype_scale
    return coarse_weight, prototype_scale, effective_gamma


def configure_fbdm_epoch(args, model, epoch_num):
    edge_weight = get_fbdm_edge_weight(args, epoch_num) if args.model in FBDM_MODELS else 0.0
    residual_scale = get_fbdm_residual_scale(args, epoch_num)
    effective_gamma = None
    if args.model in FBDM_MODELS:
        fbdm_model = model.module if isinstance(model, torch.nn.DataParallel) else model
        fbdm1 = getattr(fbdm_model, "fbdm1", None)
        if fbdm1 is not None:
            fbdm1.set_residual_scale(residual_scale)
        if args.model in HSPM_FBDM_CORRECTION_MODELS:
            fbdm_model.set_fbdm_correction_schedule_scale(
                get_fbdm_correction_schedule_scale(args, epoch_num)
            )
        effective_gamma = (
            0.0
            if fbdm1 is None
            else float(fbdm1.effective_gamma().detach().cpu())
        )
    return edge_weight, residual_scale, effective_gamma


def get_barm_residual_scale(args, epoch_num):
    if args.model not in BARM_MODELS:
        return 1.0
    warmup_epochs = int(args.barm_warmup_epochs)
    if warmup_epochs <= 0:
        return 1.0
    return min(max(float(epoch_num) / warmup_epochs, 0.0), 1.0)


def configure_barm_epoch(args, model, epoch_num):
    residual_scale = get_barm_residual_scale(args, epoch_num)
    effective_gamma = None
    if args.model in BARM_MODELS:
        barm_model = model.module if isinstance(model, torch.nn.DataParallel) else model
        barm_model.set_warmup_scale(residual_scale)
        effective_gamma = float(barm_model.barm.effective_gamma().detach().cpu())
    return residual_scale, effective_gamma


def get_hspm_fusion_diagnostics(model):
    hspm_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    diagnostics = getattr(hspm_model, "last_fusion_diagnostics", None)
    if diagnostics is None:
        return None
    return {
        name: float(value.detach().cpu())
        for name, value in diagnostics.items()
    }


def get_barm_diagnostics(model):
    barm_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    diagnostics = barm_model.get_barm_diagnostics()
    if diagnostics is None:
        return None
    return {
        name: float(value.detach().cpu())
        for name, value in diagnostics.items()
    }


def get_fbdm_v2_diagnostics(model):
    fbdm_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    diagnostics = fbdm_model.get_fbdm_v2_diagnostics()
    if diagnostics is None:
        return None
    return {
        name: float(value.detach().cpu())
        for name, value in diagnostics.items()
    }


def get_fbdm_edge_diagnostics(criterion):
    diagnostics = criterion.get_edge_diagnostics()
    if diagnostics is None:
        return None
    return {
        name: float(value.detach().cpu())
        for name, value in diagnostics.items()
    }


def get_uslgsf_diagnostics(model):
    uslgsf_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    diagnostics = getattr(uslgsf_model, "last_uslgsf_diagnostics", None)
    if diagnostics is None:
        return None
    return {
        f"{stage}_{name}": float(value.detach().cpu())
        for stage, stage_diagnostics in diagnostics.items()
        for name, value in stage_diagnostics.items()
    }


def update_loss_component_meters(avg_meters, components, prefix, batch_size):
    if components is None:
        return
    for name, value in components.items():
        meter_name = f"{prefix}_loss_{name}"
        avg_meters[meter_name].update(float(value.detach().cpu()), batch_size)


def update_fbdm_edge_diagnostic_meters(avg_meters, criterion, prefix, batch_size):
    diagnostics = get_fbdm_edge_diagnostics(criterion)
    if diagnostics is None:
        return
    for name, value in diagnostics.items():
        avg_meters[f"{prefix}_fbdm_{name}"].update(value, batch_size)


def update_barm_diagnostic_meters(avg_meters, model, prefix, batch_size):
    diagnostics = get_barm_diagnostics(model)
    if diagnostics is None:
        return
    for name, value in diagnostics.items():
        avg_meters[f"{prefix}_barm_{name}"].update(value, batch_size)


def validate_sampling_args(args):
    if args.sampling_mode not in {'uniform', 'targeted'}:
        raise ValueError("sampling_mode must be either 'uniform' or 'targeted'.")
    if not (
        0.0
        <= args.targeted_small_area_threshold
        < args.targeted_large_area_threshold
        <= 1.0
    ):
        raise ValueError(
            'Targeted sampling area thresholds must satisfy '
            '0 <= small_threshold < large_threshold <= 1.'
        )
    targeted_weights = (
        args.targeted_malignant_small_weight,
        args.targeted_malignant_medium_weight,
        args.targeted_malignant_large_weight,
    )
    if any(weight <= 0.0 for weight in targeted_weights):
        raise ValueError('Targeted sampling weights must be positive.')


def _format_sampling_groups(values, float_values=False):
    if float_values:
        return ', '.join(
            f'{group}={float(values.get(group, 0.0)):.1f}'
            for group in TARGETED_SAMPLING_GROUPS
        )
    return ', '.join(
        f'{group}={int(values.get(group, 0))}'
        for group in TARGETED_SAMPLING_GROUPS
    )


def build_train_sampler(args, dataset):
    validate_sampling_args(args)
    if args.sampling_mode == 'uniform':
        return None

    metadata = build_targeted_sampling_metadata(
        base_dir=args.base_dir,
        sample_list=dataset.sample_list,
        small_threshold=args.targeted_small_area_threshold,
        large_threshold=args.targeted_large_area_threshold,
        malignant_small_weight=args.targeted_malignant_small_weight,
        malignant_medium_weight=args.targeted_malignant_medium_weight,
        malignant_large_weight=args.targeted_malignant_large_weight,
    )
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    sampler = TrackingWeightedRandomSampler(
        weights=metadata.weights,
        num_samples=len(dataset),
        replacement=True,
        generator=generator,
        groups=metadata.groups,
    )
    logging.info(
        '=> Targeted sampling enabled: samples_per_epoch=%d, replacement=True.',
        len(dataset),
    )
    logging.info(
        '=> Targeted sampling source counts: %s',
        _format_sampling_groups(metadata.original_counts),
    )
    logging.info(
        '=> Targeted sampling expected counts: %s',
        _format_sampling_groups(metadata.expected_counts, float_values=True),
    )
    return sampler


def build_train_transform(args, img_size):
    if not args.use_extra_aug:
        logging.info("=> Using BASIC data augmentation.")
        return Compose([
            RandomRotate90(p=0.5),
            transforms.Flip(p=0.5),
            Resize(img_size, img_size),
            transforms.Normalize(),
        ])

    logging.info("=> Enabled conservative EXTRA data augmentation.")
    return Compose([
        HorizontalFlip(p=0.5),
        ShiftScaleRotate(
            shift_limit=0.04,
            scale_limit=0.08,
            rotate_limit=12,
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            mask_value=0,
            p=0.4,
        ),
        OneOf([
            RandomBrightnessContrast(brightness_limit=0.10, contrast_limit=0.10, p=1.0),
            RandomGamma(gamma_limit=(90, 110), p=1.0),
        ], p=0.35),
        MultiplicativeNoise(
            multiplier=(0.97, 1.03),
            per_channel=False,
            elementwise=True,
            p=0.20,
        ),
        GaussianBlur(blur_limit=(3, 3), p=0.05),
        Resize(img_size, img_size),
        transforms.Normalize(),
    ])


def getDataloader(args):
    img_size = args.img_size
    if args.model == "SwinUnet":
        img_size = 224

    train_transform = build_train_transform(args, img_size)

    val_transform = Compose([
        Resize(img_size, img_size),
        transforms.Normalize(),
    ])
    db_train = MedicalDataSets(base_dir=args.base_dir, split="train",
                               transform=train_transform, train_file_dir=args.train_file_dir,
                               val_file_dir=args.val_file_dir,
                               divide_image_by_255=args.model == "CMUNeXt")
    db_val = MedicalDataSets(base_dir=args.base_dir, split="val", transform=val_transform,
                             train_file_dir=args.train_file_dir, val_file_dir=args.val_file_dir,
                             divide_image_by_255=args.model == "CMUNeXt")
    # <=== 修改 5: 将 print 替换为 logging.info
    logging.info("train num:{}, val num:{}".format(len(db_train), len(db_val)))

    train_sampler = build_train_sampler(args, db_train)
    trainloader = DataLoader(
        db_train,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=8,
        pin_memory=False,
    )
    valloader = DataLoader(db_val, batch_size=args.batch_size, shuffle=False, num_workers=4)

    return trainloader, valloader


def build_optimizer(args, model):
    if args.model != HSPM_FBDM_BEST0616_MODEL:
        trainable_parameters = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        return optim.SGD(
            trainable_parameters,
            lr=args.base_lr,
            momentum=0.9,
            weight_decay=0.0001,
        )

    fbdm_parameters = [
        parameter for parameter in model.fbdm1.parameters() if parameter.requires_grad
    ]
    correction_parameters = [
        parameter
        for parameter in model.fbdm_correction.parameters()
        if parameter.requires_grad
    ]
    branch_parameter_ids = {
        id(parameter) for parameter in fbdm_parameters + correction_parameters
    }
    hspm_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in branch_parameter_ids
    ]
    parameter_groups = [
        {
            "params": hspm_parameters,
            "lr": args.base_lr,
            "lr_multiplier": 1.0,
            "group_name": "hspm",
        },
        {
            "params": fbdm_parameters,
            "lr": args.base_lr * args.fbdm_lr_multiplier,
            "lr_multiplier": args.fbdm_lr_multiplier,
            "group_name": "fbdm",
        },
        {
            "params": correction_parameters,
            "lr": args.base_lr * args.fbdm_correction_lr_multiplier,
            "lr_multiplier": args.fbdm_correction_lr_multiplier,
            "group_name": "correction",
        },
    ]
    return optim.SGD(
        parameter_groups,
        lr=args.base_lr,
        momentum=0.9,
        weight_decay=0.0001,
    )


def main(args):
    validate_sampling_args(args)
    if args.model in USLGSF_V3_MODELS:
        if args.uslgsf_residual_init_scale <= 0:
            raise ValueError("USLGSF V3 residual initialization scale must be positive.")
    if args.hspm_coarse_loss_weight < 0:
        raise ValueError("hspm_coarse_loss_weight must be non-negative.")
    if args.hspm_coarse_loss_final_weight is not None and args.hspm_coarse_loss_final_weight < 0:
        raise ValueError("hspm_coarse_loss_final_weight must be non-negative.")
    if args.hspm_coarse_loss_decay_epochs < 0 or args.hspm_prototype_warmup_epochs < 0:
        raise ValueError("HSPM schedule epochs must be non-negative.")
    if args.fbdm_semantic_uncertainty_weight < 0 or args.fbdm_semantic_coarse_weight < 0:
        raise ValueError("FBDM semantic prior weights must be non-negative.")
    if not 0.0 <= args.fbdm_semantic_gate_base <= 1.0:
        raise ValueError("fbdm_semantic_gate_base must be in [0, 1].")
    if not 0.0 < args.fbdm_gate_init < args.fbdm_gate_max:
        raise ValueError("fbdm_gate_init must be in (0, fbdm_gate_max).")
    if args.fbdm_edge_loss_weight < 0:
        raise ValueError("fbdm_edge_loss_weight must be non-negative.")
    if args.fbdm_edge_pos_weight <= 0:
        raise ValueError("fbdm_edge_pos_weight must be positive.")
    if not 0.0 < args.fbdm_edge_focal_alpha < 1.0:
        raise ValueError("fbdm_edge_focal_alpha must be in (0, 1).")
    if args.fbdm_edge_focal_gamma < 0:
        raise ValueError("fbdm_edge_focal_gamma must be non-negative.")
    if not 0.0 < args.fbdm_x2_edge_ratio <= 1.0:
        raise ValueError("fbdm_x2_edge_ratio must be in (0, 1].")
    if args.fbdm_edge_loss_final_weight is not None and args.fbdm_edge_loss_final_weight < 0:
        raise ValueError("fbdm_edge_loss_final_weight must be non-negative.")
    if args.fbdm_edge_loss_decay_epochs < 0 or args.fbdm_residual_warmup_epochs < 0:
        raise ValueError("FBDM schedule epochs must be non-negative.")
    if not 0.0 < args.fbdm_correction_scale_init < args.fbdm_correction_scale_max:
        raise ValueError("FBDM V2 correction scale init must be in (0, correction scale max).")
    if not 0.0 <= args.fbdm_boundary_gate_floor < 1.0:
        raise ValueError("fbdm_boundary_gate_floor must be in [0, 1).")
    if args.fbdm_lr_multiplier <= 0 or args.fbdm_correction_lr_multiplier <= 0:
        raise ValueError("FBDM learning-rate multipliers must be positive.")
    if args.fbdm_correction_warmup_epochs < 0 or args.fbdm_correction_start_epoch < 0:
        raise ValueError("FBDM correction schedule epochs must be non-negative.")
    if args.fbdm_refine_loss_weight < 0 or args.fbdm_preserve_loss_weight < 0:
        raise ValueError("FBDM refine and preserve loss weights must be non-negative.")
    if args.fbdm_edge_kernel_size <= 0 or args.fbdm_edge_kernel_size % 2 == 0:
        raise ValueError("fbdm_edge_kernel_size must be a positive odd integer.")
    if args.fbdm_boundary_band_loss_weight < 0:
        raise ValueError("fbdm_boundary_band_loss_weight must be non-negative.")
    if (
        args.fbdm_boundary_band_loss_final_weight is not None
        and args.fbdm_boundary_band_loss_final_weight < 0
    ):
        raise ValueError("fbdm_boundary_band_loss_final_weight must be non-negative.")
    if args.fbdm_boundary_band_loss_decay_epochs < 0:
        raise ValueError("fbdm_boundary_band_loss_decay_epochs must be non-negative.")
    if args.fbdm_boundary_band_kernel_size <= 0 or args.fbdm_boundary_band_kernel_size % 2 == 0:
        raise ValueError("fbdm_boundary_band_kernel_size must be a positive odd integer.")
    boundary_band_enabled = args.fbdm_boundary_band_loss_weight > 0 or (
        args.fbdm_boundary_band_loss_final_weight is not None
        and args.fbdm_boundary_band_loss_final_weight > 0
    )
    if boundary_band_enabled and args.model not in HSPM_FBDM_CORRECTION_MODELS:
        raise ValueError("FBDM boundary-band loss requires a logit-correction model.")
    if not 0.0 < args.barm_gate_init < args.barm_gate_max:
        raise ValueError("barm_gate_init must be in (0, barm_gate_max).")
    if not 0.0 < args.barm_hf_keep_init < 1.0:
        raise ValueError("barm_hf_keep_init must be in (0, 1).")
    if args.barm_warmup_epochs < 0:
        raise ValueError("barm_warmup_epochs must be non-negative.")
    if (
        args.barm_coarse_loss_weight < 0
        or args.barm_boundary_loss_weight < 0
        or args.barm_edge_loss_weight < 0
    ):
        raise ValueError("BARM loss weights must be non-negative.")
    if args.barm_edge_band_width < 1:
        raise ValueError("barm_edge_band_width must be positive.")
    if args.barm_edge_pos_weight <= 0:
        raise ValueError("barm_edge_pos_weight must be positive.")
    if args.early_stop_patience < 0 or args.early_stop_min_delta < 0:
        raise ValueError("Early stopping settings must be non-negative.")

    # <=== 新增 5: 确保保存目录存在 (使用 exist_ok=True 避免已存在时出错)
    os.makedirs(args.save_dir, exist_ok=True)
    config_file_path = os.path.join(args.save_dir, 'training_config.json')
    with open(config_file_path, 'w', encoding='utf-8') as config_file:
        json.dump(vars(args), config_file, indent=2, ensure_ascii=True)

    # <=== 新增 6: 配置 logging
    log_file_path = os.path.join(args.save_dir, 'training_log.txt')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file_path),  # 保存到文件
            logging.StreamHandler(sys.stdout)  # 输出到控制台
        ]
    )
    # =================================

    base_lr = args.base_lr
    model = get_model(args)
    criterion = get_criterion(args)
    val_thresholds = build_validation_thresholds(args)
    trainloader, valloader = getDataloader(args=args)

    # <=== 修改 6: 将 print 替换为 logging.info
    logging.info("Args: {}".format(args))  # 打印所有参数到日志
    logging.info("train file dir:{} val file dir:{}".format(args.train_file_dir, args.val_file_dir))
    logging.info(
        "validation threshold mode:%s thresholds:%s metric:%s",
        args.val_threshold_mode,
        val_thresholds,
        args.val_threshold_metric,
    )
    optimizer = build_optimizer(args, model)
    logging.info(
        "Optimizer groups: %s",
        ", ".join(
            "%s=%.2fx"
            % (
                group.get("group_name", "default"),
                group.get("lr_multiplier", 1.0),
            )
            for group in optimizer.param_groups
        ),
    )

    # <=== 修改 7: 将 print 替换为 logging.info
    logging.info("{} iterations per epoch".format(len(trainloader)))
    best_iou = 0
    best_base_iou = 0
    early_stop_best_iou = float("-inf")
    early_stop_wait = 0
    iter_num = 0
    max_epoch = args.epoch

    train_loss_history = []
    train_iou_history = []
    val_loss_history = []
    val_iou_history = []
    train_base_iou_history = []
    val_base_iou_history = []
    val_threshold_history = []

    max_iterations = len(trainloader) * max_epoch

    start_time = time.time()

    for epoch_num in range(max_epoch):
        current_coarse_weight, prototype_scale, effective_gamma = configure_hspm_epoch(args, model, epoch_num)
        current_edge_weight, fbdm_residual_scale, effective_fbdm_gamma = configure_fbdm_epoch(
            args,
            model,
            epoch_num,
        )
        barm_residual_scale, barm_effective_gamma = configure_barm_epoch(args, model, epoch_num)
        current_fbdm_boundary_band_weight = (
            get_fbdm_boundary_band_weight(args, epoch_num)
            if args.model in HSPM_FBDM_CORRECTION_MODELS
            else 0.0
        )
        current_band_weight = current_fbdm_boundary_band_weight
        current_correction_schedule = get_fbdm_correction_schedule_scale(args, epoch_num)
        current_refine_weight = (
            args.fbdm_refine_loss_weight * current_correction_schedule
            if args.model == HSPM_FBDM_BEST0616_MODEL
            else 0.0
        )
        current_preserve_weight = (
            args.fbdm_preserve_loss_weight * current_correction_schedule
            if args.model == HSPM_FBDM_BEST0616_MODEL
            else 0.0
        )
        model.train()
        avg_meters = {'loss': AverageMeter(),
                      'iou': AverageMeter(),
                      'base_iou': AverageMeter(),
                      'val_loss': AverageMeter(),
                      'val_iou': AverageMeter(),
                      'val_SE': AverageMeter(),
                      'val_PC': AverageMeter(),
                      'val_F1': AverageMeter(),
                      'val_SP': AverageMeter(),
                      'val_HD95': AverageMeter(),
                      'val_ASSD': AverageMeter(),
                      'val_ACC': AverageMeter(),
                      'val_base_iou': AverageMeter(),
                      'val_base_SE': AverageMeter(),
                      'val_base_HD95': AverageMeter(),
                      'val_base_ASSD': AverageMeter(),
                      'fusion_predicted_area': AverageMeter(),
                      'fusion_smallness': AverageMeter(),
                      'fusion_spatial_gate': AverageMeter(),
                      'fusion_injection_deep_rms_ratio': AverageMeter(),
                      'fusion_small_injection_deep_rms_ratio': AverageMeter(),
                      'fusion_large_injection_deep_rms_ratio': AverageMeter()}
        if args.model in FBDM_MODELS:
            component_names = ["seg", "edge_raw", "edge_weighted"]
            if args.model in HSPM_FBDM_MODELS:
                component_names.insert(1, "coarse_weighted")
                component_names.append("boundary_band_weighted")
                component_names.extend(["refine_weighted", "preserve_weighted"])
            component_names.append("total")
            for prefix in ("train", "val"):
                for component_name in component_names:
                    avg_meters[f"{prefix}_loss_{component_name}"] = AverageMeter()
                for diagnostic_name in FBDM_EDGE_DIAGNOSTIC_NAMES:
                    avg_meters[f"{prefix}_fbdm_{diagnostic_name}"] = AverageMeter()
            if args.model in HSPM_FBDM_CORRECTION_MODELS:
                for diagnostic_name in FBDM_V2_DIAGNOSTIC_NAMES:
                    avg_meters[f"fbdm_v2_{diagnostic_name}"] = AverageMeter()
        if args.model in BARM_MODELS:
            component_names = ["seg", "coarse_weighted", "boundary_weighted", "edge_weighted", "total"]
            for prefix in ("train", "val"):
                for component_name in component_names:
                    avg_meters[f"{prefix}_loss_{component_name}"] = AverageMeter()
                for diagnostic_name in BARM_DIAGNOSTIC_NAMES:
                    avg_meters[f"{prefix}_barm_{diagnostic_name}"] = AverageMeter()
        if args.model in USLGSF_V3_MODELS:
            for stage in args.uslgsf_stages:
                for diagnostic_name in USLGSF_V3_DIAGNOSTIC_NAMES:
                    avg_meters[f"uslgsf_{stage}_{diagnostic_name}"] = AverageMeter()
        # (您修改的部分)
        train_bar = tqdm(trainloader, desc=f"Epoch {epoch_num}/{max_epoch} [Train]")

        for i_batch, sampled_batch in enumerate(train_bar):
            img_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            img_batch, label_batch = img_batch.cuda(), label_batch.cuda()

            outputs = forward_with_model(args, model, img_batch)
            seg_logits = get_seg_logits(outputs)

            loss_output = compute_loss(
                args,
                criterion,
                outputs,
                label_batch,
                sampled_batch,
                aux_weight=current_coarse_weight,
                edge_weight=current_edge_weight,
                band_weight=current_band_weight,
                refine_weight=current_refine_weight,
                preserve_weight=current_preserve_weight,
            )
            loss = get_loss_tensor(loss_output)
            update_loss_component_meters(
                avg_meters,
                get_loss_components(loss_output),
                "train",
                img_batch.size(0),
            )
            if args.model in FBDM_MODELS:
                update_fbdm_edge_diagnostic_meters(
                    avg_meters,
                    criterion,
                    "train",
                    img_batch.size(0),
                )
            if args.model in BARM_MODELS:
                update_barm_diagnostic_meters(
                    avg_meters,
                    model,
                    "train",
                    img_batch.size(0),
                )
            iou, dice, _, _, _, _, _ = iou_score(seg_logits, label_batch)
            if args.model == HSPM_FBDM_BEST0616_MODEL:
                base_iou, _, _, _, _, _, _ = iou_score(
                    outputs["base_seg"],
                    label_batch,
                )
                avg_meters['base_iou'].update(base_iou, img_batch.size(0))
            optimizer.zero_grad()
            if loss.requires_grad:
                loss.backward()
                optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_ * param_group.get('lr_multiplier', 1.0)

            iter_num = iter_num + 1
            avg_meters['loss'].update(loss.item(), img_batch.size(0))
            avg_meters['iou'].update(iou, img_batch.size(0))

            train_bar.set_postfix(loss=avg_meters['loss'].avg, iou=avg_meters['iou'].avg)

        model.eval()
        with torch.no_grad():
            val_bar = tqdm(valloader, desc=f"Epoch {epoch_num}/{max_epoch} [Val  ]")
            selected_threshold = args.val_threshold
            val_prob_batches = []
            val_base_prob_batches = []
            val_target_batches = []
            val_sample_count = 0

            for i_batch, sampled_batch in enumerate(val_bar):
                img_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                img_batch, label_batch = img_batch.cuda(), label_batch.cuda()
                output = forward_with_model(args, model, img_batch)
                seg_logits = get_seg_logits(output)
                base_seg_logits = (
                    output["base_seg"]
                    if args.model == HSPM_FBDM_BEST0616_MODEL
                    else None
                )
                if args.model in USLGSF_V3_MODELS:
                    uslgsf_diagnostics = get_uslgsf_diagnostics(model)
                    if uslgsf_diagnostics is not None:
                        for diagnostic_name, diagnostic_value in uslgsf_diagnostics.items():
                            avg_meters[f"uslgsf_{diagnostic_name}"].update(
                                diagnostic_value,
                                img_batch.size(0),
                            )
                fusion_diagnostics = (
                    get_hspm_fusion_diagnostics(model)
                    if args.model in HSPM_MODELS
                    else None
                )
                if fusion_diagnostics is not None:
                    for diagnostic_name in (
                        "predicted_area",
                        "smallness",
                        "spatial_gate",
                        "injection_deep_rms_ratio",
                    ):
                        if diagnostic_name in fusion_diagnostics:
                            avg_meters[f"fusion_{diagnostic_name}"].update(
                                fusion_diagnostics[diagnostic_name],
                                img_batch.size(0),
                            )
                    for area_group in ("small", "large"):
                        count = fusion_diagnostics.get(f"{area_group}_injection_count", 0.0)
                        if count > 0:
                            ratio_sum = fusion_diagnostics[f"{area_group}_injection_deep_rms_ratio_sum"]
                            avg_meters[f"fusion_{area_group}_injection_deep_rms_ratio"].update(
                                ratio_sum / count,
                                count,
                            )
                if args.model in HSPM_FBDM_CORRECTION_MODELS:
                    fbdm_v2_diagnostics = get_fbdm_v2_diagnostics(model)
                    if fbdm_v2_diagnostics is not None:
                        for diagnostic_name, diagnostic_value in fbdm_v2_diagnostics.items():
                            avg_meters[f"fbdm_v2_{diagnostic_name}"].update(
                                diagnostic_value,
                                img_batch.size(0),
                            )
                loss_output = compute_loss(
                    args,
                    criterion,
                    output,
                    label_batch,
                    sampled_batch,
                    aux_weight=current_coarse_weight,
                    edge_weight=current_edge_weight,
                    band_weight=current_band_weight,
                    refine_weight=current_refine_weight,
                    preserve_weight=current_preserve_weight,
                )
                loss = get_loss_tensor(loss_output)
                update_loss_component_meters(
                    avg_meters,
                    get_loss_components(loss_output),
                    "val",
                    img_batch.size(0),
                )
                if args.model in FBDM_MODELS:
                    update_fbdm_edge_diagnostic_meters(
                        avg_meters,
                        criterion,
                        "val",
                        img_batch.size(0),
                    )
                if args.model in BARM_MODELS:
                    update_barm_diagnostic_meters(
                        avg_meters,
                        model,
                        "val",
                        img_batch.size(0),
                    )
                avg_meters['val_loss'].update(loss.item(), img_batch.size(0))
                if args.val_threshold_mode == "scan":
                    val_prob_batches.append(torch.sigmoid(seg_logits).detach().cpu())
                    if base_seg_logits is not None:
                        val_base_prob_batches.append(
                            torch.sigmoid(base_seg_logits).detach().cpu()
                        )
                    val_target_batches.append(label_batch.detach().cpu())
                    val_sample_count += img_batch.size(0)
                    val_bar.set_postfix(val_loss=avg_meters['val_loss'].avg)
                    continue

                iou, _, SE, PC, F1, SP, ACC = iou_score(
                    seg_logits,
                    label_batch,
                    threshold=args.val_threshold,
                )
                hd95, assd = boundary_scores(
                    seg_logits,
                    label_batch,
                    threshold=args.val_threshold,
                )
                avg_meters['val_iou'].update(iou, img_batch.size(0))
                avg_meters['val_SE'].update(SE, img_batch.size(0))
                avg_meters['val_PC'].update(PC, img_batch.size(0))
                avg_meters['val_F1'].update(F1, img_batch.size(0))
                avg_meters['val_SP'].update(SP, img_batch.size(0))
                avg_meters['val_HD95'].update(hd95, img_batch.size(0))
                avg_meters['val_ASSD'].update(assd, img_batch.size(0))
                avg_meters['val_ACC'].update(ACC, img_batch.size(0))
                if base_seg_logits is not None:
                    base_iou, _, base_SE, _, _, _, _ = iou_score(
                        base_seg_logits,
                        label_batch,
                        threshold=args.val_threshold,
                    )
                    base_hd95, base_assd = boundary_scores(
                        base_seg_logits,
                        label_batch,
                        threshold=args.val_threshold,
                    )
                    avg_meters['val_base_iou'].update(base_iou, img_batch.size(0))
                    avg_meters['val_base_SE'].update(base_SE, img_batch.size(0))
                    avg_meters['val_base_HD95'].update(base_hd95, img_batch.size(0))
                    avg_meters['val_base_ASSD'].update(base_assd, img_batch.size(0))
                val_bar.set_postfix(val_loss=avg_meters['val_loss'].avg, val_iou=avg_meters['val_iou'].avg)

            if args.val_threshold_mode == "scan":
                val_metrics = find_best_threshold(
                    val_prob_batches,
                    val_target_batches,
                    val_thresholds,
                    select_metric=args.val_threshold_metric,
                    from_logits=False,
                )
                selected_threshold = val_metrics['threshold']
                avg_meters['val_iou'].update(val_metrics['iou'], val_sample_count)
                avg_meters['val_SE'].update(val_metrics['se'], val_sample_count)
                avg_meters['val_PC'].update(val_metrics['pc'], val_sample_count)
                avg_meters['val_F1'].update(val_metrics['f1'], val_sample_count)
                avg_meters['val_SP'].update(val_metrics['sp'], val_sample_count)
                avg_meters['val_HD95'].update(val_metrics['hd95'], val_sample_count)
                avg_meters['val_ASSD'].update(val_metrics['assd'], val_sample_count)
                avg_meters['val_ACC'].update(val_metrics['acc'], val_sample_count)
                if val_base_prob_batches:
                    base_metrics = find_best_threshold(
                        val_base_prob_batches,
                        val_target_batches,
                        [selected_threshold],
                        select_metric=args.val_threshold_metric,
                        from_logits=False,
                    )
                    avg_meters['val_base_iou'].update(base_metrics['iou'], val_sample_count)
                    avg_meters['val_base_SE'].update(base_metrics['se'], val_sample_count)
                    avg_meters['val_base_HD95'].update(base_metrics['hd95'], val_sample_count)
                    avg_meters['val_base_ASSD'].update(base_metrics['assd'], val_sample_count)

        # <=== 修改 8: 将 print 替换为 logging.info
        if isinstance(trainloader.sampler, TrackingWeightedRandomSampler):
            logging.info(
                'Targeted sampling actual counts [epoch %d/%d]: %s',
                epoch_num,
                max_epoch,
                _format_sampling_groups(trainloader.sampler.last_group_counts()),
            )

        elapsed_time = time.time() - start_time
        elapsed_str = time.strftime('%H:%M:%S', time.gmtime(elapsed_time))
        if args.model in HSPM_MODELS:
            hspm_model = model.module if isinstance(model, torch.nn.DataParallel) else model
            effective_gamma = float(hspm_model.prototype_mixer.effective_gamma().detach().cpu())
            if args.hspm_mixer_mode == "stable":
                effective_gamma *= prototype_scale
            fusion_gate = hspm_model.effective_fusion_gate()
            effective_fusion_gate = None if fusion_gate is None else float(fusion_gate.detach().cpu())
            logging.info(
                "HSPM schedule: coarse_weight=%.4f - prototype_scale=%.4f - effective_gamma=%.4f"
                " - effective_fusion_gate=%s",
                current_coarse_weight,
                prototype_scale,
                effective_gamma,
                "n/a" if effective_fusion_gate is None else f"{effective_fusion_gate:.4f}",
            )
            if avg_meters['fusion_injection_deep_rms_ratio'].count > 0:
                logging.info(
                    "HSPM fusion diagnostics: predicted_area=%s - smallness=%s - spatial_gate=%s"
                    " - injection_deep_rms_ratio=%.6f",
                    (
                        "n/a"
                        if avg_meters['fusion_predicted_area'].count == 0
                        else f"{avg_meters['fusion_predicted_area'].avg:.6f}"
                    ),
                    (
                        "n/a"
                        if avg_meters['fusion_smallness'].count == 0
                        else f"{avg_meters['fusion_smallness'].avg:.6f}"
                    ),
                    (
                        "n/a"
                        if avg_meters['fusion_spatial_gate'].count == 0
                        else f"{avg_meters['fusion_spatial_gate'].avg:.6f}"
                    ),
                    avg_meters['fusion_injection_deep_rms_ratio'].avg,
                )
                if (
                    avg_meters['fusion_small_injection_deep_rms_ratio'].count > 0
                    or avg_meters['fusion_large_injection_deep_rms_ratio'].count > 0
                ):
                    logging.info(
                        "HSPM fusion area diagnostics: small_injection_deep_rms_ratio=%s"
                        " - large_injection_deep_rms_ratio=%s",
                        (
                            "n/a"
                            if avg_meters['fusion_small_injection_deep_rms_ratio'].count == 0
                            else f"{avg_meters['fusion_small_injection_deep_rms_ratio'].avg:.6f}"
                        ),
                        (
                            "n/a"
                            if avg_meters['fusion_large_injection_deep_rms_ratio'].count == 0
                            else f"{avg_meters['fusion_large_injection_deep_rms_ratio'].avg:.6f}"
                        ),
                    )
        if args.model in FBDM_MODELS:
            logging.info(
                "FBDM schedule: edge_weight=%.4f - residual_scale=%.4f - effective_gamma=%.4f",
                current_edge_weight,
                fbdm_residual_scale,
                effective_fbdm_gamma,
            )
            if args.model in HSPM_FBDM_MODELS:
                logging.info(
                    "FBDM loss components: train(seg=%.6f - coarse=%.6f"
                    " - edge_raw=%.6f - edge=%.6f - band=%.6f - refine=%.6f"
                    " - preserve=%.6f - total=%.6f)"
                    " - val(seg=%.6f - coarse=%.6f - edge_raw=%.6f - edge=%.6f"
                    " - band=%.6f - refine=%.6f - preserve=%.6f - total=%.6f)",
                    avg_meters["train_loss_seg"].avg,
                    avg_meters["train_loss_coarse_weighted"].avg,
                    avg_meters["train_loss_edge_raw"].avg,
                    avg_meters["train_loss_edge_weighted"].avg,
                    avg_meters["train_loss_boundary_band_weighted"].avg,
                    avg_meters["train_loss_refine_weighted"].avg,
                    avg_meters["train_loss_preserve_weighted"].avg,
                    avg_meters["train_loss_total"].avg,
                    avg_meters["val_loss_seg"].avg,
                    avg_meters["val_loss_coarse_weighted"].avg,
                    avg_meters["val_loss_edge_raw"].avg,
                    avg_meters["val_loss_edge_weighted"].avg,
                    avg_meters["val_loss_boundary_band_weighted"].avg,
                    avg_meters["val_loss_refine_weighted"].avg,
                    avg_meters["val_loss_preserve_weighted"].avg,
                    avg_meters["val_loss_total"].avg,
                )
            else:
                logging.info(
                    "FBDM loss components: train(seg=%.6f - edge_raw=%.6f - edge=%.6f"
                    " - total=%.6f) - val(seg=%.6f - edge_raw=%.6f"
                    " - edge=%.6f - total=%.6f)",
                    avg_meters["train_loss_seg"].avg,
                    avg_meters["train_loss_edge_raw"].avg,
                    avg_meters["train_loss_edge_weighted"].avg,
                    avg_meters["train_loss_total"].avg,
                    avg_meters["val_loss_seg"].avg,
                    avg_meters["val_loss_edge_raw"].avg,
                    avg_meters["val_loss_edge_weighted"].avg,
                    avg_meters["val_loss_total"].avg,
                )
            logging.info(
                "FBDM edge diagnostics [%s]: train(target=%.6f - prob=%.6f"
                " - positive@0.5=%.6f) - val(target=%.6f - prob=%.6f"
                " - positive@0.5=%.6f)",
                args.fbdm_edge_loss_type,
                avg_meters["train_fbdm_edge_target_ratio"].avg,
                avg_meters["train_fbdm_edge_prob_mean"].avg,
                avg_meters["train_fbdm_edge_pred_positive_ratio"].avg,
                avg_meters["val_fbdm_edge_target_ratio"].avg,
                avg_meters["val_fbdm_edge_prob_mean"].avg,
                avg_meters["val_fbdm_edge_pred_positive_ratio"].avg,
            )
            if args.model in HSPM_FBDM_CORRECTION_MODELS:
                if args.model == HSPM_FBDM_BEST0616_MODEL:
                    logging.info(
                        "FBDM-v2 correction: schedule=%.4f - scale=%.6f"
                        " - boundary_gate_mean=%.6f - boundary_gate>0.5=%.6f"
                        " - safe_gate_mean=%.6f - boundary_feature_rms=%.6f"
                        " - raw_correction_mean=%.6f - raw_correction_max=%.6f"
                        " - logit_correction_mean=%.6f - logit_correction_max=%.6f"
                        " - prediction_flip_ratio=%.6f",
                        avg_meters["fbdm_v2_correction_schedule_scale"].avg,
                        avg_meters["fbdm_v2_effective_correction_scale"].avg,
                        avg_meters["fbdm_v2_boundary_gate_mean"].avg,
                        avg_meters["fbdm_v2_boundary_gate_over_05"].avg,
                        avg_meters["fbdm_v2_safe_boundary_gate_mean"].avg,
                        avg_meters["fbdm_v2_boundary_feature_rms"].avg,
                        avg_meters["fbdm_v2_raw_correction_abs_mean"].avg,
                        avg_meters["fbdm_v2_raw_correction_abs_max"].avg,
                        avg_meters["fbdm_v2_logit_correction_abs_mean"].avg,
                        avg_meters["fbdm_v2_logit_correction_abs_max"].avg,
                        avg_meters["fbdm_v2_prediction_flip_ratio"].avg,
                    )
                else:
                    logging.info(
                        "FBDM-v2 correction: schedule=%.4f - scale=%.6f"
                        " - boundary_gate_mean=%.6f - boundary_gate>0.5=%.6f"
                        " - logit_correction_mean=%.6f - logit_correction_max=%.6f"
                        " - prediction_flip_ratio=%.6f",
                        avg_meters["fbdm_v2_correction_schedule_scale"].avg,
                        avg_meters["fbdm_v2_effective_correction_scale"].avg,
                        avg_meters["fbdm_v2_boundary_gate_mean"].avg,
                        avg_meters["fbdm_v2_boundary_gate_over_05"].avg,
                        avg_meters["fbdm_v2_logit_correction_abs_mean"].avg,
                        avg_meters["fbdm_v2_logit_correction_abs_max"].avg,
                        avg_meters["fbdm_v2_prediction_flip_ratio"].avg,
                    )
                activation_epoch = (
                    args.fbdm_correction_start_epoch
                    + args.fbdm_correction_warmup_epochs
                )
                if (
                    args.model == HSPM_FBDM_BEST0616_MODEL
                    and epoch_num == activation_epoch
                    and (
                        avg_meters["fbdm_v2_logit_correction_abs_max"].avg < 0.01
                        or avg_meters["fbdm_v2_prediction_flip_ratio"].avg <= 0.0
                    )
                ):
                    logging.warning(
                        "FBDM correction remains under-active at epoch %d:"
                        " correction_max=%.6f, prediction_flip_ratio=%.6f.",
                        epoch_num,
                        avg_meters["fbdm_v2_logit_correction_abs_max"].avg,
                        avg_meters["fbdm_v2_prediction_flip_ratio"].avg,
                    )
            if args.model == HSPM_FBDM_BEST0616_MODEL:
                logging.info(
                    "Protected HSPM-FBDM metrics: refine_weight=%.4f"
                    " - preserve_weight=%.4f - train_base_iou=%.4f"
                    " - val_base_iou=%.4f - val_base_SE=%.4f"
                    " - val_base_HD95=%.4f - val_base_ASSD=%.4f",
                    current_refine_weight,
                    current_preserve_weight,
                    avg_meters["base_iou"].avg,
                    avg_meters["val_base_iou"].avg,
                    avg_meters["val_base_SE"].avg,
                    avg_meters["val_base_HD95"].avg,
                    avg_meters["val_base_ASSD"].avg,
                )
        if args.model in BARM_MODELS:
            logging.info(
                "BARM schedule: residual_scale=%.4f - effective_gamma=%.4f",
                barm_residual_scale,
                barm_effective_gamma,
            )
            logging.info(
                "BARM loss components: train(seg=%.6f - coarse=%.6f - boundary=%.6f"
                " - edge=%.6f - total=%.6f) - val(seg=%.6f - coarse=%.6f"
                " - boundary=%.6f - edge=%.6f - total=%.6f)",
                avg_meters["train_loss_seg"].avg,
                avg_meters["train_loss_coarse_weighted"].avg,
                avg_meters["train_loss_boundary_weighted"].avg,
                avg_meters["train_loss_edge_weighted"].avg,
                avg_meters["train_loss_total"].avg,
                avg_meters["val_loss_seg"].avg,
                avg_meters["val_loss_coarse_weighted"].avg,
                avg_meters["val_loss_boundary_weighted"].avg,
                avg_meters["val_loss_edge_weighted"].avg,
                avg_meters["val_loss_total"].avg,
            )
            logging.info(
                "BARM diagnostics: train(band=%.6f - hf=%.6f - correction_mean=%.6f"
                " - correction_max=%.6f - flip=%.6f) - val(band=%.6f - hf=%.6f"
                " - correction_mean=%.6f - correction_max=%.6f - flip=%.6f)",
                avg_meters["train_barm_band_mean"].avg,
                avg_meters["train_barm_hf_energy_mean"].avg,
                avg_meters["train_barm_logit_correction_abs_mean"].avg,
                avg_meters["train_barm_logit_correction_abs_max"].avg,
                avg_meters["train_barm_prediction_flip_ratio"].avg,
                avg_meters["val_barm_band_mean"].avg,
                avg_meters["val_barm_hf_energy_mean"].avg,
                avg_meters["val_barm_logit_correction_abs_mean"].avg,
                avg_meters["val_barm_logit_correction_abs_max"].avg,
                avg_meters["val_barm_prediction_flip_ratio"].avg,
            )
        if args.model in USLGSF_V3_MODELS:
            for stage in args.uslgsf_stages:
                logging.info(
                    "USLGSF-v3 stage %s diagnostics: structure=%.6f"
                    " - relevance=%.6f - structure_weight=%.6f"
                    " - relevance_weight=%.6f - active_gate=%.6f"
                    " - route_scale=%.6f - alpha=%.6f - residual_delta=%.6f"
                    " - injection_encoder_rms_ratio=%.6f",
                    stage,
                    avg_meters[f"uslgsf_{stage}_structure_reliability_mean"].avg,
                    avg_meters[f"uslgsf_{stage}_decoder_relevance_mean"].avg,
                    avg_meters[f"uslgsf_{stage}_structure_weight_mean"].avg,
                    avg_meters[f"uslgsf_{stage}_relevance_weight_mean"].avg,
                    avg_meters[f"uslgsf_{stage}_active_gate_mean"].avg,
                    avg_meters[f"uslgsf_{stage}_route_scale"].avg,
                    avg_meters[f"uslgsf_{stage}_effective_alpha"].avg,
                    avg_meters[f"uslgsf_{stage}_residual_delta_abs_mean"].avg,
                    avg_meters[f"uslgsf_{stage}_injection_encoder_rms_ratio"].avg,
                )
        if args.val_threshold_mode == "scan":
            logging.info(
                'epoch [%d/%d] (Total time: %s)  train_loss : %.4f, train_iou: %.4f - val_loss %.4f - val_thr %.4f - '
                'val_iou %.4f - val_SE %.4f - val_PC %.4f - val_F1 %.4f - val_SP %.4f - val_HD95 %.4f - '
                'val_ASSD %.4f - val_ACC %.4f '
                % (epoch_num, max_epoch, elapsed_str,
                   avg_meters['loss'].avg, avg_meters['iou'].avg,
                   avg_meters['val_loss'].avg, selected_threshold, avg_meters['val_iou'].avg, avg_meters['val_SE'].avg,
                   avg_meters['val_PC'].avg, avg_meters['val_F1'].avg, avg_meters['val_SP'].avg,
                   avg_meters['val_HD95'].avg, avg_meters['val_ASSD'].avg, avg_meters['val_ACC'].avg))
        else:
            logging.info(
                'epoch [%d/%d] (Total time: %s)  train_loss : %.4f, train_iou: %.4f - val_loss %.4f - val_iou %.4f - '
                'val_SE %.4f - val_PC %.4f - val_F1 %.4f - val_SP %.4f - val_HD95 %.4f - val_ASSD %.4f - val_ACC %.4f '
                % (epoch_num, max_epoch, elapsed_str,
                   avg_meters['loss'].avg, avg_meters['iou'].avg,
                   avg_meters['val_loss'].avg, avg_meters['val_iou'].avg, avg_meters['val_SE'].avg,
                   avg_meters['val_PC'].avg, avg_meters['val_F1'].avg, avg_meters['val_SP'].avg,
                   avg_meters['val_HD95'].avg, avg_meters['val_ASSD'].avg, avg_meters['val_ACC'].avg))
        # <=========================================

        train_loss_history.append(avg_meters['loss'].avg)
        train_iou_history.append(avg_meters['iou'].avg)
        val_loss_history.append(avg_meters['val_loss'].avg)
        val_iou_history.append(avg_meters['val_iou'].avg)
        if args.model == HSPM_FBDM_BEST0616_MODEL:
            train_base_iou_history.append(avg_meters['base_iou'].avg)
            val_base_iou_history.append(avg_meters['val_base_iou'].avg)
        if args.val_threshold_mode == "scan":
            val_threshold_history.append(selected_threshold)

        if avg_meters['val_iou'].avg > best_iou:
            # <=== 修改 9: 使用 args.save_dir 来构建路径
            # 目录已在 main 开头创建，这里无需检查
            save_file_path = os.path.join(args.save_dir, '{}_model.pth'.format(args.model))
            torch.save(model.state_dict(), save_file_path)
            best_iou = avg_meters['val_iou'].avg

            # <=== 修改 10: 将 print 替换为 logging.info
            if args.val_threshold_mode == "scan":
                threshold_file_path = os.path.join(args.save_dir, f'{args.model}_best_threshold.txt')
                with open(threshold_file_path, 'w', encoding='utf-8') as threshold_file:
                    threshold_file.write(f"{selected_threshold:.4f}\n")
                logging.info(f"=> saved best model to {save_file_path} (threshold={selected_threshold:.4f})")
            else:
                logging.info(f"=> saved best model to {save_file_path}")

        if (
            args.model == HSPM_FBDM_BEST0616_MODEL
            and avg_meters['val_base_iou'].avg > best_base_iou
        ):
            base_save_path = os.path.join(
                args.save_dir,
                '{}_model_base_best.pth'.format(args.model),
            )
            torch.save(model.state_dict(), base_save_path)
            best_base_iou = avg_meters['val_base_iou'].avg
            logging.info(
                "=> saved best protected HSPM base model to %s (base_iou=%.4f)",
                base_save_path,
                best_base_iou,
            )

        if args.early_stop_patience > 0:
            current_iou = avg_meters['val_iou'].avg
            if current_iou > early_stop_best_iou + args.early_stop_min_delta:
                early_stop_best_iou = current_iou
                early_stop_wait = 0
            else:
                early_stop_wait += 1
                logging.info(
                    "Early stopping wait: %d/%d (best significant val_iou=%.4f)",
                    early_stop_wait,
                    args.early_stop_patience,
                    early_stop_best_iou,
                )
                if early_stop_wait >= args.early_stop_patience:
                    logging.info("=> Early stopping triggered at epoch %d", epoch_num)
                    break

    # <=== 修改 11: 将 print 替换为 logging.info，并使用 args.save_dir
    # 目录已在 main 开头创建，这里无需检查
    logging.info("Saving metric history...")
    np.save(os.path.join(args.save_dir, f'{args.model}_train_loss.npy'), np.array(train_loss_history))
    np.save(os.path.join(args.save_dir, f'{args.model}_train_iou.npy'), np.array(train_iou_history))
    np.save(os.path.join(args.save_dir, f'{args.model}_val_loss.npy'), np.array(val_loss_history))
    np.save(os.path.join(args.save_dir, f'{args.model}_val_iou.npy'), np.array(val_iou_history))
    if args.model == HSPM_FBDM_BEST0616_MODEL:
        np.save(
            os.path.join(args.save_dir, f'{args.model}_train_base_iou.npy'),
            np.array(train_base_iou_history),
        )
        np.save(
            os.path.join(args.save_dir, f'{args.model}_val_base_iou.npy'),
            np.array(val_base_iou_history),
        )
    if args.val_threshold_mode == "scan":
        np.save(os.path.join(args.save_dir, f'{args.model}_val_threshold.npy'), np.array(val_threshold_history))

    # <=== 新增 7: 绘制并保存训练曲线图
    logging.info("Saving training curve plots...")
    epochs = range(1, len(train_loss_history) + 1)

    # 绘制 Loss 曲线
    plt.figure()
    plt.plot(epochs, train_loss_history, 'b', label='Training Loss')
    plt.plot(epochs, val_loss_history, 'r', label='Validation Loss')
    plt.title(f'{args.model} - Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(args.save_dir, f'{args.model}_loss_plot.png'))
    plt.close()

    # 绘制 IoU 曲线
    plt.figure()
    plt.plot(epochs, train_iou_history, 'b', label='Training IoU')
    plt.plot(epochs, val_iou_history, 'r', label='Validation IoU')
    plt.title(f'{args.model} - Training and Validation IoU')
    plt.xlabel('Epochs')
    plt.ylabel('IoU')
    plt.legend()
    plt.savefig(os.path.join(args.save_dir, f'{args.model}_iou_plot.png'))
    plt.close()
    # =================================
    # 保存最后一个 epoch 的模型权重
    last_model_path = os.path.join(args.save_dir, '{}_model_last.pth'.format(args.model))
    logging.info(f"=> Saving last epoch model to {last_model_path}")
    torch.save(model.state_dict(), last_model_path)
    # ++++++++++++++++ 添加结束 ++++++++++++++++

    logging.info("Training Finished!")
    return "Training Finished!"


if __name__ == "__main__":
    main(args)



#  git add .
#  git commit -m "提交信息"

#  libgomp: Invalid value for environment variable OMP_NUM_THREADS：     echo $OMP_NUM_THREADS    export OMP_NUM_THREADS=4
#  15
#  启动数据增强     --use_extra_aug
#  启动阈值扫描   --val_threshold_mode scan


# python main.py --model CMUNeXt --base_dir ./data/busi --train_file_dir busi_train3.txt --val_file_dir busi_val3.txt --save_dir ./checkpoint/5.28/busi-CMUNeXt-3-d --base_lr 0.01 --epoch 300 --batch_size 8

# python main.py --model CMUNeXt --base_dir ./data/busi_1 --train_file_dir train.txt --val_file_dir val.txt --save_dir ./checkpoint/5.28/busi_1-CMUNeXt-a --base_lr 0.01 --epoch 300 --batch_size 8

# python main.py --model CMUNeXt_DualGAG --base_dir ./data/busi --train_file_dir busi_train3.txt --val_file_dir busi_val3.txt --save_dir ./checkpoint/5.28/busi-CMUNeXt_DualGAG-3-a --base_lr 0.01 --epoch 300 --batch_size 8

# python main.py --model CMUNeXt_SpeckleEnhance --base_dir ./data/busi --train_file_dir busi_train3.txt --val_file_dir busi_val3.txt --save_dir ./checkpoint/5.28/busi-CMUNeXt_SpeckleEnhance0123-3-b --base_lr 0.01 --epoch 300 --batch_size 8 --ddsr_stages 0,1,2,3

# python main.py --model CMUNeXt_DualGAG_SpeckleEnhance --base_dir ./data/busi --train_file_dir busi_train3.txt --val_file_dir busi_val3.txt --save_dir ./checkpoint/5.28/busi-CMUNeXt_DualGAG_SpeckleEnhance-3-d --base_lr 0.01 --epoch 300 --batch_size 8 --ddsr_stages 0,1 --gag_stages 0,1,2,3 --ddsr_aux_init 0.05 --use_extra_aug
