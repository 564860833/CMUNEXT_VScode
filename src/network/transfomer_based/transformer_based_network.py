from pathlib import Path

from .transUnet.transunet import TransUnet
from .swinUnet.vision_transformer import SwinUnet
from .swinUnet.config import get_config
from .medicalT.axialnet import MedT


def _resolve_swin_pretrained_path(config, cfg_path):
    pretrained_path = config.MODEL.PRETRAIN_CKPT
    if pretrained_path is None:
        return config

    path = Path(pretrained_path)
    if path.is_absolute():
        resolved_path = path
    else:
        cwd_path = (Path.cwd() / path).resolve()
        cfg_relative_path = (Path(cfg_path).resolve().parent / path).resolve()
        if cwd_path.exists():
            resolved_path = cwd_path
        elif cfg_relative_path.exists():
            resolved_path = cfg_relative_path
        else:
            raise FileNotFoundError(
                "SwinUnet pretrained checkpoint not found. Tried: "
                f"{cwd_path} and {cfg_relative_path}"
            )

    config.defrost()
    config.MODEL.PRETRAIN_CKPT = str(resolved_path)
    config.freeze()
    return config


def get_transformer_based_model(
    parser,
    model_name: str,
    img_size: int,
    num_classes: int,
    in_ch: int,
    load_pretrained: bool = False,
):
    if model_name == "MedT":
        model = MedT(img_size=img_size, imgchan=in_ch, num_classes=num_classes)
    elif model_name == "SwinUnet":
        parser.add_argument('--zip', action='store_true',
                            help='use zipped dataset instead of folder dataset')
        parser.add_argument(
            '--cfg', type=str, default="./src/network/transfomer_based/swinUnet/swin_tiny_patch4_window7_224_lite.yaml",
            help='path to config file', )
        parser.add_argument(
            "--opts",
            help="Modify config options by adding 'KEY VALUE' pairs. ",
            default=None,
            nargs='+',
        )
        parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                            help='no: no cache, '
                                 'full: cache all data, '
                                 'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
        parser.add_argument('--resume', help='resume from checkpoint')
        parser.add_argument('--accumulation-steps', type=int,
                            help="gradient accumulation steps")
        parser.add_argument('--use-checkpoint', action='store_true',
                            help="whether to use gradient checkpointing to save memory")
        parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                            help='mixed precision opt level, if O0, no amp is used')
        parser.add_argument('--tag', help='tag of experiment')
        parser.add_argument('--eval', action='store_true',
                            help='Perform evaluation only')
        parser.add_argument('--throughput', action='store_true',
                            help='Test throughput only')
        parsed_args = parser.parse_args()
        config = get_config(parsed_args)
        model = SwinUnet(config, img_size=224, num_classes=num_classes)
        if load_pretrained:
            config = _resolve_swin_pretrained_path(config, parsed_args.cfg)
            model.load_from(config)
    elif model_name == "TransUnet":
        model = TransUnet(img_ch=in_ch, output_ch=num_classes)
    else:
        model = None
        print("model err")
        exit(0)
    return model
