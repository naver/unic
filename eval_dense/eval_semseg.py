import os
import sys
import os.path as osp
import time
from argparse import ArgumentParser
from functools import partial

import torch

import mmcv
from mmseg.datasets import build_dataset
from mmseg.apis import set_random_seed, train_segmentor
from mmseg.utils import get_device
from mmengine import Config
from eval_dense import utils as dutils

from modeling.unic import load_student_encoder_from_checkpoint


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--pretrained",
        type=str,
        required=True,
        help="Path to the student checkpoint",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Path to the output directory "
        "if empty, it is set automatically to "
        "<student_dir>/semseg_<head_type>_<dataset>_<seed>",
    )
    parser.add_argument(
        "--purge_output_dir",
        action="store_true",
        help="if set, delete the output dir if it exists",
    )
    # hyperparameters
    parser.add_argument(
        "--samples_per_gpu",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--iter_with_8_gpu",
        type=int,
        default=10000,
        help="set 40000 to match dinov2",
    )
    parser.add_argument(
        "--head_type",
        type=str,
        default="linear",
        choices=["linear", "ms"],
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the data directory",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ade20k",
        choices=["ade20k"],
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0,
        help="Custom learning rate to overwrite the config file if > 0",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the random number generators",
    )
    # logging
    parser.add_argument(
        "--delete_ckpts",
        action="store_true",
        help="whether to delete checkpoints at the end",
    )
    parser.add_argument("--disable_tb", action="store_true")
    args = parser.parse_args()

    assert os.path.isfile(args.pretrained), "Student checkpoint not found"

    return args


def load_cfg(
    backbone_name,
    output_dir,
    data_dir,
    samples_per_gpu=2,
    iter_with_8_gpu=40000,
    head_type="linear",
    dataset="ade20k",
    lr=0.0,
    seed=0,
):
    HEAD_SCALE_COUNT = 3  # more scales: slower but better results, in (1,2,3,4,5)

    assert head_type in ["linear", "ms"], "unsuported head type"
    assert dataset in ["ade20k"], "unsuported dataset"

    # eg https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_ade20k_linear_config.py
    DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"
    head_config_url = f"{DINOV2_BASE_URL}/{backbone_name}/{backbone_name}_{dataset}_{head_type}_config.py"
    cfg_str = dutils.load_config_from_url(head_config_url)
    cfg = Config.fromstring(cfg_str, file_format=".py")

    if head_type == "ms":
        cfg.data.test.pipeline[1]["img_ratios"] = cfg.data.test.pipeline[1][
            "img_ratios"
        ][:HEAD_SCALE_COUNT]
        print("scales:", cfg.data.test.pipeline[1]["img_ratios"])

    cfg.work_dir = output_dir
    cfg.data.train.data_root = data_dir
    cfg.data.test.data_root = data_dir
    cfg.data.val.data_root = data_dir

    cfg.seed = seed
    set_random_seed(seed, deterministic=False)
    cfg.gpu_ids = range(1)
    cfg.device = get_device()

    cfg.data.samples_per_gpu = samples_per_gpu
    cfg.runner.max_iters = ()
    cfg.runner.max_iters = iter_with_8_gpu * int((8 / len(cfg.gpu_ids)))
    cfg.lr_config.policy = "Poly"
    if lr > 0:
        cfg.optimizer.lr = lr

    ii = min(10000, cfg.runner.max_iters)
    cfg.checkpoint_config = dict(by_epoch=False, max_keep_ckpts=2, interval=ii)
    cfg.evaluation = dict(interval=ii, metric="mIoU", pre_eval=True)

    return cfg


def build_datasets(cfg):
    datasets = [build_dataset(cfg.data.train)]
    return datasets


def build_model(cfg, backbone_model):
    from mmseg.models import build_segmentor
    from dinov2_mmcv import CenterPadding

    backbone_model.to(cfg.device)

    # Build the segmentor
    model = build_segmentor(cfg.model)

    # backbone_model.train()
    model.backbone.forward = partial(
        backbone_model.get_intermediate_layers,
        n=cfg.model.backbone.out_indices,
        reshape=True,
        return_all_tokens=False,
    )
    model.train()

    for n, param in model.named_parameters():
        param.requires_grad = True
    for n, param in model.backbone.named_parameters():
        param.requires_grad = False

    if hasattr(backbone_model, "patch_size"):
        model.backbone.register_forward_pre_hook(
            lambda _, x: CenterPadding(backbone_model.patch_size)(x[0])
        )

    model.init_weights()

    model.CLASSES = datasets[0].CLASSES

    return model


if __name__ == "__main__":
    args = get_args()
    setup_name = dutils.prepare_dirs("semseg", args)

    res_fname, miou = dutils.check_for_results(args)
    if miou is not None:
        print("Computation already done")
        print("mIoU: " + str(miou))
        sys.exit()

    encoder, epoch = load_student_encoder_from_checkpoint(args.pretrained)
    backbone_name = dutils.get_backbone_name_from_model(encoder)
    cfg = load_cfg(
        backbone_name,
        args.output_dir,
        args.data_dir,
        samples_per_gpu=args.samples_per_gpu,
        iter_with_8_gpu=args.iter_with_8_gpu,
        head_type=args.head_type,
        dataset=args.dataset,
        lr=args.lr,
        seed=args.seed,
    )

    datasets = build_datasets(cfg)

    model = build_model(cfg, encoder)

    # Create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    train_segmentor(
        model,
        datasets,
        cfg,
        distributed=False,
        validate=True,
        timestamp=timestamp,
        meta=dict(),
    )

    # parse the json to get the final results
    lines = dutils.read_final_log_file(args.output_dir)
    miou = eval(lines[-1].strip())["mIoU"]
    print("mIoU: " + str(miou))

    with open(res_fname, "w") as fid:
        fid.write(f"{miou}")

    if not args.disable_tb and os.path.isfile(args.pretrained):
        dutils.log_tensorboard(setup_name, {"mIoU": miou}, epoch, args.pretrained)

    if args.delete_ckpts:
        print("Deleting checkpoints ...")
        dutils.delete_checkpoints(args.output_dir)
