import argparse
import datetime
import math
import os
import shutil
import sys
import time
import logging

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
from torchvision.transforms import v2 as T
from torchvision.datasets import ImageFolder

import utils
from modeling.unic import build_student_from_args
from teachers import build_teachers, get_teacher_output
from modeling.losses import unic_loss
from dinov2.logging import setup_logging, ExternalLogger, MetricLogger
from dinov2.distributed import get_global_rank


logger = logging.getLogger()


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--arch",
        type=str,
        default="vit_base",
        help="Architecture of the student model. "
        "See dinov2/models/vision_transformer.py for options.",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=16,
        help="Patch size for the student model.",
    )
    parser.add_argument(
        "--drop_path_rate",
        type=float,
        default=0.1,
        help="Drop path rate for the student model.",
    )
    parser.add_argument(
        "--lp_args",
        type=str,
        default="{'std': 0.02}",
        help="Dictionary of keyword arguments for the ladder of projector.",
    )

    parser.add_argument(
        "--teachers",
        type=str,
        default="dino_vitbase_16,deit3_vitbase_16,ibot_vitbase_16,dbotft_vitbase_16",
        help="Comma-separated list of teacher names.",
    )
    parser.add_argument(
        "--tnorm_ema_momentum_start",
        type=float,
        default=1.0,
        help="Starting value for the EMA momentum for teacher feature statistics.",
    )
    parser.add_argument(
        "--tnorm_ema_momentum_end",
        type=float,
        default=0.001,
        help="Final value for the EMA momentum for teacher feature statistics.",
    )
    parser.add_argument(
        "--t_drop_prob",
        type=float,
        default=0.5,
        help="Probability of dropping a teacher in loss.",
    )
    parser.add_argument(
        "--lam_lcos",
        default=0.5,
        type=float,
        help="Coefficient for the cosine loss.",
    )
    parser.add_argument(
        "--lam_lsl1",
        default=0.5,
        type=float,
        help="Coefficient for the smooth L1 loss.",
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        help="Path for the ImageNet-1K data directory",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="Image size (for both training and validation). "
        "We assume the input images are square.",
    )

    parser.add_argument(
        "--use_fp16",
        type=utils.bool_flag,
        default=True,
        help="Whether or not to use mixed precision for training.",
    )
    parser.add_argument(
        "--clip_grad",
        type=float,
        default=0.0,
        help="Gradient clipping value.",
    )
    parser.add_argument(
        "--batch_size_per_gpu",
        default=128,
        type=int,
        help="Batch size per GPU. Total batch size is proportional to the number of GPUs.",
    )
    parser.add_argument(
        "--epochs",
        default=200,
        type=int,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--optim_args",
        type=str,
        default="{'betas':(0.9, 0.99), 'eps':1e-8}",
        help="Dictionary of keyword arguments for the optimizer.",
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=3e-2,
        help="Weight decay for the SGD optimizer.",
    )
    parser.add_argument(
        "--lr",
        default=3e-4,
        type=float,
        help="Maximum learning rate at the end of linear warmup.",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-6,
        help="Minimum learning rate at the end of training.",
    )
    parser.add_argument(
        "--warmup_epochs",
        default=10,
        type=int,
        help="Number of training epochs for the learning-rate-warm-up phase.",
    )

    parser.add_argument(
        "--output_dir",
        default="./output",
        type=str,
        help="Path to the output folder to save logs and checkpoints.",
    )
    parser.add_argument(
        "--saveckpt_freq",
        default=20,
        type=int,
        help="Frequency of intermediate checkpointing.",
    )
    parser.add_argument(
        "--seed",
        default=22,
        type=int,
        help="Random seed",
    )
    parser.add_argument(
        "--num_workers",
        default=12,
        type=int,
        help="Number of data loading workers per GPU.",
    )
    parser.add_argument(
        "--dist_url",
        default="env://",
        type=str,
        help="Url used to set up distributed training.",
    )
    parser.add_argument(
        "--local_rank",
        default=0,
        type=int,
        help="Please ignore this argument; No need to set it manually.",
    )

    args = parser.parse_args()

    args.teachers = sorted(args.teachers.split(","))
    args.num_cpus = len(os.sched_getaffinity(0))

    os.makedirs(args.output_dir, exist_ok=True)

    return args


def main(args):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed + get_global_rank())
    torch.backends.cuda.matmul.allow_tf32 = True
    cudnn.benchmark = True

    setup_logging(os.path.join(args.output_dir, "log.txt"), level=logging.INFO)
    utils.print_program_info(args)
    ext_logger = ExternalLogger(args.output_dir)

    logger.info("Creating data loaders ...")
    train_loader, val_loader = get_dataloaders(args)

    logger.info("Loading teachers ...")
    teachers, teacher_ft_stats = build_teachers(args.teachers)

    logger.info("Creating student model")
    model = build_student_from_args(args)
    model = model.cuda()
    model = nn.parallel.DistributedDataParallel(
        model, device_ids=[args.gpu], find_unused_parameters=True
    )
    utils.save_model_defn(model.module, os.path.join(args.output_dir, "model_defn.txt"))

    optimizer = torch.optim.AdamW(
        utils.get_params_groups(
            model,
            save_file_path=os.path.join(args.output_dir, "params_groups.txt"),
        ),
        lr=0,
        **eval(args.optim_args),
    )
    logger.info("Optimizer: {}".format(optimizer))

    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    args.lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * dist.get_world_size()) / 256.0,
        args.min_lr,
        args.epochs,
        len(train_loader),
        warmup_epochs=args.warmup_epochs,
    )

    args.tnorm_ema_schedule = utils.cosine_scheduler(
        args.tnorm_ema_momentum_start,
        args.tnorm_ema_momentum_end,
        args.epochs,
        len(train_loader),
        warmup_epochs=0,
    )

    to_restore = {"epoch": 0, "teacher_ft_stats": teacher_ft_stats}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        model=model,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
    )
    start_epoch = to_restore["epoch"]
    teacher_ft_stats = to_restore["teacher_ft_stats"]

    logger.info("Training starts ...")
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        train_loader.sampler.set_epoch(epoch)

        train_one_epoch(
            model,
            teachers,
            teacher_ft_stats,
            train_loader,
            optimizer,
            epoch,
            fp16_scaler,
            ext_logger,
            args,
        )

        evaluate(
            model,
            teachers,
            teacher_ft_stats,
            val_loader,
            epoch,
            ext_logger,
            args,
        )

        save_dict = {
            "model": model.state_dict(),
            "teacher_ft_stats": teacher_ft_stats,
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "args": args,
        }
        if fp16_scaler is not None:
            save_dict["fp16_scaler"] = fp16_scaler.state_dict()

        if dist.get_rank() == 0:
            torch.save(save_dict, os.path.join(args.output_dir, "checkpoint.pth"))
            if args.saveckpt_freq and epoch % args.saveckpt_freq == 0:
                shutil.copy(
                    os.path.join(args.output_dir, "checkpoint.pth"),
                    os.path.join(args.output_dir, f"checkpoint_{epoch:04}.pth"),
                )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info("Training time {}".format(total_time_str))


def train_one_epoch(
    model,
    teachers,
    teacher_ft_stats,
    data_loader,
    optimizer,
    epoch,
    fp16_scaler,
    ext_logger,
    args,
):
    logger.info("-" * 50)
    logger.info("Starting training epoch {}".format(epoch))

    metrics_file = os.path.join(args.output_dir, "metrics_training.json")
    metric_logger = MetricLogger(delimiter="  ", output_file=metrics_file)
    header = "Training - Epoch: [{}/{}]".format(epoch, args.epochs)

    model.train()

    for it, (image, target) in enumerate(
        metric_logger.log_every(
            data_loader,
            10,
            header,
            start_iteration=epoch * len(data_loader),
            ext_logger=ext_logger,
            ext_logger_prefix="train/batch/",
        )
    ):
        image = image.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        it = len(data_loader) * epoch + it
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = args.lr_schedule[it]
            if i == 0:
                param_group["weight_decay"] = args.wd

        metric_dict = {
            "lr": optimizer.param_groups[0]["lr"],
            "wd": optimizer.param_groups[0]["weight_decay"],
            "tnorm_ema_momentum": args.tnorm_ema_schedule[it],
        }

        with torch.cuda.amp.autocast(fp16_scaler is not None):
            student_output = model(image)

            with torch.no_grad():
                teacher_output = get_teacher_output(
                    image, teachers, teacher_ft_stats, args.tnorm_ema_schedule[it]
                )

            loss, _ = unic_loss(
                student_output,
                teacher_output,
                args.lam_lcos,
                args.lam_lsl1,
                args.t_drop_prob,
                metric_dict=metric_dict,
            )

        if not math.isfinite(loss.item()):
            logger.info("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        optimizer.zero_grad()
        grad_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad > 0:
                grad_norms = utils.clip_gradients(model, args.clip_grad)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad > 0:
                fp16_scaler.unscale_(optimizer)
                grad_norms = utils.clip_gradients(model, args.clip_grad)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        if grad_norms is not None:
            metric_dict.update(
                {
                    "grad_norm/mean": grad_norms.mean().item(),
                    "grad_norm/std": grad_norms.std().item(),
                    "grad_norm/max": grad_norms.max().item(),
                    "grad_norm/min": grad_norms.min().item(),
                }
            )

    metric_logger.synchronize_between_processes()
    metric_dict = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    logger.info("Averaged train stats:")
    for k, v in metric_dict.items():
        logger.info("{}: {}".format(k, v))

    ext_logger.log(
        metric_dict,
        epoch + 1,
        prefix="train/epoch/",
        save_path=os.path.join(args.output_dir, "log_train.txt"),
    )

    return metric_dict


@torch.no_grad()
def evaluate(
    model,
    teachers,
    teacher_ft_stats,
    data_loader,
    epoch,
    ext_logger,
    args,
):
    metric_logger = MetricLogger(delimiter="  ")
    header = "Test - Epoch: [{}/{}]".format(epoch, args.epochs)

    model.eval()

    for it, (image, target) in enumerate(
        metric_logger.log_every(data_loader, 10, header)
    ):
        image = image.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        student_output = model(image)
        teacher_output = get_teacher_output(image, teachers, teacher_ft_stats, 0.0)

        metric_dict = {}
        unic_loss(
            student_output,
            teacher_output,
            args.lam_lcos,
            args.lam_lsl1,
            args.t_drop_prob,
            metric_dict=metric_dict,
        )
        metric_logger.update(**metric_dict)

    metric_logger.synchronize_between_processes()
    metric_dict = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    logger.info("Averaged test stats:")
    for k, v in metric_dict.items():
        logger.info("{}: {}".format(k, v))

    ext_logger.log(
        metric_dict,
        epoch + 1,
        prefix="test/epoch/",
        save_path=os.path.join(args.output_dir, "log_test.txt"),
    )

    return metric_dict


def get_dataloaders(args):
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_dataset = ImageFolder(
        os.path.join(args.data_dir, "train"),
        # fmt:off
        transform=T.Compose(
            [
                T.ToImage(),
                T.RandomResizedCrop(args.image_size, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8),
                T.RandomApply([T.Grayscale(num_output_channels=3)], p=0.2),
                T.ToDtype(torch.float32, scale=True),
                T.RandomApply([T.GaussianBlur(kernel_size=9, sigma=(0.1, 5.0))], p=0.2),
                T.RandomSolarize(threshold=0.5, p=0.2),
                normalize,
            ]
        ),
        # fmt:on
    )
    logger.info("Training dataset:\n - {}".format(train_dataset))

    val_dataset = ImageFolder(
        os.path.join(args.data_dir, "val"),
        # fmt:off
        transform=T.Compose(
            [
                T.ToImage(),
                T.Resize(args.image_size, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
                T.CenterCrop(args.image_size),
                T.ToDtype(torch.float32, scale=True),
                normalize,
            ]
        ),
        # fmt:on
    )
    logger.info("Validation dataset:\n - {}".format(val_dataset))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=torch.utils.data.DistributedSampler(
            train_dataset, seed=args.seed, shuffle=True
        ),
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=4,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )

    return train_loader, val_loader


if __name__ == "__main__":
    args = get_args()
    main(args)
