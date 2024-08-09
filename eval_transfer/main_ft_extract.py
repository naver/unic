import argparse
import os
import sys
import logging

import torch

from modeling.unic import load_student_encoder_from_checkpoint
from dinov2.logging import setup_logging
from utils import print_program_info

from eval_transfer import data
from eval_transfer.feature_extraction import extract_features_loop


logger = logging.getLogger()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained",
        type=str,
        required=True,
        help="Path to the checkpoint file of the pretrained student model",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="in1k",
        choices=[
            "in1k",
            "cog_l1",
            "cog_l2",
            "cog_l3",
            "cog_l4",
            "cog_l5",
            "aircraft",
            "cars196",
            "dtd",
            "eurosat",
            "flowers",
            "pets",
            "food101",
            "sun397",
            "inat2018",
            "inat2019",
        ],
        help="From which datasets to extract features",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="Size of images given as input to the network before extracting features",
    )
    parser.add_argument(
        "--token_type",
        type=str,
        default="cls",
        choices=["cls", "patch"],
        help="Type of token used to extract features. "
        "cls: extract features from the [CLS] token, "
        "patch: extract features applying average pooling over the patch tokens.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Where to extract features.",
    )
    parser.add_argument(
        "--purge_output_dir",
        action="store_true",
        help="Whether to purge the output directory before extracting features",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size used during feature extraction",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of workers run for the data loader",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda"],
        help="Whether to use CUDA during feature extraction",
    )

    args = parser.parse_args()

    if not os.path.isfile(args.pretrained):
        print(
            "Checkpoint file ({}) not found. "
            "Please provide a valid checkpoint file path for the pretrained student model.".format(
                args.pretrained
            )
        )
        sys.exit(-1)

    if args.output_dir is None or args.output_dir == "":
        args.output_dir = os.path.join(
            os.path.dirname(args.pretrained),
            "features_{}_{}_{}".format(args.dataset, args.image_size, args.token_type),
        )
        print("Output directory set to {}".format(args.output_dir))

    if args.purge_output_dir:
        print("Purging output directory: {}".format(args.output_dir))
        os.system("rm -rf {}".format(args.output_dir))

    os.umask(0x0002)
    os.makedirs(args.output_dir, exist_ok=True)

    if (not torch.cuda.is_available()) or (torch.cuda.device_count() == 0):
        print("No CUDA-compatible device found, we will use CPU.")
        args.device = "cpu"

    return args


def main(args):
    logger.info("Initializing pretrained model")
    model, epoch = load_student_encoder_from_checkpoint(args.pretrained)
    model = model.cuda()
    if args.device == "cuda" and torch.cuda.device_count() > 1:
        logger.info("Using {} GPUs".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)

    # extract features from training and test sets
    for split in ("trainval", "test"):
        logger.info("Loading {} split of {}".format(split, args.dataset))
        dataset = data.load_dataset(
            args.dataset,
            split,
            args.image_size,
        )
        logger.info("Dataset size: {}".format(len(dataset)))
        logger.info("Data loading pipeline:\n{}".format(dataset.transform))

        X, Y = extract_features_loop(
            model,
            dataset,
            args.token_type,
            args.batch_size,
            args.num_workers,
            args.device,
        )
        logger.info("X.shape: {}, Y.shape: {}".format(X.shape, Y.shape))

        features_file = os.path.join(args.output_dir, "features_{}.pth".format(split))
        logger.info("Saving features under {}".format(features_file))
        torch.save({"X": X, "Y": Y}, features_file)


if __name__ == "__main__":
    args = parse_args()

    setup_logging(os.path.join(args.output_dir, "log.txt"), level=logging.INFO)
    print_program_info(args)

    main(args)
