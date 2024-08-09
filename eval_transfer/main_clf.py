import argparse
import os
import random
import logging

import numpy as np
import optuna
import torch

from dinov2.logging import setup_logging
from utils import print_program_info, save_pickle

from eval_transfer.data.utils import split_trainval
from eval_transfer.logreg_trainer import LogregSklearnTrainer, LogregTorchTrainer
from eval_transfer.knn_trainer import knn_classifier


logger = logging.getLogger()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features_dir",
        type=str,
        default="",
        help="Directory to include features_trainval.pth and features_test.pth",
    )
    parser.add_argument(
        "--features_norm",
        type=str,
        default="none",
        choices=["l2", "none"],
        help="Normalization applied to features before the classifier",
    )
    parser.add_argument(
        "--clf_type",
        type=str,
        default="logreg_sklearn",
        choices=["logreg_sklearn", "logreg_torch", "knn_torch"],
        help="Type of linear classifier to train on top of features",
    )
    parser.add_argument(
        "--val_perc",
        type=float,
        default=0.2,
        help="Percentage of the val set, sampled from the trainval set for hyper-parameter tuning",
    )
    # For the L-BFGS-based logistic regression trainer implemented in scikit-learn
    parser.add_argument(
        "--clf_C",
        type=float,
        help="""Inverse regularization strength for sklearn.linear_model.LogisticRegression.
        Note that this variable is determined by Optuna, so do not set it manually""",
    )
    parser.add_argument(
        "--clf_C_min",
        type=float,
        default=1e-5,
        help="Power of the minimum C parameter sampled by Optuna",
    )
    parser.add_argument(
        "--clf_C_max",
        type=float,
        default=1e6,
        help="Power of the maximum C parameter sampled by Optuna",
    )
    parser.add_argument(
        "--clf_max_iter",
        type=int,
        default=2000,
        help="Maximum number of iterations to run the classifier for sklearn.linear_model.LogisticRegression during the hyper-parameter tuning stage.",
    )
    # For the SGD-based logistic regression trainer implemented in PyTorch
    parser.add_argument(
        "--clf_lr",
        type=float,
        help="""Learning rate.
        Note that this variable is determined by Optuna, so do not set it manually""",
    )
    parser.add_argument(
        "--clf_lr_min",
        type=float,
        default=1e-1,
        help="Power of the minimum lr parameter sampled by Optuna",
    )
    parser.add_argument(
        "--clf_lr_max",
        type=float,
        default=1e2,
        help="Power of the maximum lr parameter sampled by Optuna",
    )
    parser.add_argument(
        "--clf_wd",
        type=float,
        help="""Weight decay.
        Note that this variable is determined by Optuna, do not set it manually""",
    )
    parser.add_argument(
        "--clf_wd_min",
        type=float,
        default=1e-12,
        help="Power of the minimum weight decay parameter sampled by Optuna",
    )
    parser.add_argument(
        "--clf_wd_max",
        type=float,
        default=1e-4,
        help="Power of the maximum weight decay parameter sampled by Optuna",
    )
    parser.add_argument(
        "--clf_mom",
        type=float,
        default=0.9,
        help="SGD momentum. We do not tune this variable.",
    )
    parser.add_argument(
        "--clf_epochs",
        type=int,
        default=100,
        help="""Number of epochs to train the linear classifier.
        We do not tune this variable""",
    )
    parser.add_argument(
        "--clf_batch_size",
        type=int,
        default=1024,
        help="""Batch size for SGD.
        We do not tune this variable""",
    )
    # For the kNN classifier implemented in PyTorch
    parser.add_argument(
        "--knn_k",
        default=[10, 20, 100, 200],
        nargs="+",
        type=int,
        help="Number of NN to use. 20 is usually working the best.",
    )
    parser.add_argument(
        "--knn_temp",
        default=0.07,
        type=float,
        help="Temperature used in the voting coefficient",
    )
    # Common for all trainers
    parser.add_argument(
        "--n_sklearn_workers",
        type=int,
        default=-1,
        help="Number of CPU cores to use in Scikit-learn jobs. -1 means to use all available cores.",
    )
    parser.add_argument(
        "--n_optuna_workers",
        type=int,
        default=1,
        help="Number of concurrent Optuna jobs",
    )
    parser.add_argument(
        "--n_optuna_trials",
        type=int,
        default=40,
        help="Number of trials run by Optuna",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=22,
        help="Seed for the random number generators",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Whether to use CUDA during feature extraction and classifier training",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Whether to save the logs",
    )
    parser.add_argument(
        "--purge_output_dir",
        action="store_true",
        help="Whether to purge the output directory before running classification",
    )

    args = parser.parse_args()
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        args.device = "cuda"
    else:
        args.device = "cpu"

    if args.clf_type == "knn_torch":
        print("Hyper-parameter search is disabled for the kNN classifier")
        args.val_perc = 0.0

    if args.output_dir == "":
        args.output_dir = os.path.join(
            args.features_dir,
            "{}_{}".format(args.clf_type, args.features_norm),
        )
        print("output_dir set to {}".format(args.output_dir))

    return args


def main(args):
    data_dict = _prepare_features(args)

    if args.clf_type == "knn_torch":
        logger.info("Starting kNN evaluation")
        for k in args.knn_k:
            top1, top5 = knn_classifier(
                data_dict["trainval"][0],
                data_dict["trainval"][1],
                data_dict["test"][0],
                data_dict["test"][1],
                k,
                normalize=args.features_norm,
                T=args.knn_temp,
            )
            logger.info(
                "{}-NN classifier result: Top1: {}, Top5: {}".format(k, top1, top5)
            )
        return

    trainer_class = (
        LogregSklearnTrainer
        if args.clf_type == "logreg_sklearn"
        else LogregTorchTrainer
    )

    # tune hyper-parameters with optuna
    logger.info("Starting hyper-parameter tuning")
    clf_trainer = trainer_class(
        data_dict["train"][0],
        data_dict["train"][1],
        data_dict["val"][0],
        data_dict["val"][1],
        args,
    )
    hps_sampler = optuna.samplers.TPESampler(
        multivariate=args.clf_type == "logreg_torch",
        group=args.clf_type == "logreg_torch",
        seed=args.seed,
    )
    study = optuna.create_study(sampler=hps_sampler, direction="maximize")
    study.optimize(
        clf_trainer,
        n_trials=args.n_optuna_trials,
        n_jobs=args.n_optuna_workers,
        show_progress_bar=False,
    )
    save_pickle(study, os.path.join(args.output_dir, "study.pkl"))

    logger.info("*" * 50)
    logger.info("Hyper-parameter search ended")
    logger.info("best_trial:")
    logger.info(str(study.best_trial))
    logger.info("best_params:")
    logger.info(str(study.best_params))
    logger.info("*" * 50)

    # train the final classifier with the tuned hyper-parameters
    logger.info("Training the final classifier")
    del clf_trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    clf_trainer = trainer_class(
        data_dict["trainval"][0],
        data_dict["trainval"][1],
        data_dict["test"][0],
        data_dict["test"][1],
        args,
    )
    clf_trainer.set_hps(study.best_params)
    clf_trainer()


def _prepare_features(args):
    data_dict = {}

    for split in ("trainval", "test"):
        features_file = os.path.join(args.features_dir, "features_{}.pth".format(split))

        if not os.path.isfile(features_file):
            raise ValueError("Features file not found: {}".format(features_file))

        logger.info("Loading pre-extracted features from {}".format(features_file))
        features_dict = torch.load(features_file, "cpu")
        X, Y = features_dict["X"], features_dict["Y"]
        data_dict[split] = [X, Y]

    # split the trainval into train and val
    if args.val_perc > 0:
        data_dict["train"], data_dict["val"] = split_trainval(
            data_dict["trainval"][0],
            data_dict["trainval"][1],
            per_val=args.val_perc,
        )

    # preprocess each feature split
    for split in data_dict.keys():
        data_dict[split][0] = _preprocess_features(
            data_dict[split][0], args.features_norm
        )

        print_feature_info(split, data_dict[split][0], data_dict[split][1])

    return data_dict


@torch.no_grad()
def _preprocess_features(features, norm_type):

    if norm_type == "none":
        pass
    elif norm_type == "l2":
        features = torch.nn.functional.normalize(features, p=2, dim=1)
    else:
        raise ValueError("Unknown normalization type: {}".format(norm_type))

    return features.detach().clone()


def print_feature_info(split, X, Y):
    logging.info(
        "Feature split: {:8s} | features.shape:{}, features.norm:{:.3f}, labels.shape:{}, labels.n_unique:{}".format(
            split,
            list(X.shape),
            X.norm(dim=1).mean(),
            list(Y.shape),
            len(torch.unique(Y)),
        )
    )


if __name__ == "__main__":
    args = parse_args()

    setup_logging(os.path.join(args.output_dir, "log.txt"), level=logging.INFO)
    print_program_info(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    main(args)
