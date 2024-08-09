import pickle
import random
import sys
import warnings
from textwrap import wrap
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageFile
from torch.utils.data import Dataset


ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)


class TransferDataset(Dataset):
    def __init__(
        self,
        samples: List[Tuple[str, int]],
        transform,
        classname_label_mapping: dict = None,
    ):
        super().__init__()
        self.samples = samples
        self.transform = transform
        self.classname_label_mapping = classname_label_mapping

        # determine the number of classes
        labels = np.array([sample[1] for sample in samples])
        self.n_classes = len(np.unique(labels))
        self.nspc = n_samples_per_class(labels=labels)

    def __getitem__(self, index):
        image_path, label = self.samples[index]

        try:
            image = Image.open(image_path).convert("RGB")

        except Exception as e:
            print("==> ERROR while loading image {}".format(image_path))
            print("==> {}".format(e))
            sys.exit(-1)

        image = self.transform(image)
        return (image, label)

    def __len__(self) -> int:
        return len(self.samples)


def split_trainval(X, Y, per_val=0.2):
    train_inds, val_inds = [], []

    labels = np.unique(Y)
    for c in labels:
        inds = np.where(Y == c)[0]
        random.shuffle(inds)
        n_val = int(len(inds) * per_val)
        if n_val == 0:
            assert len(inds) >= 2, (
                "We need at least 2 samples for class {}, "
                "to use one of them for validation set."
            )
            n_val = 1
            per_val = n_val / len(inds)
            print(
                "Validation set percentage for class {} is not enough, "
                "number of training images for this class: {}. "
                "Taking one sample for validation set by overriding per_val as {}".format(
                    c, len(inds), per_val
                )
            )
        assert n_val > 0
        train_inds.extend(inds[:-n_val].tolist())
        val_inds.extend(inds[-n_val:].tolist())

    train_inds = np.array(train_inds)
    val_inds = np.array(val_inds)
    assert (
        train_inds.shape[0] + val_inds.shape[0] == X.shape[0]
    ), "Error: Size mismatch for train ({}), val ({}) and trainval ({}) sets".format(
        train_inds.shape[0], val_inds.shape[0], X.shape[0]
    )
    assert (
        len(np.intersect1d(train_inds, val_inds)) == 0
    ), "Error: train and val sets overlap!"

    train = [X[train_inds], Y[train_inds]]
    val = [X[val_inds], Y[val_inds]]

    return [train, val]


def n_samples_per_class(
    dataset: TransferDataset = None, labels: list = None
) -> np.ndarray:
    """
    Computes the number of samples per class in a given dataset or label list
    """
    assert (dataset is not None) or (
        labels is not None
    ), "Either provide a TransferDataset or a label list"

    if labels is None:
        labels = [sample[1] for sample in dataset.samples]

    if isinstance(labels, list):
        labels = np.array(labels)

    nspc = {c: len(np.where(labels == c)[0]) for c in np.unique(labels)}
    return nspc


def load_pickle(save_path):
    with open(save_path, "rb") as fid:
        obj = pickle.load(fid)
    return obj


##################################################
# Functions from the torchvision library
# https://pytorch.org/vision/stable/_modules/torchvision/datasets/folder.html
##################################################

IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)


def has_file_allowed_extension(filename: str, extensions: Tuple[str, ...]) -> bool:
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def is_image_file(filename: str) -> bool:
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)
