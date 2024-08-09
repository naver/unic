import os
from glob import glob
from typing import List

from . import utils


def _load_classes(dataset_dir) -> List[str]:
    images_dir = os.path.join(dataset_dir, "images")
    folders = [
        path
        for path in sorted(glob(os.path.join(images_dir, "*" + os.sep)))
        if os.path.isdir(path)
    ]
    assert (
        len(folders) == 101
    ), "There should be 101 folders under {} (1 folder for each class) for the Food-101 dataset, but found {} folders.".format(
        images_dir, len(folders)
    )

    classes = sorted([path.strip(os.sep).split(os.sep)[-1] for path in folders])
    assert len(classes) == 101
    return classes


def _load_split_samples(dataset_dir, split, classes=None):
    assert split in ("train", "test")
    split_file = os.path.join(dataset_dir, "meta", "{}.txt".format(split))
    assert os.path.exists(split_file), "Food-101 {} split file does not exist.".format(
        split
    )

    if classes is None:
        classes = _load_classes(dataset_dir)

    images_dir = os.path.join(dataset_dir, "images")
    samples = []

    with open(split_file, "r") as fid:
        for line in fid:
            line = line.strip()
            class_name, image_name = line.split("/")
            label = classes.index(class_name)

            image_path = os.path.join(images_dir, class_name, image_name + ".jpg")
            assert os.path.isfile(image_path)
            samples.append((image_path, label))

    return samples


def load_split(dataset_dir: str, split: str, transform) -> utils.TransferDataset:
    """
    Loads a split of the Food-101 dataset.
    """
    assert split in ("trainval", "test")

    # Load class names
    classes = _load_classes(dataset_dir)

    # Load the split samples
    samples = _load_split_samples(
        dataset_dir, split.replace("trainval", "train"), classes
    )

    n_samples = {"trainval": 75750, "test": 25250}[split]
    assert (
        len(samples) == n_samples
    ), "Loaded {} samples for the {} split (should have been {})".format(
        len(samples), split, n_samples
    )

    return utils.TransferDataset(samples, transform)
