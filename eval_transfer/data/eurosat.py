import itertools
import random
from collections import OrderedDict
from glob import glob

from . import utils


_TRAIN_ORDER = {}
_SPLITS = None


def _load_images_per_class(
    dataset_dir: str,
) -> OrderedDict:
    # Locate image folders for each class
    # and parse class names
    image_folders = sorted(glob("{}/*/".format(dataset_dir)))
    assert (
        len(image_folders) == 10
    ), "There should be 10 folders under {} for EuroSAT (found {} folders)".format(
        dataset_dir, len(image_folders)
    )
    class_names = [folder.split("/")[-2] for folder in image_folders]
    assert len(class_names) == 10

    # Locate images for each class
    # and make sure that there are 27000 images in total
    images_per_class = OrderedDict()
    for cname in class_names:
        images_per_class[cname] = sorted(glob("{}/{}/*.jpg".format(dataset_dir, cname)))
    n_total_images = sum([len(ipc) for ipc in images_per_class.values()])
    assert (
        n_total_images == 27000
    ), "Found {} images for EuroSAT (should have been 27000)".format(n_total_images)

    return images_per_class


def _make_dataset_splits(dataset_dir: str) -> dict:
    # Make dataset splits
    # We determine the number of images for each class in each split based on
    # 13,500/5,400/8,100 split (for train/val/test) of Feng2022@ICLR
    images_per_class = _load_images_per_class(dataset_dir)
    class_names = sorted(images_per_class.keys())

    global _TRAIN_ORDER
    splits = {}

    for split in ("test", "val", "train"):
        samples = []
        for cix, cname in enumerate(class_names):
            image_list = sorted(images_per_class[cname])

            # create a random order of images
            if cname not in _TRAIN_ORDER:
                _order = list(range(len(image_list)))
                random.shuffle(_order)
                _TRAIN_ORDER[cname] = _order
            else:
                _order = _TRAIN_ORDER[cname]

            if split == "test":
                # the first 810 images are for test
                image_list = [image_list[ix] for ix in _order[:810]]
            elif split == "val":
                # the next 540 images are for val
                image_list = [image_list[ix] for ix in _order[810 : 810 + 540]]
            else:
                # the remaining images are for train
                image_list = [image_list[ix] for ix in _order[810 + 540 :]]

            for img_path in image_list:
                samples.append((img_path, cix))

        print(
            "{} samples gathered for the EuroSAT {} split".format(len(samples), split)
        )
        splits[split] = samples

    # Make sure that we have used all the images
    assert sum([len(samples) for samples in splits.values()]) == 27000

    # Make sure that the splits are disjoint
    for split_a, split_b in itertools.combinations(list(splits.values()), 2):
        assert (
            len(
                set([sample[0] for sample in split_a]).intersection(
                    set([sample[0] for sample in split_b])
                )
            )
            == 0
        )

    return splits


def load_split(dataset_dir: str, split: str, transform) -> utils.TransferDataset:
    """
    Loads a split of the EuroSAT dataset.
    """
    assert split in ("trainval", "test")

    global _SPLITS
    if _SPLITS is None:
        _SPLITS = _make_dataset_splits(dataset_dir)

    if split == "trainval":
        samples = _SPLITS["train"] + _SPLITS["val"]
        image_files = [sample[0] for sample in samples]
        assert len(set(image_files)) == len(
            image_files
        ), "Duplicate image files after merging train and val splits"
    else:
        samples = _SPLITS[split]

    n_samples = {"trainval": 18900, "test": 8100}[split]
    assert (
        len(samples) == n_samples
    ), "Loaded {} samples for the {} split (should have been {})".format(
        len(samples), split, n_samples
    )

    return utils.TransferDataset(samples, transform)
