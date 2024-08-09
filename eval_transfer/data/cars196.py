from collections import OrderedDict
from glob import glob

import scipy.io

from . import utils


def load_classname_label_mapping(dataset_dir) -> dict:
    """
    Loads the class name -> class label mapping dictionary for the Cars196 dataset.
    """
    cname_file = "{}/devkit/cars_meta.mat".format(dataset_dir)
    cname_arr = scipy.io.loadmat(cname_file)["class_names"].flatten()

    mapping = OrderedDict()
    for i in range(len(cname_arr)):
        mapping[cname_arr[i].item()] = i

    assert len(mapping) == 196
    return mapping


def load_split(dataset_dir: str, split: str, transform) -> utils.TransferDataset:
    """
    Loads a split of the Cars196 dataset.
    """
    assert split in ("trainval", "test")

    # Load the annotations
    labels_file = "{}/cars_annos.mat".format(dataset_dir)
    mat = scipy.io.loadmat(labels_file)

    # Locate all image files
    images_dir = "{}/car_ims".format(dataset_dir)
    image_files = glob("{}/*.jpg".format(images_dir))
    assert (
        len(image_files) == 16185
    ), "There should be 16185 images in total for the Cars196 dataset (found {})".format(
        len(image_files)
    )

    # Each annotation is composed of
    # - fname
    # - x1
    # - x2
    # - y1
    # - y2
    # - class
    # - split: 1 for test and 0 for training
    # Parse only image files and class names
    samples = [
        ("{}/{}".format(dataset_dir, ann[0].item()), int(ann[5].item()) - 1)
        for ann in mat["annotations"].flatten()
        if ann[-1] == (split == "test")
    ]

    n_samples = {"trainval": 8144, "test": 8041}[split]
    assert (
        len(samples) == n_samples
    ), "Loaded {} samples for the {} split (should have been {})".format(
        len(samples), split, n_samples
    )

    cname_label_mapping = load_classname_label_mapping(dataset_dir)

    return utils.TransferDataset(samples, transform, cname_label_mapping)
