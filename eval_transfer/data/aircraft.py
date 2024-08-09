from collections import OrderedDict

from . import utils


def load_classname_label_mapping(dataset_dir) -> dict:
    """
    Loads the class name -> class label mapping dictionary for the Aircrafts dataset.
    """
    mapping_file = "{}/data/variants.txt".format(dataset_dir)
    mapping = OrderedDict()
    with open(mapping_file, "r") as fid:
        for ix, line in enumerate(fid):
            mapping[line.strip().lower()] = ix

    assert len(mapping) == 100
    return mapping


def load_split(
    dataset_dir: str, split: str, transform, classname_label_mapping: dict = None
) -> utils.TransferDataset:
    """
    Loads a split of the Aircraft dataset.
    """
    assert split in ("train", "val", "trainval", "test")
    images_dir = "{}/data/images".format(dataset_dir)
    split_file = "{}/data/images_variant_{}.txt".format(dataset_dir, split)
    samples = []

    if classname_label_mapping is None:
        classname_label_mapping = load_classname_label_mapping(dataset_dir)

    with open(split_file, "r") as fid:
        for line in fid:
            line = line.strip()
            image_name = line.split(" ")[0]
            assert len(image_name) == 7
            image_path = "{}/{}.jpg".format(images_dir, image_name)

            class_name = line.replace(image_name, "").strip().lower()
            class_label = classname_label_mapping[class_name]
            samples.append((image_path, class_label))

    n_samples = {"train": 3334, "val": 3333, "trainval": 6667, "test": 3333}[split]
    assert (
        len(samples) == n_samples
    ), "Loaded {} samples for the {} split (should have been {})".format(
        len(samples), split, n_samples
    )

    return utils.TransferDataset(samples, transform, classname_label_mapping)
