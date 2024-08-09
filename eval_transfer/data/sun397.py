import os

from . import utils


def _load_class_names(dataset_dir):
    classes = []
    with open(os.path.join(dataset_dir, "ClassName.txt"), "r") as fid:
        for line in fid:
            classes.append(line.strip().strip("/"))

    assert (
        len(classes) == 397
    ), "Corrupted ClassName.txt for SUN397. Found {} classes (instead of 397)".format(
        len(classes)
    )
    return sorted(classes)


def _load_split_samples(dataset_dir, split, split_no=1, classes=None):
    assert split in ("train", "test")
    assert split_no in list(range(1, 11))

    if classes is None:
        classes = _load_class_names(dataset_dir)

    samples = []
    split_file = os.path.join(
        dataset_dir, "{}ing_{:02d}.txt".format(split.capitalize(), split_no)
    )
    with open(split_file, "r") as fid:
        for line in fid:
            line = line.strip().strip("/")
            if line == "":
                continue
            class_name = "/".join(line.split("/")[:-1])
            image_name = line.split("/")[-1]
            label = classes.index(class_name)
            image_path = os.path.join(dataset_dir, "SUN397", class_name, image_name)
            assert os.path.isfile(image_path)
            samples.append((image_path, label))

    return samples


def load_split(dataset_dir: str, split: str, transform) -> utils.TransferDataset:
    """
    Loads a split of the SUN397 dataset.
    """
    assert split in ("trainval", "test")

    # Load class names
    classes = _load_class_names(dataset_dir)

    # Load the split samples
    samples = _load_split_samples(
        dataset_dir, split.replace("trainval", "train"), classes=classes
    )

    n_samples = {"trainval": 19850, "test": 19850}[split]
    assert (
        len(samples) == n_samples
    ), "Loaded {} samples for the {} split (should have been {})".format(
        len(samples), split, n_samples
    )

    return utils.TransferDataset(samples, transform)
