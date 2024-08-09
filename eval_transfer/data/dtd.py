from glob import glob
from typing import List, Tuple

from . import utils


def _load_split_file(
    dataset_dir: str, split: str, split_no: int = 1
) -> List[Tuple[str, int]]:
    assert split in ("train", "val", "test")

    # get sorted class names to label each class
    images_dir = "{}/images".format(dataset_dir)
    image_folders = sorted(glob("{}/*".format(images_dir)))
    assert (
        len(image_folders) == 47
    ), "There are {} DTD classes (should have been 47)".format(len(image_folders))
    class_names = [folder.split("/")[-1] for folder in image_folders]

    split_file = "{}/labels/{}{}.txt".format(dataset_dir, split, split_no)
    samples = []

    with open(split_file, "r") as fid:
        for line in fid:
            line = line.strip()
            cname = line.split("/")[0]
            clbl = class_names.index(cname)
            img_path = "{}/{}".format(images_dir, line)
            samples.append((img_path, clbl))

    return samples


def load_split(
    dataset_dir: str, split: str, transform, split_no: int = 1
) -> utils.TransferDataset:
    """
    Loads a split of the DTD dataset.
    """
    assert split in ("trainval", "test")

    if split == "trainval":
        samples = _load_split_file(dataset_dir, "train", split_no) + _load_split_file(
            dataset_dir, "val", split_no
        )
        image_files = [sample[0] for sample in samples]
        assert len(set(image_files)) == len(
            image_files
        ), "Duplicate image files after merging train and val splits"
    else:
        samples = _load_split_file(dataset_dir, split, split_no)

    n_samples = {"trainval": 3760, "test": 1880}[split]
    assert (
        len(samples) == n_samples
    ), "Loaded {} samples for the {} split (should have been {})".format(
        len(samples), split, n_samples
    )

    return utils.TransferDataset(samples, transform)
