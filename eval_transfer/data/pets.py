from glob import glob

from . import utils


def _load_annotations_file(dataset_dir, split):
    assert split in ("trainval", "test")
    images_dir = "{}/images".format(dataset_dir)
    anns_file = "{}/annotations/{}.txt".format(dataset_dir, split)
    samples = []
    with open(anns_file, "r") as fid:
        for line in fid:
            image_name, label = line.strip().split(" ")[:2]
            samples.append(("{}/{}.jpg".format(images_dir, image_name), int(label) - 1))

    return samples


def load_split(dataset_dir: str, split: str, transform) -> utils.TransferDataset:
    """
    Loads a split of the Pets dataset.
    """
    assert split in ("trainval", "test")

    images_dir = "{}/images".format(dataset_dir)
    image_files = sorted(glob("{}/*.jpg".format(images_dir)))
    assert (
        len(image_files) == 7390
    ), "There should be 7390 images in total (found {})".format(len(image_files))

    # Load the split samples
    samples = _load_annotations_file(dataset_dir, split)

    n_samples = {"trainval": 3680, "test": 3669}[split]
    assert (
        len(samples) == n_samples
    ), "Loaded {} samples for the {} split (should have been {})".format(
        len(samples), split, n_samples
    )

    return utils.TransferDataset(samples, transform)
