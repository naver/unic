from glob import glob

import scipy.io

from . import utils


def load_split(dataset_dir: str, split: str, transform) -> utils.TransferDataset:
    """
    Loads a split of the Flowers dataset.
    """
    assert split in ("trainval", "test")

    images_dir = "{}/jpg".format(dataset_dir)
    labels_file = "{}/imagelabels.mat".format(dataset_dir)
    splits_file = "{}/setid.mat".format(dataset_dir)

    # Load the annotations and the train/val/test splits
    labels = scipy.io.loadmat(labels_file)["labels"].flatten()
    split_inds_dict = scipy.io.loadmat(splits_file)

    # Locate all images
    image_files = glob("{}/*.jpg".format(images_dir))
    assert (
        len(image_files) == 8189
    ), "There should be 8189 images for the Flowers dataset (but found {})".format(
        len(image_files)
    )

    # gather splits into a dictionary
    splits_dict = {}
    for _split in ("train", "val", "test"):
        inds = split_inds_dict[
            {"train": "trnid", "val": "valid", "test": "tstid"}[_split]
        ].flatten()

        samples = []
        for ix in inds:
            samples.append(
                ("{}/image_{:05d}.jpg".format(images_dir, ix), labels[ix - 1] - 1)
            )

        splits_dict[_split] = samples

    # divide the training set into two parts
    if split == "trainval":
        samples = splits_dict["train"] + splits_dict["val"]
    else:
        samples = splits_dict[split]

    n_samples = {"trainval": 2040, "test": 6149}[split]
    assert (
        len(samples) == n_samples
    ), "Loaded {} samples for the {} split (should have been {})".format(
        len(samples), split, n_samples
    )

    return utils.TransferDataset(samples, transform)
