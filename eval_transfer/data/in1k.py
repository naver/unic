import os

from torchvision.datasets import ImageFolder


def load_split(
    dataset_dir: str,
    split: str,
    transform,
) -> ImageFolder:
    # ImageNet-1K splits are "train" and "val"
    split = split.replace("trainval", "train").replace("test", "val")

    dset = ImageFolder(os.path.join(dataset_dir, split), transform=transform)

    n_samples = {"train": 1281167, "val": 50000}[split]
    assert len(dset) == n_samples

    return dset
