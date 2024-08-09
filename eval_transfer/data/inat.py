import json
import os

import numpy as np

from . import utils


def load_split(
    dataset_dir: str,
    year: str,
    split: str,
    transform,
    classname_label_mapping: dict = None,
) -> utils.TransferDataset:
    """
    Loads a split of the i-Naturalist 2018/2019 datasets.
    """
    assert year in ("2018", "2019")
    assert split in ("trainval", "test")
    split = {"trainval": "train", "test": "val"}[split]

    ann_file = os.path.join(dataset_dir, "{}{}.json".format(split, year))
    with open(ann_file, "r") as fp:
        content = json.load(fp)
        assert len(content["images"]) == len(content["annotations"])
        n_samples = len(content["images"])

    samples = []
    for i in range(n_samples):
        samples.append(
            [
                os.path.join(dataset_dir, content["images"][i]["file_name"]),
                content["annotations"][i]["category_id"],
            ]
        )

    labels = np.array([sample[1] for sample in samples])
    class_list = np.unique(labels)
    n_classes = {"2018": 8142, "2019": 1010}[year]
    assert len(class_list) == n_classes

    n_samples = {
        "2018": {"train": 437513, "val": 24426},
        "2019": {"train": 265213, "val": 3030},
    }[year][split]
    assert (
        len(samples) == n_samples
    ), "Loaded {} samples for the {} split (should have been {})".format(
        len(samples), split, n_samples
    )

    return utils.TransferDataset(samples, transform, classname_label_mapping)
