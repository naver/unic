import os

from .utils import TransferDataset, load_pickle


def load_split(
    dataset: str,
    dataset_dir: str,
    split: str,
    transform,
    cog_levels_mapping_file,
    cog_concepts_split_file,
    n_min_images=732,
    n_max_images=1300,
    n_test_images=50,
    classname_label_mapping: dict = None,
    verbose=False,
) -> TransferDataset:
    # CoG splits are "train" and "test"
    split = split.replace("trainval", "train")

    # load a concept generalization level
    # example cog_l1
    _, level = dataset.split("_")

    # level is an index of the level between 1 and 5
    assert level.startswith("l"), "level should start with l ({})".format(level)
    assert len(level) == 2, "level should be 2 chars, e.g., l1, ..., l5 ({})".format(
        level
    )
    level = int(level[1])
    assert level in list(range(1, 6))

    # load the concepts in this level
    selected_concepts = load_pickle(cog_levels_mapping_file)[level - 1]
    assert (
        len(selected_concepts) == 1000
    ), "There should be 1000 concepts in level {}, but found {}".format(
        level, len(selected_concepts)
    )

    # load the dictionary that keeps training and test splits for each concept
    images_per_concept = load_pickle(cog_concepts_split_file)
    images_per_concept = {
        k: v for k, v in images_per_concept.items() if k in selected_concepts
    }

    # go through all the images of the concepts
    image_files = []
    labels = []
    six = 0
    for concept in sorted(images_per_concept.keys()):
        _files = images_per_concept[concept][split].tolist()
        _files.sort()
        if split == "train":
            assert len(_files) >= n_min_images
            assert len(_files) <= n_max_images
        elif split == "test":
            assert len(_files) == n_test_images

        image_files += [
            os.path.join(dataset_dir, path.strip("/").strip("\\")) for path in _files
        ]
        labels += [six] * len(_files)
        six += 1
        if verbose:
            print("==> {} images found for the concept {}".format(len(_files), concept))

    samples = list(zip(image_files, labels))

    # make sure we load the correct number of images
    n_samples = {
        "cog_l1": {"train": 1118804, "test": 50000},
        "cog_l2": {"train": 1115788, "test": 50000},
        "cog_l3": {"train": 1095203, "test": 50000},
        "cog_l4": {"train": 1107128, "test": 50000},
        "cog_l5": {"train": 1091654, "test": 50000},
    }[dataset][split]
    assert len(samples) == n_samples

    # initialize the dataset
    return TransferDataset(samples, transform, classname_label_mapping)
