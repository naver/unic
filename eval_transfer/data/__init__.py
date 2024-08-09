import sys

from torchvision import transforms

from . import (
    aircraft,
    cars196,
    dtd,
    eurosat,
    flowers,
    food101,
    imagenet_cog,
    in1k,
    inat,
    pets,
    sun397,
    utils,
)

####################
# Hard coded data paths
COG_DATASET_DIR = ""

DATASET_DIR_DICT = {
    "aircraft": "",
    "cars196": "",
    "dtd": "",
    "eurosat": "",
    "flowers": "",
    "pets": "",
    "sun397": "",
    "food101": "",
    "inat2018": "",
    "inat2019": "",
    "in1k": "",
    "cog_l1": COG_DATASET_DIR,
    "cog_l2": COG_DATASET_DIR,
    "cog_l3": COG_DATASET_DIR,
    "cog_l4": COG_DATASET_DIR,
    "cog_l5": COG_DATASET_DIR,
}
COG_LEVELS_MAPPING_FILE = f"{COG_DATASET_DIR}/cog_levels_mapping_file.pkl"
COG_CONCEPTS_SPLIT_FILE = f"{COG_DATASET_DIR}/cog_concepts_split_file.pkl"
####################


def load_dataset(
    dataset,
    split,
    image_size=224,
) -> utils.TransferDataset:
    """
    Loads a split of a dataset with the center-crop augmentation.
    """

    assert split in ("trainval", "test"), "Unrecognized split: {}".format(split)
    dataset_dir = DATASET_DIR_DICT[dataset]

    transform = transforms.Compose(
        [
            transforms.Resize(
                image_size, interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    if dataset.startswith("cog_"):
        return imagenet_cog.load_split(
            dataset,
            dataset_dir,
            split,
            transform,
            COG_LEVELS_MAPPING_FILE,
            COG_CONCEPTS_SPLIT_FILE,
        )
    elif dataset.startswith("inat"):
        year = dataset.replace("inat", "")
        return inat.load_split(dataset_dir, year, split, transform)
    else:
        return (
            sys.modules[__name__]
            .__dict__[dataset]
            .load_split(dataset_dir, split, transform)
        )
