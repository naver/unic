import os
import logging
from collections import OrderedDict
from typing import List, Dict, Union

import torch

from .config import TEACHER_CFG


logger = logging.getLogger()


def build_teachers(
    teacher_names: List[str],
) -> Union[Dict[str, torch.nn.Module], Dict[str, Dict[str, Dict[str, torch.Tensor]]]]:
    teachers = OrderedDict()
    teacher_ft_stats = OrderedDict()

    for tname in teacher_names:
        logger.info("Loading teacher '{}'".format(tname))
        model = _build_teacher(tname)
        teachers[tname] = model

        # buffers for teacher feature statistics
        ft_dim = TEACHER_CFG[tname]["num_features"]
        teacher_ft_stats[tname] = {
            "cls": {
                "mean": torch.zeros(1, ft_dim).cuda(),
                "std": torch.ones(1, ft_dim).cuda(),
            },
            "patch": {
                "mean": torch.zeros(1, 1, ft_dim).cuda(),
                "std": torch.ones(1, 1, ft_dim).cuda(),
            },
        }

    return teachers, teacher_ft_stats


def _build_teacher(name):
    # name is expected to be in the following format:
    #  dino_vitbase_16
    #  <model_name>_<arch>_<patch_size>
    if name not in TEACHER_CFG.keys():
        raise ValueError(
            "Unsupported teacher name: {} (supported ones: {})".format(
                name, TEACHER_CFG.keys()
            )
        )

    ckpt_path = TEACHER_CFG[name]["ckpt_path"]
    ckpt_key = TEACHER_CFG[name]["ckpt_key"]

    if not os.path.isfile(ckpt_path):
        raise ValueError("Invalid teacher model path: {}".format(ckpt_path))

    state_dict = torch.load(ckpt_path, map_location="cpu")
    if ckpt_key != "" and ckpt_key in state_dict.keys():
        state_dict = state_dict[ckpt_key]

    _, _, patch_size = name.split("_")
    patch_size = int(patch_size)
    img_size = TEACHER_CFG[name]["resolution"]

    model = TEACHER_CFG[name]["loader"](img_size=img_size, patch_size=patch_size)
    model.load_state_dict(state_dict, strict=True)
    model = model.cuda()
    model = model.eval()
    for param in model.parameters():
        param.requires_grad = False

    return model
