import logging
from collections import defaultdict
from typing import List, Dict

import torch

from utils import standard_normalize
from .config import TEACHER_CFG
from .builder import build_teachers


logger = logging.getLogger()


@torch.no_grad()
def get_teacher_output(
    image: torch.Tensor,
    teachers: List[str],
    teacher_ft_stats: Dict[str, Dict[str, Dict[str, torch.Tensor]]],
    teacher_ft_stat_ema_momentum: float = 0.0,
) -> Dict[str, Dict[str, torch.Tensor]]:

    teacher_output = defaultdict(dict)

    for tname in teachers.keys():
        tout_dict = teachers[tname].forward_features(image)

        for ttype in ["cls", "patch"]:
            key = "x_norm_{}{}".format(ttype, "token" if ttype == "cls" else "tokens")
            tout = tout_dict[key]
            tout = standard_normalize(
                tout,
                mean_ema=teacher_ft_stats[tname][ttype]["mean"],
                std_ema=teacher_ft_stats[tname][ttype]["std"],
                ema_momentum=teacher_ft_stat_ema_momentum,
            )

            teacher_output[tname][ttype] = tout

    return teacher_output
