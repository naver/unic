from typing import Dict

import torch
import torch.nn.functional as F

from .teacher_dropping import aggregate_losses


def unic_loss(
    student_output: Dict[str, Dict[str, torch.Tensor]],
    teacher_output: Dict[str, Dict[str, torch.Tensor]],
    lam_lcos: float = 0.5,
    lam_lsl1: float = 0.5,
    t_drop_prob: float = 0.5,
    metric_dict: Dict = {},
):
    loss_pt = loss_per_teacher(
        student_output,
        teacher_output,
        lam_lcos,
        lam_lsl1,
        metric_dict=metric_dict,
    )

    loss, tcoeffs = aggregate_losses(loss_pt, drop_prob=t_drop_prob)

    for tname in teacher_output.keys():
        metric_dict["t_coeff_{}".format(tname)] = tcoeffs[tname]

    metric_dict["loss/dist"] = loss.item()

    return loss, metric_dict


def loss_per_teacher(
    student_output: Dict[str, Dict[str, torch.Tensor]],
    teacher_output: Dict[str, Dict[str, torch.Tensor]],
    lam_cos: float,
    lam_sl1: float,
    metric_dict={},
) -> Dict[str, torch.Tensor]:
    loss_pt = {}

    for tname in teacher_output.keys():

        tout_dict = teacher_output[tname]
        sout_dict = student_output[tname]
        losses = []

        for ttype in tout_dict.keys():
            tout = tout_dict[ttype]
            sout = sout_dict[ttype]

            loss_cos = cosine_loss(sout, tout, avg=False)
            loss_sl1 = smooth_l1_loss(sout, tout, avg=False)
            loss = lam_cos * loss_cos + lam_sl1 * loss_sl1

            # for patch tokens
            if len(loss.shape) == 2:
                loss = loss.mean(dim=1)

            losses.append(loss)

            # fmt:off
            metric_dict.update(
                {
                    "loss/dist_{}_cos_{}".format(ttype, tname): loss_cos.mean().item(),
                    "loss/dist_{}_sl1_{}".format(ttype, tname): loss_sl1.mean().item(),
                    "loss/dist_{}_{}".format(ttype, tname): loss.mean().item(),
                }
            )
            # fmt:on

        losses = torch.stack(losses, dim=1).mean(dim=1)
        loss_pt[tname] = losses

        metric_dict.update(
            {
                "loss/dist_{}".format(tname): losses.mean().item(),
            }
        )

    return loss_pt


def cosine_loss(pred, target, avg=False):
    sim = F.cosine_similarity(pred, target, dim=-1)
    loss = 1 - sim

    if avg:
        loss = loss.mean()

    return loss


def smooth_l1_loss(pred, target, beta=1.0, avg=False):
    loss = F.smooth_l1_loss(pred, target, reduction="none", beta=beta).mean(dim=-1)

    if avg:
        loss = loss.mean()

    return loss
