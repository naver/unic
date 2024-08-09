import math
import os
from collections import defaultdict
from typing import Dict, List

import torch
import torch.nn as nn

from dinov2.models import vision_transformer


class UNIC(nn.Module):
    def __init__(self, encoder, lp):
        super().__init__()
        self.encoder = encoder
        self.lp = lp

    def forward(self, image):
        x, num_register_tokens = self.encoder.prepare_tokens_with_masks(image)
        output_cls = [x[:, 0, :]]
        output_patch = [x[:, 1 + num_register_tokens :, :]]

        for blk in self.encoder.blocks[0]:
            x = blk(x)
            output_cls.append(x[:, 0, :])
            output_patch.append(x[:, 1 + num_register_tokens :, :])

        out = self.lp(output_cls, output_patch)

        return out


class LP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        head_dims: Dict[str, int],
        n_encoder_blocks: int,
        which_blocks: List[int] = None,
        hidden_dim: int = 768,
        last_hidden_dim: int = 3072,
        prenorm: bool = False,
        midnorm: bool = False,
        std: float = 0.02,
    ):
        super().__init__()

        if which_blocks is None:
            which_blocks = list(range(n_encoder_blocks))
        self.which_blocks = which_blocks

        def _make_head(output_dim):
            return nn.ModuleList(
                [
                    (
                        AdaptMLP(
                            hidden_dim=(
                                last_hidden_dim
                                if bix == n_encoder_blocks - 1
                                else hidden_dim
                            ),
                            prenorm=prenorm,
                            midnorm=midnorm,
                            dim=input_dim,
                            output_dim=output_dim,
                        )
                        if bix in which_blocks
                        else None
                    )
                    for bix in range(n_encoder_blocks)
                ]
            )

        self.heads = nn.ModuleDict(
            {
                hname: nn.ModuleDict(
                    {
                        "cls": _make_head(head_dims[hname]),
                        "patch": _make_head(head_dims[hname]),
                    }
                )
                for hname in sorted(head_dims.keys())
            }
        )

        for m in self.heads.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=std)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self, x_cls: List[torch.Tensor], x_patch: List[torch.Tensor]
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        out = defaultdict(dict)

        for hname, head_dict in self.heads.items():
            xc = 0
            xp = 0

            for bix in self.which_blocks:
                xc = xc + head_dict["cls"][bix](x_cls[bix + 1])
                xp = xp + head_dict["patch"][bix](x_patch[bix + 1])

            out[hname]["cls"] = xc
            out[hname]["patch"] = xp

        return out


class AdaptMLP(nn.Module):

    def __init__(
        self,
        hidden_dim,
        prenorm=False,
        midnorm=False,
        norm_fn=nn.LayerNorm,
        act_fn=nn.GELU,
        scale=1.0,
        zinit=False,
        dim=None,
        output_dim=None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.prenorm = prenorm
        self.midnorm = midnorm
        self.norm_fn = norm_fn
        self.act_fn = act_fn
        self.scale = nn.Parameter(torch.ones(1).float()) if scale == 0.0 else scale
        self.zinit = zinit
        if dim is not None:
            self.setup(dim, output_dim)

    def extra_repr(self):
        repr = "scale={}, zinit={}".format(self.scale, self.zinit)
        return repr

    def setup(self, dim, output_dim=None):
        layers = []

        if self.prenorm:
            layers.append(self.norm_fn(dim))

        layers.append(nn.Linear(dim, self.hidden_dim))
        if self.zinit:
            nn.init.kaiming_uniform_(layers[-1].weight, a=math.sqrt(5))
            nn.init.zeros_(layers[-1].bias)

        if self.midnorm:
            layers.append(self.norm_fn(self.hidden_dim))

        layers.append(self.act_fn())

        layers.append(
            nn.Linear(self.hidden_dim, dim if output_dim is None else output_dim)
        )
        if self.zinit:
            nn.init.zeros_(layers[-1].weight)
            nn.init.zeros_(layers[-1].bias)

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.scale * self.layers(x)


def _build_encoder_from_args(args):
    return vision_transformer.get_model(
        arch=args.arch,
        patch_size=args.patch_size,
        drop_path_rate=args.drop_path_rate,
        img_size=args.image_size,
    )


def load_student_encoder_from_checkpoint(ckpt_fname, ckpt_key="model"):
    assert os.path.isfile(ckpt_fname), "Student checkpoint ({}) not found!".format(
        ckpt_fname
    )
    ckpt = torch.load(ckpt_fname, "cpu")

    encoder = _build_encoder_from_args(ckpt["args"])

    state_dict = ckpt.get(ckpt_key, ckpt)
    encoder.load_state_dict(
        {
            k.replace("module.", "").replace("encoder.", ""): v
            for k, v in state_dict.items()
            if "encoder." in k
        }
    )

    return encoder, ckpt["epoch"]


def build_student_from_args(args):

    encoder = _build_encoder_from_args(args)

    from teachers import TEACHER_CFG

    lp_args = eval(args.lp_args)
    lp = LP(
        input_dim=encoder.embed_dim,
        head_dims={
            tname: TEACHER_CFG[tname]["num_features"] for tname in args.teachers
        },
        n_encoder_blocks=encoder.n_blocks,
        **lp_args,
    )

    model = UNIC(encoder, lp)

    return model


def load_student_from_checkpoint(ckpt_fname, ckpt_key="model"):
    assert os.path.isfile(ckpt_fname), ckpt_fname
    ckpt = torch.load(ckpt_fname, "cpu")

    model = build_student_from_args(ckpt["args"])
    tnorms = ckpt["teacher_ft_stats"] if "teacher_ft_stats" in ckpt else None

    state_dict = ckpt.get(ckpt_key, ckpt)
    model.load_state_dict({k.replace("module.", ""): v for k, v in state_dict.items()})

    return model, tnorms, ckpt["epoch"]
