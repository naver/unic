import logging
import time

import torch
from torch.utils.data import DataLoader

from utils import AverageMeter


logger = logging.getLogger()


@torch.inference_mode()
def extract_features_loop(
    model,
    dataset,
    token_type="cls",
    batch_size=128,
    num_workers=12,
    device="cuda",
    print_iter=50,
):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    # (feature, label) pairs to be stored under args.output_dir
    X = None
    Y = None
    six = 0  # sample index

    t_per_sample = AverageMeter("time-per-sample")
    t0 = time.time()

    model.eval()

    with torch.no_grad():
        for bix, batch in enumerate(dataloader):
            assert (
                len(batch) == 2
            ), "Data loader should return a tuple of (image, label) every iteration."
            image, label = batch
            feature = model(image.to(device))

            # feature should be a dictionary with the following keys
            # 'x_norm_clstoken', 'x_norm_regtokens', 'x_norm_patchtokens', 'x_prenorm', 'masks'
            if token_type == "cls":
                feature = feature["x_norm_clstoken"]
            elif token_type == "patch":
                feature = feature["x_norm_patchtokens"].mean(dim=1)
            else:
                raise ValueError("Invalid token type: {}".format(token_type))

            if X is None:
                logger.info(
                    "Size of the first batch: {} and features {}".format(
                        list(image.shape), list(feature.shape)
                    ),
                )
                X = torch.zeros(
                    len(dataset), feature.size(1), dtype=torch.float32, device="cpu"
                )
                Y = torch.zeros(len(dataset), dtype=torch.long, device="cpu")

            bs = feature.size(0)
            X[six : six + bs] = feature.cpu()
            Y[six : six + bs] = label
            six += bs

            t1 = time.time()
            td = t1 - t0
            t_per_sample.update(td / bs, bs)
            t0 = t1

            if (bix % print_iter == 0) or (bix == len(dataloader) - 1):
                logger.info(
                    "{:6d}/{:6d} extracted, {:5.3f} secs per sample, {:5.1f} mins remaining".format(
                        six,
                        len(dataset),
                        t_per_sample.avg,
                        (t_per_sample.avg / 60) * (len(dataset) - six),
                    ),
                )

    assert six == len(X)
    return X, Y
