# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import os
import datetime
import json
import time
import logging
from collections import defaultdict, deque

import torch

from torch.utils.tensorboard import SummaryWriter

import dinov2.distributed as distributed


logger = logging.getLogger("unic")


class ExternalLogger(object):
    """
    Class to handle logging via external loggers such as Tensorboard.
    """

    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.tb_enabled = False
        self.tb_writer = None
        self.tb_dir = None
        self._init_tb_logger()

    def _init_tb_logger(self):
        self.tb_enabled = distributed.is_main_process()
        if not self.tb_enabled:
            logger.info("Tensorboard is disabled")
            return

        self.tb_dir = os.path.join(self.output_dir, "tb")
        os.makedirs(self.tb_dir, exist_ok=True)
        logger.info("Tensorboard directory: {}".format(self.tb_dir))
        self.tb_writer = SummaryWriter(self.tb_dir, flush_secs=30)

    def log(
        self,
        stats: dict,
        step: int,
        prefix: str = "",
        save_path: str = "",
    ):
        if distributed.is_main_process() and save_path != "":
            with open(save_path, mode="a") as f:
                f.write(json.dumps(stats) + "\n")

        if prefix != "":
            stats = {prefix + k: v for k, v in stats.items()}

        if self.tb_enabled:
            for k, v in stats.items():
                self.tb_writer.add_scalar(k, v, step)


class MetricLogger(object):
    def __init__(self, delimiter="\t", output_file=None):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.output_file = output_file

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(type(self).__name__, attr)
        )

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def dump_in_output_file(self, iteration, iter_time, data_time):
        if self.output_file is None or not distributed.is_main_process():
            return
        dict_to_dump = dict(
            iteration=iteration,
            iter_time=iter_time,
            data_time=data_time,
        )
        dict_to_dump.update({k: v.median for k, v in self.meters.items()})
        with open(self.output_file, "a") as f:
            f.write(json.dumps(dict_to_dump) + "\n")

    def log_every(
        self,
        iterable,
        print_freq,
        header=None,
        n_iterations=None,
        start_iteration=0,
        ext_logger: ExternalLogger = None,
        ext_logger_prefix="",
    ):
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.6f}")
        data_time = SmoothedValue(fmt="{avg:.6f}")

        if n_iterations is None:
            n_iterations = len(iterable)

        space_fmt = ":" + str(len(str(n_iterations))) + "d"

        log_list = [
            header,
            "[{0" + space_fmt + "}/{1}]",
            "eta: {eta}",
            "{meters}",
            "time: {time}",
            "data: {data}",
            "max mem: {memory:.0f}",
        ]

        log_msg = self.delimiter.join(log_list)
        MB = 1024.0 * 1024.0
        it = 0

        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)

            if it % print_freq == 0 or it == n_iterations - 1:
                self.dump_in_output_file(
                    iteration=it + start_iteration,
                    iter_time=iter_time.avg,
                    data_time=data_time.avg,
                )

                eta_seconds = iter_time.global_avg * (n_iterations - it)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

                logger.info(
                    log_msg.format(
                        it,
                        n_iterations,
                        eta=eta_string,
                        meters=str(self),
                        time=str(iter_time),
                        data=str(data_time),
                        memory=(
                            torch.cuda.max_memory_allocated() / MB
                            if torch.cuda.is_available()
                            else 0
                        ),
                    )
                )

                # Log via external logger
                if ext_logger is not None:
                    meters = {k: v.avg for k, v in self.meters.items()}
                    meters.update(
                        {
                            "time_iter": iter_time.avg,
                            "time_data": data_time.avg,
                        }
                    )
                    ext_logger.log(
                        stats=meters,
                        step=it + start_iteration,
                        prefix=ext_logger_prefix,
                    )

            it += 1
            end = time.time()
            if it >= n_iterations:
                break

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info(
            "{} Total time: {} ({:.6f} s / it)".format(
                header, total_time_str, total_time / n_iterations
            )
        )


class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, num=1):
        self.deque.append(value)
        self.count += num
        self.total += value * num

    def synchronize_between_processes(self):
        """
        Distributed synchronization of the metric
        Warning: does not synchronize the deque!
        """
        if not distributed.is_enabled():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        torch.distributed.barrier()
        torch.distributed.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )
