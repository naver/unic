import datetime
import math
import os
import time
import logging

import numpy as np
import optuna
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

import utils


logger = logging.getLogger()


class _BaseTrainer:
    def __init__(
        self,
        train_features,
        train_labels,
        test_features,
        test_labels,
        args,
    ):
        self.args = args
        self.train_features, self.test_features = train_features, test_features
        self.train_labels, self.test_labels = train_labels, test_labels

    def set_hps(self, hps_dict: dict):
        for hp_name, hp_val in hps_dict.items():
            setattr(self.args, "clf_{}".format(hp_name), hp_val)

    def sample_hps(self, trial: optuna.trial.Trial) -> dict:
        hps_dict = {}
        for hp_name in self.hps_list:
            hp_val = trial.suggest_float(
                hp_name,
                getattr(self.args, "clf_{}_min".format(hp_name)),
                getattr(self.args, "clf_{}_max".format(hp_name)),
                log=True,
            )
            hps_dict[hp_name] = hp_val

        return hps_dict

    def fit(self, hps_dict: dict):
        raise NotImplementedError()

    def __call__(self, trial: optuna.trial.Trial = None):
        t_study_0 = time.time()

        # If we are in the HPS phase
        # sample the hyper-parameters required for the classifier
        # otherwise we expect the hyper-parameters to be already set
        if trial is not None:
            hps_dict = self.sample_hps(trial)

        else:
            hps_dict = {
                hp_name: getattr(self.args, "clf_{}".format(hp_name))
                for hp_name in self.hps_list
            }
            utils.print_program_info(self.args)

        # train classifier on the training data, and return predictions on the test data
        clf, preds = self.fit(hps_dict)

        # compute accuracy
        acc_1 = np.mean(np.equal(self.test_labels, preds).astype(np.float32)) * 100.0
        confmat = confusion_matrix(self.test_labels, preds)
        m_ap = (np.diag(confmat) / confmat.sum(axis=1)).mean() * 100.0

        # save the final evaluation logs
        if trial is None:
            utils.save_pickle(
                {"test/top1": acc_1, "test/m_ap": m_ap},
                os.path.join(self.args.output_dir, "logs.pkl"),
            )
            utils.save_pickle(
                clf.state_dict() if hasattr(clf, "state_dict") else clf,
                os.path.join(self.args.output_dir, "classifier.pth"),
            )
            utils.save_pickle(
                {"labels": self.test_labels, "predictions": preds},
                os.path.join(self.args.output_dir, "predictions.pkl"),
            )

            t_study_1 = time.time()
            t_study = str(datetime.timedelta(seconds=int(t_study_1 - t_study_0)))
            logger.info(
                " - Top-1 acc: {:3.1f}, m-AP: {:3.1f}, Runtime: {}".format(
                    acc_1, m_ap, t_study
                ),
            )

        del clf
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # return the test set accuracy to maximize
        return acc_1


class LogregSklearnTrainer(_BaseTrainer):
    hps_list = ["C"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_features = self.train_features.detach().numpy()
        self.train_labels = self.train_labels.detach().numpy()
        self.test_features = self.test_features.detach().numpy()
        self.test_labels = self.test_labels.detach().numpy()

    def fit(self, hps_dict: dict):
        # fit logistic regression classifier on training data
        clf = LogisticRegression(
            penalty="l2",
            dual=False,
            C=hps_dict["C"],
            solver="lbfgs",
            max_iter=self.args.clf_max_iter,
            random_state=self.args.seed,
            verbose=0,
            n_jobs=self.args.n_sklearn_workers,
        ).fit(self.train_features, self.train_labels)

        # make predictions on the test data
        preds = clf.predict(self.test_features)

        return clf, preds


class LogregTorchTrainer(_BaseTrainer):
    hps_list = ["lr", "wd"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_features = self.train_features.to(self.args.device)
        self.train_labels = self.train_labels.to(self.args.device)
        self.test_features = self.test_features.to(self.args.device)
        self.test_labels = self.test_labels.cpu().numpy()

    def create_model(self, feature_dim, output_dim):
        return torch.nn.Linear(feature_dim, output_dim)

    def fit(self, hps_dict: dict):
        # create training and test set iterators
        train_iter = TorchIterator(
            (self.train_features, self.train_labels),
            self.args.clf_batch_size,
            shuffle=True,
        )
        test_iter = TorchIterator(
            (self.test_features,), self.args.clf_batch_size, shuffle=False
        )

        # define logistic classifier
        clf = self.create_model(
            self.train_features.size(1), self.train_labels.max() + 1
        ).to(self.args.device)
        crit = torch.nn.CrossEntropyLoss().to(self.args.device)

        optim = torch.optim.SGD(
            clf.parameters(),
            lr=hps_dict["lr"],
            momentum=self.args.clf_mom,
            weight_decay=hps_dict["wd"],
        )

        for epoch in range(self.args.clf_epochs):
            train_top1 = train(clf, train_iter, crit, optim, epoch, self.args.device)
            test_preds = predict(clf, test_iter, self.args.device)
            lr = adjust_learning_rate(
                optim, hps_dict["lr"], self.args.clf_epochs, epoch
            )

            # if something went wrong during training
            # e.g. SGD diverged
            if train_top1 == -1:
                break

        # return the last test accuracy
        return clf, test_preds


def train(clf, dataloader, criterion, optimizer, epoch, device="cuda"):
    top1 = utils.AverageMeter("Acc@1", ":6.2f")

    # switch to train mode
    clf.train()

    for i, (feature, label) in enumerate(dataloader):
        feature = feature.to(device)
        label = label.to(device)

        # compute output
        output = clf(feature)
        loss = criterion(output, label)

        if not torch.isfinite(loss):
            logger.info("==> Loss ({}) is not finite, terminating".format(loss.item()))
            optimizer.zero_grad()
            return -1

        # measure accuracy and record loss
        acc1 = utils.accuracy(output, label, topk=(1,))[0]
        top1.update(acc1.item(), feature.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return top1.avg


def predict(clf, dataloader, device="cuda"):
    # switch to evaluate mode
    clf.eval()

    # keep predictions per class
    preds = torch.empty(len(dataloader.tensors[0]), dtype=torch.int32, device=device)
    preds.fill_(-1)
    six = 0

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            feature = batch[0].to(device)
            bs = feature.size(0)

            # compute output
            output = clf(feature)

            # store the predicted classes
            preds[six : six + bs] = torch.argmax(output, dim=1)
            six += bs

    # make sure that there is no invalid prediction
    assert torch.all(preds >= 0).item()

    return preds.detach().cpu().numpy()


def adjust_learning_rate(optimizer, lr_init, n_epochs, epoch):
    """Decay the learning rate based on cosine schedule"""
    lr = lr_init
    lr *= 0.5 * (1.0 + math.cos(math.pi * epoch / n_epochs))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


class TorchIterator:
    """
    Iterator for list of tensors whose first dimension match.
    """

    def __init__(self, tensors, batch_size, shuffle=True):
        self.tensors = tensors
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = tensors[0].device

        # number of elements in each tensor should be equal and bigger than batch size
        n_elems = [len(t) for t in tensors]
        assert np.all(np.equal(n_elems, n_elems[0]))
        self.n_sample = n_elems[0]
        if self.n_sample < batch_size:
            logger.info(
                "==> Length of tensors ({}) given to TorchIterator is less than batch size ({}). "
                "    Reducing the batch size to {}".format(
                    self.n_sample, batch_size, self.n_sample
                )
            )
            self.batch_size = self.n_sample

        self._s_ix = (
            0  # index of sample that will be fetched as the first sample in next_batch
        )
        self._order = torch.zeros(
            self.n_sample, dtype=torch.long, device=self.device
        )  # order of samples fetched in an epoch
        self.reset_batch_order()

    def __len__(self):
        return math.ceil(self.n_sample / self.batch_size)

    def __iter__(self):
        return self

    def _check_new_epoch(self):
        # check whether there is no not-fetched sample left
        return self._s_ix >= self.n_sample

    def reset_batch_order(self):
        self._s_ix = 0
        if self.shuffle:
            torch.randperm(self.n_sample, out=self._order)
        else:
            torch.arange(self.n_sample, out=self._order)

    def __next__(self):
        if self._check_new_epoch():
            self.reset_batch_order()
            raise StopIteration

        inds = self._order[self._s_ix : self._s_ix + self.batch_size]
        self._s_ix += len(inds)
        batch = [t[inds] for t in self.tensors]
        return batch
