import math
import torch
import torch.nn as nn
from .constants import PAD_INDEX
from datetime import datetime
import os
import csv
from itertools import product
from hashlib import sha256

class SimpleLossCompute:
    """A simple loss compute and train function."""

    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1))
        loss = loss / norm

        if self.opt is not None:  # training mode
            loss.backward()
            self.opt.step()
            self.opt.zero_grad()

        return loss.data.item() * norm


def run_epoch(data_loader, model, loss_compute, print_every, device="cpu"):
    """Standard Training and Logging Function"""

    total_tokens = 0
    total_loss = 0

    for i, (src_ids_BxT, src_lengths_B, trg_ids_BxL, trg_lengths_B) in enumerate(data_loader):
        # We define some notations here to help you understand the loaded tensor
        # shapes:
        #   `B`: batch size
        #   `T`: max sequence length of source sentences
        #   `L`: max sequence length of target sentences; due to our preprocessing
        #        in the beginning, `L` == `T` == 50
        # An example of `src_ids_BxT` (when B = 2):
        #   [[2, 4, 6, 7, ..., 4, 3, 0, 0, 0],
        #    [2, 8, 6, 5, ..., 9, 5, 4, 3, 0]]
        # The corresponding `src_lengths_B` would be [47, 49].
        # Note that SOS_INDEX == 2, EOS_INDEX == 3, and PAD_INDEX = 0.

        src_ids_BxT = src_ids_BxT.to(device)
        src_lengths_B = src_lengths_B.to(device)
        trg_ids_BxL = trg_ids_BxL.to(device)
        del trg_lengths_B  # unused

        _, output = model(src_ids_BxT, trg_ids_BxL, src_lengths_B)

        loss = loss_compute(x=output, y=trg_ids_BxL[:, 1:],
                            norm=src_ids_BxT.size(0))
        total_loss += loss
        total_tokens += (trg_ids_BxL[:, 1:] != PAD_INDEX).data.sum().item()

        if model.training and i % print_every == 0:
            print("Epoch Step: %d Loss: %f" % (i, loss / src_ids_BxT.size(0)))

    return math.exp(total_loss / float(total_tokens))


def train(model, num_epochs, learning_rate, print_every, train_dl, test_dl, device="cpu"):
    # Set `ignore_index` as PAD_INDEX so that pad tokens won't be included when
    # computing the loss.
    criterion = nn.NLLLoss(reduction="sum", ignore_index=PAD_INDEX)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Keep track of dev ppl for each epoch.
    dev_ppls = []

    for epoch in range(num_epochs):
        print("Epoch", epoch)

        model.train()
        train_ppl = run_epoch(data_loader=train_dl, model=model,
                              loss_compute=SimpleLossCompute(model.generator,
                                                             criterion, optim),
                              print_every=print_every, device=device)

        model.eval()
        with torch.no_grad():
            dev_ppl = run_epoch(data_loader=test_dl, model=model,
                                loss_compute=SimpleLossCompute(model.generator,
                                                               criterion, None),
                                print_every=print_every, device=device)
            print("Validation perplexity: %f" % dev_ppl)
            dev_ppls.append(dev_ppl)

    return dev_ppls


def build_model_folder(model, params, add_ts=False):
    model_name = type(model).__name__
    h = hash_dict(params)
    if add_ts:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return "{model}_{hash}_{ts}".format(model=model_name, hash=h, ts=timestamp)
    else:
        return "{model}_{hash}".format(model=model_name, hash=h)


def save_params(params, folder_path):
    csv_path = os.path.join(folder_path, "params.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(("name", "value"))
        for key, value in params.items():
            writer.writerow((key, value))


def save_stats(stats, folder_path):
    keys = stats.keys()
    values = zip(*stats.values())
    csv_path = os.path.join(folder_path, "stats.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(keys)
        for value in values:
            writer.writerow(value)


def save_model(model, folder_path):
    model_path = os.path.join(folder_path, "model.pkl")
    torch.save(model.state_dict(), model_path)


def save(model, save_path, params, stats=None):
    folder = build_model_folder(model, params)
    folder_path = os.path.join(save_path, folder)

    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    save_model(model, folder_path)
    save_params(params, folder_path)

    if stats is not None:
        save_stats(stats, folder_path)


def hash_dict(d, len=8):
    return sha256(repr(d.items()).encode("utf-8")).hexdigest()[:len]


def create_grid(grid, fixed={}):
    ret_list = []
    keys = grid.keys()
    for values in product(*grid.values()):
        ans = fixed.copy()
        for key, value in zip(keys, values):
            ans[key] = value
        ret_list.append(ans)
    return ret_list


