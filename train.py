import argparse
import random
import shutil
import sys
import os
import logging
import math
import time 
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torch.utils.data import DataLoader
from torchvision import transforms

from compressai.datasets import ImageFolder
from compressai.losses import RateDistortionLoss
import sys
sys.path.insert(0, '.')
# sys.path.insert(0, '..')
from models.msinn import MSINNCompression

lmbda_list = [0.0018, 0.0035, 0.0067, 0.0130, 0.0250, 0.0483, 0.0932, 0.1800, 0.320, 0.569, 1.012, 1.8]

class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, output, target, lmbda):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        if isinstance(lmbda, list):
            out["bpp_loss"] = sum(
                (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
                for likelihoods in output["likelihoods"].values()
            )
            out["mse_loss"] = sum(self.mse(output["x_hat"][b], target[b]) * lmbda[b] for b in range(N)) / N
            out["loss"] = 255**2 * out["mse_loss"] + out["bpp_loss"]
        else:
            out["bpp_loss"] = sum(
                (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
                for likelihoods in output["likelihoods"].values()
            )
            out["mse_loss"] = self.mse(output["x_hat"], target)
            out["loss"] = lmbda * 255**2 * out["mse_loss"] + out["bpp_loss"]

        return out

class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""
    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)
    return optimizer

def init(args):
    base_dir = f'./outputs/anchor{args.anchor_num}/epoch{args.epochs}/'
    os.makedirs(base_dir, exist_ok=True)

    return base_dir

def setup_logger(log_dir):
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    log_file_handler = logging.FileHandler(log_dir, encoding='utf-8')
    log_file_handler.setFormatter(log_formatter)
    root_logger.addHandler(log_file_handler)

    log_stream_handler = logging.StreamHandler(sys.stdout)
    log_stream_handler.setFormatter(log_formatter)
    root_logger.addHandler(log_stream_handler)

    logging.info('Logging file is %s' % log_dir)


def train_one_epoch(
    model, criterion, train_dataloader, optimizer, epoch, clip_max_norm, anchor_num
):
    model.train()
    device = next(model.parameters()).device

    for i, d in enumerate(train_dataloader):
        d = d.to(device)

        batch_size = d.shape[0]
        qp_index = np.random.choice(anchor_num, size=batch_size).tolist()
        lmbda = [lmbda_list[e] for e in qp_index]

        optimizer.zero_grad()

        out_net = model(d, qp_index)

        out_criterion = criterion(out_net, d, lmbda)
        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        num_pixels = d.numel() // 3
        latent_rates = {k: torch.log(likelihoods).sum() / (-math.log(2) * num_pixels) 
                               for k, likelihoods in out_net["likelihoods"].items()}
        latent_rates_msg = ' '.join([f'{k}: {v.item():.4f}' for k, v in latent_rates.items()])

        if i * len(d) % 5000 == 0:
            logging.info(
                f"Train epoch {epoch}: ["
                f"{i*len(d)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\tLoss: {out_criterion["loss"].item():.3f} |'
                f'\tMSE loss: {out_criterion["mse_loss"].item():.4f} |'
                f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
                f"\t{latent_rates_msg}"
            )


def test_epoch(epoch, test_dataloader, model, criterion, anchor_num):
    model.eval()
    device = next(model.parameters()).device

    test_anchor_num = 4
    loss = [AverageMeter() for _ in range(test_anchor_num)]
    bpp_loss = [AverageMeter() for _ in range(test_anchor_num)]
    mse_loss = [AverageMeter() for _ in range(test_anchor_num)]

    qp_index_list = np.linspace(0, anchor_num - 1, test_anchor_num).round().astype(int)
    with torch.no_grad():
        for d in test_dataloader:
            d = d.to(device)
            for i, qp_index in enumerate(qp_index_list):
                out_net = model(d, qp_index)
                out_criterion = criterion(out_net, d, lmbda_list[qp_index])

                bpp_loss[i].update(out_criterion["bpp_loss"])
                loss[i].update(out_criterion["loss"])
                mse_loss[i].update(out_criterion["mse_loss"])

    for i, qp_index in enumerate(qp_index_list):
        logging.info(
            f"Test epoch {epoch}: Average losses:"
            f"\tLoss: {loss[i].avg:.3f} |"
            f"\tMSE loss: {mse_loss[i].avg:.4f} |"
            f"\tBpp loss: {bpp_loss[i].avg:.2f} |"
            f"lambda: {lmbda_list[qp_index]} \n "
        )

    return sum(l.avg for l in loss)


def save_checkpoint(state, is_best, base_dir, filename="checkpoint.pth.tar"):
    torch.save(state, base_dir+filename)
    if is_best:
        logging.info("Saving BEST!")
        shutil.copyfile(base_dir+filename, base_dir+"checkpoint_best_loss.pth.tar")


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-d", "--dataset", type=str, default='./data/flicker', help="Training dataset"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=750,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=4,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "-b", "--batch-size", type=int, default=8, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=64,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", default=True, help="Use cuda")
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument("--seed", type=int, help="Set random seed for reproducibility")
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    parser.add_argument(
        '--name', 
        default=datetime.now().strftime('%Y-%m-%d_%H_%M_%S'), 
        type=str,
        help='Result dir name', 
    )

    # model configs
    parser.add_argument("--N", type=int, default=128)
    parser.add_argument("--M", type=int, default=192)
    parser.add_argument("--depth", type=int, nargs="+", default=[4, 4, 4, 4])
    parser.add_argument(
        "--flow_permutation",
        type=str,
        default="invconv",
        choices=["invconv", "shuffle", "reverse", "cross"],
        help="Type of flow permutation",
    )

    parser.add_argument(
        "--flow_coupling",
        type=str,
        default="affine",
        choices=["additive", "affine"],
        help="Type of flow coupling",
    )
    parser.add_argument("--use_act_norm", action="store_true", default=False)
    parser.add_argument("--post_process", action="store_true", default=False)
    parser.add_argument("--lrp", action="store_true")
    parser.add_argument("--sc_type", type=str, default="multi_ckbd",)
    parser.add_argument("--spatial_context_num", type=int, default=5)

    parser.add_argument("--anchor_num", type=int, default=8)

    parser.add_argument("--finetune", action="store_true", default=False)
 
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)
    base_dir = init(args)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    train_transforms = transforms.Compose(
        [transforms.RandomCrop(args.patch_size), transforms.ToTensor()]
    )
    test_transforms = transforms.Compose(
        [transforms.CenterCrop(args.patch_size), transforms.ToTensor()]
    )

    train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms)
    test_dataset = ImageFolder(args.dataset, split="test", transform=test_transforms)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    setup_logger(base_dir + '/' + time.strftime('%Y%m%d_%H%M%S') + '.log')
    msg = f'======================= {args.name} ======================='
    logging.info(msg)
    for k in args.__dict__:
        logging.info(k + ':' + str(args.__dict__[k]))
    logging.info('=' * len(msg))

    net = MSINNCompression(M=args.M, N=args.N, depths=args.depth,
        flow_permutation=args.flow_permutation, flow_coupling=args.flow_coupling, use_act_norm=args.use_act_norm,
        post_process=args.post_process, lrp=args.lrp, sc_type=args.sc_type, spatial_context_num=args.spatial_context_num,
        anchor_num=args.anchor_num
    )
    net = net.to(device)

    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)

    optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[675, 720], gamma=0.1)
    criterion = RateDistortionLoss()

    if args.finetune:
        # fine-tune a pretrained model to support a wider bitrate range (e.g. increasing anchor_num from 8 to 12)
        checkpoint = torch.load("./outputs/anchor8/epoch750/checkpoint.pth.tar", map_location=device)
        state_dict = checkpoint["state_dict"]
        new_state_dict = {}
        for k, v in state_dict.items():
            if 'gain' in k:
                gain_tensor = torch.zeros(args.anchor_num, v.shape[1], v.shape[2], v.shape[3], device=v.device)
                gain_tensor[:v.shape[0]] = v
                gain_tensor[v.shape[0]:] = v[-1:]
                new_state_dict[k] = gain_tensor
            else:
                new_state_dict[k] = v
        net.load_state_dict(new_state_dict, strict=True)

    last_epoch = 0
    if args.checkpoint:  # load from previous checkpoint
        checkpoint = torch.load(args.checkpoint, map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        net.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        best_loss = checkpoint["loss"]

    best_loss = float("inf")
    for epoch in range(last_epoch, args.epochs):
        logging.info(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            epoch,
            args.clip_max_norm,
            args.anchor_num
        )
        loss = test_epoch(epoch, test_dataloader, net, criterion, args.anchor_num)
        lr_scheduler.step()

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if args.save:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                is_best,
                base_dir
            )


if __name__ == "__main__":
    main(sys.argv[1:])
