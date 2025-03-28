
"""
Evaluate an end-to-end compression model on an image dataset.
"""
import argparse
import json
import math
import sys
import time

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from pytorch_msssim import ms_ssim
from torchvision import transforms

import compressai

from compressai.ops import compute_padding
import sys
sys.path.insert(0, '.')
# sys.path.insert(0, '..')
from models.msinn import MSINNCompression

torch.backends.cudnn.deterministic = True
torch.set_num_threads(1)

# from torchvision.datasets.folder
IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)

lmbda_list = [0.0018, 0.0035, 0.0067, 0.0130, 0.0250, 0.0483, 0.0932, 0.1800, 0.320, 0.569, 1.012, 1.8]

def collect_images(rootpath: str) -> List[str]:
    image_files = []

    for ext in IMG_EXTENSIONS:
        image_files.extend(Path(rootpath).rglob(f"*{ext}"))
    return sorted(image_files)


def psnr(a: torch.Tensor, b: torch.Tensor, max_val: int = 255) -> float:
    return 20 * math.log10(max_val) - 10 * torch.log10((a - b).pow(2).mean())


def compute_metrics(
    org: torch.Tensor, rec: torch.Tensor, max_val: int = 255
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    org = (org * max_val).clamp(0, max_val).round()
    rec = (rec * max_val).clamp(0, max_val).round()
    metrics["psnr-rgb"] = psnr(org, rec).item()
    metrics["ms-ssim-rgb"] = ms_ssim(org, rec, data_range=max_val).item()
    return metrics


def read_image(filepath: str) -> torch.Tensor:
    assert filepath.is_file()
    img = Image.open(filepath).convert("RGB")
    return transforms.ToTensor()(img)


@torch.no_grad()
def inference(model, x, qp_index):
    x = x.unsqueeze(0)

    h, w = x.size(2), x.size(3)
    pad, unpad = compute_padding(h, w, min_div=2**6)  # pad to allow 6 strides of 2

    x_padded = F.pad(x, pad, mode="constant", value=0)

    start = time.time()
    out_enc = model.compress(x_padded, qp_index=qp_index)
    enc_time = time.time() - start

    start = time.time()
    out_dec = model.decompress(
            out_enc["strings"],
            out_enc["shape"],
            qp_index=qp_index,
        )
    dec_time = time.time() - start

    out_dec["x_hat"] = F.pad(out_dec["x_hat"], unpad)

    # input images are 8bit RGB for now
    metrics = compute_metrics(x, out_dec["x_hat"], 255)
    num_pixels = x.size(0) * x.size(2) * x.size(3)
    bpp = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels

    return {
        "psnr-rgb": metrics["psnr-rgb"],
        "ms-ssim-rgb": metrics["ms-ssim-rgb"],
        "bpp": bpp,
        "encoding_time": enc_time,
        "decoding_time": dec_time,
    }


@torch.no_grad()
def inference_entropy_estimation(model, x, qp_index):
    x = x.unsqueeze(0)

    start = time.time()
    out_net = model.forward(x, qp_index=qp_index)
    elapsed_time = time.time() - start

    # input images are 8bit RGB for now
    metrics = compute_metrics(x, out_net["x_hat"], 255)
    num_pixels = x.size(0) * x.size(2) * x.size(3)
    bpp = sum(
        (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
        for likelihoods in out_net["likelihoods"].values()
    )

    return {
        "psnr-rgb": metrics["psnr-rgb"],
        "ms-ssim-rgb": metrics["ms-ssim-rgb"],
        "bpp": bpp.item(),
        "encoding_time": elapsed_time / 2.0,  # broad estimation
        "decoding_time": elapsed_time / 2.0,
    }


def load_checkpoint(checkpoint_path: str, args) -> nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint
    for key in ["network", "state_dict", "model_state_dict"]:
        if key in checkpoint:
            state_dict = checkpoint[key]
    new_state_dict = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    state_dict = new_state_dict

    net = MSINNCompression(M=args.M, N=args.N, depths=args.depth,
        flow_permutation=args.flow_permutation, flow_coupling=args.flow_coupling, use_act_norm=args.use_act_norm,
        post_process=args.post_process, lrp=args.lrp, sc_type=args.sc_type, spatial_context_num=args.spatial_context_num,
        anchor_num=args.anchor_num
    )
    net.load_state_dict(state_dict)
    net.update(force=True)

    return net.eval()


def eval_model(
    model: nn.Module,
    filepaths,
    entropy_estimation: bool = False,
    qp_index=None,
    **args: Any,
) -> Dict[str, Any]:
    device = next(model.parameters()).device
    metrics = defaultdict(float)

    for filepath in filepaths:
        x = read_image(filepath).to(device)
        if not entropy_estimation:
            if args["half"]:
                model = model.half()
                x = x.half()
            rv = inference(model, x, qp_index)
        else:
            rv = inference_entropy_estimation(model, x, qp_index)
        for k, v in rv.items():
            metrics[k] += v

    for k, v in metrics.items():
        metrics[k] = v / len(filepaths)

    return metrics


def setup_args():
    # Common options.
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("dataset", type=str, help="dataset path") # /aiarena/group/icgroup/data/kodak
    parser.add_argument(
        "-c",
        "--entropy-coder",
        choices=compressai.available_entropy_coders(),
        default=compressai.available_entropy_coders()[0],
        help="entropy coder (default: %(default)s)",
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="enable CUDA",
        default=True,
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="convert model to half floating point (fp16)",
    )
    parser.add_argument(
        "--entropy_estimation",
        action="store_true",
        help="use evaluated entropy estimation (no entropy coding)",
    )

    parser.add_argument(
        "-p",
        "--checkpoint_path",
        dest="checkpoint_paths",
        type=str,
        nargs="*",
        required=True,
        help="checkpoint path",
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

    return parser


def main(argv):  # noqa: C901
    parser = setup_args()
    args = parser.parse_args(argv)

    description = (
        "entropy-estimation" if args.entropy_estimation else args.entropy_coder
    )

    filepaths = collect_images(args.dataset)
    if len(filepaths) == 0:
        print("Error: no images found in directory.", file=sys.stderr)
        raise SystemExit(1)

    compressai.set_entropy_coder(args.entropy_coder)

    runs = args.checkpoint_paths
    results = defaultdict(list)
    for run in runs:
        print(f"Evaluating {run:s} ")
        for qp_index in range(args.anchor_num):
            model = load_checkpoint(run, args)
            if args.cuda and torch.cuda.is_available():
                model = model.to("cuda")
            args_dict = vars(args)
            metrics = eval_model(
                model,
                filepaths,
                qp_index=qp_index,
                **args_dict,
            )
            for k, v in metrics.items():
                results[k].append(v)


    description = (
        "entropy estimation" if args.entropy_estimation else args.entropy_coder
    )
    output = {
        "description": f"Inference ({description})",
        "results": results,
    }
    print(json.dumps(output, indent=2))

    psnr_list = results['psnr-rgb']
    bpp_list = results['bpp']
    psnr_str = ','.join(f"{p:.2f}" for p in psnr_list)
    bpp_str = ','.join(f"{p:.3f}" for p in bpp_list)

    print('PSNR:', psnr_str)
    print('bpp:', bpp_str)

if __name__ == "__main__":
    main(sys.argv[1:])