"""GPU-resident SEM data, generic augmentation and DINO multi-crop.

Images are cached as (N, 1, 256, 256) uint8 on the GPU (21k images = 1.4 GB). All
augmentation runs batched on the GPU, so the CPU never decodes JPEGs during training.
"""
import json
import math

import numpy as np
import torch
import torch.nn.functional as F

import config as C
from physics import physics_chain

MEAN, STD = 0.449, 0.226   # ImageNet mean/std averaged over RGB (grayscale input replicated x3)


class SEMData:
    def __init__(self, device):
        d = C.DATA_DIR
        self.meta = json.load(open(d / "meta.json"))
        self.splits = json.load(open(d / "splits.json"))
        self.classes = self.meta["classes"]
        self.n_classes = len(self.classes)
        X = torch.from_numpy(np.load(d / "images.npy")).unsqueeze(1).contiguous()
        self.X = X.to(device)
        self.y = torch.from_numpy(np.load(d / "labels.npy")).long().to(device)
        self.device = device

    def indices(self, split):
        return np.asarray(self.splits[split])

    def budget(self, seed, frac):
        return np.asarray(self.splits["budgets"][str(seed)][str(frac)])

    def raw(self, idx):
        b = torch.as_tensor(np.asarray(idx), device=self.device)
        return self.X[b].float().div_(255), self.y[b]

    def batches(self, idx, bs, train, seed=0, aug=None):
        order = np.random.default_rng(seed).permutation(idx) if train else np.asarray(idx)
        n = len(order) // bs * bs if (train and len(order) >= bs) else len(order)
        for s in range(0, n, bs):
            x, y = self.raw(order[s:s + bs])
            x = aug(x) if aug is not None else center(x)
            yield x, y


def to_input(x):
    """(B,1,H,W) in [0,1] -> normalised 3-channel network input."""
    return ((x - MEAN) / STD).expand(-1, 3, -1, -1)


def center(x, size=C.IMG_SIZE):
    o = (x.shape[-1] - size) // 2
    return x[:, :, o:o + size, o:o + size]


def random_resized_crop(x, size, scale, ratio=(3 / 4, 4 / 3), flip=True, rot90=True):
    """Per-sample RandomResizedCrop (+ flips, 90 degree rotations: SEM has no canonical
    orientation) via one batched affine grid."""
    B, dev = x.shape[0], x.device
    area = torch.empty(B, device=dev).uniform_(*scale)
    r = torch.exp(torch.empty(B, device=dev).uniform_(math.log(ratio[0]), math.log(ratio[1])))
    sw = torch.sqrt(area * r).clamp(max=1.0)
    sh = torch.sqrt(area / r).clamp(max=1.0)
    tx = (torch.rand(B, device=dev) * 2 - 1) * (1 - sw)
    ty = (torch.rand(B, device=dev) * 2 - 1) * (1 - sh)
    fx = torch.where(torch.rand(B, device=dev) < 0.5, -1.0, 1.0) if flip else torch.ones(B, device=dev)
    k = torch.randint(0, 4 if rot90 else 1, (B,), device=dev).float() * (math.pi / 2)
    cos, sin = torch.cos(k), torch.sin(k)
    theta = torch.stack([torch.stack([sw * cos * fx, -sh * sin, tx], 1),
                         torch.stack([sw * sin * fx, sh * cos, ty], 1)], 1)
    grid = F.affine_grid(theta, (B, 1, size, size), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)


def gaussian_blur(x, p, sigma=(0.1, 2.0)):
    B, dev = x.shape[0], x.device
    on = torch.rand(B, device=dev) < p
    if not on.any():
        return x
    s = torch.empty(int(on.sum()), device=dev).uniform_(*sigma)
    r = torch.arange(-7, 8, device=dev).float()
    k = torch.exp(-0.5 * (r.view(1, -1) / s.view(-1, 1)) ** 2)
    k = k / k.sum(1, keepdim=True)
    xi = x[on]
    n = xi.shape[0]
    xi = F.conv2d(F.pad(xi, (7, 7, 0, 0), mode="reflect").transpose(0, 1), k.view(n, 1, 1, 15), groups=n)
    xi = F.conv2d(F.pad(xi, (0, 0, 7, 7), mode="reflect"), k.view(n, 1, 15, 1), groups=n).transpose(0, 1)
    y = x.clone()
    y[on] = xi
    return y


def photometric(x, p_jitter=0.8, blur=0.5, solarize=0.0):
    """Generic (natural-image) photometric augmentation adapted to grayscale:
    brightness/contrast jitter, Gaussian blur, solarisation."""
    B, dev = x.shape[0], x.device
    on = (torch.rand(B, 1, 1, 1, device=dev) < p_jitter).float()
    br = 1 + (torch.rand(B, 1, 1, 1, device=dev) * 2 - 1) * 0.4 * on
    ct = 1 + (torch.rand(B, 1, 1, 1, device=dev) * 2 - 1) * 0.4 * on
    x = x * br
    m = x.mean((1, 2, 3), keepdim=True)
    x = ((x - m) * ct + m).clamp(0, 1)
    x = gaussian_blur(x, blur)
    if solarize > 0:
        sol = torch.rand(B, 1, 1, 1, device=dev) < solarize
        x = torch.where(sol & (x > 0.5), 1 - x, x)
    return x


def finetune_aug(x):
    """Supervised fine-tuning augmentation (identical for every model)."""
    x = random_resized_crop(x, C.IMG_SIZE, (0.35, 1.0))
    return photometric(x, p_jitter=0.8, blur=0.2)


def multicrop(x, physics=True, ops=None):
    """DINO views: 2 global (224) + N_LOCAL local (96) crops. Generic DINO photometrics
    (asymmetric blur / solarisation) and, for the physics variant, the SEM operator chain."""
    g1 = random_resized_crop(x, C.IMG_SIZE, C.GLOBAL_SCALE)
    g2 = random_resized_crop(x, C.IMG_SIZE, C.GLOBAL_SCALE)
    locs = [random_resized_crop(x, C.LOCAL_SIZE, C.LOCAL_SCALE) for _ in range(C.N_LOCAL)]
    if physics:
        g1, g2 = physics_chain(g1, ops=ops), physics_chain(g2, ops=ops)
        locs = [physics_chain(l, ops=ops) for l in locs]
    g1 = photometric(g1, blur=1.0)
    g2 = photometric(g2, blur=0.1, solarize=0.2)
    locs = [photometric(l, blur=0.5) for l in locs]
    return [g1, g2] + locs
