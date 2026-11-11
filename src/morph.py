"""Quantitative morphology: particle segmentation -> instances -> size distribution (PSD).

Ground truth for particle sizes does not exist in the NFFA dataset, so accuracy is measured on
synthetic SEM particle images with exactly known instance masks (log-normal sizes, secondary-
electron edge contrast, the same artefact forward model at controlled severity). The trained
pipeline is then applied to real NFFA 'Particles' test images, converted to nm with the
scale bar, and compared with the classical threshold+watershed workflow.

Methods compared (same decoder, same labelled synthetic images):
  classical  Gaussian blur + Otsu + distance-transform watershed (ImageJ-style workflow)
  scratch    small CNN decoder, no pretrained features
  dino_in    decoder on frozen ImageNet DINO ViT-S/16 patch features
  ours       decoder on frozen SEM physics-aware DINO ViT-S/16 patch features
"""
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage as ndi
from scipy.stats import wasserstein_distance
from skimage.filters import gaussian, threshold_otsu
from skimage.measure import label, regionprops
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

import config as C
from physics import OPS, ORDER

SEG_SIZE = 256


# ------------------------------------------------------------------ synthetic SEM particles
def render_particles(size, rng, mean_d=14.0, sd_log=0.3, density=0.3, max_tries=4000):
    """Returns (uint8 image, int32 instance mask, equivalent diameters in px)."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    bg = 0.18 + 0.10 * rng.random() + 0.08 * (xx / size) * rng.standard_normal()
    tex = ndi.gaussian_filter(rng.standard_normal((size, size)), 2.0) * 0.03
    img = np.full((size, size), bg, np.float32) + tex
    inst = np.zeros((size, size), np.int32)
    target = density * size * size
    area, k, tries = 0.0, 0, 0
    while area < target and tries < max_tries:
        tries += 1
        d = float(np.exp(np.log(mean_d) + sd_log * rng.standard_normal()))
        r = max(2.0, d / 2)
        ecc = 1 + 0.25 * rng.random()
        th = rng.random() * np.pi
        cx, cy = rng.random() * size, rng.random() * size
        u = (xx - cx) * np.cos(th) + (yy - cy) * np.sin(th)
        v = -(xx - cx) * np.sin(th) + (yy - cy) * np.cos(th)
        rho = np.sqrt((u / (r * ecc)) ** 2 + (v * ecc / r) ** 2)
        m = rho < 1
        if not m.any():
            continue
        overlap = (inst[m] > 0).mean()
        if overlap > 0.05:         # touching allowed, heavy overlap rejected
            continue
        m &= inst == 0
        k += 1
        inst[m] = k
        # secondary-electron contrast: bright body, brighter rim (edge effect), slight shading
        body = 0.55 + 0.2 * rng.random()
        img[m] = body + 0.30 * rho[m] ** 4 - 0.08 * (u[m] / (r * ecc))
        area += m.sum()
    img = np.clip(img, 0, 1)
    ids = np.unique(inst)[1:]
    diam = np.array([2 * np.sqrt((inst == i).sum() / np.pi) for i in ids])
    return (img * 255).astype(np.uint8), inst, diam


def apply_artefacts(img_u8, sev, rng_seed, device="cpu"):
    """Full SEM forward model at severity sev (0 = clean)."""
    x = torch.from_numpy(img_u8).float().div(255).view(1, 1, *img_u8.shape).to(device)
    if sev <= 0:
        return img_u8
    g = torch.Generator(device="cpu").manual_seed(rng_seed)
    torch.manual_seed(int(torch.randint(0, 2 ** 31, (1,), generator=g)))
    s = torch.full((1,), float(sev), device=device)
    for o in ORDER:
        x = OPS[o](x, s).clamp(0, 1)
    return (x[0, 0].cpu().numpy() * 255).astype(np.uint8)


def synth_set(n, seed, sev_choices):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        md = float(rng.uniform(8, 24))
        img, inst, d = render_particles(SEG_SIZE, rng, mean_d=md, density=float(rng.uniform(0.15, 0.4)))
        sev = float(rng.choice(sev_choices))
        out.append({"img": apply_artefacts(img, sev, seed * 10000 + i), "inst": inst, "diam": d,
                     "sev": sev, "mean_d": md})
    return out


def targets(inst):
    """3-class target: 0 background, 1 interior, 2 boundary (2 px)."""
    fg = inst > 0
    er = ndi.grey_erosion(inst, size=3)
    di = ndi.grey_dilation(inst, size=3)
    boundary = fg & ((er != inst) | (di != inst))
    t = np.zeros(inst.shape, np.int64)
    t[fg] = 1
    t[boundary] = 2
    return t


# ------------------------------------------------------------------ instance extraction
def instances_from_probs(p_int, p_fg, min_area=6):
    markers = label(p_int > 0.5)
    fg = p_fg > 0.5
    lab = watershed(-p_int, markers, mask=fg | (markers > 0))
    return remove_small(lab, min_area)


def remove_small(lab, min_area):
    ids, cnt = np.unique(lab, return_counts=True)
    for i, c in zip(ids, cnt):
        if i and c < min_area:
            lab[lab == i] = 0
    return label(lab > 0) if lab.max() == 0 else lab


def classical(img_u8, min_area=6):
    g = gaussian(img_u8 / 255.0, 1.0)
    fg = g > threshold_otsu(g)
    dist = ndi.distance_transform_edt(fg)
    pk = peak_local_max(dist, min_distance=3, labels=label(fg), exclude_border=False)
    mk = np.zeros(fg.shape, np.int32)
    mk[tuple(pk.T)] = np.arange(1, len(pk) + 1)
    lab = watershed(-dist, mk, mask=fg)
    return remove_small(lab, min_area)


def diameters(lab, exclude_border=False):
    ds = []
    for r in regionprops(lab):
        if exclude_border:
            y0, x0, y1, x1 = r.bbox
            if y0 == 0 or x0 == 0 or y1 == lab.shape[0] or x1 == lab.shape[1]:
                continue
        ds.append(2 * np.sqrt(r.area / np.pi))
    return np.array(ds)


def shape_stats(lab):
    out = []
    for r in regionprops(lab):
        if r.perimeter > 0:
            out.append({"d": 2 * np.sqrt(r.area / np.pi), "ar": r.major_axis_length / max(r.minor_axis_length, 1e-6),
                        "circ": 4 * np.pi * r.area / r.perimeter ** 2})
    return out


def match_f1(pred, gt, thr=0.5):
    """Instance F1 at IoU >= thr (greedy one-to-one matching via overlap table)."""
    p_ids, g_ids = np.unique(pred)[1:], np.unique(gt)[1:]
    if len(p_ids) == 0 or len(g_ids) == 0:
        return 0.0
    pairs = np.stack([pred.ravel(), gt.ravel()], 1)
    pairs = pairs[(pairs[:, 0] > 0) & (pairs[:, 1] > 0)]
    inter = {}
    for a, b in pairs:
        inter[(a, b)] = inter.get((a, b), 0) + 1
    pa = dict(zip(*np.unique(pred, return_counts=True)))
    ga = dict(zip(*np.unique(gt, return_counts=True)))
    cand = sorted(((v / (pa[a] + ga[b] - v), a, b) for (a, b), v in inter.items()), reverse=True)
    used_p, used_g, tp = set(), set(), 0
    for iou, a, b in cand:
        if iou < thr:
            break
        if a in used_p or b in used_g:
            continue
        used_p.add(a); used_g.add(b); tp += 1
    return 2 * tp / (len(p_ids) + len(g_ids))


def evaluate(lab, s):
    gt = s["inst"]
    dice = 2 * ((lab > 0) & (gt > 0)).sum() / max((lab > 0).sum() + (gt > 0).sum(), 1)
    dp, dg = diameters(lab), s["diam"]
    w1 = wasserstein_distance(dp, dg) if len(dp) else float(np.median(dg))
    med_err = abs(np.median(dp) - np.median(dg)) / np.median(dg) if len(dp) else 1.0
    return {"dice": float(dice), "f1_50": match_f1(lab, gt), "w1_px": float(w1), "med_rel_err": float(med_err),
            "count_rel_err": float(abs(len(dp) - len(dg)) / len(dg))}


# ------------------------------------------------------------------ learned decoder
class SegNet(nn.Module):
    """Full-resolution CNN stem + (optional) frozen ViT patch features, fused by a light decoder."""

    def __init__(self, vit=None, feat_dim=384):
        super().__init__()
        self.vit = vit
        if vit is not None:
            for p in vit.parameters():
                p.requires_grad = False
        self.stem = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                                  nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU())
        self.down = nn.Sequential(nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                                  nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.proj = nn.Conv2d(feat_dim, 64, 1) if vit is not None else None
        cin = 32 + 64 + (64 if vit is not None else 0)
        self.head = nn.Sequential(nn.Conv2d(cin, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                                  nn.Conv2d(64, 3, 1))

    def vit_feats(self, x):
        from data import to_input
        H = x.shape[-1]
        xi = F.interpolate(x, size=(448, 448), mode="bilinear", align_corners=False)   # 28x28 tokens
        with torch.no_grad():
            self.vit.eval()
            t = self.vit.forward_features(to_input(xi))[:, self.vit.num_prefix_tokens:]
        n = int(t.shape[1] ** 0.5)
        f = t.transpose(1, 2).reshape(x.shape[0], -1, n, n).float()
        return F.interpolate(self.proj(f), size=(H, H), mode="bilinear", align_corners=False)

    def forward(self, x):
        s = self.stem(x)
        d = F.interpolate(self.down(s), size=s.shape[-2:], mode="bilinear", align_corners=False)
        parts = [s, d]
        if self.vit is not None:
            parts.append(self.vit_feats(x))
        return self.head(torch.cat(parts, 1))


def train_seg(net, train_set, device, epochs=C.SEG_EPOCHS, seed=0):
    torch.manual_seed(seed)
    X = torch.stack([torch.from_numpy(s["img"]).float().div(255) for s in train_set]).unsqueeze(1).to(device)
    T = torch.stack([torch.from_numpy(targets(s["inst"])) for s in train_set]).to(device)
    w = torch.tensor([1.0, 1.0, 3.0], device=device)     # boundaries are rare but decide splitting
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=2e-3, weight_decay=1e-4)
    bs = min(8, len(X))
    steps = max(epochs * int(np.ceil(len(X) / bs)), 10 if C.SMOKE else 600)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps)
    g = torch.Generator().manual_seed(seed)
    net.train()
    for _ in range(steps):
        i = torch.randint(0, len(X), (bs,), generator=g).to(device)
        x, t = X[i], T[i]
        k = int(torch.randint(0, 4, (1,), generator=g))
        x, t = torch.rot90(x, k, (2, 3)), torch.rot90(t, k, (1, 2))
        if torch.rand(1, generator=g) < 0.5:
            x, t = x.flip(3), t.flip(2)
        with torch.autocast(device.type, dtype=torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported()
                            else torch.float16, enabled=device.type == "cuda"):
            loss = F.cross_entropy(net(x).float(), t, weight=w)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return net


@torch.no_grad()
def seg_predict(net, imgs_u8, device, bs=16):
    net.eval()
    out = []
    for s in range(0, len(imgs_u8), bs):
        x = torch.stack([torch.from_numpy(i).float().div(255) for i in imgs_u8[s:s + bs]]).unsqueeze(1).to(device)
        p = net(x).float().softmax(1).cpu().numpy()
        out.extend(p)
    return out


def labs_from(probs):
    return [instances_from_probs(p[1], p[1] + p[2]) for p in probs]


# ------------------------------------------------------------------ experiment
def run_synthetic(device, backbones):
    """backbones: {name: frozen ViT or None}. Returns per-method, per-budget, per-severity metrics."""
    sevs = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]
    train_pool = synth_set(max(C.SEG_LABEL_BUDGETS), seed=1, sev_choices=[0.0, 0.25, 0.5, 0.75])
    test = synth_set(C.SYN_TEST, seed=2, sev_choices=sevs)
    res = {"classical": [], "sev": [s["sev"] for s in test]}
    t0 = time.time()
    res["classical"] = [evaluate(classical(s["img"]), s) for s in test]
    res["classical_time_s"] = time.time() - t0
    nets = {}
    for name, vit in backbones.items():
        for nb in C.SEG_LABEL_BUDGETS:
            for seed in C.SEEDS[: (1 if C.SMOKE else 3)]:
                net = SegNet(vit, vit.num_features if vit is not None else 384).to(device)
                train_seg(net, train_pool[:nb], device, seed=seed)
                t0 = time.time()
                labs = labs_from(seg_predict(net, [s["img"] for s in test], device))
                res[f"{name}_{nb}_{seed}"] = [evaluate(l, s) for l, s in zip(labs, test)]
                res[f"{name}_{nb}_{seed}_time_s"] = time.time() - t0
                print(f"    seg {name} n={nb} seed={seed}: dice "
                      f"{np.mean([r['dice'] for r in res[f'{name}_{nb}_{seed}']]):.3f} f1 "
                      f"{np.mean([r['f1_50'] for r in res[f'{name}_{nb}_{seed}']]):.3f}", flush=True)
                if nb == max(C.SEG_LABEL_BUDGETS) and seed == 0:
                    nets[name] = net
    # example panel for the figure
    ex = test[: 4]
    res["examples"] = {"img": [s["img"] for s in ex], "gt": [s["inst"] for s in ex],
                       "classical": [classical(s["img"]) for s in ex]}
    for name, net in nets.items():
        res["examples"][name] = labs_from(seg_predict(net, [s["img"] for s in ex], device))
    return res, nets
