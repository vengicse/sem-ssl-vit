"""Label-efficient evaluation: end-to-end fine-tuning, linear probe and k-NN, plus the SEM-C
robustness benchmark. One fixed recipe for every model; no early stopping on validation labels
(at a 1 % budget a 3,000-image labelled validation set would be unrealistic)."""
import math
import time

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

import config as C
from data import finetune_aug, center, to_input
from dino import amp_dtype, extract, knn_acc
from physics import ORDER, corrupt


def build(arch, n_cls, ckpt=None):
    """arch: timm name; ckpt: optional domain-pretrained ViT backbone state_dict."""
    name = C.SMOKE_MODEL if C.SMOKE else arch
    kw = dict(pretrained=(ckpt is None and not C.SMOKE), num_classes=n_cls)
    try:
        m = timm.create_model(name, drop_path_rate=C.FT_DROP_PATH, **kw)
    except TypeError:
        m = timm.create_model(name, **kw)
    if ckpt is not None:
        sd = torch.load(ckpt, map_location="cpu")
        missing, unexpected = m.load_state_dict(sd, strict=False)
        assert all(k.startswith("head") for k in missing), missing
    return m


def param_groups(model, lr):
    blocks = getattr(model, "blocks", None)
    L = len(blocks) + 1 if blocks is not None else None
    groups = {}
    for n, p in model.named_parameters():
        if L is None:
            lid, scale = 0, 1.0
        else:
            if n.startswith(("cls_token", "pos_embed", "patch_embed", "reg_token")):
                lid = 0
            elif n.startswith("blocks."):
                lid = int(n.split(".")[1]) + 1
            else:
                lid = L
            scale = C.FT_LAYER_DECAY ** (L - lid)
        no_wd = p.ndim <= 1 or n.endswith(".bias") or "pos_embed" in n or "cls_token" in n
        key = (lid, no_wd)
        if key not in groups:
            groups[key] = {"params": [], "base_lr": lr * scale, "lr": lr * scale,
                           "weight_decay": 0.0 if no_wd else C.FT_WD}
        groups[key]["params"].append(p)
    return list(groups.values())


@torch.no_grad()
def predict(model, data, idx, transform=None, bs=256):
    model.eval()
    dt = amp_dtype()
    out = []
    for s in range(0, len(idx), bs):
        x, _ = data.raw(idx[s:s + bs])
        x = center(x)
        if transform is not None:
            x = transform(x)
        with torch.autocast(data.device.type, dtype=dt, enabled=data.device.type == "cuda"):
            out.append(model(to_input(x)).float().softmax(-1))
    return torch.cat(out).cpu().numpy()


def semc(predict_fn, data, idx, y):
    """Accuracy under each artefact x severity (SEM-C)."""
    res = {}
    for op in ORDER:
        for sev in C.SEMC_SEVERITIES:
            p = predict_fn(lambda x, op=op, sev=sev: corrupt(x, op, sev))
            res[f"{op}_{sev}"] = float((p.argmax(1) == y).mean())
    return res


def finetune(data, arch, frac, seed, ckpt=None, robust=False, keep=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = data.device
    tr = data.budget(seed, frac)
    te = data.indices("test")
    model = build(arch, data.n_classes, ckpt).to(dev).to(memory_format=torch.channels_last)
    is_vit = hasattr(model, "blocks")
    lr = C.FT_LR["vit" if is_vit else "cnn"]
    opt = torch.optim.AdamW(param_groups(model, lr if is_vit else lr * 0.5))
    crit = nn.CrossEntropyLoss(label_smoothing=C.FT_LABEL_SMOOTH)
    bs = min(C.FT_BATCH, len(tr))
    ipe = max(1, len(tr) // bs)
    epochs = max(C.FT_EPOCHS[frac], math.ceil(C.MIN_STEPS / ipe))
    total, warm = epochs * ipe, min(C.FT_WARMUP_EPOCHS * ipe, epochs * ipe // 5)
    dt = amp_dtype()
    scaler = torch.amp.GradScaler(enabled=(dev.type == "cuda" and dt == torch.float16))
    step, t0, hist = 0, time.time(), []
    for ep in range(epochs):
        model.train()
        tl = 0.0
        for x, y in data.batches(tr, bs, train=True, seed=seed * 7919 + ep, aug=finetune_aug):
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * ((step + 1) / warm if step < warm else
                                          0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm))))
            with torch.autocast(dev.type, dtype=dt, enabled=dev.type == "cuda"):
                loss = crit(model(to_input(x).contiguous(memory_format=torch.channels_last)).float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            step += 1
            tl += loss.item()
        hist.append({"epoch": ep + 1, "train_loss": tl / ipe})
    train_time = time.time() - t0
    y = data.y[torch.as_tensor(te, device=dev)].cpu().numpy()
    t1 = time.time()
    p = predict(model, data, te)
    infer_ms = (time.time() - t1) / len(te) * 1000
    res = {"arch": arch, "frac": frac, "seed": seed, "n_train": int(len(tr)), "epochs": epochs,
           "train_time_s": train_time, "infer_ms_per_img": infer_ms, "history": hist,
           "probs": p.astype(np.float16), "y": y,
           "params_M": sum(q.numel() for q in model.parameters()) / 1e6}
    if robust:
        res["semc"] = semc(lambda tf: predict(model, data, te, transform=tf), data, te, y)
    if keep:
        res["model"] = model
    else:
        del model
        torch.cuda.empty_cache() if dev.type == "cuda" else None
    return res


# ------------------------------------------------------------------ frozen-feature protocols
def frozen_backbone(arch, ckpt=None):
    name = C.SMOKE_MODEL if C.SMOKE else arch
    m = timm.create_model(name, pretrained=(ckpt is None and not C.SMOKE), num_classes=0)
    if ckpt is not None:
        m.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True)
    return m


def linear_probe(trf, try_, tef, n_cls, epochs=C.LINEAR_EPOCHS, wd=1e-4, seed=0):
    torch.manual_seed(seed)
    lin = nn.Linear(trf.shape[1], n_cls).to(trf.device)
    opt = torch.optim.AdamW(lin.parameters(), lr=1e-3, weight_decay=wd)
    bs = min(256, len(trf))
    steps = max(epochs * math.ceil(len(trf) / bs), C.MIN_STEPS)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    g = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(steps):
        i = torch.randint(0, len(trf), (bs,), generator=g).to(trf.device)
        loss = F.cross_entropy(lin(trf[i]), try_[i])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    with torch.no_grad():
        return lin, lin(tef).softmax(-1)


def frozen_eval(data, arch, ckpt=None, robust=True):
    """k-NN and linear probe for every budget x seed on frozen features; linear-probe SEM-C."""
    dev = data.device
    bb = frozen_backbone(arch, ckpt).to(dev)
    tr_all, te = data.indices("train"), data.indices("test")
    ftr = extract(bb, data, tr_all)
    fte = extract(bb, data, te)
    pos = {int(i): k for k, i in enumerate(tr_all)}
    yte = data.y[torch.as_tensor(te, device=dev)]
    out = {"knn": {}, "linear": {}, "linear_probs": {}}
    lin_full = None
    for seed in C.SEEDS:
        for frac in C.BUDGETS:
            b = data.budget(seed, frac)
            sel = torch.as_tensor([pos[int(i)] for i in b], device=dev)
            yb = data.y[torch.as_tensor(b, device=dev)]
            key = f"{frac}_{seed}"
            out["knn"][key] = knn_acc(ftr[sel], yb, fte, yte, data.n_classes)
            lin, p = linear_probe(ftr[sel], yb, fte, data.n_classes, seed=seed)
            out["linear"][key] = float((p.argmax(1) == yte).float().mean())
            out["linear_probs"][key] = p.cpu().numpy().astype(np.float16)
            if frac == 0.10 and seed == 0:
                lin_full = lin
    if robust:
        yn = yte.cpu().numpy()

        def pf(tf):
            f = extract(bb, data, te, transform=tf)
            return lin_full(f).softmax(-1).detach().cpu().numpy()
        out["semc_linear10"] = semc(pf, data, te, yn)
    out["y"] = yte.cpu().numpy()
    del bb
    return out
