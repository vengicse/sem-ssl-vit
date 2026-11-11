"""DINO self-distillation on unlabelled SEM images (labels of the training split are never read).

Student and teacher are ViT-S/16 initialised from ImageNet DINO weights; the teacher is an EMA
of the student (Eq. 8), its outputs are centred and sharpened (Eqs. 9-10), and the student
matches the teacher across all global/local view pairs (Eq. 7).
"""
import json
import math
import time

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

import config as C
from data import multicrop, to_input


class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=2048, bottleneck=256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, bottleneck))
        self.last = nn.utils.parametrizations.weight_norm(nn.Linear(bottleneck, out_dim, bias=False))
        self.last.parametrizations.weight.original0.data.fill_(1.0)
        self.last.parametrizations.weight.original0.requires_grad = False

    def forward(self, x):
        return self.last(F.normalize(self.mlp(x), dim=-1))


class Wrapper(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone, self.head = backbone, head

    def forward(self, crops):
        # group crops of equal resolution into one forward pass
        sizes = [c.shape[-1] for c in crops]
        out, i = [], 0
        while i < len(crops):
            j = i
            while j < len(crops) and sizes[j] == sizes[i]:
                j += 1
            out.append(self.backbone(to_input(torch.cat(crops[i:j]))))
            i = j
        return self.head(torch.cat(out))


def backbone(pretrained=True):
    name = C.SMOKE_MODEL if C.SMOKE else C.BACKBONE
    return timm.create_model(name, pretrained=pretrained and not C.SMOKE, num_classes=0,
                             dynamic_img_size=True, drop_path_rate=0.1)


class DINOLoss(nn.Module):
    def __init__(self, out_dim, n_crops, epochs):
        super().__init__()
        self.n_crops = n_crops
        self.register_buffer("center", torch.zeros(1, out_dim))
        w = min(30, epochs)
        self.tt = np.concatenate([np.linspace(C.PT_TEACHER_TEMP[0], C.PT_TEACHER_TEMP[1], w),
                                  np.full(max(0, epochs - w), C.PT_TEACHER_TEMP[1])])

    def forward(self, s_out, t_out, epoch):
        s = (s_out / C.PT_STUDENT_TEMP).chunk(self.n_crops)
        t = F.softmax((t_out - self.center) / self.tt[epoch], dim=-1).detach().chunk(2)
        loss, n = 0.0, 0
        for iq, q in enumerate(t):
            for v in range(self.n_crops):
                if v == iq:
                    continue
                loss = loss + torch.sum(-q * F.log_softmax(s[v], dim=-1), dim=-1).mean()
                n += 1
        with torch.no_grad():
            self.center.mul_(0.9).add_(t_out.mean(0, keepdim=True), alpha=0.1)
        return loss / n


def cos_sched(base, final, n, warm=0, warm_start=0.0):
    w = np.linspace(warm_start, base, warm) if warm > 0 else np.array([])
    it = np.arange(n - warm)
    return np.concatenate([w, final + 0.5 * (base - final) * (1 + np.cos(np.pi * it / max(1, len(it))))])


def amp_dtype():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


@torch.no_grad()
def extract(bb, data, idx, bs=256, transform=None):
    bb.eval()
    feats = []
    dt = amp_dtype()
    for s in range(0, len(idx), bs):
        x, _ = data.raw(idx[s:s + bs])
        x = x[:, :, 16:240, 16:240] if x.shape[-1] == C.STORE_SIZE else x
        if transform is not None:
            x = transform(x)
        with torch.autocast(data.device.type, dtype=dt, enabled=data.device.type == "cuda"):
            f = bb(to_input(x))
        feats.append(F.normalize(f.float(), dim=-1))
    return torch.cat(feats)


@torch.no_grad()
def knn_acc(train_f, train_y, test_f, test_y, n_cls, k=C.KNN_K, T=C.KNN_T):
    preds = []
    for s in range(0, len(test_f), 512):
        sim = test_f[s:s + 512] @ train_f.T
        v, i = sim.topk(min(k, train_f.shape[0]), dim=1)
        w = (v / T).exp()
        votes = torch.zeros(len(v), n_cls, device=v.device).scatter_add_(1, train_y[i], w)
        preds.append(votes.argmax(1))
    return (torch.cat(preds) == test_y).float().mean().item()


def pretrain(data, name, epochs, physics=True, ops=None, seed=0):
    """Returns path of the teacher backbone state_dict."""
    out = C.CKPT_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    final = out / "teacher_backbone.pt"
    if final.exists():
        print(f"  [{name}] exists, skipping")
        return final
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = data.device
    s_bb, t_bb = backbone(), backbone()
    dim = s_bb.num_features
    student = Wrapper(s_bb, DINOHead(dim, C.PT_OUT_DIM)).to(dev)
    teacher = Wrapper(t_bb, DINOHead(dim, C.PT_OUT_DIM)).to(dev)
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False
    n_crops = 2 + C.N_LOCAL
    crit = DINOLoss(C.PT_OUT_DIM, n_crops, epochs).to(dev)
    reg, noreg = [], []
    for n, p in student.named_parameters():
        if p.requires_grad:
            (noreg if (n.endswith(".bias") or p.ndim == 1) else reg).append(p)
    opt = torch.optim.AdamW([{"params": reg}, {"params": noreg, "weight_decay": 0.0}])
    idx = data.indices("train")            # labels are never accessed during pretraining
    bs = min(C.PT_BATCH, len(idx))
    ipe = max(1, len(idx) // bs)
    total = epochs * ipe
    lr_s = cos_sched(C.PT_LR * bs / 256, C.PT_MIN_LR, total, min(C.PT_WARMUP, epochs) * ipe)
    wd_s = cos_sched(C.PT_WD[0], C.PT_WD[1], total)
    mom_s = cos_sched(C.PT_TEACHER_MOM[0], C.PT_TEACHER_MOM[1], total)
    dt = amp_dtype()
    scaler = torch.amp.GradScaler(enabled=(dev.type == "cuda" and dt == torch.float16))
    start, log = 0, []
    ck = out / "last.pt"
    if ck.exists():
        st = torch.load(ck, map_location=dev, weights_only=False)
        student.load_state_dict(st["student"]); teacher.load_state_dict(st["teacher"])
        opt.load_state_dict(st["opt"]); crit.center.copy_(st["center"])
        start, log = st["epoch"], st["log"]
        print(f"  [{name}] resumed at epoch {start}")
    # monitoring set: kNN with the 10 % budget as memory bank, validation split as queries
    mem_idx, val_idx = data.budget(0, 0.10), data.indices("val")
    t0 = time.time()
    for ep in range(start, epochs):
        student.train()
        tot, nb = 0.0, 0
        for it, (x, _) in enumerate(data.batches(idx, bs, train=True, seed=seed * 1000 + ep,
                                                 aug=lambda z: z)):
            step = ep * ipe + it
            for i, g in enumerate(opt.param_groups):
                g["lr"] = lr_s[step]
                if i == 0:
                    g["weight_decay"] = wd_s[step]
            crops = multicrop(x, physics=physics, ops=ops)
            with torch.autocast(dev.type, dtype=dt, enabled=dev.type == "cuda"):
                t_out = teacher(crops[:2])
                s_out = student(crops)
            loss = crit(s_out.float(), t_out.float(), ep)
            if not math.isfinite(loss.item()):
                raise RuntimeError(f"loss is {loss.item()} at epoch {ep}")
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(student.parameters(), 3.0)
            if ep == 0:   # freeze last layer during the first epoch (DINO practice)
                for p in student.head.last.parameters():
                    p.grad = None
            scaler.step(opt)
            scaler.update()
            with torch.no_grad():
                m = mom_s[step]
                for ps, pt in zip(student.parameters(), teacher.parameters()):
                    pt.mul_(m).add_(ps.detach(), alpha=1 - m)
            tot += loss.item()
            nb += 1
        rec = {"epoch": ep + 1, "loss": tot / max(nb, 1), "time_s": time.time() - t0}
        if (ep + 1) % (1 if C.SMOKE else 10) == 0 or ep + 1 == epochs:
            mf = extract(teacher.backbone, data, mem_idx)
            vf = extract(teacher.backbone, data, val_idx)
            rec["knn_val_acc"] = knn_acc(mf, data.y[torch.as_tensor(mem_idx, device=dev)], vf,
                                         data.y[torch.as_tensor(val_idx, device=dev)], data.n_classes)
            rec["feat_std"] = vf.std(0).mean().item()   # collapse monitor
        log.append(rec)
        print(f"  [{name}] ep {ep + 1}/{epochs} " + " ".join(f"{k} {v:.4f}" for k, v in rec.items()
                                                          if k != "epoch"), flush=True)
        if (ep + 1) % 5 == 0 or ep + 1 == epochs:
            torch.save({"student": student.state_dict(), "teacher": teacher.state_dict(),
                        "opt": opt.state_dict(), "center": crit.center, "epoch": ep + 1, "log": log}, ck)
    torch.save(teacher.backbone.state_dict(), final)
    json.dump({"name": name, "epochs": epochs, "physics": physics, "ops": ops, "log": log,
               "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu"},
              open(C.RESULTS_DIR / "pretrain" / f"{name}.json", "w"), indent=1)
    ck.unlink(missing_ok=True)
    return final
