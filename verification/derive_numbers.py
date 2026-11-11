"""Derive every secondary number the NTREV manuscript cites, straight from the result files.

Usage: python derive_numbers.py <sem-ssl-vit/results> <out.json>
"""
import glob
import json
import os
import sys

import numpy as np
from scipy.stats import binomtest

R, OUT = sys.argv[1], sys.argv[2]
CLASSES = None
d = {}


def load_ft(name, frac, seed):
    j = os.path.join(R, "finetune", f"{name}_{frac}_{seed}")
    if not os.path.exists(j + ".json"):
        return None, None
    z = np.load(j + ".npz")
    return json.load(open(j + ".json")), z


def mcnemar(pa, pb, y):
    a, b = pa == y, pb == y
    n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
    p = binomtest(min(n01, n10), n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return n01, n10, p


models = ["ResNet-50", "Inception-v3", "EfficientNet-B0", "ConvNeXt-T", "ViT-S-16_sup_IN",
          "DINO_ViT-S-16_IN", "DINO-generic_SEM"]
ours = "Ours_DINO-physics_SEM"
# McNemar per matched (budget, seed)
mc = {}
for m in models:
    for frac in ["0.01", "0.05", "0.1", "1.0"]:
        for seed in [0, 1]:
            _, zo = load_ft(ours, frac, seed)
            _, zm = load_ft(m, frac, seed)
            if zo is None or zm is None:
                continue
            y = zo["y"]
            assert (y == zm["y"]).all()
            po, pm = zo["probs"].argmax(1), zm["probs"].argmax(1)
            n01, n10, p = mcnemar(po, pm, y)
            mc[f"{m}|{frac}|{seed}"] = {"ours_only": n01, "other_only": n10, "p": p,
                                       "acc_ours": float((po == y).mean() * 100), "acc_other": float((pm == y).mean() * 100)}
d["mcnemar"] = mc

# per-class recall + macro-F1 of ours
from sklearn.metrics import confusion_matrix, f1_score
for frac in ["0.01", "0.1", "1.0"]:
    _, z = load_ft(ours, frac, 0)
    y, p = z["y"], z["probs"].argmax(1)
    cm = confusion_matrix(y, p)
    rec = (cm.diagonal() / cm.sum(1) * 100).round(1).tolist()
    d[f"ours_recall_{frac}"] = rec
    d[f"ours_macroF1_{frac}"] = round(float(f1_score(y, p, average="macro") * 100), 2)
    d[f"ours_cm_{frac}"] = cm.tolist()

# SEM-C per severity, fine-tuned at 10 %
semc = {}
for m in models[:1] + models[4:] + [ours]:
    rows = []
    for seed in [0, 1]:
        j, _ = load_ft(m, "0.1", seed)
        if j and "semc" in j:
            rows.append(j["semc"])
    if rows:
        semc[m] = {k: round(float(np.mean([r[k] for r in rows]) * 100), 1) for k in rows[0]}
d["semc_ft10"] = semc

# frozen linear SEM-C mCA
fz = {}
for f in glob.glob(os.path.join(R, "frozen", "*.json")):
    j = json.load(open(f))
    if "semc_linear10" in j:
        v = j["semc_linear10"]
        fz[os.path.basename(f)[:-5]] = {"mCA": round(float(np.mean(list(v.values())) * 100), 2),
                                       "noise_5": round(v["noise_5"] * 100, 1),
                                       "psf_5": round(v["psf_5"] * 100, 1),
                                       "charging_5": round(v["charging_5"] * 100, 1),
                                       "noise_mean": round(float(np.mean([v[f'noise_{s}'] for s in range(1, 6)])) * 100, 1)}
d["frozen_semc"] = fz

# pretraining logs
for f in glob.glob(os.path.join(R, "pretrain", "*.json")):
    j = json.load(open(f))
    log = j["log"]
    d["pretrain_" + j["name"]] = {"epochs": j["epochs"], "first_loss": round(log[0]["loss"], 2),
                                  "last_loss": round(log[-1]["loss"], 2),
                                  "knn": {e["epoch"]: round(e["knn"] * 100, 2) for e in log if "knn" in e},
                                  "feat_std": [round(e["feat_std"], 3) for e in log if "feat_std" in e][-1:],
                                  "hours": round(sum(e["time_s"] for e in log) / 3600, 2)}

# training times
tt = {}
for f in glob.glob(os.path.join(R, "finetune", "*.json")):
    j = json.load(open(f))
    tt[os.path.basename(f)[:-5]] = round(j.get("train_time_s", 0) / 60, 2)
d["ft_minutes"] = tt
d["ft_total_hours"] = round(sum(tt.values()) / 60, 2)
d["pretrain_total_hours"] = round(sum(v["hours"] for k, v in d.items() if k.startswith("pretrain_") and isinstance(v, dict)), 2)

# morphology json keys
for f in ["synthetic.json", "real_particles.json"]:
    p = os.path.join(R, "morph", f)
    if os.path.exists(p):
        j = json.load(open(p))
        d["morph_" + f[:-5] + "_keys"] = list(j.keys())[:40] if isinstance(j, dict) else type(j).__name__
json.dump(d, open(OUT, "w"), indent=1, default=float)
print("ok", len(mc), "mcnemar pairs")
