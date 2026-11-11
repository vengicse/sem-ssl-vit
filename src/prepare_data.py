"""Download the NFFA-EUROPE 100% SEM dataset, remove the instrument info bar, de-duplicate,
and build group-aware stratified 70/15/15 splits with nested 1/5/10/100 % label budgets.

Outputs in DATA_DIR:
  images.npy  (N, 256, 256) uint8 grayscale      labels.npy (N,) int64
  meta.json   classes, file names, info-bar rows, original sizes, duplicate groups
  splits.json train/val/test indices + budgets[seed][budget] (nested, stratified)
  raw/<Class>/*.jpg (kept for morphology / scale-bar reading)
"""
import hashlib
import os
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

import config as C

API = f"https://b2share.eudat.eu/api/records/{C.B2SHARE_RECORD}/files"


# ------------------------------------------------------------------ download
def md5(path, chunk=1 << 24):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def download(raw):
    import urllib.request
    raw.mkdir(parents=True, exist_ok=True)
    entries = json.load(urllib.request.urlopen(API))["entries"]
    jobs = []
    for e in entries:
        cls = e["key"].replace(".tar", "")
        if (raw / cls / ".done").exists():
            continue
        sumf = raw / f"{cls}.md5"
        # stream: the tar is never stored, its md5 is computed on the fly (disk-light)
        cmd = (f"set -o pipefail; curl -sfL --retry 8 --retry-delay 5 '{e['links']['content']}'"
               f" | tee >(md5sum | cut -d' ' -f1 > '{sumf.as_posix()}') | tar x --force-local -C '{raw.as_posix()}'")
        jobs.append((e, sumf, cmd))
    procs = [(e, sumf, subprocess.Popen([os.environ.get("BASH_EXE", "bash"), "-c", cmd])) for e, sumf, cmd in jobs]
    for e, sumf, p in procs:
        if p.wait() != 0:
            sys.exit(f"download/extract failed: {e['key']}")
        want = e["checksum"].split(":")[1]
        for _ in range(60):                       # process substitution may finish a moment later
            if sumf.exists() and sumf.read_text().strip():
                break
            time.sleep(1)
        got = sumf.read_text().strip()
        if got != want:
            sys.exit(f"md5 mismatch {e['key']}: {got} != {want}")
        (raw / e["key"].replace(".tar", "") / ".done").touch()
        print(f"  {e['key']}: ok (md5 {got})", flush=True)


# ------------------------------------------------------------------ info bar
def infobar_top(g):
    """Row index where the instrument data bar starts (image is kept above it).
    A bar row is nearly all saturated (white background or black background with text)."""
    H = g.shape[0]
    sat = ((g > 235) | (g < 12)).mean(1)
    bar = sat > 0.6
    lo = int(H * 0.6)
    for r in range(lo, H - 8):
        if bar[r:r + 16].mean() >= 0.75:
            return r
    return H


def load_clean(path):
    g = np.asarray(Image.open(path).convert("L"))
    top = infobar_top(g)
    return g[:top], top, g.shape


def square(g, size):
    h, w = g.shape
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    return np.asarray(Image.fromarray(g[y0:y0 + s, x0:x0 + s]).resize((size, size), Image.BICUBIC))


def dhash(g):
    a = np.asarray(Image.fromarray(g).resize((9, 8), Image.BILINEAR), dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return np.packbits(bits)  # 8 bytes


def process(path):
    g, top, shape = load_clean(path)
    return square(g, C.STORE_SIZE), dhash(g), top, shape


# ------------------------------------------------------------------ de-duplication
POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def dup_groups(hashes, thr):
    n = len(hashes)
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for s in range(0, n, 512):
        x = hashes[s:s + 512, None, :] ^ hashes[None, :, :]
        d = POP[x].sum(-1, dtype=np.int32)
        for a, b in zip(*np.nonzero(d <= thr)):
            a += s
            if a < b:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
    return np.array([find(i) for i in range(n)])


# ------------------------------------------------------------------ splits
def make_splits(labels, groups, seed=0):
    rng = np.random.default_rng(seed)
    n_cls = labels.max() + 1
    gids = np.unique(groups)
    rng.shuffle(gids)
    members = {g: np.nonzero(groups == g)[0] for g in gids}
    g_label = {g: np.bincount(labels[m], minlength=n_cls).argmax() for g, m in members.items()}
    split = {"train": [], "val": [], "test": []}
    for c in range(n_cls):
        gs = [g for g in gids if g_label[g] == c]
        tot = sum(len(members[g]) for g in gs)
        acc = 0
        for g in gs:
            f = acc / max(tot, 1)
            k = "train" if f < C.SPLIT[0] else ("val" if f < C.SPLIT[0] + C.SPLIT[1] else "test")
            split[k].extend(members[g].tolist())
            acc += len(members[g])
    for k in split:
        split[k] = sorted(int(i) for i in split[k])
    tr = np.array(split["train"])
    budgets = {}
    for s in C.SEEDS:
        r = np.random.default_rng(1000 + s)
        order = {c: r.permutation(tr[labels[tr] == c]) for c in range(n_cls)}
        budgets[str(s)] = {}
        for b in C.BUDGETS:
            keep = []
            for c in range(n_cls):
                k = max(1, int(np.ceil(b * len(order[c]))))
                keep.extend(order[c][:k].tolist())
            budgets[str(s)][str(b)] = sorted(int(i) for i in keep)
    split["budgets"] = budgets
    return split


# ------------------------------------------------------------------ smoke data
def make_smoke(out):
    """Tiny synthetic stand-in so every stage can be tested on a CPU in minutes."""
    from morph import render_particles
    rng = np.random.default_rng(0)
    n_per = 12
    X, y = [], []
    for c in range(len(C.CLASSES)):
        for i in range(n_per):
            img, _, _ = render_particles(C.STORE_SIZE, rng, mean_d=6 + 3 * c, density=0.2 + 0.03 * c)
            X.append(img)
            y.append(c)
    X, y = np.stack(X), np.array(y)
    groups = np.arange(len(y))
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "images.npy", X)
    np.save(out / "labels.npy", y)
    json.dump({"classes": C.CLASSES, "files": [f"smoke_{i}" for i in range(len(y))], "smoke": True},
              open(out / "meta.json", "w"))
    json.dump(make_splits(y, groups), open(out / "splits.json", "w"))
    print(f"smoke data: {X.shape}")


def main():
    out = C.DATA_DIR
    if C.SMOKE:
        return make_smoke(out)
    if (out / "splits.json").exists():
        print("data already prepared")
        return
    raw = out / "raw"
    download(raw)
    files, labels = [], []
    for ci, c in enumerate(C.CLASSES):
        fs = sorted(p for p in (raw / c).rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff"))
        files += fs
        labels += [ci] * len(fs)
        print(f"  {c}: {len(fs)} images", flush=True)
    labels = np.array(labels)
    with ProcessPoolExecutor() as ex:
        res = list(ex.map(process, files, chunksize=64))
    X = np.stack([r[0] for r in res])
    hashes = np.stack([r[1] for r in res])
    tops = [int(r[2]) for r in res]
    shapes = [list(map(int, r[3])) for r in res]
    groups = dup_groups(hashes, C.DEDUP_HAMMING)
    n_groups = len(np.unique(groups))
    cross = sum(len(np.unique(labels[groups == g])) > 1 for g in np.unique(groups))
    print(f"  {len(files)} images, {n_groups} duplicate groups, {cross} groups with conflicting labels")
    np.save(out / "images.npy", X)
    np.save(out / "labels.npy", labels)
    meta = {"classes": C.CLASSES, "files": [str(p.relative_to(raw)) for p in files],
            "infobar_top": tops, "orig_shape": shapes, "dup_group": groups.tolist(),
            "n_dup_groups": int(n_groups), "n_label_conflict_groups": int(cross),
            "infobar_found": float(np.mean([t < s[0] for t, s in zip(tops, shapes)]))}
    json.dump(meta, open(out / "meta.json", "w"))
    json.dump(make_splits(labels, groups), open(out / "splits.json", "w"))
    print(f"  info bar detected in {meta['infobar_found']:.1%} of images")
    # only the Particles originals are needed later (morphology + scale bar); free the rest
    import shutil
    for c in C.CLASSES:
        if c != "Particles":
            shutil.rmtree(raw / c, ignore_errors=True)


if __name__ == "__main__":
    main()
