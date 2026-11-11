"""Whole pipeline in resumable stages. Every stage skips work whose result file already exists.

    python run_all.py                                  # all stages
    python run_all.py --stages pretrain finetune       # selected stages
    SMOKE=1 python run_all.py                          # tiny CPU test on synthetic data
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import config as C  # noqa: E402

STAGES = ["data", "pretrain", "frozen", "finetune", "morph", "xai", "report"]
OURS, GENERIC = "dino_physics", "dino_generic"


def dev():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def jdump(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(obj, open(path, "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


def pretrain_jobs():
    jobs = [(OURS, C.PT_EPOCHS, True, None), (GENERIC, C.PT_EPOCHS, False, None),
            (f"abl_physics_{C.PT_ABL_EPOCHS}", C.PT_ABL_EPOCHS, True, None),
            (f"abl_generic_{C.PT_ABL_EPOCHS}", C.PT_ABL_EPOCHS, False, None)]
    from physics import ORDER
    for op in ORDER:
        jobs.append((f"abl_no_{op}_{C.PT_ABL_EPOCHS}", C.PT_ABL_EPOCHS, True, [o for o in ORDER if o != op]))
    return jobs


def ckpt(name):
    return C.CKPT_DIR / name / "teacher_backbone.pt"


# ------------------------------------------------------------------ stages
def stage_data():
    import prepare_data
    prepare_data.main()


def stage_pretrain(only=None):
    from data import SEMData
    from dino import pretrain
    data = SEMData(dev())
    (C.RESULTS_DIR / "pretrain").mkdir(parents=True, exist_ok=True)
    for name, ep, phys, ops in pretrain_jobs():
        if only and name not in only:
            continue
        print(f"[pretrain] {name}: {ep} epochs, physics={phys}, ops={ops}", flush=True)
        pretrain(data, name, ep, physics=phys, ops=ops)


def frozen_models():
    arch = C.BACKBONE
    m = {"DINO ViT-S/16 (IN)": (arch, None), "ViT-S/16 (sup. IN)": (C.BASELINES["ViT-S/16 (sup. IN)"], None)}
    for name, *_ in pretrain_jobs():
        if ckpt(name).exists():
            m[name] = (arch, ckpt(name))
    return m


def stage_frozen():
    from data import SEMData
    from train import frozen_eval
    data = SEMData(dev())
    (C.RESULTS_DIR / "frozen").mkdir(parents=True, exist_ok=True)
    for name, (arch, ck) in frozen_models().items():
        out = C.RESULTS_DIR / "frozen" / f"{safe(name)}.json"
        if out.exists():
            continue
        t0 = time.time()
        r = frozen_eval(data, arch, ck)
        probs = r.pop("linear_probs")
        np.savez_compressed(out.with_suffix(".npz"), y=r["y"], **{k: v for k, v in probs.items()})
        r.pop("y")
        r["time_s"] = time.time() - t0
        jdump(r, out)
        print(f"[frozen] {name}: knn@1% {np.mean([r['knn'][f'0.01_{s}'] for s in C.SEEDS]):.4f} "
              f"lin@1% {np.mean([r['linear'][f'0.01_{s}'] for s in C.SEEDS]):.4f}", flush=True)


def safe(s):
    return s.replace(":", "").replace("/", "-").replace(" ", "_").replace("(", "").replace(")", "").replace(".", "")


def finetune_models():
    m = {k: (v, None) for k, v in C.BASELINES.items()}
    if ckpt(GENERIC).exists():
        m["DINO-generic (SEM)"] = (C.BACKBONE, ckpt(GENERIC))
    if ckpt(OURS).exists():
        m["Ours: DINO-physics (SEM)"] = (C.BACKBONE, ckpt(OURS))
    return m


def stage_finetune(only=None):
    from data import SEMData
    from train import finetune
    data = SEMData(dev())
    (C.RESULTS_DIR / "finetune").mkdir(parents=True, exist_ok=True)
    keep_for_xai = {"Ours: DINO-physics (SEM)", "DINO ViT-S/16 (IN)", "DINO-generic (SEM)"}
    for seed in C.SEEDS:                         # seed-major order: a partial run still covers all models
        for frac in C.BUDGETS:
            for name, (arch, ck) in finetune_models().items():
                if only and name not in only:
                    continue
                out = C.RESULTS_DIR / "finetune" / f"{safe(name)}_{frac}_{seed}.json"
                if out.exists():
                    continue
                keep = name in keep_for_xai and frac == 1.0 and seed == 0
                r = finetune(data, arch, frac, seed, ck, robust=frac == 0.10, keep=keep)
                if keep:
                    (C.CKPT_DIR / "ft").mkdir(parents=True, exist_ok=True)
                    torch.save(r.pop("model").state_dict(), C.CKPT_DIR / "ft" / f"{safe(name)}.pt")
                np.savez_compressed(out.with_suffix(".npz"), probs=r.pop("probs"), y=r.pop("y"))
                jdump(r, out)
                p = np.load(out.with_suffix(".npz"))
                acc = float((p["probs"].argmax(1) == p["y"]).mean())
                print(f"[finetune] {name} frac={frac} seed={seed}: acc {acc:.4f} "
                      f"({r['train_time_s']:.0f}s, {r['epochs']} ep, n={r['n_train']})", flush=True)


def stage_morph():
    import timm
    from data import SEMData
    import morph
    out = C.RESULTS_DIR / "morph"
    out.mkdir(parents=True, exist_ok=True)
    d = dev()
    name = C.SMOKE_MODEL if C.SMOKE else C.BACKBONE

    def vit(ck=None):
        m = timm.create_model(name, pretrained=(ck is None and not C.SMOKE), num_classes=0,
                              dynamic_img_size=True)
        if ck is not None:
            m.load_state_dict(torch.load(ck, map_location="cpu"))
        return m.to(d).eval()

    bbs = {"scratch": None, "dino_in": vit()}
    if ckpt(OURS).exists():
        bbs["ours"] = vit(ckpt(OURS))
    if not (out / "synthetic.json").exists():
        res, nets = morph.run_synthetic(d, bbs)
        ex = res.pop("examples")
        np.savez_compressed(out / "synthetic_examples.npz", **{k: np.stack(v) for k, v in ex.items()})
        jdump(res, out / "synthetic.json")
        for k, n in nets.items():
            torch.save(n.state_dict(), out / f"segnet_{k}.pt")
    if not C.SMOKE and not (out / "real_particles.json").exists():
        real_particles(d, bbs, out)


def real_particles(d, bbs, out):
    """Apply the segmentation pipeline to NFFA Particles test images and convert to nm."""
    import morph
    from prepare_data import load_clean, square
    from scalebar import read_scale
    meta = json.load(open(C.DATA_DIR / "meta.json"))
    split = json.load(open(C.DATA_DIR / "splits.json"))
    pc = meta["classes"].index("Particles")
    labels = np.load(C.DATA_DIR / "labels.npy")
    idx = [i for i in split["test"] if labels[i] == pc]
    raw = C.DATA_DIR / "raw"
    nets = {}
    for k in ("ours", "dino_in"):
        if k in bbs and (out / f"segnet_{k}.pt").exists():
            n = morph.SegNet(bbs[k], bbs[k].num_features).to(d)
            n.load_state_dict(torch.load(out / f"segnet_{k}.pt", map_location=d))
            nets[k] = n
    recs, panel = [], []
    for j, i in enumerate(idx):
        f = raw / meta["files"][i]
        g, top, shape = load_clean(f)
        side = min(g.shape)
        img = square(g, morph.SEG_SIZE)
        sc = read_scale(f, top, gpu=d.type == "cuda")
        nm_px = sc["nm_per_px"] * side / morph.SEG_SIZE if sc and sc.get("nm_per_px") else None
        r = {"idx": int(i), "file": meta["files"][i], "scale": sc, "nm_per_px_256": nm_px}
        labs = {"classical": morph.classical(img)}
        for k, n in nets.items():
            labs[k] = morph.labs_from(morph.seg_predict(n, [img], d))[0]
        for k, lab in labs.items():
            ds = morph.diameters(lab, exclude_border=True)
            st = morph.shape_stats(lab)
            r[k] = {"n": int(len(ds)), "d_px": ds.round(2).tolist(),
                    "ar_median": float(np.median([s["ar"] for s in st])) if st else None,
                    "circ_median": float(np.median([s["circ"] for s in st])) if st else None}
        recs.append(r)
        if j < 6:
            panel.append({"img": img, **{k: v for k, v in labs.items()}})
    jdump(recs, out / "real_particles.json")
    np.savez_compressed(out / "real_examples.npz", **{k: np.stack([p[k] for p in panel]) for k in panel[0]})


def stage_xai():
    """CLS-token self-attention maps of the fine-tuned models on test images of every class."""
    import timm
    import torch.nn.functional as F
    from data import SEMData, center, to_input
    out = C.RESULTS_DIR / "xai"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "attention.npz").exists():
        return
    d = dev()
    data = SEMData(d)
    te = data.indices("test")
    y = data.y[torch.as_tensor(te, device=d)].cpu().numpy()
    pick = np.array([te[np.nonzero(y == c)[0][0]] for c in range(data.n_classes) if (y == c).any()])
    x, _ = data.raw(pick)
    x = center(x)
    maps = {"img": x[:, 0].cpu().numpy()}
    for name in ("Ours: DINO-physics (SEM)", "DINO ViT-S/16 (IN)", "DINO-generic (SEM)"):
        p = C.CKPT_DIR / "ft" / f"{safe(name)}.pt"
        if not p.exists():
            continue
        m = timm.create_model(C.SMOKE_MODEL if C.SMOKE else C.BACKBONE, num_classes=data.n_classes)
        m.load_state_dict(torch.load(p, map_location="cpu"))
        m.to(d).eval()
        att = {}

        def hook(mod, inp, outp):
            B, N, _ = inp[0].shape
            qkv = mod.qkv(inp[0]).reshape(B, N, 3, mod.num_heads, -1).permute(2, 0, 3, 1, 4)
            a = (qkv[0] @ qkv[1].transpose(-2, -1)) * mod.scale
            att["a"] = a.softmax(-1)
        h = m.blocks[-1].attn.register_forward_hook(hook)
        with torch.no_grad():
            m(to_input(x))
        h.remove()
        a = att["a"][:, :, 0, m.num_prefix_tokens:].mean(1)
        n = int(a.shape[-1] ** 0.5)
        a = F.interpolate(a.reshape(-1, 1, n, n), size=x.shape[-2:], mode="bilinear")[:, 0]
        maps[safe(name)] = a.cpu().numpy()
    np.savez_compressed(out / "attention.npz", idx=pick, **maps)
    # frozen test-set features for the embedding figure
    from dino import extract
    from train import frozen_backbone
    feats = {"y": y}
    for name, (arch, ck) in frozen_models().items():
        if name in ("DINO ViT-S/16 (IN)", OURS, GENERIC):
            feats[safe(name)] = extract(frozen_backbone(arch, ck).to(d), data, te).cpu().numpy().astype(np.float16)
    np.savez_compressed(out / "test_features.npz", **feats)


def stage_report():
    import report
    report.main()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="+", default=STAGES)
    ap.add_argument("--only", nargs="+", default=None, help="restrict pretrain/finetune to these names")
    a = ap.parse_args()
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"device {dev()}  smoke={C.SMOKE}  data={C.DATA_DIR}  results={C.RESULTS_DIR}", flush=True)
    for s in a.stages:
        t0 = time.time()
        print(f"=== stage {s}", flush=True)
        fn = globals()[f"stage_{s}"]
        fn(a.only) if s in ("pretrain", "finetune") else fn()
        print(f"=== stage {s} done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
