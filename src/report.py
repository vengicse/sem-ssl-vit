"""Turn raw results into paper-ready figures (PNG 300 dpi + PDF) and tables (CSV, Markdown, LaTeX),
plus results/RESULTS.md with every number quoted in the manuscript."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

import config as C  # noqa: E402
from metrics import summary, paired, mcnemar, holm, bootstrap_ci, macro_f1  # noqa: E402

RES = C.RESULTS_DIR
FIG, TAB = RES / "figures", RES / "tables"
OURS = "Ours: DINO-physics (SEM)"
GEN = "DINO-generic (SEM)"
MODELS = list(C.BASELINES) + [GEN, OURS]
COL = {"ResNet-50": "#8c8c8c", "Inception-v3": "#b07aa1", "EfficientNet-B0": "#9c755f", "ConvNeXt-T": "#76b7b2",
       "ViT-S/16 (sup. IN)": "#edc948", "DINO ViT-S/16 (IN)": "#4e79a7", GEN: "#f28e2b", OURS: "#e15759"}
BUD = C.BUDGETS
BLAB = {0.01: "1%", 0.05: "5%", 0.10: "10%", 1.00: "100%"}
plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 300, "savefig.bbox": "tight"})
NUM = {}   # every number for RESULTS.md


def J(p):
    return json.load(open(p))


def safe(s):
    return s.replace(":", "").replace("/", "-").replace(" ", "_").replace("(", "").replace(")", "").replace(".", "")


def savefig(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png")
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)


def write_table(df, name, caption, bold_max=None, bold_min=None):
    TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB / f"{name}.csv", index=False)
    md = df.copy().astype(str)
    for cols, fn in ((bold_max or [], np.nanargmax), (bold_min or [], np.nanargmin)):
        for col in cols:
            if col in md:
                vals = md[col].str.extract(r"(-?[\d.]+)")[0].astype(float).values
                if np.isfinite(vals).any():
                    i = int(fn(vals))
                    md.iloc[i, md.columns.get_loc(col)] = f"**{md.iloc[i, md.columns.get_loc(col)]}**"
    with open(TAB / f"{name}.md", "w", encoding="utf-8") as f:
        f.write(f"**{caption}**\n\n" + md.to_markdown(index=False) + "\n")
    with open(TAB / f"{name}.tex", "w", encoding="utf-8") as f:
        f.write(df.to_latex(index=False, caption=caption, label=f"tab:{name}", escape=True))


def ms(v, pct=True, d=2):
    v = np.asarray(v, float) * (100 if pct else 1)
    if len(v) == 1:                      # a single run has no spread to report
        return f"{v[0]:.{d}f}"
    return f"{v.mean():.{d}f} ± {v.std(ddof=1):.{d}f}"


# ------------------------------------------------------------------ loading
def ft_runs():
    """{(model, frac, seed): {"p": probs, "y": y, "meta": json}}"""
    out = {}
    for m in MODELS:
        for f in BUD:
            for s in C.SEEDS:
                p = RES / "finetune" / f"{safe(m)}_{f}_{s}.json"
                if p.exists():
                    z = np.load(p.with_suffix(".npz"))
                    out[(m, f, s)] = {"p": z["probs"].astype(np.float64), "y": z["y"], "meta": J(p)}
    return out


def frozen():
    out = {}
    for p in sorted((RES / "frozen").glob("*.json")):
        out[p.stem] = J(p)
    return out


# ------------------------------------------------------------------ tables
def have_data():
    return (C.DATA_DIR / "meta.json").exists() and (C.DATA_DIR / "labels.npy").exists()


def table_dataset_stats():
    """Same table from results/dataset_stats.json (class totals logged by the data stage on the pod and the
    saved test labels) when the 14 GB image archive is not present locally. Exact; no split is re-drawn."""
    st = J(RES / "dataset_stats.json")
    rows = []
    for name, n, t in zip(st["classes"], st["images_per_class"], st["test_per_class"]):
        rows.append({"Class": name.replace("_", " "), "Images": n, "Share (%)": round(100 * n / st["n_images"], 1),
                     "Train + val": n - t, "Test": t})
    df = pd.DataFrame(rows)
    df.loc[len(df)] = {"Class": "Total", "Images": st["n_images"], "Share (%)": 100.0,
                       "Train + val": st["n_images"] - st["n_test"], "Test": st["n_test"]}
    b = st["n_train_budget_seed0"]
    write_table(df, "t1_dataset", "NFFA-EUROPE 100% SEM dataset after info-bar removal: images per class and "
                f"group-aware stratified split (train {b['1.0']}, validation {st['n_val']}, test {st['n_test']} images). "
                f"Labelled training budgets (seed 0): 1% = {b['0.01']}, 5% = {b['0.05']}, 10% = {b['0.1']} and "
                f"100% = {b['1.0']} images.")
    NUM.update({"n_images": st["n_images"], "n_train": b["1.0"], "n_val": st["n_val"], "n_test": st["n_test"],
                "n_dup_groups": st["n_dup_groups"], "n_label_conflict_groups": st["n_label_conflict_groups"],
                "infobar_found_pct": st["infobar_found_pct"], "class_min": min(st["images_per_class"]),
                "class_max": max(st["images_per_class"])})
    return df


def table_dataset():
    if not have_data():
        return table_dataset_stats()
    meta = J(C.DATA_DIR / "meta.json")
    sp = J(C.DATA_DIR / "splits.json")
    y = np.load(C.DATA_DIR / "labels.npy")
    rows = []
    for c, name in enumerate(meta["classes"]):
        r = {"Class": name.replace("_", " "), "Images": int((y == c).sum())}
        for k in ("train", "val", "test"):
            r[k.capitalize()] = int((y[sp[k]] == c).sum())
        r["1% budget"] = int((y[sp["budgets"]["0"]["0.01"]] == c).sum())
        r["10% budget"] = int((y[sp["budgets"]["0"]["0.1"]] == c).sum())
        rows.append(r)
    df = pd.DataFrame(rows)
    tot = df.drop(columns="Class").sum()
    df.loc[len(df)] = {"Class": "Total", **tot.to_dict()}
    write_table(df, "t1_dataset", "NFFA-EUROPE 100% SEM dataset after info-bar removal and de-duplication: "
                "images per class, group-aware stratified split and the labelled budgets (seed 0).")
    NUM["n_images"] = int(len(y))
    NUM["n_train"], NUM["n_val"], NUM["n_test"] = (len(sp[k]) for k in ("train", "val", "test"))
    for k in ("n_dup_groups", "n_label_conflict_groups", "infobar_found"):
        if k in meta:
            NUM[k] = meta[k]
    return df


def table_physics():
    P = C.PHYS
    rows = [
        ("Drift / scan jitter", "T_δ", f"global shift 0–{P['drift']['shift_px'][1]:.0f} px, row random walk 0–{P['drift']['jitter_px'][1]:.0f} px",
         "stage drift, scan-coil instability"),
        ("Defocus / astigmatism", "h_(σx,σy,θ)", f"σ {P['psf']['sigma'][0]}–{P['psf']['sigma'][1]} px, anisotropy 1–{P['psf']['aniso'][1]:.0f}, random θ",
         "focus error, stigmator mis-set"),
        ("Charging", "C(x,y)", f"amplitude 0–{P['charging']['amp'][1]}, correlation length {P['charging']['sigma_frac'][0]}–{P['charging']['sigma_frac'][1]} × FOV, scan streaks",
         "insulating samples, beam-induced charge"),
        ("Detector gain", "g(·)", f"brightness ±{P['gain']['brightness'][1]}, contrast {P['gain']['contrast'][0]}–{P['gain']['contrast'][1]}, γ {P['gain']['gamma'][0]}–{P['gain']['gamma'][1]}",
         "brightness/contrast settings, detector drift"),
        ("Shot + read noise", "N_PG", f"dose {P['noise']['photons'][1]:.0f}→{P['noise']['photons'][0]:.0f} e⁻/px (log), read σ 0–{P['noise']['read'][1]}",
         "low dose / fast scan, amplifier noise"),
    ]
    df = pd.DataFrame(rows, columns=["Operator", "Symbol", "Range (severity s ∈ [0, 1])", "Physical origin"])
    write_table(df, "t2_physics", f"Physics-aware SEM augmentation operators, applied in imaging-chain order with "
                f"probability {C.P_PHYSICS} per view and 0.5 per operator.")


def table_main(R):
    rows = []
    for m in MODELS:
        r = {"Model": m}
        for f in BUD:
            acc = [summary(R[(m, f, s)]["y"], R[(m, f, s)]["p"])["acc"] for s in C.SEEDS if (m, f, s) in R]
            r[f"Acc {BLAB[f]}"] = (f"{100 * acc[0]:.2f}†" if len(acc) == 1 else ms(acc)) if acc else "–"
        f1 = [summary(R[(m, 0.01, s)]["y"], R[(m, 0.01, s)]["p"])["f1"] for s in C.SEEDS if (m, 0.01, s) in R]
        r["Macro-F1 1%"] = ms(f1) if f1 else "–"
        ec = [summary(R[(m, 0.10, s)]["y"], R[(m, 0.10, s)]["p"])["ece"] for s in C.SEEDS if (m, 0.10, s) in R]
        r["ECE 10%"] = ms(ec) if ec else "–"
        rows.append(r)
    df = pd.DataFrame(rows)
    cols = [c for c in df.columns if c.startswith("Acc") or c.startswith("Macro")]
    write_table(df, "t3_main", "Test accuracy (%) after end-to-end fine-tuning with 1, 5, 10 and 100% of the training "
                "labels (mean ± s.d. over the seeds that were run; † = single seed because of the compute budget; identical recipe "
                "for every model). Macro-F1 at 1% and expected calibration error (ECE, %) at 10%.", bold_max=cols, bold_min=["ECE 10%"])
    for m in MODELS:
        for f in BUD:
            a = [summary(R[(m, f, s)]["y"], R[(m, f, s)]["p"])["acc"] for s in C.SEEDS if (m, f, s) in R]
            if a:
                NUM[f"ft_acc_{safe(m)}_{f}"] = round(100 * float(np.mean(a)), 2)
    return df


def table_stats(R):
    """Ours vs every other model: paired over the 12 (budget, seed) runs + McNemar per budget."""
    rows, pv = [], []
    for m in MODELS:
        if m == OURS:
            continue
        a, b, mc = [], [], {}
        for f in BUD:
            ps = []
            for s in C.SEEDS:
                if (OURS, f, s) in R and (m, f, s) in R:
                    yo = R[(OURS, f, s)]["y"]
                    a.append((R[(OURS, f, s)]["p"].argmax(1) == yo).mean())
                    b.append((R[(m, f, s)]["p"].argmax(1) == yo).mean())
                    ps.append(mcnemar(yo, R[(OURS, f, s)]["p"].argmax(1), R[(m, f, s)]["p"].argmax(1))["p"])
            mc[f] = ps
        if len(a) < 2:
            continue
        st = paired(a, b)
        pv.append(st["t_p"])
        r = {"Ours vs": m, "Δ acc (pp)": f"{100 * st['mean_diff']:+.2f}", "Wins": f"{st['wins']}/{st['n']}",
             "Paired t p": st["t_p"], "Wilcoxon p": f"{st['wilcoxon_p']:.2g}", "d_z": f"{st['d_z']:.2f}"}
        for f in (0.01, 0.10):
            if mc.get(f):
                r[f"McNemar p<0.05 @{BLAB[f]}"] = f"{sum(p < 0.05 for p in mc[f])}/{len(mc[f])}"
        rows.append(r)
    if not rows:
        return None
    adj = holm(pv)
    for r, p in zip(rows, adj):
        r["Holm p"] = f"{p:.2g}"
        r["Paired t p"] = f"{r['Paired t p']:.2g}"
    df = pd.DataFrame(rows).fillna("–")
    write_table(df, "t4_stats", "Statistical comparison of the proposed model with each existing model over all "
                "matched (budget × seed) fine-tuning runs available for both models (count in the Wins column): mean accuracy difference, wins, paired t-test, Wilcoxon "
                "signed-rank test, Holm-adjusted p (t-test), effect size d_z, and the number of seeds where the exact "
                "McNemar test on test-set predictions gives p < 0.05.")
    for r in rows:
        NUM[f"stats_{safe(r['Ours vs'])}"] = {k: v for k, v in r.items() if k != "Ours vs"}
    return df


def table_frozen(Fz):
    order = [("DINO_ViT-S-16_IN", "DINO ViT-S/16 (IN, no SEM pretraining)"),
             ("ViT-S-16_sup_IN", "ViT-S/16 (supervised IN)"),
             ("dino_generic", "DINO-generic (SEM)"), ("dino_physics", "Ours: DINO-physics (SEM)")]
    rows = []
    for k, name in order:
        if k not in Fz:
            continue
        r = {"Backbone (frozen)": name}
        for proto in ("knn", "linear"):
            for f in BUD:
                v = [Fz[k][proto][f"{f}_{s}"] for s in C.SEEDS]
                r[f"{'k-NN' if proto == 'knn' else 'Linear'} {BLAB[f]}"] = ms(v)
                NUM[f"frozen_{proto}_{k}_{f}"] = round(100 * float(np.mean(v)), 2)
        rows.append(r)
    if not rows:
        return None
    df = pd.DataFrame(rows)
    write_table(df, "t5_frozen", "Frozen-feature evaluation: weighted k-NN (k = 20) and linear probe accuracy (%) on "
                "the test split, mean ± s.d. over the 3 label subsets.", bold_max=[c for c in df.columns if "%" in c])
    return df


def semc_mean(d, op=None):
    ks = [k for k in d if op is None or k.startswith(op + "_")]
    return float(np.mean([d[k] for k in ks]))


def table_robust(R, Fz):
    from physics import ORDER
    rows = []
    for m in MODELS:
        for f in (0.10, 1.00):
            runs = [R[(m, f, s)] for s in C.SEEDS if (m, f, s) in R and "semc" in R[(m, f, s)]["meta"]]
            if not runs:
                continue
            clean = [summary(r["y"], r["p"])["acc"] for r in runs]
            r = {"Model": m, "Labels": BLAB[f], "Clean": ms(clean)}
            for op in ORDER:
                r[op.capitalize()] = ms([semc_mean(x["meta"]["semc"], op) for x in runs])
            mca = [semc_mean(x["meta"]["semc"]) for x in runs]
            r["mCA"] = ms(mca)
            r["Rel. drop"] = ms([1 - a / c for a, c in zip(mca, clean)])
            NUM[f"mca_{safe(m)}_{f}"] = round(100 * float(np.mean(mca)), 2)
            rows.append(r)
    if not rows:
        return None
    df = pd.DataFrame(rows)
    write_table(df, "t6_robustness", "SEM-C robustness benchmark: accuracy (%) under each artefact averaged over "
                "severities 1–5 (severity 5 exceeds the training range), mean corruption accuracy (mCA) and relative "
                "drop from clean accuracy, for models fine-tuned with 10% of the labels (seeds as in Table 3).",
                bold_max=["mCA"], bold_min=["Rel. drop"])
    return df


def table_ablation(Fz):
    from physics import ORDER
    e = C.PT_ABL_EPOCHS
    names = [(f"abl_generic_{e}", "Generic augmentation only"), (f"abl_physics_{e}", "Full physics chain")] + \
            [(f"abl_no_{op}_{e}", f"Physics chain without {op}") for op in ORDER]
    rows = []
    for k, lab in names:
        if k not in Fz:
            continue
        d = Fz[k]
        r = {"Pretraining augmentation": lab,
             "k-NN 1%": ms([d["knn"][f"0.01_{s}"] for s in C.SEEDS]),
             "Linear 1%": ms([d["linear"][f"0.01_{s}"] for s in C.SEEDS]),
             "Linear 10%": ms([d["linear"][f"0.1_{s}"] for s in C.SEEDS])}
        if "semc_linear10" in d:
            r["Linear mCA 10%"] = f"{100 * semc_mean(d['semc_linear10']):.2f}"
            if k.startswith("abl_no_"):
                op = k[len("abl_no_"):].rsplit("_", 1)[0]
                r["Acc. on removed artefact"] = f"{100 * semc_mean(d['semc_linear10'], op):.2f}"
        rows.append(r)
        NUM[f"abl_{k}"] = {kk: v for kk, v in r.items() if kk != "Pretraining augmentation"}
    if not rows:
        return None
    df = pd.DataFrame(rows)
    write_table(df, "t7_ablation", f"Augmentation ablation ({e}-epoch DINO pretraining, frozen features): k-NN and "
                "linear-probe accuracy (%), linear-probe mean corruption accuracy on SEM-C, and accuracy on the "
                "artefact removed from pretraining (leave-one-operator-out).",
                bold_max=["k-NN 1%", "Linear 1%", "Linear 10%", "Linear mCA 10%"])
    return df


def table_morph():
    p = RES / "morph" / "synthetic.json"
    if not p.exists():
        return None
    S = J(p)
    sev = np.array(S["sev"])
    rows = []

    def add(name, recs_list):
        mets = ["dice", "f1_50", "w1_px", "med_rel_err", "count_rel_err"]
        r = {"Method": name}
        for k in mets:
            v = [np.mean([x[k] for x in recs]) for recs in recs_list]
            r[{"dice": "Dice", "f1_50": "Instance F1@0.5", "w1_px": "PSD W1 (px)", "med_rel_err": "Median-d error",
               "count_rel_err": "Count error"}[k]] = ms(v, pct=k in ("dice", "f1_50", "med_rel_err", "count_rel_err"))
        hi = sev >= 0.75
        r["F1 (s ≥ 0.75)"] = ms([np.mean([x["f1_50"] for x, h in zip(recs, hi) if h]) for recs in recs_list])
        rows.append(r)
    add("Otsu + watershed (classical)", [S["classical"]])
    nb = max(C.SEG_LABEL_BUDGETS)
    for k, name in (("scratch", "CNN decoder, no pretraining"), ("dino_in", "Decoder on DINO ViT-S/16 (IN)"),
                    ("ours", "Decoder on ours (DINO-physics SEM)")):
        runs = [S[f"{k}_{nb}_{s}"] for s in C.SEEDS if f"{k}_{nb}_{s}" in S]
        if runs:
            add(f"{name}, {nb} labels", runs)
    df = pd.DataFrame(rows)
    write_table(df, "t8_morph_synth", f"Particle segmentation and size-distribution accuracy on {len(sev)} synthetic "
                "SEM test images with exact ground truth (mean ± s.d. over training seeds). Dice, F1, median-diameter "
                "and count errors in %; W1 = Wasserstein distance between predicted and true diameter distributions.",
                bold_max=["Dice", "Instance F1@0.5", "F1 (s ≥ 0.75)"], bold_min=["PSD W1 (px)", "Median-d error", "Count error"])
    # label-budget table
    rows2 = []
    for k in ("scratch", "dino_in", "ours"):
        r = {"Features": k}
        for b in C.SEG_LABEL_BUDGETS:
            runs = [S[f"{k}_{b}_{s}"] for s in C.SEEDS if f"{k}_{b}_{s}" in S]
            if runs:
                r[f"F1 @ {b} masks"] = ms([np.mean([x["f1_50"] for x in rr]) for rr in runs])
        rows2.append(r)
    write_table(pd.DataFrame(rows2), "t9_morph_budget", "Instance F1@0.5 (%) versus the number of labelled synthetic "
                "masks used to train the segmentation decoder.")
    for r in rows:
        NUM[f"morph_{safe(r['Method'])}"] = {k: v for k, v in r.items() if k != "Method"}
    return df


def table_real():
    p = RES / "morph" / "real_particles.json"
    if not p.exists():
        return None
    R = J(p)
    n = len(R)
    sc = [r for r in R if r["nm_per_px_256"]]
    NUM["real_n"] = n
    NUM["real_scale_read"] = len(sc)
    rows = []
    for k, name in (("classical", "Otsu + watershed"), ("dino_in", "Decoder on DINO (IN)"), ("ours", "Decoder on ours")):
        if k not in R[0]:
            continue
        med = [np.median(np.array(r[k]["d_px"]) * r["nm_per_px_256"]) for r in sc if r[k]["n"] >= 5]
        cnt = [r[k]["n"] for r in R]
        rows.append({"Method": name, "Images with ≥5 particles": f"{sum(r[k]['n'] >= 5 for r in R)}/{n}",
                     "Particles / image (median)": f"{np.median(cnt):.0f}",
                     "Median diameter (nm), median over images": f"{np.median(med):.0f}" if med else "–",
                     "Circularity (median)": f"{np.nanmedian([r[k]['circ_median'] or np.nan for r in R]):.2f}"})
    # agreement ours vs classical (per-image median diameter, nm)
    if "ours" in R[0]:
        pairs = [(np.median(np.array(r["ours"]["d_px"])) * r["nm_per_px_256"],
                  np.median(np.array(r["classical"]["d_px"])) * r["nm_per_px_256"])
                 for r in sc if r["ours"]["n"] >= 5 and r["classical"]["n"] >= 5]
        if len(pairs) > 2:
            a, b = np.array(pairs).T
            la, lb = np.log10(a), np.log10(b)
            ccc = 2 * np.cov(la, lb)[0, 1] / (la.var() + lb.var() + (la.mean() - lb.mean()) ** 2)
            NUM["real_agree_n"] = len(pairs)
            NUM["real_ccc_log"] = round(float(ccc), 3)
            NUM["real_ratio_median"] = round(float(np.median(a / b)), 3)
    df = pd.DataFrame(rows)
    write_table(df, "t10_morph_real", f"Morphology extraction on the {n} real NFFA 'Particles' test images; physical "
                f"scale read automatically from the info bar for {len(sc)} images.")
    return df


def table_efficiency(R):
    rows = []
    pt = {p.stem: J(p) for p in (RES / "pretrain").glob("*.json")}
    for m in MODELS:
        runs = [R[(m, 0.05, s)]["meta"] for s in C.SEEDS if (m, 0.05, s) in R]   # 5 % budget: run for every model
        if not runs:
            continue
        r = {"Model": m, "Params (M)": f"{runs[0]['params_M']:.1f}",
             "Fine-tune 5% (min)": f"{np.mean([x['train_time_s'] for x in runs]) / 60:.1f}",
             "Inference (ms/img)": f"{np.mean([x['infer_ms_per_img'] for x in runs]):.2f}"}
        key = "dino_physics" if m == OURS else ("dino_generic" if m == GEN else None)
        r["SSL pretraining (GPU-h)"] = f"{pt[key]['log'][-1]['time_s'] / 3600:.2f}" if key in pt else "0 (public weights)"
        rows.append(r)
    if rows:
        gpu = next(iter(pt.values()))["gpu"] if pt else "?"
        write_table(pd.DataFrame(rows), "t11_efficiency", f"Computational cost on one {gpu} (batch inference, mixed "
                    "precision).")
        NUM["gpu"] = gpu
        NUM["pretrain_hours"] = {k: round(v["log"][-1]["time_s"] / 3600, 2) for k, v in pt.items()}


# ------------------------------------------------------------------ figures
def fig_framework():
    fig, ax = plt.subplots(figsize=(13, 5.4))
    ax.set_xlim(0, 13); ax.set_ylim(0, 5.4); ax.axis("off")

    def box(x, y, w, h, t, c, fs=8.5):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12", fc=c, ec="#333", lw=1))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs)

    def arr(x0, y0, x1, y1):
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="->", lw=1.3, color="#333"))
    box(0.1, 3.9, 2.1, 1.2, "NFFA-EUROPE SEM\n21k images, 10 classes\ninfo bar removed,\nde-duplicated", "#e8eef7")
    box(0.1, 2.2, 2.1, 1.2, "Unlabelled pool\n(training split,\nlabels hidden)", "#e8eef7")
    box(0.1, 0.5, 2.1, 1.2, "Scale bar OCR\n→ nm per pixel", "#e8eef7")
    box(2.8, 2.0, 3.0, 3.1, "Physics-aware views\n\nT_drift → h_PSF → + C_charge\n→ g(gain) → N_Poisson-Gauss\n\n"
        "+ multi-crop (2 global, 4 local)", "#fde9d9")
    box(6.4, 3.3, 2.7, 1.8, "DINO self-distillation\nViT-S/16 student\nEMA teacher\n(init. ImageNet-DINO)", "#fbe5e5")
    box(6.4, 0.5, 2.7, 2.2, "Morphology head\nfrozen patch tokens\n+ CNN stem → 3-class\nmask → watershed\n(trained on synthetic SEM)",
        "#e6f2e6")
    box(9.7, 3.9, 3.1, 1.2, "Label-efficient classification\nfine-tune with 1–10% labels", "#f3e6f7")
    box(9.7, 2.3, 3.1, 1.2, "Robustness (SEM-C)\n5 artefacts × 5 severities", "#f3e6f7")
    box(9.7, 0.5, 3.1, 1.4, "Quantitative morphology\nPSD (nm), aspect ratio,\ncircularity", "#f3e6f7")
    arr(2.2, 2.8, 2.8, 3.2); arr(5.8, 4.2, 6.4, 4.2); arr(9.1, 4.4, 9.7, 4.5); arr(9.1, 3.9, 9.7, 2.9)
    arr(7.75, 3.3, 7.75, 2.7); arr(9.1, 1.4, 9.7, 1.2); arr(2.2, 1.1, 9.7, 0.9); arr(2.2, 4.5, 2.8, 4.2)
    savefig(fig, "fig1_framework")


def fig_dataset_stats():
    st = J(RES / "dataset_stats.json")
    z = np.load(RES / "xai" / "attention.npz")
    cls = st["classes"]
    fig, axs = plt.subplots(2, 5, figsize=(10, 4.4))
    for c, ax in enumerate(axs.ravel()):
        ax.imshow(z["img"][c], cmap="gray"); ax.axis("off")
        ax.set_title(cls[c].replace("_", " ").replace("devices and ", "& "), fontsize=8)
    savefig(fig, "fig2_samples")
    n, t = np.array(st["images_per_class"]), np.array(st["test_per_class"])
    o = np.argsort(n)[::-1]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    lab = [cls[i].replace("_", " ").replace("devices and ", "& ") for i in o]
    ax.bar(lab, (n - t)[o], color="#4e79a7", label="train + validation")
    ax.bar(lab, t[o], bottom=(n - t)[o], color="#f28e2b", label="test")
    for i, v in enumerate(n[o]):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Images"); ax.legend(frameon=False); plt.xticks(rotation=35, ha="right")
    savefig(fig, "fig3_class_distribution")


def fig_dataset():
    if not have_data():
        return fig_dataset_stats()
    X = np.load(C.DATA_DIR / "images.npy", mmap_mode="r")
    y = np.load(C.DATA_DIR / "labels.npy")
    cls = J(C.DATA_DIR / "meta.json")["classes"]
    rng = np.random.default_rng(0)
    fig, axs = plt.subplots(3, len(cls), figsize=(2 * len(cls), 6.2))
    for c in range(len(cls)):
        idx = rng.choice(np.nonzero(y == c)[0], 3, replace=len(np.nonzero(y == c)[0]) < 3)
        for r in range(3):
            axs[r, c].imshow(X[idx[r]], cmap="gray"); axs[r, c].axis("off")
        axs[0, c].set_title(cls[c].replace("_", " ").replace("devices and ", "& "), fontsize=8)
    savefig(fig, "fig2_samples")
    cnt = np.bincount(y, minlength=len(cls))
    fig, ax = plt.subplots(figsize=(8, 3.4))
    o = np.argsort(cnt)[::-1]
    ax.bar([cls[i].replace("_", " ") for i in o], cnt[o], color="#4e79a7")
    for i, v in enumerate(cnt[o]):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Images"); plt.xticks(rotation=35, ha="right")
    savefig(fig, "fig3_class_distribution")


def fig_physics():
    import torch
    from physics import OPS, ORDER, physics_chain
    torch.manual_seed(3)
    if have_data():
        X = np.load(C.DATA_DIR / "images.npy", mmap_mode="r")
        x = torch.from_numpy(np.array(X[len(X) // 3])).float().div(255).view(1, 1, *X.shape[1:])
    else:   # a real NFFA test image (Particles class) kept with the attention results
        im = np.load(RES / "xai" / "attention.npz")["img"][5].astype(np.float32)
        im = (im - im.min()) / (im.max() - im.min() + 1e-8)
        x = torch.from_numpy(im).view(1, 1, *im.shape)
    sev = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]
    fig, axs = plt.subplots(len(ORDER) + 1, len(sev), figsize=(1.9 * len(sev), 1.9 * (len(ORDER) + 1)))
    for i, o in enumerate(ORDER):
        for j, s in enumerate(sev):
            im = x if s == 0 else OPS[o](x, torch.tensor([s])).clamp(0, 1)
            axs[i, j].imshow(im[0, 0], cmap="gray", vmin=0, vmax=1); axs[i, j].set_xticks([]); axs[i, j].set_yticks([])
            if i == 0:
                axs[i, j].set_title("clean" if s == 0 else f"s = {s}" + (" (test only)" if s > 1 else ""), fontsize=8)
        axs[i, 0].set_ylabel(o, fontsize=9)
    for j in range(len(sev)):
        axs[-1, j].imshow(physics_chain(x, p_chain=1.0)[0, 0], cmap="gray", vmin=0, vmax=1)
        axs[-1, j].set_xticks([]); axs[-1, j].set_yticks([])
    axs[-1, 0].set_ylabel("random chain", fontsize=9)
    savefig(fig, "fig4_physics_augmentation")


def fig_pretrain():
    pt = {p.stem: J(p) for p in (RES / "pretrain").glob("*.json")}
    if "dino_physics" not in pt:
        return
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
    for k, lab, c in (("dino_physics", "DINO-physics (ours)", COL[OURS]), ("dino_generic", "DINO-generic", COL[GEN])):
        if k not in pt:
            continue
        L = pt[k]["log"]
        axs[0].plot([r["epoch"] for r in L], [r["loss"] for r in L], c=c, label=lab)
        kk = [r for r in L if "knn_val_acc" in r]
        axs[1].plot([r["epoch"] for r in kk], [100 * r["knn_val_acc"] for r in kk], "o-", c=c, label=lab)
    Fz = frozen()
    if "DINO_ViT-S-16_IN" in Fz:
        axs[1].axhline(100 * np.mean([Fz["DINO_ViT-S-16_IN"]["knn"][f"0.1_{s}"] for s in C.SEEDS]), ls="--", c=COL["DINO ViT-S/16 (IN)"],
                       label="ImageNet DINO (start), test")
    axs[0].set_xlabel("Epoch"); axs[0].set_ylabel("DINO loss"); axs[0].legend()
    axs[1].set_xlabel("Epoch"); axs[1].set_ylabel("k-NN accuracy (%), 10% bank"); axs[1].legend(fontsize=8)
    savefig(fig, "fig5_pretraining")


def fig_label_efficiency(R):
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    x = np.array(BUD) * 100
    for m in MODELS:
        mu, sd = [], []
        for f in BUD:
            a = [100 * (R[(m, f, s)]["p"].argmax(1) == R[(m, f, s)]["y"]).mean() for s in C.SEEDS if (m, f, s) in R]
            mu.append(np.mean(a) if a else np.nan); sd.append(np.std(a) if a else np.nan)
        ax.errorbar(x, mu, yerr=sd, marker="o", capsize=3, lw=2.6 if m == OURS else 1.4, c=COL[m], label=m,
                    zorder=5 if m == OURS else 2)
    ax.set_xscale("log"); ax.set_xticks(x); ax.set_xticklabels([BLAB[f] for f in BUD])
    ax.set_xlabel("Labelled training images used"); ax.set_ylabel("Test accuracy (%)")
    ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.3)
    savefig(fig, "fig6_label_efficiency")


def fig_frozen(Fz):
    keys = [("DINO_ViT-S-16_IN", "DINO (IN)", COL["DINO ViT-S/16 (IN)"]), ("ViT-S-16_sup_IN", "ViT-S sup. (IN)", COL["ViT-S/16 (sup. IN)"]),
            ("dino_generic", "DINO-generic (SEM)", COL[GEN]), ("dino_physics", "Ours", COL[OURS])]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    x = np.array(BUD) * 100
    for ax, proto, t in ((axs[0], "knn", "k-NN"), (axs[1], "linear", "Linear probe")):
        for k, lab, c in keys:
            if k in Fz:
                mu = [100 * np.mean([Fz[k][proto][f"{f}_{s}"] for s in C.SEEDS]) for f in BUD]
                ax.plot(x, mu, "o-", c=c, label=lab, lw=2.4 if k == "dino_physics" else 1.4)
        ax.set_xscale("log"); ax.set_xticks(x); ax.set_xticklabels([BLAB[f] for f in BUD]); ax.set_title(t)
        ax.set_xlabel("Labels"); ax.grid(alpha=0.3)
    axs[0].set_ylabel("Test accuracy (%)"); axs[1].legend(fontsize=8)
    savefig(fig, "fig7_frozen")


def fig_confusion(R):
    cls = [c.replace("_", " ").replace("MEMS devices and electrodes", "MEMS") for c in C.CLASSES]
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    for ax, f in zip(axs, (0.01, 1.0)):
        if (OURS, f, 0) not in R:
            continue
        r = R[(OURS, f, 0)]
        n = len(cls)
        cm = np.zeros((n, n))
        for t, p in zip(r["y"], r["p"].argmax(1)):
            cm[t, p] += 1
        cm = cm / cm.sum(1, keepdims=True).clip(1)
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        for i in range(n):
            for j in range(n):
                if cm[i, j] >= 0.01:
                    ax.text(j, i, f"{100 * cm[i, j]:.0f}", ha="center", va="center", fontsize=7,
                            color="white" if cm[i, j] > 0.6 else "black")
        ax.set_xticks(range(n)); ax.set_xticklabels(cls, rotation=60, ha="right", fontsize=7)
        ax.set_yticks(range(n)); ax.set_yticklabels(cls, fontsize=7)
        ax.set_title(f"Ours, {BLAB[f]} labels (seed 0)"); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    savefig(fig, "fig8_confusion")


def fig_robust(R):
    from physics import ORDER
    sel = [m for m in (OURS, GEN, "DINO ViT-S/16 (IN)", "ViT-S/16 (sup. IN)", "ConvNeXt-T", "ResNet-50")]
    fig, axs = plt.subplots(1, len(ORDER), figsize=(3.1 * len(ORDER), 3.2), sharey=True)
    for ax, op in zip(axs, ORDER):
        for m in sel:
            runs = [R[(m, 0.10, s)]["meta"]["semc"] for s in C.SEEDS if (m, 0.10, s) in R and "semc" in R[(m, 0.10, s)]["meta"]]
            if not runs:
                continue
            clean = np.mean([100 * (R[(m, 0.10, s)]["p"].argmax(1) == R[(m, 0.10, s)]["y"]).mean() for s in C.SEEDS if (m, 0.10, s) in R])
            mu = [clean] + [100 * np.mean([r[f"{op}_{v}"] for r in runs]) for v in C.SEMC_SEVERITIES]
            ax.plot([0] + C.SEMC_SEVERITIES, mu, "o-", ms=3, c=COL[m], lw=2.4 if m == OURS else 1.2, label=m)
        ax.axvspan(4.5, 5.5, color="#ddd", alpha=0.5)
        ax.set_title(op); ax.set_xlabel("Severity"); ax.grid(alpha=0.3)
    axs[0].set_ylabel("Test accuracy (%), 10% labels"); axs[-1].legend(fontsize=7)
    savefig(fig, "fig9_robustness")


def fig_embed():
    p = RES / "xai" / "test_features.npz"
    if not p.exists():
        return
    from sklearn.manifold import TSNE
    z = np.load(p)
    y = z["y"]
    cls = C.CLASSES
    keys = [k for k in ("DINO_ViT-S-16_IN", "dino_generic", "dino_physics") if k in z]
    fig, axs = plt.subplots(1, len(keys), figsize=(6 * len(keys), 5))
    axs = np.atleast_1d(axs)
    from sklearn.metrics import silhouette_score
    for ax, k in zip(axs, keys):
        f = z[k].astype(np.float32)
        e = TSNE(2, init="pca", perplexity=min(30, len(f) // 4), random_state=0).fit_transform(f)
        sil = silhouette_score(f, y, metric="cosine") if len(np.unique(y)) > 1 else np.nan
        NUM[f"silhouette_{k}"] = round(float(sil), 3)
        for c in np.unique(y):
            ax.scatter(*e[y == c].T, s=3, label=cls[c].replace("_", " "), alpha=0.7)
        ax.set_title({"DINO_ViT-S-16_IN": "ImageNet DINO", "dino_generic": "DINO-generic (SEM)",
                      "dino_physics": "Ours (DINO-physics SEM)"}[k] + f" — silhouette {sil:.3f}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    axs[-1].legend(markerscale=4, fontsize=7, loc="center left", bbox_to_anchor=(1, 0.5))
    savefig(fig, "fig10_tsne")


def fig_attention():
    p = RES / "xai" / "attention.npz"
    if not p.exists():
        return
    z = np.load(p)
    keys = [k for k in ("DINO_ViT-S-16_IN", "DINO-generic_SEM", "Ours_DINO-physics_SEM") if k in z]
    n = len(z["img"])
    fig, axs = plt.subplots(1 + len(keys), n, figsize=(1.7 * n, 1.8 * (1 + len(keys))))
    for j in range(n):
        axs[0, j].imshow(z["img"][j], cmap="gray")
        for i, k in enumerate(keys):
            axs[i + 1, j].imshow(z["img"][j], cmap="gray")
            a = z[k][j]
            axs[i + 1, j].imshow((a - a.min()) / (np.ptp(a) + 1e-9), cmap="inferno", alpha=0.55)
    for ax in axs.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    axs[0, 0].set_ylabel("image", fontsize=8)
    for i, k in enumerate(keys):
        axs[i + 1, 0].set_ylabel({"DINO_ViT-S-16_IN": "DINO (IN)", "DINO-generic_SEM": "DINO-generic",
                                  "Ours_DINO-physics_SEM": "Ours"}[k], fontsize=8)
    for j in range(len(z["idx"])):      # stage_xai picks the first test image of each class, in class order
        axs[0, j].set_title(C.CLASSES[j].replace("_", " ")[:14], fontsize=7)
    savefig(fig, "fig11_attention")


def fig_ablation(Fz):
    from physics import ORDER
    e = C.PT_ABL_EPOCHS
    ks = [(f"abl_generic_{e}", "generic only"), (f"abl_physics_{e}", "full physics")] + \
         [(f"abl_no_{o}_{e}", f"− {o}") for o in ORDER]
    ks = [(k, l) for k, l in ks if k in Fz]
    if not ks:
        return
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
    v1 = [100 * np.mean([Fz[k]["linear"][f"0.01_{s}"] for s in C.SEEDS]) for k, _ in ks]
    s1 = [100 * np.std([Fz[k]["linear"][f"0.01_{s}"] for s in C.SEEDS]) for k, _ in ks]
    cols = ["#999" if "generic" in k else (COL[OURS] if "abl_physics" in k else "#f4a582") for k, _ in ks]
    axs[0].bar(range(len(ks)), v1, yerr=s1, color=cols, capsize=3)
    axs[0].set_ylim(min(v1) - 3, max(v1) + 2)
    axs[0].set_ylabel("Linear probe acc. (%), 1% labels")
    v2 = [100 * semc_mean(Fz[k]["semc_linear10"]) if "semc_linear10" in Fz[k] else np.nan for k, _ in ks]
    axs[1].bar(range(len(ks)), v2, color=cols)
    axs[1].set_ylim(np.nanmin(v2) - 3, np.nanmax(v2) + 2)
    axs[1].set_ylabel("SEM-C mCA (%), linear 10%")
    for ax in axs:
        ax.set_xticks(range(len(ks))); ax.set_xticklabels([l for _, l in ks], rotation=30, ha="right")
    savefig(fig, "fig12_ablation")


def fig_morph():
    p = RES / "morph" / "synthetic.json"
    if not p.exists():
        return
    S = J(p)
    sev = np.array(S["sev"])
    nb = max(C.SEG_LABEL_BUDGETS)
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    series = [("classical", [S["classical"]], "Otsu + watershed", "#7f7f7f")]
    for k, lab, c in (("scratch", "CNN, no pretraining", "#9c755f"), ("dino_in", "DINO (IN) features", COL["DINO ViT-S/16 (IN)"]),
                      ("ours", "Ours", COL[OURS])):
        runs = [S[f"{k}_{nb}_{s}"] for s in C.SEEDS if f"{k}_{nb}_{s}" in S]
        if runs:
            series.append((k, runs, lab, c))
    for k, runs, lab, c in series:
        for ax, met, sc in ((axs[0], "f1_50", 100), (axs[1], "w1_px", 1)):
            us = np.unique(sev)
            mu = [np.mean([np.mean([r[met] for r, s in zip(rr, sev) if s == u]) for rr in runs]) * sc for u in us]
            ax.plot(us, mu, "o-", c=c, label=lab, lw=2.4 if k == "ours" else 1.3)
    axs[0].set_ylabel("Instance F1@0.5 (%)"); axs[1].set_ylabel("PSD Wasserstein-1 (px)")
    for ax in axs:
        ax.set_xlabel("Artefact severity s"); ax.axvspan(1.0, 1.3, color="#ddd", alpha=0.5); ax.grid(alpha=0.3)
    axs[0].legend(fontsize=8)
    savefig(fig, "fig13_morph_synthetic")
    ex = RES / "morph" / "synthetic_examples.npz"
    if ex.exists():
        from skimage.color import label2rgb
        z = np.load(ex)
        keys = [k for k in ("img", "gt", "classical", "dino_in", "ours") if k in z]
        n = len(z["img"])
        fig, axs = plt.subplots(n, len(keys), figsize=(2.2 * len(keys), 2.2 * n))
        for i in range(n):
            for j, k in enumerate(keys):
                im = z["img"][i] if k == "img" else label2rgb(z[k][i], image=z["img"][i] / 255, bg_label=0, alpha=0.4)
                axs[i, j].imshow(im, cmap="gray"); axs[i, j].axis("off")
                if i == 0:
                    axs[i, j].set_title({"img": "input", "gt": "ground truth", "classical": "Otsu+watershed",
                                         "dino_in": "DINO (IN)", "ours": "ours"}[k], fontsize=9)
        savefig(fig, "fig14_morph_examples")


def fig_real():
    p = RES / "morph" / "real_particles.json"
    if not p.exists():
        return
    R = J(p)
    from scipy import stats
    sc = [r for r in R if r["nm_per_px_256"] and r.get("ours", {}).get("n", 0) >= 20]
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.8))
    for ax, r in zip(axs[:2], sorted(sc, key=lambda r: -r["ours"]["n"])[:2]):
        d = np.array(r["ours"]["d_px"]) * r["nm_per_px_256"]
        dc = np.array(r["classical"]["d_px"]) * r["nm_per_px_256"]
        bins = np.logspace(np.log10(min(d.min(), dc.min())), np.log10(max(d.max(), dc.max())), 25)
        ax.hist(dc, bins, alpha=0.45, color="#7f7f7f", label=f"Otsu+watershed (n={len(dc)})")
        ax.hist(d, bins, alpha=0.6, color=COL[OURS], label=f"ours (n={len(d)})")
        s, loc, scale = stats.lognorm.fit(d, floc=0)
        xx = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), 200)
        ax.plot(xx, stats.lognorm.pdf(xx, s, 0, scale) * len(d) * np.diff(np.log(bins)).mean() * xx, c="k",
                label=f"log-normal: median {scale:.0f} nm, σ_g {np.exp(s):.2f}")
        ax.set_xscale("log"); ax.set_xlabel("Equivalent diameter (nm)"); ax.set_ylabel("Count"); ax.legend(fontsize=7)
        ax.set_title(Path(r["file"]).name[:22], fontsize=8)
    pairs = [(np.median(r["ours"]["d_px"]) * r["nm_per_px_256"], np.median(r["classical"]["d_px"]) * r["nm_per_px_256"])
             for r in R if r["nm_per_px_256"] and r.get("ours", {}).get("n", 0) >= 5 and r["classical"]["n"] >= 5]
    if len(pairs) > 2:
        a, b = np.log10(np.array(pairs).T)
        m, dd = (a + b) / 2, a - b
        axs[2].scatter(10 ** m, dd, s=8, c=COL[OURS])
        for v, ls in ((dd.mean(), "-"), (dd.mean() + 1.96 * dd.std(), "--"), (dd.mean() - 1.96 * dd.std(), "--")):
            axs[2].axhline(v, ls=ls, c="k", lw=1)
        axs[2].set_xscale("log"); axs[2].set_xlabel("Mean of per-image median diameters (nm)")
        axs[2].set_ylabel("log10(ours / classical)"); axs[2].set_title("Bland–Altman agreement", fontsize=9)
        NUM["real_ba_bias_log10"] = round(float(dd.mean()), 3)
        NUM["real_ba_loa_log10"] = round(float(1.96 * dd.std()), 3)
    savefig(fig, "fig15_morph_real")
    ex = RES / "morph" / "real_examples.npz"
    if ex.exists():
        from skimage.color import label2rgb
        z = np.load(ex)
        keys = [k for k in ("img", "classical", "ours") if k in z]
        n = min(4, len(z["img"]))
        fig, axs = plt.subplots(len(keys), n, figsize=(2.6 * n, 2.6 * len(keys)))
        for j in range(n):
            for i, k in enumerate(keys):
                im = z["img"][j] if k == "img" else label2rgb(z[k][j], image=z["img"][j] / 255, bg_label=0, alpha=0.4)
                axs[i, j].imshow(im, cmap="gray"); axs[i, j].axis("off")
            axs[0, j].set_title(f"NFFA Particles #{j + 1}", fontsize=8)
        for i, k in enumerate(keys):
            axs[i, 0].text(-0.05, 0.5, k, transform=axs[i, 0].transAxes, rotation=90, ha="right", va="center")
        savefig(fig, "fig16_real_examples")


# ------------------------------------------------------------------ main
def main():
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    R, Fz = ft_runs(), frozen()
    NUM.update({"n_local": C.N_LOCAL, "pt_epochs": C.PT_EPOCHS, "pt_abl_epochs": C.PT_ABL_EPOCHS,
                "ft_epochs": ", ".join(f"{BLAB[k]}: {v}" for k, v in C.FT_EPOCHS.items()),
                "min_steps": C.MIN_STEPS, "syn_test": C.SYN_TEST, "profile": "low" if C.LOW else "full"})
    meta = J(C.DATA_DIR / "meta.json") if have_data() else {}
    if "infobar_found" in meta:
        NUM["infobar_found_pct"] = round(100 * meta["infobar_found"], 1)
    steps = [table_dataset, table_physics, lambda: table_main(R), lambda: table_stats(R), lambda: table_frozen(Fz),
             lambda: table_robust(R, Fz), lambda: table_ablation(Fz), table_morph, table_real, lambda: table_efficiency(R),
             fig_framework, fig_dataset, fig_physics, fig_pretrain, lambda: fig_label_efficiency(R), lambda: fig_frozen(Fz),
             lambda: fig_confusion(R), lambda: fig_robust(R), fig_embed, fig_attention, lambda: fig_ablation(Fz),
             fig_morph, fig_real]
    for s in steps:
        try:
            s()
        except Exception as e:          # one missing result must not block the rest of the report
            import traceback
            print(f"[report] step failed: {getattr(s, '__name__', s)}: {e}")
            traceback.print_exc()
    # bootstrap CI for our headline numbers
    for f in BUD:
        if (OURS, f, 0) in R:
            r = R[(OURS, f, 0)]
            lo, hi = bootstrap_ci(r["y"], r["p"].argmax(1), lambda a, b: (a == b).mean())
            NUM[f"ours_seed0_acc_ci_{f}"] = [round(100 * lo, 2), round(100 * hi, 2)]
    json.dump(NUM, open(RES / "numbers.json", "w"), indent=1, default=str)
    with open(RES / "RESULTS.md", "w", encoding="utf-8") as fh:
        fh.write("# Results (generated by src/report.py — do not edit by hand)\n\n")
        for t in sorted(TAB.glob("*.md")):
            fh.write(t.read_text(encoding="utf-8") + "\n")
        fh.write("## Numbers\n\n```json\n" + json.dumps(NUM, indent=1, default=str) + "\n```\n")
    print(f"report: {len(list(FIG.glob('*.png')))} figures, {len(list(TAB.glob('*.md')))} tables")


if __name__ == "__main__":
    main()
