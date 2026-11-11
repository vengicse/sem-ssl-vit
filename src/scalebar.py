"""Read the physical pixel size from the instrument info bar: OCR the scale label
(e.g. '200 nm', '10 µm') and measure the scale-bar line length in pixels."""
import re

import numpy as np
from PIL import Image

UNIT = {"nm": 1.0, "µm": 1e3}
PAT = re.compile(r"(\d+(?:[.,]\d+)?)\s*(nm|µm)")
_reader = None


def reader(gpu=False):
    global _reader
    if _reader is None:
        import easyocr
        _reader = easyocr.Reader(["en"], gpu=gpu, verbose=False)
    return _reader


def norm(t):
    """OCR renders the micro sign as u, H, p or an unknown glyph, and 0 as O/Q."""
    t = t.replace("�", "µ")
    t = re.sub(r"(?<![A-Za-z])[ZzOoQ](?=\d)", lambda m: {"Z": "2", "z": "2"}.get(m.group(0), "0"), t)
    t = re.sub(r"(?<=\d)[OoQ]", "0", t)
    t = re.sub(r"(\d)\s*[uµμHhpU]+\s*m(?![a-z])", lambda m: m.group(1) + " µm", t)
    return re.sub(r"(\d)\s+m(?![a-z])", lambda m: m.group(1) + " µm", t)   # dropped micro sign


def bar_band(g, top):
    """Rows of the white info bar (stop where the bar ends)."""
    H, W = g.shape
    white = lambda r: (g[r, W // 4: W // 2] > 200).mean() > 0.4   # bar-free columns
    r0 = top
    while r0 < min(H, top + 20) and not (white(r0) and white(min(r0 + 1, H - 1))):
        r0 += 1
    r1 = r0
    while r1 < H and white(r1):
        r1 += 1
    return g[r0:r1, : W // 4]


def bar_length(band):
    """Longest horizontal run of non-white pixels that does not touch the band edges."""
    W = band.shape[1]
    best = 0
    for row in band < 200:
        d = np.diff(np.concatenate([[0], row.astype(np.int8), [0]]))
        for s, e in zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]):
            if s > 0 and e < W:
                best = max(best, int(e - s))
    return best


def read_scale(path, top, gpu=False):
    g = np.asarray(Image.open(path).convert("L"))
    band = bar_band(g, top)
    if band.shape[0] < 10:
        return {"text": "", "bar_px": 0, "nm_per_px": None}
    dets = sorted(reader(gpu).readtext(band), key=lambda d: min(p[0] for p in d[0]))
    txt = " | ".join(t for _, t, c in dets)
    m = None
    for box, t, c in dets:            # left-most detection holding '<number> <unit>'
        if min(p[0] for p in box) < band.shape[1] * 0.5 and c > 0.2:
            m = PAT.search(norm(t))
            if m:
                break
    if m is None:                      # label split over two detections, e.g. '20' + 'Hm'
        left = [t for box, t, c in dets if min(p[0] for p in box) < band.shape[1] * 0.5]
        m = PAT.search(norm(" ".join(left)).split("EHT")[0])
    L = bar_length(band)
    if not m or L < 15:
        return {"text": txt, "bar_px": L, "nm_per_px": None}
    val = float(m.group(1).replace(",", ".")) * UNIT[m.group(2)]
    return {"text": txt, "bar_px": L, "value_nm": val, "nm_per_px": val / L}
