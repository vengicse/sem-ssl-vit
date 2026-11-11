"""Physics-aware SEM artefact operators, batched on the GPU.

All operators act on x of shape (B, 1, H, W) in [0, 1] and take a per-sample severity
s in [0, 1] (training) or s > 1 (extrapolated test severities of the SEM-C benchmark).
Forward model (Eq. 2):  I' = N_PG( g( h_psf * T_drift(I) + C ) )
"""
import math

import torch
import torch.nn.functional as F

import config as C


def _lerp(rng, s):
    lo, hi = rng
    return lo + (hi - lo) * s


def _v(s, B, dev):
    return s.view(B, 1, 1, 1)


def charging(x, s):
    """Low-frequency additive field (surface charge) plus scan-line streaks: a bright
    band that decays along the fast scan direction, as charge builds up then leaks away."""
    B, _, H, W = x.shape
    dev = x.device
    amp = _lerp(C.PHYS["charging"]["amp"], s)
    frac = torch.empty(B, device=dev).uniform_(*C.PHYS["charging"]["sigma_frac"])
    k = (1.0 / frac).clamp(2, 12).round().int()
    field = torch.empty(B, 1, H, W, device=dev)
    for n in k.unique().tolist():            # one batched upsample per correlation length
        sel = k == n
        f = torch.randn(int(sel.sum()), 1, n, n, device=dev)
        field[sel] = F.interpolate(f, size=(H, W), mode="bicubic", align_corners=False)
    field = field / field.flatten(1).std(1).clamp_min(1e-6).view(B, 1, 1, 1)
    # streaks: sparse rows that brighten and decay exponentially along x
    rows = (torch.rand(B, 1, H, 1, device=dev) < 0.02).float()
    rows = F.max_pool2d(rows, (3, 1), 1, (1, 0))
    decay = torch.exp(-torch.arange(W, device=dev).float() / (0.3 * W)).view(1, 1, 1, W)
    streak = rows * decay
    return x + _v(amp, B, dev) * (0.6 * field.clamp(min=0) + 0.8 * streak)


def drift(x, s):
    """Sub-pixel global shift plus row-wise horizontal jitter (random walk) from stage drift
    and scan instability during the slow (y) scan."""
    B, _, H, W = x.shape
    dev = x.device
    shift = _lerp(C.PHYS["drift"]["shift_px"], s)
    jit = _lerp(C.PHYS["drift"]["jitter_px"], s)
    ang = torch.rand(B, device=dev) * 2 * math.pi
    dx = (shift * torch.cos(ang)).view(B, 1, 1) * 2 / W
    dy = (shift * torch.sin(ang)).view(B, 1, 1) * 2 / H
    walk = torch.randn(B, H, device=dev).cumsum(1)
    walk = walk - walk.mean(1, keepdim=True)
    walk = walk / walk.std(1, keepdim=True).clamp_min(1e-6)
    row = (walk * jit.view(B, 1)).view(B, H, 1) * 2 / W
    ys = torch.linspace(-1, 1, H, device=dev).view(1, H, 1).expand(B, H, W)
    xs = torch.linspace(-1, 1, W, device=dev).view(1, 1, W).expand(B, H, W)
    grid = torch.stack([xs + dx + row, ys + dy], -1)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=True)


def psf(x, s, ksize=15):
    """Anisotropic, rotated Gaussian blur: defocus plus astigmatism."""
    B, _, H, W = x.shape
    dev = x.device
    sig = _lerp(C.PHYS["psf"]["sigma"], s).clamp_min(0.2)
    an = _lerp(C.PHYS["psf"]["aniso"], s * torch.rand(B, device=dev)).clamp_min(1.0)
    th = torch.rand(B, device=dev) * math.pi
    r = torch.arange(ksize, device=dev).float() - ksize // 2
    yy, xx = torch.meshgrid(r, r, indexing="ij")
    c, sn = torch.cos(th).view(B, 1, 1), torch.sin(th).view(B, 1, 1)
    u = c * xx + sn * yy
    v = -sn * xx + c * yy
    sx = (sig * an.sqrt()).view(B, 1, 1)
    sy = (sig / an.sqrt()).view(B, 1, 1)
    k = torch.exp(-0.5 * ((u / sx) ** 2 + (v / sy) ** 2))
    k = (k / k.sum((1, 2), keepdim=True)).unsqueeze(1)
    xp = F.pad(x, [ksize // 2] * 4, mode="reflect")
    out = F.conv2d(xp.transpose(0, 1), k, groups=B)
    return out.transpose(0, 1)


def noise(x, s):
    """Poisson (shot) noise at a photon budget lam, plus Gaussian read noise."""
    B = x.shape[0]
    dev = x.device
    lo, hi = C.PHYS["noise"]["photons"]
    lam = torch.exp(math.log(hi) + (math.log(lo) - math.log(hi)) * s).clamp_min(5.0)   # log-spaced dose
    rd = _lerp(C.PHYS["noise"]["read"], s)
    lam4 = _v(lam, B, dev)
    y = torch.poisson(x.clamp(0, 1) * lam4) / lam4
    return y + torch.randn_like(y) * _v(rd, B, dev)


def gain(x, s):
    """Detector brightness / contrast / gamma drift."""
    B = x.shape[0]
    dev = x.device
    P = C.PHYS["gain"]
    sign = lambda: torch.where(torch.rand(B, device=dev) < 0.5, -1.0, 1.0)
    br = sign() * _lerp((0, P["brightness"][1]), s)
    ct = 1 + sign() * _lerp((0, P["contrast"][1] - 1), s)
    gm = torch.exp(sign() * _lerp((0, math.log(P["gamma"][1])), s))
    m = x.mean((1, 2, 3), keepdim=True)
    y = (x - m) * _v(ct, B, dev) + m + _v(br, B, dev)
    return y.clamp(0, 1) ** _v(gm, B, dev)


OPS = {"drift": drift, "psf": psf, "charging": charging, "gain": gain, "noise": noise}
ORDER = ["drift", "psf", "charging", "gain", "noise"]   # physical order of the imaging chain


def physics_chain(x, p_chain=C.P_PHYSICS, p_op=0.5, ops=None):
    """Training augmentation: with prob p_chain apply each enabled operator with prob p_op at a
    random severity, in imaging-chain order. `ops` restricts the operator set (ablation)."""
    ops = ORDER if ops is None else [o for o in ORDER if o in ops]
    B, dev = x.shape[0], x.device
    on = torch.rand(B, device=dev) < p_chain
    if not on.any() or not ops:
        return x
    y = x.clone()
    xi = x[on]
    n = xi.shape[0]
    for o in ops:
        sel = torch.rand(n, device=dev) < p_op
        if sel.any():
            s = torch.rand(int(sel.sum()), device=dev)
            xi[sel] = OPS[o](xi[sel], s).clamp(0, 1)
    y[on] = xi
    return y


def corrupt(x, op, severity, seed=0):
    """SEM-C benchmark corruption at a fixed integer severity 1..5 (s = 0.25 * severity:
    severity 4 equals the strongest training setting, severity 5 lies beyond it)."""
    with torch.random.fork_rng(devices=[x.device.index or 0] if x.is_cuda else []):
        torch.manual_seed(seed * 100 + severity)
        s = torch.full((x.shape[0],), 0.25 * severity, device=x.device)
        return OPS[op](x, s).clamp(0, 1)
