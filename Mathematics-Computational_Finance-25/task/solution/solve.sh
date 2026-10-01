#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

import numpy as np


def discrete_gaussian_weights(x, sigma, centre):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size < 2:
        raise ValueError("x must be a 1-D grid with at least two nodes")
    if not np.all(np.isfinite(x)):
        raise ValueError("x must be finite")
    sigma = float(sigma)
    if not (sigma > 0.0) or not np.isfinite(sigma):
        raise ValueError("sigma must be finite and strictly positive")
    centre = float(centre)
    if not np.isfinite(centre):
        raise ValueError("centre must be finite")
    w = np.exp(-((x - centre) ** 2) / (2.0 * sigma * sigma))
    s = float(w.sum())
    if not (s > 0.0):
        raise ValueError("weights underflowed to zero mass")
    return w / s

import numpy as np


def discrete_second_moment(x, w, centre):
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    if x.shape != w.shape:
        raise ValueError("x and w must have the same shape")
    if np.any(w < 0.0):
        raise ValueError("w must be non-negative")
    tot = float(w.sum())
    if not (tot > 0.0):
        raise ValueError("w must carry positive total mass")
    centre = float(centre)
    return float((w * (x - centre) ** 2).sum() / tot)

import numpy as np


def active_state_mask(w, mass_frac):
    w = np.asarray(w, dtype=float)
    if w.ndim != 1:
        raise ValueError("w must be 1-D")
    if np.any(w < 0.0):
        raise ValueError("w must be non-negative")
    peak = float(w.max()) if w.size else 0.0
    if not (peak > 0.0):
        raise ValueError("w has no positive peak")
    mass_frac = float(mass_frac)
    if not (0.0 <= mass_frac <= 1.0):
        raise ValueError("mass_frac must lie in [0, 1]")
    return (w >= mass_frac * peak).astype(float)

import numpy as np


def jensen_feasibility_gap(x, w_source, w_target, mask, centre):
    x = np.asarray(x, dtype=float)
    mask = np.asarray(mask, dtype=float)
    if mask.shape != x.shape:
        raise ValueError("mask must match the grid shape")
    if np.any((mask != 0.0) & (mask != 1.0)):
        raise ValueError("mask must be an indicator of zeros and ones")
    lhs = discrete_second_moment(x, w_target, centre)
    w_src = np.asarray(w_source, dtype=float)
    if w_src.shape != x.shape:
        raise ValueError("w_source must match the grid shape")
    tot = float(w_src.sum())
    if not (tot > 0.0):
        raise ValueError("w_source must carry positive total mass")
    centre = float(centre)
    rhs = float(((w_src / tot) * mask * (x - centre) ** 2).sum())
    return float(lhs - rhs)

import numpy as np


def product_coupling(w_source, w_target):
    a = np.asarray(w_source, dtype=float)
    b = np.asarray(w_target, dtype=float)
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("both marginals must be 1-D")
    if np.any(a < 0.0) or np.any(b < 0.0):
        raise ValueError("marginals must be non-negative")
    if not (float(a.sum()) > 0.0 and float(b.sum()) > 0.0):
        raise ValueError("both marginals must carry positive mass")
    P = np.outer(a / a.sum(), b / b.sum())
    return P / float(P.sum())

import numpy as np


def capped_marginal_tilt(P, axis, index, target, cap):
    P = np.array(P, dtype=float, copy=True)
    if P.ndim != 2:
        raise ValueError("P must be 2-D")
    axis = int(axis)
    if axis not in (0, 1):
        raise ValueError("axis must be 0 or 1")
    index = int(index)
    if not (0 <= index < P.shape[1 - axis]):
        raise ValueError("index out of range for the requested axis")
    cap = float(cap)
    if not (cap > 0.0) or not np.isfinite(cap):
        raise ValueError("cap must be finite and strictly positive")
    target = float(target)
    if not (target >= 0.0) or not np.isfinite(target):
        raise ValueError("target must be finite and non-negative")
    sl = P[index, :] if axis == 1 else P[:, index]
    s = float(sl.sum())
    if s <= 0.0:
        return P
    theta = float(np.clip(np.log(max(target, 1e-300) / s), -cap, cap))
    sl *= float(np.exp(theta))
    return P

import numpy as np


def martingale_soft_gradient(P, x, mask, penalty):
    P = np.asarray(P, dtype=float)
    x = np.asarray(x, dtype=float)
    mask = np.asarray(mask, dtype=float)
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError("P must be a square 2-D coupling")
    if x.shape != (P.shape[0],) or mask.shape != x.shape:
        raise ValueError("x and mask must match the coupling dimension")
    penalty = float(penalty)
    if not np.isfinite(penalty) or penalty < 0.0:
        raise ValueError("penalty must be finite and non-negative")
    d = x[None, :] - x[:, None]
    resid = (P * d).sum(axis=1) * mask
    return (penalty * resid)[:, None] * d * mask[:, None]

import numpy as np


def hybrid_sweep(P, x, w_source, w_target, mask, hard_cap, penalty,
                         base_step, soft_cap):
    P = np.array(P, dtype=float, copy=True)
    x = np.asarray(x, dtype=float)
    a = np.asarray(w_source, dtype=float)
    b = np.asarray(w_target, dtype=float)
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError("P must be a square 2-D coupling")
    if a.shape != (P.shape[0],) or b.shape != (P.shape[0],):
        raise ValueError("marginals must match the coupling dimension")
    base_step = float(base_step)
    soft_cap = float(soft_cap)
    if not (base_step > 0.0) or not (soft_cap > 0.0):
        raise ValueError("base_step and soft_cap must be strictly positive")
    n = P.shape[0]
    # L5: the source-marginal family is visited first, then the target family.
    # L4: total mass is restored after EVERY row, not once at the end of the sweep.
    for j in range(n):
        P = capped_marginal_tilt(P, 1, j, float(a[j]), hard_cap)
        P /= float(P.sum())
    for k in range(n):
        P = capped_marginal_tilt(P, 0, k, float(b[k]), hard_cap)
        P /= float(P.sum())
    # L2: exactly ONE batch exponentiated-gradient update follows the hard sweep.
    # No dual ascent: the multiplier stays at zero throughout.
    g = martingale_soft_gradient(P, x, mask, penalty)
    gmax = float(np.abs(g).max())
    eta = min(base_step, soft_cap / gmax) if gmax > 0.0 else 0.0
    P *= np.exp(np.clip(-eta * g, -700.0, 700.0))
    return P / float(P.sum())

import numpy as np


def residual_pair(P, x, w_source, w_target, mask):
    P = np.asarray(P, dtype=float)
    x = np.asarray(x, dtype=float)
    a = np.asarray(w_source, dtype=float)
    b = np.asarray(w_target, dtype=float)
    mask = np.asarray(mask, dtype=float)
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError("P must be a square 2-D coupling")
    if a.shape != (P.shape[0],) or b.shape != (P.shape[0],) or mask.shape != a.shape:
        raise ValueError("marginals, mask and grid must match the coupling dimension")
    r_marg = max(float(np.abs(P.sum(axis=1) - a).max()),
                 float(np.abs(P.sum(axis=0) - b).max()))
    d = x[None, :] - x[:, None]
    row = np.abs((P * d).sum(axis=1)) * mask
    r_cond = float(row.max()) if mask.any() else 0.0
    return np.array([r_marg, r_cond], dtype=float)

import numpy as np


def finite_budget_priority_audit(n_sweeps):
    n_sweeps = int(n_sweeps)
    if n_sweeps < 1:
        raise ValueError("n_sweeps must be a positive integer")
    # Evaluation configuration: GIVEN in the prompt.
    x = 0.25 + 0.0375 * np.arange(41, dtype=float)
    centre, sigma_x, mass_frac = 1.0, 0.15, 0.01
    hard_cap, soft_cap = 0.02, 1.0
    # L1: the paper's soft-update magnitude. penalty and base step are ONE lock:
    # their product sets the step actually taken.
    penalty, base_step = 200.0, 0.5

    a = discrete_gaussian_weights(x, sigma_x, centre)
    b = discrete_gaussian_weights(x, sigma_x * np.sqrt(0.70), centre)
    mask = active_state_mask(a, mass_frac)
    gap = jensen_feasibility_gap(x, a, b, mask, centre)
    # the two dispersions the certificate compares, reported alongside the slack
    m_target = discrete_second_moment(x, b, centre)
    m_source_active = discrete_second_moment(x, a * mask, centre) * float((a * mask).sum())

    P = product_coupling(a, b)
    # The product coupling already meets both marginals, so a tilt applied to it is
    # the identity. Exercise the tilt on the coupling AFTER one sweep, where the soft
    # update has broken the marginals and the cap actually binds.
    grad0 = martingale_soft_gradient(P, x, mask, penalty)
    grad0_max = float(np.abs(grad0).max())
    P1 = hybrid_sweep(P, x, a, b, mask, hard_cap, penalty, base_step, soft_cap)
    dev_before = abs(float(P1[0, :].sum()) - float(a[0]))
    P1t = capped_marginal_tilt(P1, 1, 0, float(a[0]), hard_cap)
    dev_after = abs(float(P1t[0, :].sum()) - float(a[0]))
    tilt_gain = float(dev_before - dev_after)

    hist = np.empty((n_sweeps, 2), dtype=float)
    for _ in range(n_sweeps):
        P = hybrid_sweep(P, x, a, b, mask, hard_cap, penalty,
                                 base_step, soft_cap)
        hist[_, :] = residual_pair(P, x, a, b, mask)
    tail = max(1, int(round(0.10 * n_sweeps)))
    r_marg = float(np.median(hist[-tail:, 0]))
    r_cond = float(np.median(hist[-tail:, 1]))
    return np.array([r_marg, r_cond, gap, float(mask.sum()),
                     m_target, m_source_active, tilt_gain, grad0_max], dtype=float)
SCICODE_GOLD_EOF
