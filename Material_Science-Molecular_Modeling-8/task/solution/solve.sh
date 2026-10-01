#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

"""Step 1: bin aggregates of the merge group."""

import numpy as np


def bin_aggregates(w, v, x):
    w = np.asarray(w, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if w.ndim != 1 or v.shape != (w.shape[0], 3) or x.shape != w.shape:
        raise ValueError("shape mismatch")
    if not (np.all(np.isfinite(w)) and np.all(np.isfinite(v)) and np.all(np.isfinite(x))):
        raise ValueError("nonfinite input")
    if np.any(w <= 0):
        raise ValueError("nonpositive weight")
    wS = float(np.sum(w))
    vS = (w @ v) / wS
    xS = float(w @ x) / wS
    return np.concatenate([[wS], vS, [xS]])

"""Step 2: reference scales for the scaled moment system."""

import numpy as np


def reference_scales(w, v, x):
    w = np.asarray(w, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    agg = bin_aggregates(w, v, x)
    wS, vS, xS = agg[0], agg[1:4], agg[4]
    vref = np.sqrt((w @ (v - vS) ** 2) / wS)
    xref = float(np.sqrt(w @ (x - xS) ** 2 / wS))
    if np.any(vref <= 0) or xref <= 0:
        raise ValueError("degenerate reference scale")
    return np.concatenate([vref, [xref]])

"""Step 3: scaled moment block with consistent right-hand side."""

import numpy as np

_IV = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1),
       (2, 0, 0), (0, 2, 0), (0, 0, 2), (1, 1, 0), (1, 0, 1), (0, 1, 1)]
_IX = [1, 2]


def scaled_moment_system(w, v, x):
    w = np.asarray(w, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    n = w.shape[0]
    agg = bin_aggregates(w, v, x)
    ref = reference_scales(w, v, x)
    wS, vS, xS = agg[0], agg[1:4], agg[4]
    vref, xref = ref[0:3], ref[3]
    vh = (v - vS) / vref
    xh = (x - xS) / xref
    out = np.empty((len(_IV) + len(_IX), n + 1))
    for j, (mx, my, mz) in enumerate(_IV):
        out[j, :n] = (vh[:, 0] ** mx) * (vh[:, 1] ** my) * (vh[:, 2] ** mz)
    for k, mx_ in enumerate(_IX):
        out[len(_IV) + k, :n] = xh ** mx_
    out[:, n] = out[:, :n] @ (w / wS)
    return out

"""Step 4: rate-preservation rows against the fixed background species."""

import numpy as np


def _sigma(r, g):
    if r == 0:
        return 1.0 / (1.0 + g * g)
    return g / (1.0 + g)


def rate_rows(w, v, w2, v2):
    w = np.asarray(w, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    w2 = np.asarray(w2, dtype=np.float64)
    v2 = np.asarray(v2, dtype=np.float64)
    if v.ndim != 2 or v.shape[1] != 3 or w.shape[0] != v.shape[0] or v2.shape != (w2.shape[0], 3):
        raise ValueError("shape mismatch")
    n = v.shape[0]
    wS = float(np.sum(w))
    wS2 = float(np.sum(w2))
    out = np.empty((2, n + 1))
    for r in range(2):
        c = np.empty(n)
        for i in range(n):
            g = np.linalg.norm(v[i] - v2, axis=1)
            c[i] = float(np.sum(w2 * g * _sigma(r, g)))
        m_r = float(np.sum(w * c))
        scale = wS * wS2
        out[r, :n] = (wS * c) / scale
        out[r, n] = m_r / scale
    return out

"""Step 5: column scaling of the stacked constraint matrix."""

import numpy as np


def column_scaling(A):
    A = np.asarray(A, dtype=np.float64)
    nrm2 = np.sum(A ** 2, axis=0)
    if np.any(nrm2 <= 0):
        raise ValueError("zero column in scaled system")
    return nrm2 ** -0.5

"""Step 6: active-set nonnegative least squares with declared conventions."""

import numpy as np


def _ls_solve(A, b):
    Q, R = np.linalg.qr(A)
    return np.linalg.solve(R, Q.T @ b)


def nnls_lawson_hanson(A, b):
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = A.shape[1]
    P = []
    Z = list(range(n))
    w = np.zeros(n)
    it = 0
    while True:
        grad = -(A.T @ (A @ w - b))
        if not Z:
            break
        gz = np.full(n, -np.inf)
        gz[Z] = grad[Z]
        if np.max(gz) <= 1e-10:
            break
        it += 1
        if it > 200:
            raise ValueError("NNLS iteration limit exceeded")
        tau = int(np.argmax(gz))
        Z.remove(tau)
        P.append(tau)
        z = np.zeros(n)
        z[P] = _ls_solve(A[:, P], b)
        inner = 0
        while P and np.min(z[P]) <= 0:
            inner += 1
            if inner > 200:
                raise ValueError("NNLS inner iteration limit exceeded")
            Qset = [i for i in P if z[i] <= 0]
            alpha = min(w[i] / (w[i] - z[i]) for i in Qset)
            w = w + alpha * (z - w)
            drop = [i for i in P if w[i] <= 0]
            for i in drop:
                P.remove(i)
                Z.append(i)
            Z.sort()
            z = np.zeros(n)
            if P:
                z[P] = _ls_solve(A[:, P], b)
        w = z.copy()
    return w

"""Step 7: the full rate-preserving NNLS merge."""

import numpy as np


def nnls_merge(w, v, x, w2, v2, eps):
    w = np.asarray(w, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("invalid threshold")
    n = w.shape[0]
    wS = float(np.sum(w))
    M = scaled_moment_system(w, v, x)
    R = rate_rows(w, v, w2, v2)
    A = np.vstack([M[:, :n], R[:, :n]])
    b = np.concatenate([M[:, n], R[:, n]])
    s = column_scaling(A)
    sol = nnls_lawson_hanson(A * s[None, :], b)
    wh = s * sol
    keep = [i for i in range(n) if wh[i] >= eps]
    rows = [[wS * wh[i], v[i, 0], v[i, 1], v[i, 2], x[i]] for i in keep]
    return np.asarray(rows, dtype=np.float64)

"""Step 8 (final orchestrator): merge audit over the two declared datasets."""

import numpy as np

_N = 24
_NB = 10
_IV = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1),
       (2, 0, 0), (0, 2, 0), (0, 0, 2), (1, 1, 0), (1, 0, 1), (0, 1, 1)]
_IX = [1, 2]


def _build(variant):
    a = 3 if variant == 1 else 5
    w = np.empty(_N)
    v = np.empty((_N, 3))
    x = np.empty(_N)
    for i in range(_N):
        w[i] = 0.5 + (((i + 1) * (i + 3) + a) % 7) / 10.0
        for c in range(3):
            v[i, c] = (((i + 1) * (i + 2) + (c + 2) * (i + 5) + a) % 47) / 23.5 - 1.0
        x[i] = (((i + 2) * (i + 4) * a + 1) % 41) / 41.0
    w2 = np.empty(_NB)
    v2 = np.empty((_NB, 3))
    for k in range(_NB):
        w2[k] = 0.8 + (((k + 1) * (k + 2) + a) % 5) / 10.0
        for c in range(3):
            v2[k, c] = (((k + 2) * (k + 3) + (c + 3) * (k + 1) + 2 * a) % 31) / 25.0 - 0.6
    return w, v, x, w2, v2


def _probe():
    return np.array([0.5, -0.25, 0.75])


def _sigma8(r, g):
    if r == 0:
        return 1.0 / (1.0 + g * g)
    return g / (1.0 + g)


def _conserved_vec(ww, vv, xx, w2, v2):
    m = np.empty(len(_IV) + len(_IX) + 2)
    for j, (mx, my, mz) in enumerate(_IV):
        m[j] = float(np.sum(ww * (vv[:, 0] ** mx) * (vv[:, 1] ** my) * (vv[:, 2] ** mz)))
    for k, mx_ in enumerate(_IX):
        m[len(_IV) + k] = float(np.sum(ww * xx ** mx_))
    for r in range(2):
        tot = 0.0
        for i in range(len(ww)):
            g = np.linalg.norm(vv[i] - v2, axis=1)
            tot += float(ww[i] * np.sum(w2 * g * _sigma8(r, g)))
        m[len(_IV) + len(_IX) + r] = tot
    return m


def merge_audit(eps):
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("invalid threshold")
    u = _probe()
    rows = []
    for variant in (1, 2):
        w, v, x, w2, v2 = _build(variant)
        merged = nnls_merge(w, v, x, w2, v2, eps)
        wp = merged[:, 0]
        vp = merged[:, 1:4]
        xp = merged[:, 4]
        res = float(np.linalg.norm(_conserved_vec(wp, vp, xp, w2, v2) - _conserved_vec(w, v, x, w2, v2)))
        uv = vp @ u
        rows.append([float(len(wp)), float(np.sum(wp)), res,
                     float(wp @ uv ** 3), float(wp @ xp ** 3),
                     float(np.max(wp)), float(np.min(wp))])
    return np.asarray(rows, dtype=np.float64)
SCICODE_GOLD_EOF
