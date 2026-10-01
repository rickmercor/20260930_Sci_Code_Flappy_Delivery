#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

"""Step 1: Morse bond-stretching potential (Eq. S8)."""

import numpy as np


def morse_potential(l, De, a, le):
    l = np.asarray(l, dtype=np.float64)
    return De * (1.0 - np.exp(-a * (l - le))) ** 2

"""Step 2: angular coupling kernel (Eq. S61, with Eq. S9)."""

import numpy as np

_NTH = 121


def _sw(n, h):
    w = np.ones(n)
    w[1:-1:2] = 4.0
    w[2:-2:2] = 2.0
    return w * h / 3.0


_TH = np.linspace(0.0, np.pi, _NTH)
_WTH = _sw(_NTH, _TH[1] - _TH[0])

_NW = 121


def bending_kernel(beta, kphi, phi_e):
    with np.errstate(all="ignore"):
        w = np.linspace(0.0, 2.0 * np.pi, _NW)
        ww = _sw(_NW, w[1] - w[0])
        ct, st = np.cos(_TH), np.sin(_TH)
        cphi = (ct[:, None, None] * ct[None, :, None]
                + st[:, None, None] * st[None, :, None] * np.cos(w)[None, None, :])
        phi = np.arccos(np.clip(cphi, -1.0, 1.0))
        return np.einsum('ijk,k->ij', np.exp(-0.5 * beta * kphi * (phi - phi_e) ** 2), ww)

"""Step 3: local intact weight (Eq. S60), returned as logs."""

import numpy as np

_NTH = 121


def _sw(n, h):
    w = np.ones(n)
    w[1:-1:2] = 4.0
    w[2:-2:2] = 2.0
    return w * h / 3.0


_TH = np.linspace(0.0, np.pi, _NTH)
_WTH = _sw(_NTH, _TH[1] - _TH[0])

_NL = 1201


def local_intact_weight(lt, f, beta, De, a, le):
    if lt <= 0:
        raise ValueError("threshold must be positive")
    with np.errstate(all="ignore"):
        l = np.linspace(0.0, lt, _NL)
        wl = _sw(_NL, l[1] - l[0])
        ex = -beta * (morse_potential(l, De, a, le)[None, :] - f * l[None, :] * np.cos(_TH)[:, None])
        ex = ex + np.where(l > 0, np.log(np.where(l > 0, l, 1.0) ** 2), -np.inf)[None, :]
        m = np.max(np.where(np.isfinite(ex), ex, -np.inf), axis=1, keepdims=True)
        val = np.log((np.exp(ex - m) * wl[None, :]).sum(axis=1)) + m[:, 0]
        s = np.sin(_TH)
        ok = s > 1e-12
        out = np.where(ok, np.log(np.where(ok, s, 1.0)), -1e300) + val
        return np.where(np.isfinite(out), np.maximum(out, -1e300), -1e300)

"""Step 4: transfer-matrix angular weights (Eqs. S72-S79)."""

import numpy as np

_NTH = 121


def _sw(n, h):
    w = np.ones(n)
    w[1:-1:2] = 4.0
    w[2:-2:2] = 2.0
    return w * h / 3.0


_TH = np.linspace(0.0, np.pi, _NTH)
_WTH = _sw(_NTH, _TH[1] - _TH[0])


def tm_angular_weights(thresholds, f, beta, De, a, le, Q):
    with np.errstate(all="ignore"):
        thr = np.asarray(thresholds, dtype=np.float64)
        N = thr.size
        logI = np.array([local_intact_weight(thr[i], f, beta, De, a, le) for i in range(N)])
        mx = np.max(np.where(np.isfinite(logI), logI, -np.inf), axis=1, keepdims=True)
        I = np.where(np.isfinite(logI), np.exp(np.minimum(logI - mx, 0.0)), 0.0)
        PL = np.zeros((N, _NTH))
        PR = np.zeros((N, _NTH))
        PL[0] = 1.0 / np.pi
        PR[N - 1] = 1.0 / np.pi
        for i in range(N - 1):
            v = Q @ (PL[i] * I[i] * _WTH)
            PL[i + 1] = v / np.sum(v * _WTH)
        for i in range(N - 1, 0, -1):
            v = Q.T @ (PR[i] * I[i] * _WTH)
            PR[i - 1] = v / np.sum(v * _WTH)
        return PL * PR

"""Step 5: bond-length PMF from the angular weight (Eq. S82)."""

import numpy as np

_NTH = 121


def _sw(n, h):
    w = np.ones(n)
    w[1:-1:2] = 4.0
    w[2:-2:2] = 2.0
    return w * h / 3.0


_TH = np.linspace(0.0, np.pi, _NTH)
_WTH = _sw(_NTH, _TH[1] - _TH[0])


def tm_pmf(l, w_i, f, beta, De, a, le):
    with np.errstate(all="ignore"):
        l = np.atleast_1d(np.asarray(l, dtype=np.float64))
        ex = beta * f * l[:, None] * np.cos(_TH)[None, :]
        m = ex.max(axis=1, keepdims=True)
        ang = np.log(((np.exp(ex - m) * (np.asarray(w_i) * np.sin(_TH))[None, :]) * _WTH[None, :]).sum(axis=1)) + m[:, 0]
        return morse_potential(l, De, a, le) - (1.0 / beta) * (np.log(l ** 2) + ang)

"""Step 6: self-consistent rupture thresholds (Sec. S2.2 procedure)."""

import numpy as np

_NSCAN, _LO, _HI, _KITER = 24001, 0.6, 6.0, 8


def _stationary(w_i, f, beta, De, a, le):
    l = np.linspace(_LO, _HI, _NSCAN)
    W = tm_pmf(l, w_i, f, beta, De, a, le)
    d = np.diff(W)
    s = np.sign(d)
    idx = np.where(s[:-1] != s[1:])[0] + 1
    lm = lb = None
    for k in idx:
        if s[k - 1] < 0 and s[k] > 0 and lm is None:
            lm = l[k]
        elif s[k - 1] > 0 and s[k] < 0 and lm is not None and lb is None:
            lb = l[k]
    if lm is None or lb is None:
        raise ValueError("no minimum/threshold pair; force may exceed f_c")

    def ref(l0, kind):
        A, B = l0 - 0.02, l0 + 0.02
        for _ in range(80):
            m1 = A + (B - A) / 3.0
            m2 = B - (B - A) / 3.0
            f1 = float(tm_pmf(m1, w_i, f, beta, De, a, le)[0])
            f2 = float(tm_pmf(m2, w_i, f, beta, De, a, le)[0])
            if kind == 'min':
                if f1 < f2: B = m2
                else: A = m1
            else:
                if f1 > f2: B = m2
                else: A = m1
        return 0.5 * (A + B)

    return ref(lm, 'min'), ref(lb, 'max')


def self_consistent_thresholds(N, f, beta, De, a, le, kphi, phi_e):
    if isinstance(N, bool) or not isinstance(N, (int, np.integer)) or N < 2:
        raise ValueError("N must be an integer >= 2")
    Q = bending_kernel(beta, kphi, phi_e)
    thr = np.full(int(N), 2.4, dtype=np.float64)
    for _ in range(_KITER):
        w = tm_angular_weights(thr, f, beta, De, a, le, Q)
        thr = np.array([_stationary(w[i], f, beta, De, a, le)[1] for i in range(int(N))])
    return thr

"""Step 7: bond-resolved activation barriers in kT."""

import numpy as np


_NSCAN, _LO, _HI = 24001, 0.6, 6.0


def _stationary(w_i, f, beta, De, a, le):
    l = np.linspace(_LO, _HI, _NSCAN)
    W = tm_pmf(l, w_i, f, beta, De, a, le)
    d = np.diff(W)
    s = np.sign(d)
    idx = np.where(s[:-1] != s[1:])[0] + 1
    lm = lb = None
    for k in idx:
        if s[k - 1] < 0 and s[k] > 0 and lm is None:
            lm = l[k]
        elif s[k - 1] > 0 and s[k] < 0 and lm is not None and lb is None:
            lb = l[k]
    if lm is None or lb is None:
        raise ValueError("no minimum/threshold pair; force may exceed f_c")

    def ref(l0, kind):
        A, B = l0 - 0.02, l0 + 0.02
        for _ in range(80):
            m1 = A + (B - A) / 3.0
            m2 = B - (B - A) / 3.0
            f1 = float(tm_pmf(m1, w_i, f, beta, De, a, le)[0])
            f2 = float(tm_pmf(m2, w_i, f, beta, De, a, le)[0])
            if kind == 'min':
                if f1 < f2: B = m2
                else: A = m1
            else:
                if f1 > f2: B = m2
                else: A = m1
        return 0.5 * (A + B)

    return ref(lm, 'min'), ref(lb, 'max')


def bond_barriers(N, f, beta, De, a, le, kphi, phi_e):
    thr = self_consistent_thresholds(N, f, beta, De, a, le, kphi, phi_e)
    Q = bending_kernel(beta, kphi, phi_e)
    w = tm_angular_weights(thr, f, beta, De, a, le, Q)
    out = np.empty(int(N))
    for i in range(int(N)):
        lm, lb = _stationary(w[i], f, beta, De, a, le)
        out[i] = beta * float(tm_pmf(lb, w[i], f, beta, De, a, le)[0]
                              - tm_pmf(lm, w[i], f, beta, De, a, le)[0])
    return out

"""Step 8 (final orchestrator): finite-bending chain-scission audit."""

import numpy as np

_DE, _AA, _LE, _BETA = 1.0, 2.15, 1.0, 279.0
_PHI_E = 69.0 * np.pi / 180.0
_KPHI = 1820.0 / (np.pi ** 2 * 279.0)
_BASE = {1: (5, 0.20), 2: (4, 0.15), 3: (3, 0.25)}


def scission_audit(force_scale):
    if isinstance(force_scale, bool) or not np.isfinite(force_scale) or force_scale <= 0:
        raise ValueError("force_scale must be positive and finite")
    rows = []
    for v in (1, 2, 3):
        N, f0 = _BASE[v]
        f = f0 * force_scale
        thr = self_consistent_thresholds(N, f, _BETA, _DE, _AA, _LE, _KPHI, _PHI_E)
        bb = bond_barriers(N, f, _BETA, _DE, _AA, _LE, _KPHI, _PHI_E)
        rows.append([float(thr[0]), float(thr[N // 2]), float(bb[0]),
                     float(bb.max()), float(bb.sum())])
    return np.asarray(rows, dtype=np.float64)
SCICODE_GOLD_EOF
