#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

"""Step 1: 1D dual-horizon bond micro-modulus (Eq. 25)."""

import numpy as np


def micro_modulus(delta, E, A):
    d = np.asarray(delta, dtype=np.float64)
    if np.any(d <= 0) or not np.all(np.isfinite(d)):
        raise ValueError("horizon must be positive and finite")
    return E / (d ** 2 * A)

"""Step 2: family membership under spatially varying horizons (Eq. 1)."""

import numpy as np


def families(X, delta):
    X = np.asarray(X, dtype=np.float64)
    d = np.asarray(delta, dtype=np.float64)
    R = np.abs(X[None, :] - X[:, None])
    H = (R <= d[:, None]).astype(np.float64)
    np.fill_diagonal(H, 0.0)
    return H

"""Step 3: bond stretch (Eq. 22)."""

import numpy as np


def bond_stretch(X, u):
    X = np.asarray(X, dtype=np.float64)
    x = X + np.asarray(u, dtype=np.float64)
    Xij = X[None, :] - X[:, None]
    xij = x[None, :] - x[:, None]
    d0 = np.abs(Xij)
    with np.errstate(divide='ignore', invalid='ignore'):
        s = (np.abs(xij) - d0) / d0
    np.fill_diagonal(s, 0.0)
    return s

"""Step 4: bond damage indicator (Eq. 27)."""

import numpy as np


def damage_state(s, H, sc):
    s = np.asarray(s, dtype=np.float64)
    Hb = np.asarray(H, dtype=np.float64) > 0
    mu = (s < sc).astype(np.float64)
    mu[~Hb] = 0.0
    return mu

"""Step 5: local damage at a point (Eq. 30)."""

import numpy as np


def point_damage(mu, H, V):
    mu = np.asarray(mu, dtype=np.float64)
    Hb = (np.asarray(H, dtype=np.float64) > 0).astype(np.float64)
    V = np.asarray(V, dtype=np.float64)
    num = (mu * Hb * V[None, :]).sum(axis=1)
    den = (Hb * V[None, :]).sum(axis=1)
    out = np.ones_like(num)
    nz = den > 0
    out[nz] = 1.0 - num[nz] / den[nz]
    return out

"""Step 6: dual-horizon internal force density (Eq. 17, 11, 24)."""

import numpy as np

_E = 1.0
_A = 1.0


def _stretch(X, u):
    X = np.asarray(X, dtype=np.float64)
    x = X + np.asarray(u, dtype=np.float64)
    R = np.abs(X[None, :] - X[:, None])
    r = np.abs(x[None, :] - x[:, None])
    with np.errstate(divide='ignore', invalid='ignore'):
        s = np.where(R > 0, (r - R) / np.where(R > 0, R, 1.0), 0.0)
    return s


def internal_force(X, u, V, delta, H, mu):
    X = np.asarray(X, dtype=np.float64)
    x = X + np.asarray(u, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    c = micro_modulus(delta, _E, _A)
    Hb = (np.asarray(H, dtype=np.float64) > 0).astype(np.float64)
    Hd = Hb.T
    mu = np.asarray(mu, dtype=np.float64)
    s = _stretch(X, u)
    xhat = np.sign(x[None, :] - x[:, None])
    A1 = (Hb * mu * s * xhat) * (c[None, :] * V[None, :])
    A2 = (Hd * mu.T * s * xhat) * (V[None, :] * c[:, None])
    return A1.sum(axis=1) + A2.sum(axis=1)

"""Step 7: critical time step (Eq. 54)."""

import numpy as np

_E = 1.0
_A = 1.0


def critical_time_step(X, V, delta, H, rho):
    X = np.asarray(X, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    c = micro_modulus(delta, _E, _A)
    Hb = (np.asarray(H, dtype=np.float64) > 0)
    R = np.abs(X[None, :] - X[:, None])
    with np.errstate(divide='ignore', invalid='ignore'):
        term = np.where(Hb, c[None, :] * V[None, :] / np.where(R > 0, R, 1.0), 0.0)
    den = term.sum(axis=1)
    if np.any(den <= 0):
        raise ValueError("point with no family members")
    return float(np.min(np.sqrt(2.0 * rho / den)))

"""Step 8 (final orchestrator): dual-horizon fracture audit."""

import numpy as np

_E = 1.0
_A = 1.0
_RHO = 1.0
_SC_BASE = 0.16125
_PAR = {1: (4, 4, 2.515, 3, 0.10), 2: (3, 5, 2.515, 5, 0.08), 3: (5, 3, 3.015, 7, 0.10)}


def _build(variant):
    nC, nF, m, a, dC = _PAR[variant]
    dF = dC / 2.0
    xs = [0.0]
    for _ in range(nC - 1):
        xs.append(xs[-1] + dC)
    for _ in range(nF):
        xs.append(xs[-1] + dF)
    X = np.array(xs, dtype=np.float64)
    N = X.size
    sp = np.empty(N)
    sp[:nC] = dC
    sp[nC:] = dF
    sp[nC - 1] = 0.5 * (dC + dF)
    V = sp * _A
    delta = m * sp
    pert = np.array([(((i + 1) * (i + 2) + a * (i + 3)) % 17) - 8 for i in range(N)], dtype=np.float64)
    u = 0.010 * X + 0.0015 * pert
    return X, V, delta, u


def pd_fracture_audit(sc_scale):
    if isinstance(sc_scale, bool) or not np.isfinite(sc_scale) or sc_scale <= 0:
        raise ValueError("sc_scale must be positive and finite")
    rows = []
    for variant in (1, 2, 3):
        X, V, delta, u = _build(variant)
        H = families(X, delta)
        s = bond_stretch(X, u)
        mu = damage_state(s, H, _SC_BASE * sc_scale)
        F = internal_force(X, u, V, delta, H, mu)
        D = point_damage(mu, H, V)
        hc = critical_time_step(X, V, delta, H, _RHO)
        rows.append([float(np.sum(F * X * V)), hc, float(np.mean(D)), float(np.sum(mu))])
    return np.asarray(rows, dtype=np.float64)
SCICODE_GOLD_EOF
