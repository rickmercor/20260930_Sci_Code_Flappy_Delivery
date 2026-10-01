#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

"""Step 1: normal fracture deformation relation, linear and Barton-Bandis (Eq. 15)."""

import numpy as np


def normal_stiffness(q_n, K_n, du_max, model):
    q = np.asarray(q_n, dtype=np.float64)
    if model == "Lin":
        return q / K_n
    if model == "BB":
        den = K_n - q / du_max
        if np.any(den <= 0):
            raise ValueError("Barton-Bandis denominator must stay positive")
        return q / den
    raise ValueError("model must be 'Lin' or 'BB'")

"""Step 2: normal contact traction from the displacement jump (Eqs. 9, 12, 15)."""

import numpy as np


def normal_traction(jump_n, K_n, du_max, model, contact):
    jn = np.asarray(jump_n, dtype=np.float64)
    if model == "Lin":
        q = K_n * jn
    elif model == "BB":
        fac = 1.0 + jn / du_max
        if np.any(fac <= 0):
            raise ValueError("closure at or beyond the maximum allowed closure")
        q = K_n * jn / fac
    else:
        raise ValueError("model must be 'Lin' or 'BB'")
    if contact:
        q = np.where(jn >= 0.0, 0.0, q)
    return q

"""Step 3: Coulomb friction bound (Eq. 10)."""

import numpy as np


def friction_bound(q_n, F):
    q = np.asarray(q_n, dtype=np.float64)
    if q.size == 0 or not np.all(np.isfinite(q)):
        raise ValueError("q_n must be finite and non-empty")
    Ff = float(F)
    if not np.isfinite(Ff) or Ff < 0.0:
        raise ValueError("F must be finite and non-negative")
    return -Ff * q

"""Step 4: radial-return map for the tangential traction (Eq. 13)."""

import numpy as np


def coulomb_return(q_tau, jump_vel_tau, b, c):
    if not np.isfinite(c) or c <= 0:
        raise ValueError("c must be positive and finite")
    qt = np.asarray(q_tau, dtype=np.float64)
    dv = np.asarray(jump_vel_tau, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    trial = qt + c * dv
    nrm = np.abs(trial)
    cap = np.maximum(bb, 0.0)
    fac = np.ones_like(trial)
    nz = nrm > 0
    fac[nz] = np.minimum(1.0, cap[nz] / nrm[nz])
    return fac * trial

"""Step 5: fracture opening, slip tendency and contact state (Section 2.3.3, Eq. 12)."""

import numpy as np


def _gn(q, K_n, du_max, model):
    q = np.asarray(q, dtype=np.float64)
    if model == "Lin":
        return q / K_n
    if model == "BB":
        den = K_n - q / du_max
        if np.any(den <= 0):
            raise ValueError("Barton-Bandis denominator must stay positive")
        return q / den
    raise ValueError("model must be 'Lin' or 'BB'")


def contact_state(jump_n, q_n, q_tau, K_n, du_max, model, F):
    jn = np.asarray(jump_n, dtype=np.float64)
    qn = np.asarray(q_n, dtype=np.float64)
    qt = np.asarray(q_tau, dtype=np.float64)
    if jn.shape != qn.shape or jn.shape != qt.shape or jn.size == 0:
        raise ValueError("jump_n, q_n and q_tau must share one non-empty shape")
    if not (np.all(np.isfinite(jn)) and np.all(np.isfinite(qn)) and np.all(np.isfinite(qt))):
        raise ValueError("non-finite input")
    if model not in ("Lin", "BB"):
        raise ValueError("model must be 'Lin' or 'BB'")
    if not np.isfinite(K_n) or K_n <= 0.0 or not np.isfinite(du_max) or du_max <= 0.0:
        raise ValueError("K_n and du_max must be positive and finite")
    if not np.isfinite(F) or F < 0.0:
        raise ValueError("F must be finite and non-negative")
    delta = jn - _gn(qn, K_n, du_max, model)
    b = -F * qn
    s = np.zeros(jn.shape)
    nz = np.abs(b) > 0
    s[nz] = np.abs(qt[nz]) / np.abs(b[nz])
    state = np.ones(jn.shape)
    state[qn == 0.0] = 0.0
    contact = qn != 0.0
    state[contact & (s >= 1.0)] = 2.0
    return np.vstack([delta, s, state])

"""Step 6: Newmark velocity and acceleration update (Eqs. 20-21)."""

import numpy as np


def newmark_update(u_new, u_old, v_old, a_old, dt, beta, gamma):
    if not np.isfinite(dt) or dt <= 0 or beta <= 0:
        raise ValueError("dt and beta must be positive")
    un = np.asarray(u_new, dtype=np.float64)
    uo = np.asarray(u_old, dtype=np.float64)
    vo = np.asarray(v_old, dtype=np.float64)
    ao = np.asarray(a_old, dtype=np.float64)
    v_new = ((1.0 - gamma / beta) * vo
             + dt * (1.0 - gamma / (2.0 * beta)) * ao
             + (gamma / (beta * dt)) * (un - uo))
    a_new = (un - uo - dt * vo - (1.0 - 2.0 * beta) * (dt ** 2 / 2.0) * ao) / (beta * dt ** 2)
    return np.vstack([v_new, a_new])

"""Step 7: absorbing boundary coefficient matrix (Eq. 19)."""

import numpy as np


def absorbing_matrix(rho, lam, mu, n):
    nv = np.asarray(n, dtype=np.float64)
    nn = float(nv @ nv)
    if not np.isfinite(nn) or abs(nn - 1.0) > 1e-10:
        raise ValueError("n must be a unit vector")
    P = np.outer(nv, nv)
    I = np.eye(nv.size)
    return np.sqrt(rho * (lam + 2.0 * mu)) * P + np.sqrt(rho * mu) * (I - P)

"""Step 8 (final orchestrator): fracture deformation model audit."""

import numpy as np

_KN, _KT, _DUMAX, _F = 2.0e11, 2.0e11, 5.0e-5, 1.0
_C = 3.1e6
_RHO, _LAM, _MU = 2600.0, 4.0e9, 4.0e9
_NBND = np.array([1.0, 0.0])
_DT, _BETA, _GAMMA = 1.0e-7, 0.25, 0.5
_JN = np.array([1.8e-5, -1.2e-5, -2.4e-5, 0.6e-5, -3.0e-5, -0.9e-5])
_JT = np.array([0.8e-5, -1.1e-5, 1.6e-5, -0.4e-5, 2.2e-5, 0.5e-5])
_DVT = np.array([0.8, -1.1, 1.6, -0.4, 2.2, 0.5])
_V0 = np.array([0.2, -0.4, 0.6, 0.1, -0.3, 0.5])
_A0 = np.array([1.0e6, 0.0, -1.0e6, 2.0e6, 5.0e5, -5.0e5])
_VARIANTS = (("Lin", False), ("BB", False), ("Lin", True), ("BB", True))


def fracture_audit(load_scale):
    if isinstance(load_scale, bool) or not np.isfinite(load_scale) or load_scale <= 0:
        raise ValueError("load_scale must be positive and finite")
    ls = float(load_scale)
    jn = ls * _JN
    jt = ls * _JT
    dvt = ls * _DVT
    D = absorbing_matrix(_RHO, _LAM, _MU, _NBND)
    rows = []
    for model, contact in _VARIANTS:
        qn = normal_traction(jn, _KN, _DUMAX, model, contact)
        b = friction_bound(qn, _F)
        if contact:
            qt = coulomb_return(np.zeros_like(jt), dvt, b, _C)
        else:
            qt = _KT * jt
        st = contact_state(jn, qn, qt, _KN, _DUMAX, model, _F)
        delta, s, state = st[0], st[1], st[2]
        max_s = float(np.max(s)) if s.size else 0.0
        vt = newmark_update(jt + qt / _KT, jt, _V0, _A0, _DT, _BETA, _GAMMA)[0]
        vn = newmark_update(jn + qn / _KN, jn, _V0, _A0, _DT, _BETA, _GAMMA)[0]
        tabs = 0.0
        for i in range(vt.size):
            tabs += float(np.linalg.norm(D @ np.array([vn[i], vt[i]])))
        gn = normal_stiffness(qn, _KN, _DUMAX, model)
        rows.append([float(np.sum(qn)), float(np.sum(np.abs(qt))), float(np.sum(delta)),
                     max_s, float(np.sum(state == 0.0)), float(np.sum(state == 1.0)),
                     float(np.sum(state == 2.0)), float(np.linalg.norm(vt)),
                     tabs, float(np.sum(gn))])
    return np.asarray(rows, dtype=np.float64)
SCICODE_GOLD_EOF
