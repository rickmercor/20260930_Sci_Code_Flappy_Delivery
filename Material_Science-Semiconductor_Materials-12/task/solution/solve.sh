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


def _checked_params(params):
    """Return params as a validated float array of shape (7,)."""
    import numpy as np
    try:
        p = np.asarray(params, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("params must be a real array")
    if p.shape != (7,) or not np.all(np.isfinite(p)):
        raise ValueError("params must be a finite array of shape (7,)")
    if p[2] <= 0.0 or p[3] <= 0.0 or p[5] <= 0.0 or p[6] <= 0.0:
        raise ValueError("lead widths and strengths must be positive")
    return p


def _real_float(x, name):
    """Return x as a finite float or raise ValueError."""
    import numpy as np
    try:
        v = float(x)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a real number")
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite")
    return v


def reaction_coordinate_drift(params: np.ndarray, g1: float, g2: float) -> np.ndarray:
    """Reference implementation."""
    import numpy as np
    p = _checked_params(params)
    g1 = _real_float(g1, "g1")
    g2 = _real_float(g2, "g2")
    if not (0.0 <= g1 <= 1.0 and 0.0 <= g2 <= 1.0):
        raise ValueError("switching factors must lie in [0, 1]")
    eps, w1, l1, G1, w2, l2, G2 = p
    m = -np.diag([1j * eps, l1 + 1j * w1, l2 + 1j * w2]).astype(complex)
    k1 = g1 * np.sqrt(G1 * l1 / 2.0)
    k2 = g2 * np.sqrt(G2 * l2 / 2.0)
    m[0, 1] += -1j * k1
    m[1, 0] += -1j * k1
    m[0, 2] += -1j * k2
    m[2, 0] += -1j * k2
    return m

import numpy as np
from scipy.linalg import expm


def segment_response(drift: np.ndarray, tau: float, omega: np.ndarray) -> np.ndarray:
    """Reference implementation."""
    import numpy as np
    from scipy.linalg import expm
    try:
        m = np.asarray(drift, dtype=complex)
        om = np.asarray(omega, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("drift and omega must be numeric arrays")
    if m.shape != (3, 3) or not np.all(np.isfinite(m)):
        raise ValueError("drift must be a finite 3 x 3 array")
    if np.max(np.linalg.eigvals(m).real) >= 0.0:
        raise ValueError("every eigenvalue of the drift must have negative real part")
    tau = _real_float(tau, "tau")
    if tau < 0.0:
        raise ValueError("tau must be non-negative")
    if om.ndim != 1 or not np.all(np.isfinite(om)):
        raise ValueError("omega must be a finite one-dimensional array")
    eye = np.eye(3)
    em = expm(m * tau)
    lhs = m[None, :, :] + 1j * om[:, None, None] * eye[None, :, :]
    rhs = em[None, :, :] - np.exp(-1j * om * tau)[:, None, None] * eye[None, :, :]
    return np.linalg.solve(lhs, rhs)

import numpy as np
from scipy.linalg import expm


def cycle_propagator(params: np.ndarray, t1: float, t2: float) -> np.ndarray:
    """Reference implementation."""
    import numpy as np
    from scipy.linalg import expm
    t1 = _real_float(t1, "t1")
    t2 = _real_float(t2, "t2")
    if t1 <= 0.0 or t2 <= 0.0:
        raise ValueError("step durations must be positive")
    m1 = reaction_coordinate_drift(params, 1.0, 0.0)
    m2 = reaction_coordinate_drift(params, 0.0, 1.0)
    return expm(m2 * t2) @ expm(m1 * t1)

import numpy as np
from scipy.linalg import expm


def limit_cycle_weights(params: np.ndarray, t1: float, t2: float, omega: np.ndarray,
                                lead: int) -> np.ndarray:
    """Reference implementation."""
    import numpy as np
    from scipy.linalg import expm
    if isinstance(lead, (bool, np.bool_)) or not isinstance(lead, (int, np.integer)) or int(lead) not in (1, 2):
        raise ValueError("lead must be 1 or 2")
    nu = int(lead)
    a = cycle_propagator(params, t1, t2)
    t1 = float(t1)
    t2 = float(t2)
    m1 = reaction_coordinate_drift(params, 1.0, 0.0)
    m2 = reaction_coordinate_drift(params, 0.0, 1.0)
    om = np.asarray(omega, dtype=float)
    s1 = segment_response(m1, t1, om)[:, :, nu]
    s2 = segment_response(m2, t2, om)[:, :, nu]
    e1 = expm(m1 * t1)
    e2 = expm(m2 * t2)
    period = t1 + t2
    u_cycle = np.einsum('nj,ij->ni', s1, e2) + np.exp(-1j * om * t1)[:, None] * s2
    eye = np.eye(3)
    lhs = eye[None, :, :] - np.exp(1j * om * period)[:, None, None] * a[None, :, :]
    w_end2 = np.linalg.solve(lhs, u_cycle[:, :, None])[:, :, 0]
    w_end1 = np.einsum('nj,ij->ni', w_end2, e1) + np.exp(-1j * om * period)[:, None] * s1
    out = np.empty((om.size, 2, 3, 3), dtype=complex)
    out[:, 0] = np.conj(w_end2)[:, :, None] * w_end2[:, None, :]
    out[:, 1] = np.conj(w_end1)[:, :, None] * w_end1[:, None, :]
    return out

import numpy as np
from scipy.linalg import expm


def _fermi_sea_nodes(mu, cutoff, near=30.0, near_panel=0.04, far_panel=0.4, order=8):
    """Composite Gauss-Legendre nodes and weights on [mu - cutoff, mu], finer within `near` of mu."""
    import numpy as np
    x, w = np.polynomial.legendre.leggauss(order)
    edges_near = np.linspace(mu - near, mu, int(round(near / near_panel)) + 1)
    edges_far = np.linspace(mu - cutoff, mu - near, int(round((cutoff - near) / far_panel)) + 1)
    edges = np.unique(np.concatenate([edges_far, edges_near]))
    lo = edges[:-1]
    hi = edges[1:]
    nodes = ((hi - lo) / 2)[:, None] * x[None, :] + ((hi + lo) / 2)[:, None]
    weights = ((hi - lo) / 2)[:, None] * w[None, :]
    return nodes.ravel(), weights.ravel()


def limit_cycle_correlations(params: np.ndarray, t1: float, t2: float, bias: float) -> np.ndarray:
    """Reference implementation."""
    import numpy as np
    p = _checked_params(params)
    v = _real_float(bias, "bias")
    if v <= 0.0:
        raise ValueError("bias must be positive")
    z_max = np.max(np.abs(np.linalg.eigvals(cycle_propagator(p, t1, t2))))
    scale = -np.log(z_max) / (float(t1) + float(t2))
    near_panel = min(0.04, 0.5 * scale)
    far_panel = min(0.4, 5.0 * scale)
    widths = {1: p[2], 2: p[5]}
    potentials = {1: 0.0, 2: v}
    estimates = []
    for cutoff in (400.0, 800.0):
        corr = np.zeros((2, 3, 3), dtype=complex)
        for nu in (1, 2):
            om, wq = _fermi_sea_nodes(potentials[nu], cutoff, near_panel=near_panel, far_panel=far_panel)
            q = limit_cycle_weights(p, t1, t2, om, nu)
            corr += (2.0 * widths[nu] / (2.0 * np.pi)) * np.tensordot(wq, q, axes=(0, 0))
        estimates.append(corr)
    # the omitted tail beyond the cutoff decays as cutoff**-2
    best = estimates[1] + (estimates[1] - estimates[0]) / 3.0
    return np.ascontiguousarray(best[:, :, 0])

import numpy as np
from scipy.linalg import expm


def pumping_performance(params: np.ndarray, t1: float, t2: float, bias: float) -> np.ndarray:
    """Reference implementation."""
    import numpy as np
    p = _checked_params(params)
    corr = limit_cycle_correlations(p, t1, t2, bias)
    v = float(bias)
    m_both = reaction_coordinate_drift(p, 1.0, 1.0)
    kappa1 = abs(m_both[0, 1])
    kappa2 = abs(m_both[0, 2])
    end2, end1 = corr[0], corr[1]
    n_pump = end1[0].real - end2[0].real
    w_a = 2.0 * kappa2 * end1[2].real - 2.0 * kappa1 * end1[1].real
    w_b = 2.0 * kappa1 * end2[1].real - 2.0 * kappa2 * end2[2].real
    return np.array([n_pump, w_a, w_b, n_pump * v / (w_a + w_b)])

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize_scalar


def maximal_pumping_efficiency(params: np.ndarray, t1: float, t2_min: float, t2_max: float,
                                       bias: float) -> float:
    """Reference implementation."""
    import numpy as np
    from scipy.optimize import minimize_scalar
    lo = _real_float(t2_min, "t2_min")
    hi = _real_float(t2_max, "t2_max")
    if not 0.0 < lo < hi:
        raise ValueError("the step-2 window must satisfy 0 < t2_min < t2_max")
    efficiency = lambda t2: pumping_performance(params, t1, t2, bias)[3]
    grid = np.linspace(lo, hi, int(np.ceil((hi - lo) / 0.25)) + 1)
    values = np.array([efficiency(x) for x in grid])
    best = int(np.argmax(values))
    a = grid[max(best - 1, 0)]
    b = grid[min(best + 1, grid.size - 1)]
    res = minimize_scalar(lambda x: -efficiency(x), bounds=(a, b), method="bounded", options={"xatol": 1e-6})
    return float(max(-res.fun, values[best]))
SCICODE_GOLD_EOF
