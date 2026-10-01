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
from scipy.stats import norm


def bs_call_price(spot: "float | np.ndarray", strike: "float | np.ndarray", total_var: "float | np.ndarray") -> "np.ndarray":
    """Paper eq (3). total_var is TOTAL variance v, not a volatility.

    Call(s,k,v) = s N(d+) - k N(d-),  d+- = (-ln(k/s) +- v/2) / sqrt(v)
    """
    sp = np.asarray(spot, dtype=np.float64)
    st = np.asarray(strike, dtype=np.float64)
    tv = np.asarray(total_var, dtype=np.float64)
    if np.any(tv < 0.0):
        raise ValueError("total_var must be non-negative")
    if np.any(st < 0.0) or np.any(sp <= 0.0):
        raise ValueError("spot must be positive and strike non-negative")
    if not (np.all(np.isfinite(sp)) and np.all(np.isfinite(st)) and np.all(np.isfinite(tv))):
        raise ValueError("spot, strike and total_var must all be finite")

    shape = np.broadcast(sp, st, tv).shape
    s, k, v = (np.atleast_1d(np.broadcast_to(x, shape)).astype(np.float64)
               for x in (sp, st, tv))
    out = np.maximum(s - k, 0.0).astype(np.float64)
    live = (v > 0.0) & (k > 0.0) & (s > 0.0)
    if np.any(live):
        sv = np.sqrt(v[live])
        dp = (-np.log(k[live] / s[live]) + 0.5 * sv ** 2) / sv
        dm = (-np.log(k[live] / s[live]) - 0.5 * sv ** 2) / sv
        out[live] = s[live] * norm.cdf(dp) - k[live] * norm.cdf(dm)
    return out.reshape(shape)

import numpy as np
from scipy.stats import norm


def _ladder(strikes):
    """Validate and return a strike ladder. Shared by several steps."""
    K = np.asarray(strikes, dtype=np.float64)
    if K.ndim != 1 or K.size == 0:
        raise ValueError("strikes must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(K)):
        raise ValueError("strikes must all be finite")
    if np.any(K <= 0.0):
        raise ValueError("strikes must be strictly positive")
    if K.size > 1 and not np.all(np.diff(K) > 0.0):
        raise ValueError("strikes must be strictly increasing")
    return K


def model_price_matrix(strikes: "np.ndarray", atm_var: float, eta: float) -> "np.ndarray":
    """C[l,i] = Call(K^i, K^l, eta*V) for INTERIOR rows l = 1..N-2. Shape (N-2, N)."""
    K = _ladder(strikes)
    if K.size < 3:
        raise ValueError("the ladder needs at least three strikes to have an interior")
    av, e = float(atm_var), float(eta)
    if not np.isfinite(av) or av < 0.0:
        raise ValueError("atm_var must be finite and non-negative")
    if not np.isfinite(e) or e < 0.0:
        raise ValueError("eta must be finite and non-negative")
    rows = np.arange(1, K.size - 1)
    C = np.empty((rows.size, K.size), dtype=np.float64)
    for a, l in enumerate(rows):
        C[a, :] = bs_call_price(K, K[l], e * av)
    return C

import numpy as np


def _ladder(strikes):
    """Validate and return a strike ladder. Shared by several steps."""
    K = np.asarray(strikes, dtype=np.float64)
    if K.ndim != 1 or K.size == 0:
        raise ValueError("strikes must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(K)):
        raise ValueError("strikes must all be finite")
    if np.any(K <= 0.0):
        raise ValueError("strikes must be strictly positive")
    if K.size > 1 and not np.all(np.diff(K) > 0.0):
        raise ValueError("strikes must be strictly increasing")
    return K


def payoff_matrix(strikes: "np.ndarray") -> "np.ndarray":
    """U[l,i] = (K^i - K^l)+ . Intrinsic payoff, ALL rows. Shape (N, N)."""
    K = _ladder(strikes)
    U = np.maximum(K[None, :] - K[:, None], 0.0).astype(np.float64)
    if U.shape[0] != U.shape[1]:
        raise ValueError("the ordering map must be square on the ladder")
    return U

import numpy as np


def _ladder(strikes):
    """Validate and return a strike ladder. Shared by several steps."""
    K = np.asarray(strikes, dtype=np.float64)
    if K.ndim != 1 or K.size == 0:
        raise ValueError("strikes must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(K)):
        raise ValueError("strikes must all be finite")
    if np.any(K <= 0.0):
        raise ValueError("strikes must be strictly positive")
    if K.size > 1 and not np.all(np.diff(K) > 0.0):
        raise ValueError("strikes must be strictly increasing")
    return K


def marginal_constraints(strikes: "np.ndarray") -> "tuple[np.ndarray, np.ndarray]":
    """Two equality rows per expiry: 1'q = 1 (mass) and K'q = 1 (unit mean)."""
    K = _ladder(strikes)
    A = np.vstack([np.ones_like(K), K]).astype(np.float64)
    b = np.array([1.0, 1.0], dtype=np.float64)
    if A.shape[0] != b.size or A.shape[1] != K.size:
        raise ValueError("the equality block and its right-hand side disagree in shape")
    return A, b

import numpy as np
from scipy.optimize import linprog


def solve_marginals(price_matrices: "np.ndarray", ordering_map: "np.ndarray", constraint_block: "tuple[np.ndarray, np.ndarray]", mid_quotes: "np.ndarray", weights: "np.ndarray") -> "np.ndarray":
    """L1 fit of the marginals under mass, unit mean and the cross-expiry ordering."""
    C = np.asarray(price_matrices, dtype=np.float64)
    U = np.asarray(ordering_map, dtype=np.float64)
    Q = np.asarray(mid_quotes, dtype=np.float64)
    W = np.asarray(weights, dtype=np.float64)
    if C.ndim != 3:
        raise ValueError("price_matrices must be stacked per expiry with shape (M, R, N)")
    M, R, N = C.shape
    if U.shape != (N, N):
        raise ValueError(f"ordering_map must be ({N}, {N}), got {U.shape}")
    if Q.shape != (M, N) or W.shape != (M, N):
        raise ValueError(f"mid_quotes and weights must both be ({M}, {N})")
    Aeq_blk, beq_blk = constraint_block
    Aeq_blk = np.asarray(Aeq_blk, dtype=np.float64)
    beq_blk = np.asarray(beq_blk, dtype=np.float64)
    if Aeq_blk.ndim != 2 or Aeq_blk.shape[1] != N:
        raise ValueError(f"constraint block must have {N} columns")
    if R != N - 2:
        raise ValueError("price_matrices must carry the interior row set")

    rows = np.arange(1, N - 1)
    nq, nt = M * N, M * R
    qs = lambda j: slice(j * N, (j + 1) * N)
    ts = lambda j: slice(nq + j * R, nq + (j + 1) * R)

    c = np.zeros(nq + nt, dtype=np.float64)
    c[nq:] = 1.0
    Aub, bub = [], []
    for j in range(M):
        for sgn in (1.0, -1.0):
            A = np.zeros((R, nq + nt))
            A[:, qs(j)] = sgn * (W[j, rows][:, None] * C[j])
            A[:, ts(j)] = -np.eye(R)
            Aub.append(A)
            bub.append(sgn * (W[j, rows] * Q[j, rows]))
    for j in range(1, M):
        A = np.zeros((N, nq + nt))
        A[:, qs(j - 1)] = U
        A[:, qs(j)] = -U
        Aub.append(A)
        bub.append(np.zeros(N))
    nr = Aeq_blk.shape[0]
    Aeq, beq = [], []
    for j in range(M):
        Z = np.zeros((nr, nq + nt))
        Z[:, qs(j)] = Aeq_blk
        Aeq.append(Z)
        beq.append(beq_blk)

    res = linprog(c, A_ub=np.vstack(Aub), b_ub=np.concatenate(bub),
                  A_eq=np.vstack(Aeq), b_eq=np.concatenate(beq),
                  bounds=[(0.0, None)] * (nq + nt), method="highs")
    if not res.success:
        return np.zeros((M, N), dtype=np.float64)
    return np.asarray(res.x[:nq].reshape(M, N), dtype=np.float64)

import numpy as np
from scipy.stats import norm


def _ladder(strikes):
    """Validate and return a strike ladder. Shared by several steps."""
    K = np.asarray(strikes, dtype=np.float64)
    if K.ndim != 1 or K.size == 0:
        raise ValueError("strikes must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(K)):
        raise ValueError("strikes must all be finite")
    if np.any(K <= 0.0):
        raise ValueError("strikes must be strictly positive")
    if K.size > 1 and not np.all(np.diff(K) > 0.0):
        raise ValueError("strikes must be strictly increasing")
    return K


def smooth_surface_price(strikes: "np.ndarray", expiries: "np.ndarray", atm_vars: "np.ndarray", q: "np.ndarray", t_query: float, k_query: float, eta: float) -> float:
    """Paper eq (2): blend the two bracketing marginals, BOTH priced at the
    INTERPOLATED dispersion V(T) = a*V_j + (1-a)*V_{j-1}, scaled by eta."""
    K = _ladder(strikes)
    T = np.asarray(expiries, dtype=np.float64)
    V = np.asarray(atm_vars, dtype=np.float64)
    Qd = np.asarray(q, dtype=np.float64)
    if T.ndim != 1 or T.size == 0 or not np.all(np.diff(T) > 0.0):
        raise ValueError("expiries must be a non-empty strictly increasing array")
    if V.shape != T.shape:
        raise ValueError("atm_vars must have one entry per expiry node")
    if Qd.shape != (T.size, K.size):
        raise ValueError(f"q must be ({T.size}, {K.size}), got {Qd.shape}")
    tq, kq, e = float(t_query), float(k_query), float(eta)
    if not (np.isfinite(tq) and np.isfinite(kq)) or kq <= 0.0 or tq <= 0.0:
        raise ValueError("t_query and k_query must be finite and strictly positive")

    j = int(np.searchsorted(T, tq))
    if j <= 0:
        return float(bs_call_price(K, kq, e * V[0]) @ Qd[0])
    if j >= T.size:
        return float(bs_call_price(K, kq, e * V[-1]) @ Qd[-1])
    a = (min(tq, T[j]) - T[j - 1]) / (T[j] - T[j - 1])
    vt = a * V[j] + (1.0 - a) * V[j - 1]
    px = bs_call_price(K, kq, e * vt)
    return float(a * (px @ Qd[j]) + (1.0 - a) * (px @ Qd[j - 1]))

import numpy as np
from scipy.stats import norm


def _ladder(strikes):
    """Validate and return a strike ladder. Shared by several steps."""
    K = np.asarray(strikes, dtype=np.float64)
    if K.ndim != 1 or K.size == 0:
        raise ValueError("strikes must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(K)):
        raise ValueError("strikes must all be finite")
    if np.any(K <= 0.0):
        raise ValueError("strikes must be strictly positive")
    if K.size > 1 and not np.all(np.diff(K) > 0.0):
        raise ValueError("strikes must be strictly increasing")
    return K


def discrete_local_variance(strikes: "np.ndarray", expiries: "np.ndarray", atm_vars: "np.ndarray", q: "np.ndarray", t_query: float, k_query: float, eta: float, h_rel: float = 0.02) -> float:
    """Local-variance ratio on the smooth surface: centred RELATIVE strike bump, and a
    FORWARD one-sided time difference toward the next expiry node."""
    if not np.isfinite(h_rel) or h_rel <= 0.0:
        raise ValueError("h_rel must be finite and strictly positive")
    T = np.asarray(expiries, dtype=np.float64)
    tq, kq = float(t_query), float(k_query)
    h = h_rel * kq
    f = lambda t, k: smooth_surface_price(strikes, expiries, atm_vars, q, t, k, eta)

    j = int(np.searchsorted(T, tq))
    t_up = float(T[min(j, T.size - 1)])
    dt = t_up - tq
    if dt <= 0.0:
        return 0.0
    dCdT = (f(t_up, kq) - f(tq, kq)) / dt
    d2CdK2 = (f(tq, kq + h) - 2.0 * f(tq, kq) + f(tq, kq - h)) / (h * h)
    if d2CdK2 <= 0.0:
        return 0.0
    return float(2.0 * dCdT / (kq * kq * d2CdK2))

import numpy as np
from scipy.stats import norm
from scipy.optimize import linprog


def _ladder(strikes):
    """Validate and return a strike ladder. Shared by several steps."""
    K = np.asarray(strikes, dtype=np.float64)
    if K.ndim != 1 or K.size == 0:
        raise ValueError("strikes must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(K)):
        raise ValueError("strikes must all be finite")
    if np.any(K <= 0.0):
        raise ValueError("strikes must be strictly positive")
    if K.size > 1 and not np.all(np.diff(K) > 0.0):
        raise ValueError("strikes must be strictly increasing")
    return K


def sanos_surface_audit(scenario: int) -> "np.ndarray":
    """Chain steps 1-7 and return a (n_probe, 2) matrix of computed quantities:
    [smooth model value, local variance] per probe point."""
    s = int(scenario)
    if s < 0:
        raise ValueError("scenario must be a non-negative integer")

    K = np.array([0.70, 0.80, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20, 1.30])
    T = np.array([0.20, 0.50, 1.00])
    V = np.array([0.040, 0.075, 0.140])
    eta = 0.25

    shift = 0.002 * ((s % 5) - 2)
    Vq = np.array([V[j] + 0.030 * (1.0 - K) for j in range(T.size)])
    Vq[1] -= (0.022 + shift) * np.exp(-((K - 1.00) / 0.16) ** 2)
    Vq[2] -= (0.030 + shift) * np.exp(-((K - 1.05) / 0.14) ** 2)
    mids = np.vstack([bs_call_price(np.ones_like(K), K, Vq[j])
                      for j in range(T.size)])
    W = np.ones_like(mids)

    # chain the maps explicitly so every earlier step is reached from here
    C = np.stack([model_price_matrix(K, V[j], eta) for j in range(T.size)])
    U = payoff_matrix(K)
    blk = marginal_constraints(K)
    q = solve_marginals(C, U, blk, mids, W)

    probes = [(0.75, 1.075), (0.40, 0.925), (0.90, 1.150), (0.30, 0.850)]
    out = np.zeros((len(probes), 2), dtype=np.float64)
    for r, (tq, kq) in enumerate(probes):
        out[r, 0] = smooth_surface_price(K, T, V, q, tq, kq, eta)
        out[r, 1] = discrete_local_variance(K, T, V, q, tq, kq, eta)
    return out
SCICODE_GOLD_EOF
