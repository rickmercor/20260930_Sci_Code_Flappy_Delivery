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


def density_pair(P_i: np.ndarray, P_f: np.ndarray) -> np.ndarray:
    P_i = np.asarray(P_i, dtype=float)
    P_f = np.asarray(P_f, dtype=float)
    if P_i.ndim != 2 or P_i.shape[0] != P_i.shape[1] or P_i.shape[0] == 0:
        raise ValueError("P_i must be a nonempty square matrix")
    if P_f.shape != P_i.shape:
        raise ValueError("P_i and P_f must have the same shape")
    if not np.all(np.isfinite(P_i)) or not np.all(np.isfinite(P_f)):
        raise ValueError("P_i and P_f must be finite")
    if np.max(np.abs(P_i - P_i.T)) > 1e-10:
        raise ValueError("P_i must be symmetric")
    if np.max(np.abs(P_f - P_f.T)) > 1e-10:
        raise ValueError("P_f must be symmetric")
    dP = P_f - P_i
    return 0.5 * (dP + dP.T)

import numpy as np


def ordered_pair(S: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    S = np.asarray(S, dtype=float)
    if S.ndim != 2 or S.shape[0] != S.shape[1] or S.shape[0] == 0:
        raise ValueError("S must be a nonempty square matrix")
    if not np.all(np.isfinite(S)):
        raise ValueError("S must be finite")
    if np.max(np.abs(S - S.T)) > 1e-10:
        raise ValueError("S must be symmetric")
    S = 0.5 * (S + S.T)
    w, u = np.linalg.eigh(S)
    if np.any(w <= 1e-12):
        raise ValueError("S must be positive definite")
    sqrt_w = np.sqrt(w)
    G = (u * sqrt_w) @ u.T
    H = (u * (1.0 / sqrt_w)) @ u.T
    return G.astype(float), H.astype(float)

import numpy as np


def coefficient_pair(dP: np.ndarray, G: np.ndarray, H: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    G = np.asarray(G, dtype=float)
    H = np.asarray(H, dtype=float)
    if dP.ndim != 2 or dP.shape[0] != dP.shape[1] or dP.shape[0] == 0:
        raise ValueError("dP must be a nonempty square matrix")
    n = dP.shape[0]
    if G.shape != (n, n) or H.shape != (n, n):
        raise ValueError("dP and the overlap factors must have the same shape")
    if not np.all(np.isfinite(dP)):
        raise ValueError("dP must be finite")
    if not np.all(np.isfinite(G)) or not np.all(np.isfinite(H)):
        raise ValueError("overlap factors must be finite")
    if np.max(np.abs(dP - dP.T)) > 1e-10:
        raise ValueError("dP must be symmetric")
    if np.max(np.abs(G - G.T)) > 1e-10:
        raise ValueError("G must be symmetric")
    if np.max(np.abs(H - H.T)) > 1e-10:
        raise ValueError("H must be symmetric")
    dP = 0.5 * (dP + dP.T)
    G = 0.5 * (G + G.T)
    H = 0.5 * (H + H.T)
    dP_tilde = G @ dP @ G
    dP_tilde = 0.5 * (dP_tilde + dP_tilde.T)
    delta, U = np.linalg.eigh(dP_tilde)
    V = H @ U
    return delta.astype(float), V.astype(float)

import numpy as np


def image_matrix(P0: np.ndarray, G: np.ndarray, H: np.ndarray) -> np.ndarray:
    P0 = np.asarray(P0, dtype=float)
    G = np.asarray(G, dtype=float)
    H = np.asarray(H, dtype=float)
    if P0.ndim != 2 or P0.shape[0] != P0.shape[1] or P0.shape[0] == 0:
        raise ValueError("P0 must be a nonempty square matrix")
    n = P0.shape[0]
    if G.shape != (n, n) or H.shape != (n, n):
        raise ValueError("P0, G, and H must have the same shape")
    if not np.all(np.isfinite(P0)) or not np.all(np.isfinite(G)) or not np.all(np.isfinite(H)):
        raise ValueError("P0, G, and H must be finite")
    if np.max(np.abs(P0 - P0.T)) > 1e-10:
        raise ValueError("P0 must be symmetric")
    if np.max(np.abs(G - G.T)) > 1e-10:
        raise ValueError("G must be symmetric")
    if np.max(np.abs(H - H.T)) > 1e-10:
        raise ValueError("H must be symmetric")
    P0 = 0.5 * (P0 + P0.T)
    G = 0.5 * (G + G.T)
    P_t = G @ P0 @ G
    return 0.5 * (P_t + P_t.T)

import numpy as np


def frame_matrix(P_t: np.ndarray) -> np.ndarray:
    P_t = np.asarray(P_t, dtype=float)
    if P_t.ndim != 2 or P_t.shape[0] != P_t.shape[1] or P_t.shape[0] == 0:
        raise ValueError("P_t must be a nonempty square matrix")
    if not np.all(np.isfinite(P_t)):
        raise ValueError("P_t must be finite")
    if np.max(np.abs(P_t - P_t.T)) > 1e-10:
        raise ValueError("P_t must be symmetric")
    P_t = 0.5 * (P_t + P_t.T)
    return np.eye(P_t.shape[0]) - P_t

import numpy as np


def group_values(
    delta: np.ndarray,
    V: np.ndarray,
    Q: np.ndarray,
    G: np.ndarray,
    H: np.ndarray,
    eps: float,
) -> tuple[float, float]:
    delta = np.asarray(delta, dtype=float).reshape(-1)
    V = np.asarray(V, dtype=float)
    Q = np.asarray(Q, dtype=float)
    G = np.asarray(G, dtype=float)
    H = np.asarray(H, dtype=float)
    if V.ndim != 2 or V.shape[0] != V.shape[1] or V.shape[0] == 0:
        raise ValueError("V must be a nonempty square matrix")
    n = V.shape[0]
    if delta.size != n:
        raise ValueError("delta must have one entry per column of V")
    if Q.shape != (n, n) or G.shape != (n, n) or H.shape != (n, n):
        raise ValueError("Q, G, H, and V must have the same shape")
    if not np.all(np.isfinite(delta)) or not np.all(np.isfinite(V)):
        raise ValueError("delta and V must be finite")
    if not np.all(np.isfinite(Q)) or not np.all(np.isfinite(G)) or not np.all(np.isfinite(H)):
        raise ValueError("Q, G, and H must be finite")
    if np.max(np.abs(Q - Q.T)) > 1e-10:
        raise ValueError("Q must be symmetric")
    if np.max(np.abs(G - G.T)) > 1e-10:
        raise ValueError("G must be symmetric")
    if np.max(np.abs(H - H.T)) > 1e-10:
        raise ValueError("H must be symmetric")
    eps = float(eps)
    if not np.isfinite(eps) or eps <= 0.0 or eps >= 1.0:
        raise ValueError("eps must be a positive number strictly less than 1")
    x = np.zeros_like(delta)
    r = np.zeros_like(delta)
    unit = 1.0 - eps
    for i, d in enumerate(delta):
        ad = abs(d)
        if ad >= unit:
            x[i] = d
        elif ad > 0.0:
            r[i] = d
    D1 = V @ np.diag(x) @ V.T
    D2 = V @ np.diag(r) @ V.T
    D1 = 0.5 * (D1 + D1.T)
    D2 = 0.5 * (D2 + D2.T)
    D1_t = image_matrix(D1, G, H)
    D2_t = image_matrix(D2, G, H)
    Q = 0.5 * (Q + Q.T)
    s1 = float(np.trace(Q @ D1_t @ Q))
    s2 = float(np.trace(Q @ D2_t @ Q))
    if not np.isfinite(s1) or not np.isfinite(s2):
        raise ValueError("block scalars must be finite")
    return s1, s2

import numpy as np


def evaluate_instance(P_i: np.ndarray, P_f: np.ndarray, S: np.ndarray, eps: float) -> float:
    dP = density_pair(P_i, P_f)
    dP = np.asarray(dP, dtype=float)
    P_i = np.asarray(P_i, dtype=float)
    S = np.asarray(S, dtype=float)
    if dP.shape != P_i.shape:
        raise ValueError("difference density must match P_i")
    if S.shape != P_i.shape:
        raise ValueError("S must match P_i")

    G, H = ordered_pair(S)
    G = np.asarray(G, dtype=float)
    H = np.asarray(H, dtype=float)
    n = dP.shape[0]
    if G.shape != (n, n) or H.shape != (n, n):
        raise ValueError("overlap factors must have shape (n, n)")

    delta, V = coefficient_pair(dP, G, H)
    delta = np.asarray(delta, dtype=float).reshape(-1)
    V = np.asarray(V, dtype=float)
    if delta.size != n or V.shape != (n, n):
        raise ValueError("DDNO factors must cover the full AO basis")

    P_t = image_matrix(P_i, G, H)
    P_t = np.asarray(P_t, dtype=float)
    if P_t.shape != (n, n):
        raise ValueError("mapped initial density must have shape (n, n)")

    Q = frame_matrix(P_t)
    Q = np.asarray(Q, dtype=float)
    if Q.shape != (n, n):
        raise ValueError("unoccupied projector must have shape (n, n)")

    s1, s2 = group_values(delta, V, Q, G, H, eps)
    s1 = float(s1)
    s2 = float(s2)
    if not np.isfinite(s1) or not np.isfinite(s2):
        raise ValueError("block scalars must be finite")
    if s2 < -1e-12:
        raise ValueError("second-group scalar must be nonnegative")
    return s2
SCICODE_GOLD_EOF
