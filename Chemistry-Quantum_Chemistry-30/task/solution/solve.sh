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


def validate_roks_orbitals(S: np.ndarray, C_GS: np.ndarray, C_d_ES: np.ndarray, h_ES: np.ndarray, l_ES: np.ndarray) -> int:
    S = np.asarray(S, dtype=float)
    C_GS = np.asarray(C_GS, dtype=float)
    C_d_ES = np.asarray(C_d_ES, dtype=float)
    h_ES = np.asarray(h_ES, dtype=float).reshape(-1)
    l_ES = np.asarray(l_ES, dtype=float).reshape(-1)
    if S.ndim != 2 or S.shape[0] != S.shape[1] or S.shape[0] == 0:
        raise ValueError("S must be a nonempty square matrix")
    n = S.shape[0]
    if not np.all(np.isfinite(S)):
        raise ValueError("S must be finite")
    if np.max(np.abs(S - S.T)) > 1e-10:
        raise ValueError("S must be symmetric")
    S = 0.5 * (S + S.T)
    w = np.linalg.eigvalsh(S)
    if np.any(w <= 1e-12):
        raise ValueError("S must be positive definite")
    if C_GS.ndim != 2 or C_GS.shape[0] != n or C_GS.shape[1] < 2:
        raise ValueError("C_GS must have shape (n, n_occ) with n_occ >= 2")
    n_occ = int(C_GS.shape[1])
    if C_d_ES.ndim != 2 or C_d_ES.shape[0] != n:
        raise ValueError("C_d_ES must have shape (n, n_d)")
    if C_d_ES.shape[1] != n_occ - 1:
        raise ValueError("the final-state doubly occupied set must have n_occ - 1 columns")
    if h_ES.size != n or l_ES.size != n:
        raise ValueError("SOMO coefficient vectors must have length n")
    if not np.all(np.isfinite(C_GS)) or not np.all(np.isfinite(C_d_ES)):
        raise ValueError("MO coefficients must be finite")
    if not np.all(np.isfinite(h_ES)) or not np.all(np.isfinite(l_ES)):
        raise ValueError("SOMO coefficients must be finite")

    def _metric_ok(C, ident, name):
        G = C.T @ S @ C
        if np.max(np.abs(G - ident)) > 1e-8:
            raise ValueError(f"{name} must be S-orthonormal")

    _metric_ok(C_GS, np.eye(n_occ), "C_GS")
    _metric_ok(C_d_ES, np.eye(n_occ - 1), "C_d_ES")
    C_hl = np.column_stack([h_ES, l_ES])
    _metric_ok(C_hl, np.eye(2), "open-shell pair")
    if np.max(np.abs(C_d_ES.T @ S @ C_hl)) > 1e-8:
        raise ValueError("final-state doubly occupied orbitals must be orthogonal to both SOMOs")
    return n_occ

import numpy as np


def ground_state_density(C_GS: np.ndarray) -> np.ndarray:
    C_GS = np.asarray(C_GS, dtype=float)
    if C_GS.ndim != 2 or C_GS.shape[0] == 0 or C_GS.shape[1] == 0:
        raise ValueError("C_GS must be a nonempty coefficient matrix")
    if not np.all(np.isfinite(C_GS)):
        raise ValueError("C_GS must be finite")
    P_GS = 2.0 * (C_GS @ C_GS.T)
    return 0.5 * (P_GS + P_GS.T)

import numpy as np


def frozen_projection_density(P_GS: np.ndarray, h_ES: np.ndarray, l_ES: np.ndarray, S: np.ndarray) -> np.ndarray:
    P_GS = np.asarray(P_GS, dtype=float)
    S = np.asarray(S, dtype=float)
    h_ES = np.asarray(h_ES, dtype=float).reshape(-1)
    l_ES = np.asarray(l_ES, dtype=float).reshape(-1)
    if P_GS.ndim != 2 or P_GS.shape[0] != P_GS.shape[1] or P_GS.shape[0] == 0:
        raise ValueError("P_GS must be a nonempty square matrix")
    n = P_GS.shape[0]
    if S.shape != (n, n):
        raise ValueError("S must match P_GS")
    if h_ES.size != n or l_ES.size != n:
        raise ValueError("SOMO vectors must match P_GS")
    if not np.all(np.isfinite(P_GS)) or not np.all(np.isfinite(S)):
        raise ValueError("P_GS and S must be finite")
    if not np.all(np.isfinite(h_ES)) or not np.all(np.isfinite(l_ES)):
        raise ValueError("SOMO vectors must be finite")
    if np.max(np.abs(P_GS - P_GS.T)) > 1e-8:
        raise ValueError("P_GS must be symmetric")
    if np.max(np.abs(S - S.T)) > 1e-10:
        raise ValueError("S must be symmetric")
    P_GS = 0.5 * (P_GS + P_GS.T)
    S = 0.5 * (S + S.T)
    C_hl = np.column_stack([h_ES, l_ES])
    ident = np.eye(n)
    left = ident - C_hl @ C_hl.T @ S
    right = ident - S @ C_hl @ C_hl.T
    P_PRJ = left @ P_GS @ right
    return 0.5 * (P_PRJ + P_PRJ.T)

import numpy as np


def purify_intermediate_state(P_PRJ: np.ndarray, h_ES: np.ndarray, l_ES: np.ndarray, S: np.ndarray, n_occ: int) -> np.ndarray:
    P_PRJ = np.asarray(P_PRJ, dtype=float)
    S = np.asarray(S, dtype=float)
    h_ES = np.asarray(h_ES, dtype=float).reshape(-1)
    l_ES = np.asarray(l_ES, dtype=float).reshape(-1)
    n_occ = int(n_occ)
    if P_PRJ.ndim != 2 or P_PRJ.shape[0] != P_PRJ.shape[1] or P_PRJ.shape[0] == 0:
        raise ValueError("P_PRJ must be a nonempty square matrix")
    n = P_PRJ.shape[0]
    if S.shape != (n, n):
        raise ValueError("S must match P_PRJ")
    if h_ES.size != n or l_ES.size != n:
        raise ValueError("SOMO vectors must match P_PRJ")
    if n_occ < 2 or n_occ >= n:
        raise ValueError("n_occ must satisfy 2 <= n_occ < n")
    if not np.all(np.isfinite(P_PRJ)) or not np.all(np.isfinite(S)):
        raise ValueError("P_PRJ and S must be finite")
    if not np.all(np.isfinite(h_ES)) or not np.all(np.isfinite(l_ES)):
        raise ValueError("SOMO vectors must be finite")
    if np.max(np.abs(S - S.T)) > 1e-10:
        raise ValueError("S must be symmetric")
    P_PRJ = 0.5 * (P_PRJ + P_PRJ.T)
    S = 0.5 * (S + S.T)
    w_s, u_s = np.linalg.eigh(S)
    if np.any(w_s <= 1e-12):
        raise ValueError("S must be positive definite")
    sh = (u_s * np.sqrt(w_s)) @ u_s.T
    sih = (u_s * (1.0 / np.sqrt(w_s))) @ u_s.T
    a = 0.5 * ((sh @ P_PRJ @ sh) + (sh @ P_PRJ @ sh).T)
    w, y = np.linalg.eigh(a)
    C = sih @ y[:, np.argsort(w)[::-1]]
    n_d = n_occ - 1
    C_d = C[:, :n_d]
    C_d = C_d - np.outer(h_ES, h_ES @ S @ C_d) - np.outer(l_ES, l_ES @ S @ C_d)
    g = 0.5 * ((C_d.T @ S @ C_d) + (C_d.T @ S @ C_d).T)
    gw, gu = np.linalg.eigh(g)
    if np.any(gw <= 1e-12):
        raise ValueError("selected intermediate orbitals are linearly dependent")
    C_d = C_d @ (gu * (1.0 / np.sqrt(gw))) @ gu.T
    P_INT = 2.0 * (C_d @ C_d.T) + np.outer(h_ES, h_ES) + np.outer(l_ES, l_ES)
    return 0.5 * (P_INT + P_INT.T)

import numpy as np


def excited_state_density(C_d_ES: np.ndarray, h_ES: np.ndarray, l_ES: np.ndarray) -> np.ndarray:
    C_d_ES = np.asarray(C_d_ES, dtype=float)
    h_ES = np.asarray(h_ES, dtype=float).reshape(-1)
    l_ES = np.asarray(l_ES, dtype=float).reshape(-1)
    if C_d_ES.ndim != 2 or C_d_ES.shape[0] == 0 or C_d_ES.shape[1] == 0:
        raise ValueError("C_d_ES must be a nonempty coefficient matrix")
    n = C_d_ES.shape[0]
    if h_ES.size != n or l_ES.size != n:
        raise ValueError("SOMO vectors must match C_d_ES")
    if not np.all(np.isfinite(C_d_ES)):
        raise ValueError("C_d_ES must be finite")
    if not np.all(np.isfinite(h_ES)) or not np.all(np.isfinite(l_ES)):
        raise ValueError("SOMO vectors must be finite")
    P_ES = 2.0 * (C_d_ES @ C_d_ES.T) + np.outer(h_ES, h_ES) + np.outer(l_ES, l_ES)
    return 0.5 * (P_ES + P_ES.T)

import numpy as np


def ovocv_relaxation_promotions(P_INT: np.ndarray, P_ES: np.ndarray, S: np.ndarray) -> np.ndarray:
    P_INT = np.asarray(P_INT, dtype=float)
    P_ES = np.asarray(P_ES, dtype=float)
    S = np.asarray(S, dtype=float)
    if P_INT.ndim != 2 or P_INT.shape[0] != P_INT.shape[1] or P_INT.shape[0] == 0:
        raise ValueError("P_INT must be a nonempty square matrix")
    n = P_INT.shape[0]
    if P_ES.shape != (n, n) or S.shape != (n, n):
        raise ValueError("P_INT, P_ES, and S must have the same shape")
    if not np.all(np.isfinite(P_INT)) or not np.all(np.isfinite(P_ES)):
        raise ValueError("densities must be finite")
    if not np.all(np.isfinite(S)):
        raise ValueError("S must be finite")
    if np.max(np.abs(S - S.T)) > 1e-10:
        raise ValueError("S must be symmetric")
    P_INT = 0.5 * (P_INT + P_INT.T)
    P_ES = 0.5 * (P_ES + P_ES.T)
    S = 0.5 * (S + S.T)
    w_s, u_s = np.linalg.eigh(S)
    if np.any(w_s <= 1e-12):
        raise ValueError("S must be positive definite")
    sh = (u_s * np.sqrt(w_s)) @ u_s.T
    sih = (u_s * (1.0 / np.sqrt(w_s))) @ u_s.T

    def _doubles(P):
        a = 0.5 * ((sh @ P @ sh) + (sh @ P @ sh).T)
        w, y = np.linalg.eigh(a)
        order = np.argsort(w)[::-1]
        w = w[order]
        C = sih @ y[:, order]
        mask = np.abs(w - 2.0) <= 1e-6
        if np.count_nonzero(mask) == 0:
            raise ValueError("no occupation-2 spectator orbitals were found")
        return C[:, mask]

    C_i = _doubles(P_INT)
    C_f = _doubles(P_ES)
    if C_i.shape[1] != C_f.shape[1]:
        raise ValueError("intermediate and final spectator ranks must match")
    s = np.linalg.svd(C_i.T @ S @ C_f, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    dQ = 1.0 - s**2
    return np.sort(dQ)[::-1].astype(float)

import numpy as np


def run_roks_excitation_eda(S: np.ndarray, C_GS: np.ndarray, C_d_ES: np.ndarray, h_ES: np.ndarray, l_ES: np.ndarray) -> float:
    n_occ = validate_roks_orbitals(S, C_GS, C_d_ES, h_ES, l_ES)
    n_occ = int(np.asarray(n_occ).reshape(-1)[0])
    P_GS = np.asarray(ground_state_density(C_GS), dtype=float)
    P_PRJ = np.asarray(
        frozen_projection_density(P_GS, h_ES, l_ES, S), dtype=float
    )
    P_INT = np.asarray(
        purify_intermediate_state(P_PRJ, h_ES, l_ES, S, n_occ),
        dtype=float,
    )
    P_ES = np.asarray(
        excited_state_density(C_d_ES, h_ES, l_ES), dtype=float
    )
    if P_GS.shape != P_PRJ.shape or P_INT.shape != P_GS.shape:
        raise ValueError("densities must share one AO dimension")
    if P_ES.shape != P_GS.shape:
        raise ValueError("final-state density must match P_GS")
    dQ = np.asarray(
        ovocv_relaxation_promotions(P_INT, P_ES, S), dtype=float
    ).reshape(-1)
    if dQ.size == 0 or not np.all(np.isfinite(dQ)):
        raise ValueError("OVOCV promotions must be finite")
    if np.any(dQ < -1e-12):
        raise ValueError("OVOCV promotions must be nonnegative")
    return float(np.sum(dQ))
SCICODE_GOLD_EOF
