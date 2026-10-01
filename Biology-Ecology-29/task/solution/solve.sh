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


def use_probabilities(resource_matrix: "numpy.ndarray") -> "numpy.ndarray":
    """Conditional resource-use probabilities p_ij = N_ij / Y_i (Eq 6); a species with Y_i = 0 gets a zero row."""
    N = np.asarray(resource_matrix, dtype=np.float64)
    if N.ndim != 2 or N.shape[0] < 1 or N.shape[1] < 1:
        raise ValueError("resource_matrix must be a 2-D array with at least one species and one state")
    if not np.all(np.isfinite(N)) or np.any(N < 0.0):
        raise ValueError("resource_matrix entries must be finite and nonnegative")
    if N.sum() <= 0.0:
        raise ValueError("resource_matrix must contain at least one positive entry")
    Y = N.sum(axis=1)
    return np.where(Y[:, None] > 0.0, N / np.where(Y[:, None] > 0.0, Y[:, None], 1.0), 0.0)

import numpy as np


def _xlogx(a):
    """x ln x with the convention 0 ln 0 = 0 (vectorised)."""
    a = np.asarray(a, dtype=np.float64)
    return np.where(a > 0.0, a * np.log(np.where(a > 0.0, a, 1.0)), 0.0)


def resource_entropies(resource_matrix: "numpy.ndarray") -> "numpy.ndarray":
    """[H(X), H(Y), H(XY), H_Y(X)] of Eqs 10-13 in nats, from the pooled probabilities of Eqs 5, 7 and 9."""
    N = np.asarray(resource_matrix, dtype=np.float64)
    if N.ndim != 2 or N.shape[0] < 1 or N.shape[1] < 1:
        raise ValueError("resource_matrix must be a 2-D array with at least one species and one state")
    if not np.all(np.isfinite(N)) or np.any(N < 0.0):
        raise ValueError("resource_matrix entries must be finite and nonnegative")
    if N.sum() <= 0.0:
        raise ValueError("resource_matrix must contain at least one positive entry")
    Z = N.sum()
    P = N.sum(axis=0) / Z                                   # marginal of the resource states
    Q = N.sum(axis=1) / Z                                   # marginal of the species
    pi = N / Z
    HX = -_xlogx(P).sum()
    HY = -_xlogx(Q).sum()
    HXY = -_xlogx(pi).sum()
    return np.array([HX, HY, HXY, HXY - HY])

import numpy as np


def _xlogx(a):
    """x ln x with the convention 0 ln 0 = 0 (vectorised)."""
    a = np.asarray(a, dtype=np.float64)
    return np.where(a > 0.0, a * np.log(np.where(a > 0.0, a, 1.0)), 0.0)


def _check_matrix(resource_matrix):
    N = np.asarray(resource_matrix, dtype=np.float64)
    if N.ndim != 2 or N.shape[0] < 1 or N.shape[1] < 1:
        raise ValueError("resource_matrix must be a 2-D array with at least one species and one state")
    if not np.all(np.isfinite(N)) or np.any(N < 0.0):
        raise ValueError("resource_matrix entries must be finite and nonnegative")
    if N.sum() <= 0.0:
        raise ValueError("resource_matrix must contain at least one positive entry")
    return N


def state_contributions(resource_matrix: "numpy.ndarray", use_probs: "numpy.ndarray",
                                entropies: "numpy.ndarray") -> "numpy.ndarray":
    """delta_j of Eq 16: the contribution of resource state j to the standardized resource heterogeneity M(X)."""
    N = _check_matrix(resource_matrix)
    p = np.asarray(use_probs, dtype=np.float64)
    ent = np.asarray(entropies, dtype=np.float64)
    if p.shape != N.shape:
        raise ValueError("use_probs must have the shape of resource_matrix")
    if ent.shape != (4,) or not np.all(np.isfinite(ent)):
        raise ValueError("entropies must be the finite array [H(X), H(Y), H(XY), H_Y(X)]")
    if not np.all(np.isfinite(p)) or np.any(p < 0.0):
        raise ValueError("use_probs must be finite and nonnegative")
    HX = float(ent[0])
    if HX <= 0.0:
        raise ValueError("H(X) must be positive: the matrix needs at least two occupied resource states")
    Z = N.sum()
    pi = N / Z
    P = N.sum(axis=0) / Z
    # sum_i pi_ij ln p_ij - P_j ln P_j = sum_i pi_ij ln(p_ij / P_j): the per-state share of the shared
    # information m(X) = H(X) - H_Y(X); the states sum to M(X) = m(X)/H(X)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(p > 0.0, pi * np.log(np.where(p > 0.0, p, 1.0)), 0.0).sum(axis=0)
    return (term - _xlogx(P)) / HX

import numpy as np


def modified_weights(contributions: "numpy.ndarray", n_occupied: int) -> "numpy.ndarray":
    """ed_j of Eq 33: exp(delta_j r'/r) normalised to unit sum, r' the occupied states of the complete matrix."""
    delta = np.asarray(contributions, dtype=np.float64)
    if delta.ndim != 1 or delta.size < 1 or not np.all(np.isfinite(delta)):
        raise ValueError("contributions must be a non-empty finite 1-D array")
    r = delta.size
    if int(n_occupied) != n_occupied or not (1 <= n_occupied <= r):
        raise ValueError("n_occupied must be an integer between 1 and the number of resource states")
    # the exponent is scaled by the occupancy fraction r'/r of the COMPLETE matrix (not of a reduced one)
    w = np.exp(delta * (float(int(n_occupied)) / r))
    return w / w.sum()

import numpy as np


def _check_matrix(resource_matrix):
    N = np.asarray(resource_matrix, dtype=np.float64)
    if N.ndim != 2 or N.shape[0] < 1 or N.shape[1] < 1:
        raise ValueError("resource_matrix must be a 2-D array with at least one species and one state")
    if not np.all(np.isfinite(N)) or np.any(N < 0.0):
        raise ValueError("resource_matrix entries must be finite and nonnegative")
    if N.sum() <= 0.0:
        raise ValueError("resource_matrix must contain at least one positive entry")
    return N


def adjusted_probabilities(resource_matrix: "numpy.ndarray", weights: "numpy.ndarray",
                                   k: float) -> "numpy.ndarray":
    """p*_ij of Eq 19: N_ij divided by the weighted, k-scaled abundance Y*_i = sum_j w_j k N_ij."""
    N = _check_matrix(resource_matrix)
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != (N.shape[1],) or not np.all(np.isfinite(w)) or np.any(w < 0.0):
        raise ValueError("weights must be a finite nonnegative array with one entry per resource state")
    if not np.isfinite(k) or k <= 0.0:
        raise ValueError("k must be positive and finite")
    Ystar = (w[None, :] * float(k) * N).sum(axis=1)
    if np.any((N.sum(axis=1) > 0.0) & (Ystar <= 0.0)):
        raise ValueError("a species with positive abundance has zero weighted abundance: undefined p*")
    return np.where(Ystar[:, None] > 0.0, N / np.where(Ystar[:, None] > 0.0, Ystar[:, None], 1.0), 0.0)

import numpy as np


def _xlogx(a):
    """x ln x with the convention 0 ln 0 = 0 (vectorised)."""
    a = np.asarray(a, dtype=np.float64)
    return np.where(a > 0.0, a * np.log(np.where(a > 0.0, a, 1.0)), 0.0)


def niche_breadth(adjusted_row: "numpy.ndarray", weights: "numpy.ndarray", k: float) -> float:
    """Standardized niche breadth beta' of Eq 37 for one species: -(k/ln k) sum_j w_j p*_j ln p*_j."""
    ps = np.asarray(adjusted_row, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if ps.ndim != 1 or ps.size < 1 or w.shape != ps.shape:
        raise ValueError("adjusted_row and weights must be 1-D arrays of the same length")
    if not np.all(np.isfinite(ps)) or np.any(ps < 0.0) or not np.all(np.isfinite(w)) or np.any(w < 0.0):
        raise ValueError("adjusted_row and weights must be finite and nonnegative")
    if not np.isfinite(k) or k <= 1.0:
        raise ValueError("k must exceed 1 (ln k must be positive)")
    if ps.sum() <= 0.0:
        raise ValueError("adjusted_row must contain a positive entry")
    return float(-(float(k) / np.log(float(k))) * (w * _xlogx(ps)).sum())

import numpy as np


def _xlogx(a):
    """x ln x with the convention 0 ln 0 = 0 (vectorised)."""
    a = np.asarray(a, dtype=np.float64)
    return np.where(a > 0.0, a * np.log(np.where(a > 0.0, a, 1.0)), 0.0)


def niche_overlap(adjusted_row_i: "numpy.ndarray", adjusted_row_h: "numpy.ndarray",
                          weights: "numpy.ndarray", k: float) -> float:
    """Horn-type niche overlap gamma' of Eq 39 between two species from their adjusted probabilities."""
    a = np.asarray(adjusted_row_i, dtype=np.float64)
    b = np.asarray(adjusted_row_h, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if a.ndim != 1 or a.size < 1 or b.shape != a.shape or w.shape != a.shape:
        raise ValueError("the two adjusted rows and weights must be 1-D arrays of the same length")
    for arr in (a, b, w):
        if not np.all(np.isfinite(arr)) or np.any(arr < 0.0):
            raise ValueError("adjusted rows and weights must be finite and nonnegative")
    if not np.isfinite(k) or k <= 0.0:
        raise ValueError("k must be positive and finite")
    if a.sum() <= 0.0 or b.sum() <= 0.0:
        raise ValueError("each adjusted row must contain a positive entry")
    # I(x) = x ln x; the k inside the bracket cancels against the 1/k of p*, so gamma' is k-free
    val = -(1.0 / (2.0 * np.log(2.0))) * (w * float(k) * (_xlogx(a) + _xlogx(b) - _xlogx(a + b))).sum()
    return float(val)

import numpy as np


def _check_matrix(resource_matrix):
    N = np.asarray(resource_matrix, dtype=np.float64)
    if N.ndim != 2 or N.shape[0] < 1 or N.shape[1] < 1:
        raise ValueError("resource_matrix must be a 2-D array with at least one species and one state")
    if not np.all(np.isfinite(N)) or np.any(N < 0.0):
        raise ValueError("resource_matrix entries must be finite and nonnegative")
    if N.sum() <= 0.0:
        raise ValueError("resource_matrix must contain at least one positive entry")
    return N


def generalist_pair_overlap(resource_matrix: "numpy.ndarray", k: float) -> float:
    """ORCHESTRATOR: noncircular ed-weighted Horn overlap (Eq 39) between the two species with the largest
    noncircular niche breadths beta' (Eq 37), Sec 6 noncircularity throughout."""
    N = _check_matrix(resource_matrix)
    if not np.isfinite(k) or k <= 1.0:
        raise ValueError("k must exceed 1")
    s, r = N.shape
    if s < 3:
        raise ValueError("at least three species are needed for a noncircular pair overlap")
    n_occupied = int((N.sum(axis=0) > 0.0).sum())              # r' of the COMPLETE matrix
    # noncircular niche breadth of every species: factors from the matrix without that species
    breadth = np.empty(s)
    for i in range(s):
        reduced = np.delete(N, i, axis=0)
        p = use_probabilities(reduced)
        ent = resource_entropies(reduced)
        delta = state_contributions(reduced, p, ent)
        ed = modified_weights(delta, n_occupied)
        pstar = adjusted_probabilities(N[i:i + 1], ed, k)  # the focal species' own row
        breadth[i] = niche_breadth(pstar[0], ed, k)
    order = np.argsort(-breadth, kind="stable")
    if breadth[order[1]] == breadth[order[2]] or breadth[order[0]] == breadth[order[1]]:
        raise ValueError("the two broadest niches are not uniquely determined (tied breadths)")
    i, h = int(order[0]), int(order[1])
    # noncircular overlap of the pair: factors from the matrix without both species
    reduced = np.delete(N, [i, h], axis=0)
    p = use_probabilities(reduced)
    ent = resource_entropies(reduced)
    delta = state_contributions(reduced, p, ent)
    ed = modified_weights(delta, n_occupied)
    pstar = adjusted_probabilities(N[[i, h]], ed, k)
    return niche_overlap(pstar[0], pstar[1], ed, k)
SCICODE_GOLD_EOF
