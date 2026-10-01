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

def construct_indefinite_symmetric_matrix(n, s, seed):
    if not isinstance(n, (int, np.integer)):
        raise ValueError("n must be an integer")
    n = int(n)
    if n < 2:
        raise ValueError("require n >= 2")
    s = np.asarray(s, dtype=float).reshape(-1)
    if s.shape != (n,):
        raise ValueError("s must have shape (n,)")
    if not np.all(np.isfinite(s)):
        raise ValueError("s must be finite")
    if not (np.any(s > 0.0) and np.any(s < 0.0)):
        raise ValueError("s must contain both a positive and a negative entry")
    rng = np.random.default_rng(int(seed))
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)), mode="reduced")
    A = Q @ np.diag(s) @ Q.T
    return 0.5 * (A + A.T)

import numpy as np

def extract_column_subset(A, indices):
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square 2D array")
    n = A.shape[0]
    if n < 2:
        raise ValueError("require n >= 2")
    indices = np.asarray(indices, dtype=int).reshape(-1)
    r = indices.size
    if r < 1 or r >= n:
        raise ValueError("require 1 <= r < n")
    if np.unique(indices).size != r:
        raise ValueError("indices must be distinct")
    if np.any(indices < 0) or np.any(indices >= n):
        raise ValueError("indices must lie in 0, ..., n-1")
    return A[:, indices]

import numpy as np

def draw_gaussian_sketch(t, n, seed):
    if not isinstance(t, (int, np.integer)) or not isinstance(n, (int, np.integer)):
        raise ValueError("t and n must be integers")
    t, n = int(t), int(n)
    if n < 2:
        raise ValueError("require n >= 2")
    if t < 1 or t >= n:
        raise ValueError("require 1 <= t < n")
    rng = np.random.default_rng(int(seed))
    return rng.standard_normal((t, n))

import numpy as np

def residual_bound_factor(C, X):
    C = np.asarray(C, dtype=float)
    X = np.asarray(X, dtype=float)
    if C.ndim != 2:
        raise ValueError("C must be a 2D array")
    n, r = C.shape
    if r < 1 or r >= n:
        raise ValueError("require 1 <= r < n")
    if X.ndim != 2 or X.shape[1] != n:
        raise ValueError("X must have shape (t, n)")
    t = X.shape[0]
    if t <= r:
        raise ValueError("require t > r")
    Q, R = np.linalg.qr(C, mode="reduced")
    if np.min(np.abs(np.diag(R))) < 1e-14:
        raise ValueError("C is rank deficient")
    sv = np.linalg.svd(X @ Q, compute_uv=False)
    smin = float(sv[-1])
    if smin <= 0.0:
        raise ValueError("sketch does not embed the column span")
    return float(np.sqrt(1.0 + 1.0 / (smin ** 4)))

import numpy as np

def sketched_middle_matrix(A, C, X):
    A = np.asarray(A, dtype=float)
    C = np.asarray(C, dtype=float)
    X = np.asarray(X, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square 2D array")
    n = A.shape[0]
    if C.ndim != 2 or C.shape[0] != n:
        raise ValueError("C must have shape (n, r)")
    r = C.shape[1]
    if r < 1 or r >= n:
        raise ValueError("require 1 <= r < n")
    if X.ndim != 2 or X.shape[1] != n:
        raise ValueError("X must have shape (t, n)")
    t = X.shape[0]
    if t <= r:
        raise ValueError("require t > r")
    if t >= n:
        raise ValueError("require t < n")
    XC = X @ C
    Msk = X @ A @ X.T
    Q, R = np.linalg.qr(XC, mode="reduced")
    if np.min(np.abs(np.diag(R))) < 1e-14:
        raise ValueError("XC is rank deficient")
    core = Q.T @ Msk @ Q
    Y = np.linalg.solve(R, core)
    M = np.linalg.solve(R, Y.T).T
    return 0.5 * (M + M.T)

import numpy as np

def reconstruction_entry(C, M, row, col):
    C = np.asarray(C, dtype=float)
    M = np.asarray(M, dtype=float)
    if C.ndim != 2:
        raise ValueError("C must be 2D")
    n, r = C.shape
    if r < 1 or n <= r:
        raise ValueError("require n > r >= 1")
    if M.shape != (r, r):
        raise ValueError("M must have shape (r, r)")
    if not isinstance(row, (int, np.integer)) or not isinstance(col, (int, np.integer)):
        raise ValueError("row and col must be integers")
    row, col = int(row), int(col)
    if row < 0 or col < 0 or row >= n or col >= n:
        raise ValueError("row and col must lie in 0, ..., n-1")
    Ahat = C @ M @ C.T
    return float(Ahat[row, col])

import numpy as np

def run_sketched_core_entry(
    n, s, data_seed, sketch_seed, indices, t, row, col
):
    if not isinstance(t, (int, np.integer)):
        raise ValueError("t must be an integer")
    t = int(t)
    indices = np.asarray(indices, dtype=int).reshape(-1)
    r = indices.size
    if t <= r:
        raise ValueError("require t > r")
    A = construct_indefinite_symmetric_matrix(n, s, data_seed)
    C = extract_column_subset(A, indices)
    X = draw_gaussian_sketch(t, n, sketch_seed)
    factor = residual_bound_factor(C, X)
    if not np.isfinite(factor) or factor <= 1.0:
        raise ValueError("sketch does not embed the column span")
    Mhat = sketched_middle_matrix(A, C, X)
    entry = reconstruction_entry(C, Mhat, row, col)
    if not np.isfinite(entry):
        raise ValueError("reconstruction entry is not finite")
    return float(factor)
SCICODE_GOLD_EOF
