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


def build_kaczmarz_instance(seed: int = 20260920, m: int = 600, n: int = 150, density: float = 0.20, noise_ratio: float = 0.10) -> dict:
    """Reference implementation of build_kaczmarz_instance."""
    if not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    if not isinstance(m, (int, np.integer)) or int(m) < 1:
        raise ValueError("m must be a positive integer")
    if not isinstance(n, (int, np.integer)) or int(n) < 1:
        raise ValueError("n must be a positive integer")
    m = int(m)
    n = int(n)
    if m < n:
        raise ValueError("m must be >= n (A must have full column rank)")
    if not (isinstance(density, (int, float, np.floating, np.integer)) and 0.0 < float(density) <= 1.0):
        raise ValueError("density must satisfy 0 < density <= 1")
    if not (isinstance(noise_ratio, (int, float, np.floating, np.integer)) and float(noise_ratio) > 0.0):
        raise ValueError("noise_ratio must be a positive real number")

    seed = int(seed)
    rng = np.random.default_rng(seed)
    values = rng.standard_normal((m, n))
    mask = rng.random((m, n)) < float(density)
    A = values * mask
    x_true = rng.standard_normal(n)
    w = rng.standard_normal(m)

    xi = np.linalg.lstsq(A, w, rcond=None)[0]
    r = w - A @ xi
    r_norm = np.linalg.norm(r)
    if r_norm == 0.0:
        raise ValueError("degenerate instance: noise vector has zero norm")
    r = r * (float(noise_ratio) * np.linalg.norm(A @ x_true) / r_norm)
    b = A @ x_true + r

    col_sq_norms = np.sum(A ** 2, axis=0)
    row_sq_norms = np.sum(A ** 2, axis=1)
    b_norm = float(np.linalg.norm(b))

    return {
        "A": A,
        "b": b,
        "x_star": x_true,
        "r_opt": r,
        "col_sq_norms": col_sq_norms,
        "row_sq_norms": row_sq_norms,
        "b_norm": b_norm,
    }

import numpy as np


def sample_index_block(probabilities: np.ndarray, block_size: int, rng: np.random.Generator) -> np.ndarray:
    """Reference implementation of sample_index_block."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 1:
        raise ValueError("probabilities must be 1-D")
    if np.any(probabilities < 0.0):
        raise ValueError("probabilities must be nonnegative")
    if not isinstance(block_size, (int, np.integer)) or int(block_size) < 1:
        raise ValueError("block_size must be a positive integer")
    block_size = int(block_size)
    n_positive = int(np.sum(probabilities > 0.0))
    if n_positive < block_size:
        raise ValueError(
            f"fewer than block_size={block_size} entries have strictly "
            f"positive weight (have {n_positive})"
        )

    w = probabilities.copy()
    chosen = np.empty(block_size, dtype=np.int64)
    for t in range(block_size):
        total = w.sum()
        u = rng.random()
        cum = np.cumsum(w)
        idx = int(np.searchsorted(cum, u * total, side="left"))
        idx = min(idx, w.shape[0] - 1)
        chosen[t] = idx
        w[idx] = 0.0
    return chosen

import numpy as np


def _sample_block(weights, block_size, rng):
    """Sequential inverse-CDF sampling, the same convention pinned for
    sample_index_block, reproduced locally so this step is
    self-contained."""
    weights = np.asarray(weights, dtype=np.float64)
    if not isinstance(block_size, (int, np.integer)) or int(block_size) < 1:
        raise ValueError("block_size must be a positive integer")
    block_size = int(block_size)
    n_positive = int(np.sum(weights > 0.0))
    if n_positive < block_size:
        raise ValueError(
            f"fewer than block_size={block_size} entries have strictly "
            f"positive weight (have {n_positive})"
        )
    w = weights.copy()
    chosen = np.empty(block_size, dtype=np.int64)
    for t in range(block_size):
        total = w.sum()
        u = rng.random()
        cum = np.cumsum(w)
        idx = int(np.searchsorted(cum, u * total, side="left"))
        idx = min(idx, w.shape[0] - 1)
        chosen[t] = idx
        w[idx] = 0.0
    return chosen


def column_phase_update(A: np.ndarray, z: np.ndarray, col_sq_norms: np.ndarray, block_size: int, rng: np.random.Generator) -> np.ndarray:
    """Reference implementation of column_phase_update."""
    A = np.asarray(A, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    col_sq_norms = np.asarray(col_sq_norms, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError("A must be 2-D")
    m, n = A.shape
    if z.shape != (m,):
        raise ValueError("z must have length equal to A's number of rows")
    if col_sq_norms.shape != (n,):
        raise ValueError("col_sq_norms must have length equal to A's number of columns")
    if np.any(col_sq_norms <= 0.0):
        raise ValueError("every entry of col_sq_norms must be strictly positive")

    eps_z = (A.T @ z) ** 2 / col_sq_norms
    if float(np.sum(eps_z)) <= 0.0:
        raise ValueError("degenerate auxiliary vector: every column weight is zero")

    col_idx = _sample_block(eps_z, block_size, rng)
    A_U = A[:, col_idx]
    step = np.linalg.lstsq(A_U, z, rcond=None)[0]
    return z - A_U @ step

import numpy as np


def _sample_block(weights, block_size, rng):
    """Sequential inverse-CDF sampling, the same convention pinned for
    sample_index_block, reproduced locally so this step is
    self-contained."""
    weights = np.asarray(weights, dtype=np.float64)
    if not isinstance(block_size, (int, np.integer)) or int(block_size) < 1:
        raise ValueError("block_size must be a positive integer")
    block_size = int(block_size)
    n_positive = int(np.sum(weights > 0.0))
    if n_positive < block_size:
        raise ValueError(
            f"fewer than block_size={block_size} entries have strictly "
            f"positive weight (have {n_positive})"
        )
    w = weights.copy()
    chosen = np.empty(block_size, dtype=np.int64)
    for t in range(block_size):
        total = w.sum()
        u = rng.random()
        cum = np.cumsum(w)
        idx = int(np.searchsorted(cum, u * total, side="left"))
        idx = min(idx, w.shape[0] - 1)
        chosen[t] = idx
        w[idx] = 0.0
    return chosen


def row_phase_update(A: np.ndarray, b: np.ndarray, z: np.ndarray, x: np.ndarray, row_sq_norms: np.ndarray, block_size: int, rng: np.random.Generator) -> np.ndarray:
    """Reference implementation of row_phase_update."""
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    row_sq_norms = np.asarray(row_sq_norms, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError("A must be 2-D")
    m, n = A.shape
    if b.shape != (m,):
        raise ValueError("b must have length equal to A's number of rows")
    if z.shape != (m,):
        raise ValueError("z must have length equal to A's number of rows")
    if x.shape != (n,):
        raise ValueError("x must have length equal to A's number of columns")
    if row_sq_norms.shape != (m,):
        raise ValueError("row_sq_norms must have length equal to A's number of rows")
    if np.any(row_sq_norms <= 0.0):
        raise ValueError("every entry of row_sq_norms must be strictly positive")

    resid = b - z - A @ x
    eps_x = resid ** 2 / row_sq_norms
    if float(np.sum(eps_x)) <= 0.0:
        raise ValueError("degenerate row residual: every row weight is zero")

    row_idx = _sample_block(eps_x, block_size, rng)
    A_J = A[row_idx, :]
    rhs = b[row_idx] - z[row_idx] - A_J @ x
    step = np.linalg.lstsq(A_J, rhs, rcond=None)[0]
    return x + step

import numpy as np


def march_rgdbek(A: np.ndarray, b: np.ndarray, eta: float, iterations: int, march_seed: int) -> dict:
    """Reference implementation of march_rgdbek: chains
    column_phase_update and row_phase_update into the
    fixed two-phase iteration, feeding the row phase the auxiliary
    vector the column phase just produced (z_{k+1}, not the pre-update
    z_k)."""
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError("A must be 2-D")
    m, n = A.shape
    if b.shape != (m,):
        raise ValueError("b must have length equal to A's number of rows")
    if not (isinstance(eta, (int, float, np.floating, np.integer)) and 0.0 < float(eta) <= 1.0):
        raise ValueError("eta must satisfy 0 < eta <= 1")
    if not isinstance(iterations, (int, np.integer)) or int(iterations) < 1:
        raise ValueError("iterations must be a positive integer")
    if not isinstance(march_seed, (int, np.integer)) or int(march_seed) < 0:
        raise ValueError("march_seed must be a non-negative integer")

    eta = float(eta)
    iterations = int(iterations)
    march_seed = int(march_seed)
    n_eta = max(1, round(eta * n))
    m_eta = max(1, round(eta * m))

    rng = np.random.default_rng(march_seed)
    col_sq_norms = np.sum(A ** 2, axis=0)
    row_sq_norms = np.sum(A ** 2, axis=1)

    x = np.zeros(n)
    z = b.copy()
    error_history = []

    for _k in range(iterations):
        z_next = column_phase_update(A, z, col_sq_norms, n_eta, rng)
        # The row phase consumes z_next (the auxiliary vector the column
        # phase just produced this same iteration), never the pre-update z.
        x_next = row_phase_update(A, b, z_next, x, row_sq_norms, m_eta, rng)

        error_history.append(float(np.linalg.norm(x_next - x)))
        x = x_next
        z = z_next

    return {"x": x, "z": z, "error_history": error_history}

import numpy as np


def _row_block_bounds(A, n_ranks):
    """Contiguous, nonzero-count-balanced row partition of A's rows into
    n_ranks blocks, in index order (S22: same-named function redefined
    elsewhere must be byte-identical). Row i's nonzero count c_i is
    computed once over the whole matrix; C = sum(c_i) is the total
    nonzero count. The running prefix sum S(i) = c_0 + ... + c_i is a
    single monotone pass over the whole matrix, taken from row 0 and
    never reset at a block boundary. For p = 0, ..., n_ranks - 2 (in
    increasing p order), boundary p closes immediately after the first
    row index i such that S(i) >= (p + 1) * C / n_ranks (i.e. the first
    row whose global cumulative nonzero count reaches or exceeds a
    (p + 1) / n_ranks share of the whole matrix's total nonzero count);
    the final rank takes every remaining row. Non-degeneracy guard,
    applied left to right in increasing p order: if the scan would ever
    produce an empty block, the boundary is pulled forward by one row so
    every rank owns at least one row."""
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError("A must be 2-D")
    m = A.shape[0]
    if not isinstance(n_ranks, (int, np.integer)) or int(n_ranks) < 1:
        raise ValueError("n_ranks must be a positive integer")
    n_ranks = int(n_ranks)
    if m < n_ranks:
        raise ValueError(f"m={m} rows fewer than n_ranks={n_ranks}")
    c = np.count_nonzero(A, axis=1).astype(np.int64)
    C = int(c.sum())
    if C == 0:
        raise ValueError("A has no nonzero entries; cannot balance the row partition")
    S = np.cumsum(c)
    bounds = []
    lo = 0
    for p in range(n_ranks - 1):
        target = (p + 1) * C / n_ranks
        i = int(np.searchsorted(S, target, side="left"))
        i = min(i, m - 1)
        hi = i + 1
        if hi <= lo:
            hi = lo + 1
        bounds.append((lo, hi))
        lo = hi
    bounds.append((lo, m))
    return bounds


def _sample_block(weights, block_size, rng):
    """Sequential inverse-CDF sampling, the same convention pinned for
    sample_index_block, reproduced locally so this step is
    self-contained."""
    weights = np.asarray(weights, dtype=np.float64)
    if not isinstance(block_size, (int, np.integer)) or int(block_size) < 1:
        raise ValueError("block_size must be a positive integer")
    block_size = int(block_size)
    n_positive = int(np.sum(weights > 0.0))
    if n_positive < block_size:
        raise ValueError(
            f"fewer than block_size={block_size} entries have strictly "
            f"positive weight (have {n_positive})"
        )
    w = weights.copy()
    chosen = np.empty(block_size, dtype=np.int64)
    for t in range(block_size):
        total = w.sum()
        u = rng.random()
        cum = np.cumsum(w)
        idx = int(np.searchsorted(cum, u * total, side="left"))
        idx = min(idx, w.shape[0] - 1)
        chosen[t] = idx
        w[idx] = 0.0
    return chosen


def parallel_column_phase(A: np.ndarray, z: np.ndarray, n_ranks: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    """Reference implementation of parallel_column_phase."""
    A = np.asarray(A, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError("A must be 2-D")
    m = A.shape[0]
    if z.shape != (m,):
        raise ValueError("z must have length equal to A's number of rows")
    bounds = _row_block_bounds(A, n_ranks)

    col_sq_norms = np.sum(A ** 2, axis=0)
    if np.any(col_sq_norms <= 0.0):
        raise ValueError("every column of A must have strictly positive squared norm")
    eps_z = (A.T @ z) ** 2 / col_sq_norms
    col_idx = _sample_block(eps_z, block_size, rng)

    z_next = z.copy()
    for (lo, hi) in bounds:
        A_p = A[lo:hi, :]
        z_p = z[lo:hi]
        A_p_U = A_p[:, col_idx]
        z_sol_p = np.linalg.lstsq(A_p_U, z_p, rcond=None)[0]
        z_next[lo:hi] = z_p - A_p_U @ z_sol_p
    return z_next

import numpy as np


def _row_block_bounds(A, n_ranks):
    """Contiguous, nonzero-count-balanced row partition of A's rows into
    n_ranks blocks, in index order (S22: same-named function redefined
    elsewhere must be byte-identical). Row i's nonzero count c_i is
    computed once over the whole matrix; C = sum(c_i) is the total
    nonzero count. The running prefix sum S(i) = c_0 + ... + c_i is a
    single monotone pass over the whole matrix, taken from row 0 and
    never reset at a block boundary. For p = 0, ..., n_ranks - 2 (in
    increasing p order), boundary p closes immediately after the first
    row index i such that S(i) >= (p + 1) * C / n_ranks (i.e. the first
    row whose global cumulative nonzero count reaches or exceeds a
    (p + 1) / n_ranks share of the whole matrix's total nonzero count);
    the final rank takes every remaining row. Non-degeneracy guard,
    applied left to right in increasing p order: if the scan would ever
    produce an empty block, the boundary is pulled forward by one row so
    every rank owns at least one row."""
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError("A must be 2-D")
    m = A.shape[0]
    if not isinstance(n_ranks, (int, np.integer)) or int(n_ranks) < 1:
        raise ValueError("n_ranks must be a positive integer")
    n_ranks = int(n_ranks)
    if m < n_ranks:
        raise ValueError(f"m={m} rows fewer than n_ranks={n_ranks}")
    c = np.count_nonzero(A, axis=1).astype(np.int64)
    C = int(c.sum())
    if C == 0:
        raise ValueError("A has no nonzero entries; cannot balance the row partition")
    S = np.cumsum(c)
    bounds = []
    lo = 0
    for p in range(n_ranks - 1):
        target = (p + 1) * C / n_ranks
        i = int(np.searchsorted(S, target, side="left"))
        i = min(i, m - 1)
        hi = i + 1
        if hi <= lo:
            hi = lo + 1
        bounds.append((lo, hi))
        lo = hi
    bounds.append((lo, m))
    return bounds


def _sample_block(weights, block_size, rng):
    """Sequential inverse-CDF sampling, the same convention pinned for
    sample_index_block, reproduced locally so this step is
    self-contained."""
    weights = np.asarray(weights, dtype=np.float64)
    if not isinstance(block_size, (int, np.integer)) or int(block_size) < 1:
        raise ValueError("block_size must be a positive integer")
    block_size = int(block_size)
    n_positive = int(np.sum(weights > 0.0))
    if n_positive < block_size:
        raise ValueError(
            f"fewer than block_size={block_size} entries have strictly "
            f"positive weight (have {n_positive})"
        )
    w = weights.copy()
    chosen = np.empty(block_size, dtype=np.int64)
    for t in range(block_size):
        total = w.sum()
        u = rng.random()
        cum = np.cumsum(w)
        idx = int(np.searchsorted(cum, u * total, side="left"))
        idx = min(idx, w.shape[0] - 1)
        chosen[t] = idx
        w[idx] = 0.0
    return chosen


def parallel_row_phase(A: np.ndarray, b: np.ndarray, z: np.ndarray, x: np.ndarray, n_ranks: int, block_size_per_rank: int, rng: np.random.Generator) -> np.ndarray:
    """Reference implementation of parallel_row_phase."""
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError("A must be 2-D")
    m, n = A.shape
    if b.shape != (m,):
        raise ValueError("b must have length equal to A's number of rows")
    if z.shape != (m,):
        raise ValueError("z must have length equal to A's number of rows")
    if x.shape != (n,):
        raise ValueError("x must have length equal to A's number of columns")
    bounds = _row_block_bounds(A, n_ranks)
    n_ranks = int(n_ranks)

    updates = []
    for (lo, hi) in bounds:
        A_p = A[lo:hi, :]
        b_p = b[lo:hi]
        z_p = z[lo:hi]
        row_sq_norms_p = np.sum(A_p ** 2, axis=1)
        if np.any(row_sq_norms_p <= 0.0):
            raise ValueError("every row of A must have strictly positive squared norm")
        resid_p = b_p - z_p - A_p @ x
        eps_x_p = resid_p ** 2 / row_sq_norms_p
        if float(np.sum(eps_x_p)) <= 0.0:
            raise ValueError("degenerate row residual on one rank: every row weight is zero")

        idx_p = _sample_block(eps_x_p, block_size_per_rank, rng)
        A_Jp = A_p[idx_p, :]
        rhs = b_p[idx_p] - z_p[idx_p] - A_Jp @ x
        step_p = np.linalg.lstsq(A_Jp, rhs, rcond=None)[0]
        updates.append(step_p)

    x_next = x + np.sum(updates, axis=0) / n_ranks
    return x_next

import numpy as np


def _row_block_bounds(A, n_ranks):
    """Contiguous, nonzero-count-balanced row partition of A's rows into
    n_ranks blocks, in index order (S22: same-named function redefined
    elsewhere must be byte-identical). Row i's nonzero count c_i is
    computed once over the whole matrix; C = sum(c_i) is the total
    nonzero count. The running prefix sum S(i) = c_0 + ... + c_i is a
    single monotone pass over the whole matrix, taken from row 0 and
    never reset at a block boundary. For p = 0, ..., n_ranks - 2 (in
    increasing p order), boundary p closes immediately after the first
    row index i such that S(i) >= (p + 1) * C / n_ranks (i.e. the first
    row whose global cumulative nonzero count reaches or exceeds a
    (p + 1) / n_ranks share of the whole matrix's total nonzero count);
    the final rank takes every remaining row. Non-degeneracy guard,
    applied left to right in increasing p order: if the scan would ever
    produce an empty block, the boundary is pulled forward by one row so
    every rank owns at least one row."""
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError("A must be 2-D")
    m = A.shape[0]
    if not isinstance(n_ranks, (int, np.integer)) or int(n_ranks) < 1:
        raise ValueError("n_ranks must be a positive integer")
    n_ranks = int(n_ranks)
    if m < n_ranks:
        raise ValueError(f"m={m} rows fewer than n_ranks={n_ranks}")
    c = np.count_nonzero(A, axis=1).astype(np.int64)
    C = int(c.sum())
    if C == 0:
        raise ValueError("A has no nonzero entries; cannot balance the row partition")
    S = np.cumsum(c)
    bounds = []
    lo = 0
    for p in range(n_ranks - 1):
        target = (p + 1) * C / n_ranks
        i = int(np.searchsorted(S, target, side="left"))
        i = min(i, m - 1)
        hi = i + 1
        if hi <= lo:
            hi = lo + 1
        bounds.append((lo, hi))
        lo = hi
    bounds.append((lo, m))
    return bounds


def march_parallel_rgdbek(A: np.ndarray, b: np.ndarray, eta: float, iterations: int, march_seed: int, n_ranks: int = 4) -> dict:
    """Reference implementation of march_parallel_rgdbek: chains
    parallel_column_phase and parallel_row_phase into the
    fixed two-phase iteration, feeding the row phase the auxiliary vector
    the column phase just produced (z_{k+1}, not the pre-update z_k), the
    same timing rule the serial march follows."""
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError("A must be 2-D")
    m, n = A.shape
    if b.shape != (m,):
        raise ValueError("b must have length equal to A's number of rows")
    if not (isinstance(eta, (int, float, np.floating, np.integer)) and 0.0 < float(eta) <= 1.0):
        raise ValueError("eta must satisfy 0 < eta <= 1")
    if not isinstance(iterations, (int, np.integer)) or int(iterations) < 1:
        raise ValueError("iterations must be a positive integer")
    if not isinstance(march_seed, (int, np.integer)) or int(march_seed) < 0:
        raise ValueError("march_seed must be a non-negative integer")
    _row_block_bounds(A, n_ranks)  # validates n_ranks against A up front
    n_ranks = int(n_ranks)

    eta = float(eta)
    iterations = int(iterations)
    march_seed = int(march_seed)
    n_eta = max(1, round(eta * n))
    d_eta = max(1, round(eta * m / n_ranks))

    rng = np.random.default_rng(march_seed)
    x = np.zeros(n)
    z = b.copy()
    error_history = []

    for _k in range(iterations):
        z_next = parallel_column_phase(A, z, n_ranks, n_eta, rng)
        # The row phase consumes z_next (the auxiliary vector the column
        # phase just produced this same iteration), never the pre-update z.
        x_next = parallel_row_phase(A, b, z_next, x, n_ranks, d_eta, rng)

        error_history.append(float(np.linalg.norm(x_next - x)))
        x = x_next
        z = z_next

    return {"x": x, "z": z, "error_history": error_history}

import numpy as np


def convergence_diagnostics(A: np.ndarray, b: np.ndarray, x_serial: np.ndarray, x_parallel: np.ndarray, z_serial: np.ndarray, z_parallel: np.ndarray, x_star: np.ndarray, r_opt: np.ndarray) -> dict:
    """Reference implementation of convergence_diagnostics."""
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    x_serial = np.asarray(x_serial, dtype=np.float64)
    x_parallel = np.asarray(x_parallel, dtype=np.float64)
    z_serial = np.asarray(z_serial, dtype=np.float64)
    z_parallel = np.asarray(z_parallel, dtype=np.float64)
    x_star = np.asarray(x_star, dtype=np.float64)
    r_opt = np.asarray(r_opt, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError("A must be 2-D")
    m, n = A.shape
    if b.shape != (m,):
        raise ValueError("b must have length equal to A's number of rows")
    if r_opt.shape != (m,):
        raise ValueError("r_opt must have length equal to A's number of rows")
    if z_serial.shape != (m,):
        raise ValueError("z_serial must have length equal to A's number of rows")
    if z_parallel.shape != (m,):
        raise ValueError("z_parallel must have length equal to A's number of rows")
    if x_serial.shape != (n,):
        raise ValueError("x_serial must have length equal to A's number of columns")
    if x_parallel.shape != (n,):
        raise ValueError("x_parallel must have length equal to A's number of columns")
    if x_star.shape != (n,):
        raise ValueError("x_star must have length equal to A's number of columns")
    x_star_norm = np.linalg.norm(x_star)
    if x_star_norm == 0.0:
        raise ValueError("x_star must have nonzero norm")
    z0_minus_r = b - r_opt
    denom = np.linalg.norm(z0_minus_r)
    if denom == 0.0:
        raise ValueError("b must not equal r_opt exactly (degenerate z-error-ratio denominator)")

    rel_error_serial = float(np.linalg.norm(x_serial - x_star) / x_star_norm)
    if rel_error_serial == 0.0:
        raise ValueError("x_serial must not equal x_star exactly (degenerate error_ratio denominator)")
    rel_error_parallel = float(np.linalg.norm(x_parallel - x_star) / x_star_norm)
    error_ratio = rel_error_parallel / rel_error_serial
    z_error_ratio_serial = float(np.linalg.norm(z_serial - r_opt) / denom)
    z_error_ratio_parallel = float(np.linalg.norm(z_parallel - r_opt) / denom)

    return {
        "rel_error_serial": rel_error_serial,
        "rel_error_parallel": rel_error_parallel,
        "error_ratio": error_ratio,
        "z_error_ratio_serial": z_error_ratio_serial,
        "z_error_ratio_parallel": z_error_ratio_parallel,
    }

import numpy as np


def _row_block_bounds(A, n_ranks):
    """Contiguous, nonzero-count-balanced row partition of A's rows into
    n_ranks blocks, in index order (S22: same-named function redefined
    elsewhere must be byte-identical). Row i's nonzero count c_i is
    computed once over the whole matrix; C = sum(c_i) is the total
    nonzero count. The running prefix sum S(i) = c_0 + ... + c_i is a
    single monotone pass over the whole matrix, taken from row 0 and
    never reset at a block boundary. For p = 0, ..., n_ranks - 2 (in
    increasing p order), boundary p closes immediately after the first
    row index i such that S(i) >= (p + 1) * C / n_ranks (i.e. the first
    row whose global cumulative nonzero count reaches or exceeds a
    (p + 1) / n_ranks share of the whole matrix's total nonzero count);
    the final rank takes every remaining row. Non-degeneracy guard,
    applied left to right in increasing p order: if the scan would ever
    produce an empty block, the boundary is pulled forward by one row so
    every rank owns at least one row."""
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError("A must be 2-D")
    m = A.shape[0]
    if not isinstance(n_ranks, (int, np.integer)) or int(n_ranks) < 1:
        raise ValueError("n_ranks must be a positive integer")
    n_ranks = int(n_ranks)
    if m < n_ranks:
        raise ValueError(f"m={m} rows fewer than n_ranks={n_ranks}")
    c = np.count_nonzero(A, axis=1).astype(np.int64)
    C = int(c.sum())
    if C == 0:
        raise ValueError("A has no nonzero entries; cannot balance the row partition")
    S = np.cumsum(c)
    bounds = []
    lo = 0
    for p in range(n_ranks - 1):
        target = (p + 1) * C / n_ranks
        i = int(np.searchsorted(S, target, side="left"))
        i = min(i, m - 1)
        hi = i + 1
        if hi <= lo:
            hi = lo + 1
        bounds.append((lo, hi))
        lo = hi
    bounds.append((lo, m))
    return bounds


def run_rgdbek_comparison(seed: int = 20260920, march_seed: int = 7, m: int = 600, n: int = 150, density: float = 0.20, noise_ratio: float = 0.10, eta: float = 0.2, iterations: int = 12, n_ranks: int = 4) -> float:
    """Reference implementation of run_rgdbek_comparison: chains
    build_kaczmarz_instance, then runs the serial march both step by
    step (through column_phase_update / row_phase_update)
    and via the fused march_rgdbek, cross-checking the two agree;
    runs the parallel march both step by step (through
    parallel_column_phase / parallel_row_phase) and via the
    fused march_parallel_rgdbek, cross-checking the two agree; calls
    sample_index_block on a fixed input as a standalone conformance
    check of the shared sampling primitive's own no-repeat/in-range
    contract; and finally calls convergence_diagnostics on both
    marches' results."""
    # Standalone conformance check on the shared sampling primitive: every
    # step that samples (02's own function, and the locally-reproduced
    # copies inside 03/04/06/07) must draw distinct, in-range indices. A
    # broken sample_index_block oracle would violate this on this fixed
    # input regardless of which march consumes it.
    _sample_check = sample_index_block(np.array([0.2, 0.3, 0.5]), 2, np.random.default_rng(0))
    if len(set(_sample_check.tolist())) != 2 or np.any(_sample_check < 0) or np.any(_sample_check >= 3):
        raise ValueError("sample_index_block oracle violated its own no-repeat/in-range contract")

    instance = build_kaczmarz_instance(seed=seed, m=m, n=n, density=density, noise_ratio=noise_ratio)
    A, b = instance["A"], instance["b"]
    m_rows, n_cols = A.shape
    n_eta = max(1, round(eta * n_cols))
    m_eta = max(1, round(eta * m_rows))

    # --- serial march: step-by-step, then cross-checked against the fused oracle ---
    rng = np.random.default_rng(march_seed)
    col_sq_norms = np.sum(A ** 2, axis=0)
    row_sq_norms = np.sum(A ** 2, axis=1)
    x = np.zeros(n_cols)
    z = b.copy()
    for _k in range(iterations):
        z_next = column_phase_update(A, z, col_sq_norms, n_eta, rng)
        x_next = row_phase_update(A, b, z_next, x, row_sq_norms, m_eta, rng)
        x = x_next
        z = z_next

    fused_serial = march_rgdbek(A, b, eta, iterations, march_seed)
    if not (np.allclose(x, fused_serial["x"], rtol=1e-9, atol=1e-12)
            and np.allclose(z, fused_serial["z"], rtol=1e-9, atol=1e-12)):
        raise ValueError("step-by-step serial march disagrees with the fused march (march_rgdbek) beyond tolerance")

    # --- parallel march: step-by-step, then cross-checked against the fused oracle ---
    _row_block_bounds(A, n_ranks)  # validates n_ranks against A up front
    d_eta = max(1, round(eta * m_rows / n_ranks))
    rng_par = np.random.default_rng(march_seed)
    x_par = np.zeros(n_cols)
    z_par = b.copy()
    for _k in range(iterations):
        z_par_next = parallel_column_phase(A, z_par, n_ranks, n_eta, rng_par)
        x_par_next = parallel_row_phase(A, b, z_par_next, x_par, n_ranks, d_eta, rng_par)
        x_par = x_par_next
        z_par = z_par_next

    fused_parallel = march_parallel_rgdbek(A, b, eta, iterations, march_seed, n_ranks=n_ranks)
    if not (np.allclose(x_par, fused_parallel["x"], rtol=1e-9, atol=1e-12)
            and np.allclose(z_par, fused_parallel["z"], rtol=1e-9, atol=1e-12)):
        raise ValueError("step-by-step parallel march disagrees with the fused march (march_parallel_rgdbek) beyond tolerance")

    diagnostics = convergence_diagnostics(
        A, b, x, x_par, z, z_par, instance["x_star"], instance["r_opt"]
    )
    return diagnostics["error_ratio"]
SCICODE_GOLD_EOF
