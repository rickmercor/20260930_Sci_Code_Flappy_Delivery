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


def network_summary(A: "np.ndarray") -> "np.ndarray":
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape[0] < 2:
        raise ValueError("A must be a square matrix with at least two nodes")
    if not np.all(np.isfinite(A)):
        raise ValueError("A must contain only finite values")
    if not np.all((A == 0.0) | (A == 1.0)):
        raise ValueError("A must be binary")
    if not np.allclose(A, A.T, rtol=0.0, atol=0.0):
        raise ValueError("A must be symmetric")
    if not np.all(np.diag(A) == 0.0):
        raise ValueError("A must have a zero diagonal")

    N = A.shape[0]
    seen = np.zeros(N, dtype=bool)
    stack = [0]
    seen[0] = True
    while stack:
        i = stack.pop()
        for j in np.flatnonzero(A[i]):
            j = int(j)
            if not seen[j]:
                seen[j] = True
                stack.append(j)
    if not np.all(seen):
        raise ValueError("A must describe a connected graph")

    M = float(np.sum(A) / 2.0)
    return np.array([float(N), M], dtype=float)

import numpy as np
from scipy.integrate import solve_ivp


def double_well_states(
    A: "np.ndarray",
    D_grid: "np.ndarray",
    T: float,
) -> "np.ndarray":
    A = np.asarray(A, dtype=float)
    D_grid = np.asarray(D_grid, dtype=float)
    network_summary(A)

    if D_grid.ndim != 1 or D_grid.size < 2 or not np.all(np.isfinite(D_grid)):
        raise ValueError("D_grid must be a finite one-dimensional grid with at least two values")
    if not np.all(np.diff(D_grid) > 0.0):
        raise ValueError("D_grid must be strictly increasing")
    if D_grid[0] < 0.0 or D_grid[-1] > 1.0:
        raise ValueError("D_grid must lie in [0,1]")
    if not np.isfinite(T) or T <= 0.0:
        raise ValueError("T must be positive and finite")

    N = A.shape[0]
    initial = np.ones(N, dtype=float)
    out = np.empty((D_grid.size, N), dtype=float)

    def _rhs(_t, x, D):
        return -(x - 1.0) * (x - 3.0) * (x - 5.0) + D * (A @ x)

    for k, D in enumerate(D_grid):
        sol = solve_ivp(
            _rhs,
            (0.0, float(T)),
            initial,
            args=(float(D),),
            method="DOP853",
            rtol=1e-11,
            atol=1e-13,
            max_step=0.1,
        )
        if not sol.success or not np.all(np.isfinite(sol.y[:, -1])):
            raise ValueError("double-well integration failed")
        out[k] = sol.y[:, -1]

    return out

import numpy as np


def network_average(states: "np.ndarray") -> "np.ndarray":
    states = np.asarray(states, dtype=float)
    if states.ndim != 2 or states.shape[0] < 1 or states.shape[1] < 1:
        raise ValueError("states must have shape (L,N) with L,N >= 1")
    if not np.all(np.isfinite(states)):
        raise ValueError("states must be finite")
    return np.mean(states, axis=1)

import numpy as np


def sentinel_error(states: "np.ndarray", mean_activity: "np.ndarray", sentinel_indices: "np.ndarray") -> float:
    states = np.asarray(states, dtype=float)
    mean_activity = np.asarray(mean_activity, dtype=float)
    sentinel_indices = np.asarray(sentinel_indices)
    if states.ndim != 2 or states.shape[0] < 1 or states.shape[1] < 1:
        raise ValueError("states must have shape (L,N)")
    if mean_activity.ndim != 1 or mean_activity.shape[0] != states.shape[0]:
        raise ValueError("mean_activity must have length L")
    if sentinel_indices.ndim != 1 or sentinel_indices.size < 1:
        raise ValueError("sentinel_indices must be a nonempty one-dimensional array")
    if not np.issubdtype(sentinel_indices.dtype, np.integer):
        raise ValueError("sentinel_indices must contain integers")
    if np.any(sentinel_indices < 0) or np.any(sentinel_indices >= states.shape[1]):
        raise ValueError("sentinel index out of bounds")
    if np.unique(sentinel_indices).size != sentinel_indices.size:
        raise ValueError("sentinel_indices must not contain duplicates")
    if not np.all(np.isfinite(states)) or not np.all(np.isfinite(mean_activity)):
        raise ValueError("states and mean_activity must be finite")
    denominator = float(np.sum(mean_activity))
    if denominator <= 0.0:
        raise ValueError("sum of mean_activity must be positive")
    approximation = np.mean(states[:, sentinel_indices], axis=1)
    return float(np.sum((approximation - mean_activity) ** 2) / denominator)

import numpy as np


def select_sentinels(
    states: "np.ndarray",
    mean_activity: "np.ndarray",
    n: int,
    seed: int,
) -> "np.ndarray":
    states = np.asarray(states, dtype=float)
    mean_activity = np.asarray(mean_activity, dtype=float)

    if states.ndim != 2 or states.shape[0] < 2 or states.shape[1] < 2:
        raise ValueError("states must have shape (L,N) with L,N >= 2")
    if mean_activity.shape != (states.shape[0],):
        raise ValueError("mean_activity must have length L")
    if not np.all(np.isfinite(states)) or not np.all(np.isfinite(mean_activity)):
        raise ValueError("states and mean_activity must be finite")
    if not isinstance(n, (int, np.integer)) or n < 1 or n >= states.shape[1]:
        raise ValueError("n must satisfy 1 <= n < N")
    if not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")

    denominator = float(np.sum(mean_activity))
    if denominator <= 0.0:
        raise ValueError("sum of mean_activity must be positive")

    rng = np.random.default_rng(int(seed))
    N = states.shape[1]
    sentinel = np.sort(rng.choice(N, size=int(n), replace=False)).astype(int)

    def _error_of(S):
        approx = np.mean(states[:, S], axis=1)
        return float(np.sum((approx - mean_activity) ** 2) / denominator)

    current_error = _error_of(sentinel)
    best_sentinel = sentinel.copy()
    best_error = current_error
    h_max = 50 * N

    for h in range(1, h_max + 1):
        replace_pos = int(rng.integers(0, n))
        outside = np.setdiff1d(
            np.arange(N, dtype=int),
            sentinel,
            assume_unique=True,
        )
        replacement = int(rng.choice(outside))
        candidate = sentinel.copy()
        candidate[replace_pos] = replacement
        candidate.sort()
        candidate_error = _error_of(candidate)

        if candidate_error < current_error:
            accept = True
        else:
            temperature = 10.0 / np.log(h + np.e - 1.0)
            probability = np.exp(
                -(candidate_error - current_error) / temperature
            )
            accept = bool(rng.random() < probability)

        if accept:
            sentinel = candidate
            current_error = candidate_error

        if (
            candidate_error < best_error
            or (
                candidate_error == best_error
                and tuple(candidate.tolist())
                < tuple(best_sentinel.tolist())
            )
        ):
            best_sentinel = candidate.copy()
            best_error = candidate_error

    return best_sentinel

import numpy as np
from scipy.integrate import solve_ivp


def sis_states(
    A: "np.ndarray",
    lambda_grid: "np.ndarray",
    T: float,
) -> "np.ndarray":
    A = np.asarray(A, dtype=float)
    lambda_grid = np.asarray(lambda_grid, dtype=float)
    network_summary(A)

    if (
        lambda_grid.ndim != 1
        or lambda_grid.size < 2
        or not np.all(np.isfinite(lambda_grid))
    ):
        raise ValueError(
            "lambda_grid must be a finite one-dimensional grid with at least two values"
        )
    if not np.all(np.diff(lambda_grid) > 0.0):
        raise ValueError("lambda_grid must be strictly increasing")
    if lambda_grid[0] < 0.0 or lambda_grid[-1] > 1.0:
        raise ValueError("lambda_grid must lie in [0,1]")
    if not np.isfinite(T) or T <= 0.0:
        raise ValueError("T must be positive and finite")

    N = A.shape[0]
    initial = np.full(N, 0.01, dtype=float)
    out = np.empty((lambda_grid.size, N), dtype=float)

    def _rhs(_t, x, lam):
        return -x + lam * (1.0 - x) * (A @ x)

    for k, lam in enumerate(lambda_grid):
        sol = solve_ivp(
            _rhs,
            (0.0, float(T)),
            initial,
            args=(float(lam),),
            method="DOP853",
            rtol=1e-11,
            atol=1e-13,
            max_step=0.1,
        )
        if not sol.success or not np.all(np.isfinite(sol.y[:, -1])):
            raise ValueError("SIS integration failed")
        out[k] = sol.y[:, -1]

    return out

import numpy as np


def transfer_error(test_states: "np.ndarray", sentinel_indices: "np.ndarray") -> float:
    test_states = np.asarray(test_states, dtype=float)
    sentinel_indices = np.asarray(sentinel_indices)
    if test_states.ndim != 2 or test_states.shape[0] < 1 or test_states.shape[1] < 1:
        raise ValueError("test_states must have shape (L,N)")
    if sentinel_indices.ndim != 1 or sentinel_indices.size < 1:
        raise ValueError("sentinel_indices must be nonempty and one-dimensional")
    if not np.issubdtype(sentinel_indices.dtype, np.integer):
        raise ValueError("sentinel_indices must contain integers")
    if np.any(sentinel_indices < 0) or np.any(sentinel_indices >= test_states.shape[1]):
        raise ValueError("sentinel index out of bounds")
    if np.unique(sentinel_indices).size != sentinel_indices.size:
        raise ValueError("sentinel_indices must be unique")
    if not np.all(np.isfinite(test_states)):
        raise ValueError("test_states must be finite")

    full_mean = np.mean(test_states, axis=1)
    denominator = float(np.sum(full_mean))
    if denominator <= 0.0:
        raise ValueError("test-dynamics normalization must be positive")
    sentinel_mean = np.mean(test_states[:, sentinel_indices], axis=1)
    return float(np.sum((sentinel_mean - full_mean) ** 2) / denominator)

import numpy as np


def transfer_penalty(training_error: float, test_error: float) -> float:
    if not np.isfinite(training_error) or not np.isfinite(test_error):
        raise ValueError("errors must be finite")
    if training_error <= 0.0:
        raise ValueError("training_error must be positive")
    if test_error < 0.0:
        raise ValueError("test_error must be non-negative")
    return float(test_error / training_error)

import numpy as np


def sentinel_transfer_pipeline(A: "np.ndarray", D_grid: "np.ndarray", lambda_grid: "np.ndarray", seed: int = 123) -> float:
    summary = network_summary(A)
    N = int(round(summary[0]))
    n = int(np.floor(np.log(N)))
    if n < 1:
        raise ValueError("network is too small for a sentinel set")

    training_states = double_well_states(A, D_grid, 15.0)
    training_mean = network_average(training_states)
    sentinel = select_sentinels(training_states, training_mean, n, seed)
    training_error = sentinel_error(training_states, training_mean, sentinel)

    test_states = sis_states(A, lambda_grid, 15.0)
    test_error = transfer_error(test_states, sentinel)
    return transfer_penalty(training_error, test_error)
SCICODE_GOLD_EOF
