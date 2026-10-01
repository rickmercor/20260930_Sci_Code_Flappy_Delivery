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

def construct_rs_deflation(
    B: np.ndarray,
    T: np.ndarray,
) -> np.ndarray:
    """Reference implementation of the deterministic RS construction."""
    B = np.asarray(B, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)

    if B.ndim != 2 or T.ndim != 2:
        raise ValueError("B and T must be 2D arrays.")

    if B.shape[1] != T.shape[1]:
        raise ValueError("B and T must have the same number of modes.")

    if B.shape[0] < 1 or T.shape[0] < 1:
        raise ValueError("B and T must be non-empty.")

    return T @ B.T

import numpy as np

def split_deflation_blocks(
    P_tilde: np.ndarray,
    groups: tuple[np.ndarray, ...],
) -> np.ndarray:
    """Validate groups while preserving values and original global row order."""
    P_tilde = np.asarray(P_tilde, dtype=np.float64)

    if P_tilde.ndim != 2:
        raise ValueError("P_tilde must be 2D.")

    n = P_tilde.shape[0]
    preserved = np.empty_like(P_tilde, dtype=np.float64)

    seen = set()

    for group in groups:
        idx = np.asarray(group, dtype=int)

        if idx.ndim != 1 or idx.size == 0:
            raise ValueError("Each group must be a non-empty 1D index array.")

        if np.any(idx < 0) or np.any(idx >= n):
            raise ValueError("Group index out of range.")

        for i in idx:
            if int(i) in seen:
                raise ValueError("Groups must be disjoint.")
            seen.add(int(i))

        preserved[idx, :] = P_tilde[idx, :]

    if len(seen) != n:
        raise ValueError("Groups must cover all degrees of freedom.")

    return preserved

import numpy as np

def build_block_deflation_operator(
    grouped_p_tilde: np.ndarray,
    groups: tuple[np.ndarray, ...],
) -> np.ndarray:
    """Reference Eq. (25): place each local Q block at its original rows."""
    grouped_p_tilde = np.asarray(
        grouped_p_tilde,
        dtype=np.float64,
    )

    n = grouped_p_tilde.shape[0]
    k = grouped_p_tilde.shape[1]
    S = len(groups)

    P = np.zeros(
        (n, k * S),
        dtype=np.float64,
    )

    seen = set()

    for s, group in enumerate(groups):
        idx = np.asarray(group, dtype=int)

        if idx.ndim != 1 or idx.size == 0:
            raise ValueError("Each group must be a non-empty 1D index array.")

        if idx.size < k:
            raise ValueError(
                "Each block must have at least k rows."
            )

        if any(int(i) in seen for i in idx):
            raise ValueError("Groups must be disjoint.")

        for i in idx:
            seen.add(int(i))

        block = grouped_p_tilde[idx, :]

        Q, _ = np.linalg.qr(
            block,
            mode="reduced",
        )

        P[idx, s * k:(s + 1) * k] = Q

    if len(seen) != n:
        raise ValueError("Groups must cover all degrees of freedom.")

    return P

import numpy as np

def build_coarse_operators(
    A: np.ndarray,
    P: np.ndarray,
) -> np.ndarray:
    """Reference coarse-space construction with deterministic C-order packing."""
    A = np.asarray(A, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square.")

    if P.ndim != 2 or P.shape[0] != A.shape[0]:
        raise ValueError("P must have the same row count as A.")

    R = P.T
    Ac = R @ A @ P
    C = P @ np.linalg.solve(Ac, R)

    return np.concatenate([
        R.ravel(order="C"),
        Ac.ravel(order="C"),
        C.ravel(order="C"),
    ])

import numpy as np

def initialize_dpcg(
    A: np.ndarray,
    f: np.ndarray,
    u00: np.ndarray,
    M: np.ndarray,
    P: np.ndarray,
    coarse_data: np.ndarray,
) -> np.ndarray:
    """Reference implementation of DPCG initialization."""
    A = np.asarray(A, dtype=np.float64)
    f = np.asarray(f, dtype=np.float64)
    u00 = np.asarray(u00, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    coarse_data = np.asarray(coarse_data, dtype=np.float64)

    n = A.shape[0]
    k = P.shape[1]

    if A.ndim != 2 or A.shape != (n, n):
        raise ValueError("A must be square.")

    if f.shape != (n,) or u00.shape != (n,):
        raise ValueError("f and u00 must have shape (n,).")

    if M.shape != (n, n):
        raise ValueError("M must have shape (n, n).")

    if P.ndim != 2 or P.shape[0] != n:
        raise ValueError("P has incompatible dimensions.")

    r_size = k * n
    ac_size = k * k
    c_size = n * n

    expected_coarse_size = r_size + ac_size + c_size

    if coarse_data.size != expected_coarse_size:
        raise ValueError("Invalid packed coarse-data size.")

    offset = 0

    R = coarse_data[offset:offset + r_size].reshape(k, n)
    offset += r_size

    Ac = coarse_data[offset:offset + ac_size].reshape(k, k)
    offset += ac_size

    C = coarse_data[offset:offset + c_size].reshape(n, n)

    u0 = u00 + C @ (f - A @ u00)
    r0 = f - A @ u0
    z0 = M @ r0
    mu0 = np.linalg.solve(Ac, R @ A @ z0)
    p0 = z0 - P @ mu0

    return np.concatenate([
        u0,
        r0,
        z0,
        p0,
        mu0,
    ])

import numpy as np

def dpcg_update(
    A: np.ndarray,
    M: np.ndarray,
    P: np.ndarray,
    coarse_data: np.ndarray,
    state: np.ndarray,
) -> np.ndarray:
    """Reference implementation of one complete DPCG update."""
    A = np.asarray(A, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    coarse_data = np.asarray(coarse_data, dtype=np.float64)
    state = np.asarray(state, dtype=np.float64)

    n = A.shape[0]
    k = P.shape[1]

    # Unpack coarse operators.
    r_size = k * n
    ac_size = k * k
    c_size = n * n

    offset = 0
    R = coarse_data[offset:offset + r_size].reshape(k, n)
    offset += r_size

    Ac = coarse_data[offset:offset + ac_size].reshape(k, k)

    # Unpack current state.
    offset = 0

    u = state[offset:offset + n]
    offset += n

    r = state[offset:offset + n]
    offset += n

    z = state[offset:offset + n]
    offset += n

    p = state[offset:offset + n]
    offset += n

    mu = state[offset:offset + k]

    alpha = np.dot(r, z) / np.dot(p, A @ p)

    u_new = u + alpha * p
    r_new = r - alpha * (A @ p)
    z_new = M @ r_new

    mu_new = np.linalg.solve(
        Ac,
        R @ A @ z_new,
    )

    beta = np.dot(r_new, z_new) / np.dot(r, z)

    p_new = (
        beta * p
        + z_new
        - P @ mu_new
    )

    return np.concatenate([
        u_new,
        r_new,
        z_new,
        p_new,
        mu_new,
        np.array([alpha, beta], dtype=np.float64),
    ])

import numpy as np

def run_three_dpcg_updates(
    A: np.ndarray,
    M: np.ndarray,
    P: np.ndarray,
    coarse_data: np.ndarray,
    state: np.ndarray,
) -> np.ndarray:
    """Reference execution of exactly three DPCG updates."""
    current = np.asarray(state, dtype=np.float64)

    for _ in range(3):
        current = dpcg_update(
            A,
            M,
            P,
            coarse_data,
            current,
        )

        current = current[:-2]

    n = A.shape[0]
    return current[:n].copy()

import numpy as np

def compute_final_component(
    A: np.ndarray,
    f: np.ndarray,
    u00: np.ndarray,
    M: np.ndarray,
    B: np.ndarray,
    T: np.ndarray,
    groups: tuple[np.ndarray, ...],
) -> float:
    """Reference end-to-end computation."""
    P_tilde = construct_rs_deflation(B, T)

    grouped = split_deflation_blocks(
        P_tilde,
        groups,
    )

    P = build_block_deflation_operator(
        grouped,
        groups,
    )

    coarse_data = build_coarse_operators(
        A,
        P,
    )

    state = initialize_dpcg(
        A,
        f,
        u00,
        M,
        P,
        coarse_data,
    )

    u3 = run_three_dpcg_updates(
        A,
        M,
        P,
        coarse_data,
        state,
    )

    return float(u3[5])
SCICODE_GOLD_EOF
