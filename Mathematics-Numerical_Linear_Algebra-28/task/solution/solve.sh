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


def build_instance(
    n: int = 96,
    sketch_size: int = 45,
    zeta: int = 4,
    seed: int = 2026,
) -> tuple:
    """Reference implementation."""
    for name, value in (("n", n), ("sketch_size", sketch_size), ("zeta", zeta), ("seed", seed)):
        if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} must be an integer")

    n = int(n)
    s = int(sketch_size)
    z = int(zeta)

    if n < 4 or n % 4 != 0:
        raise ValueError("n must satisfy n >= 4 and be divisible by 4")

    if s < 1 or z < 1 or z > s:
        raise ValueError("sketch parameters must satisfy 1 <= zeta <= sketch_size")

    A = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        idx = i + 1
        A[i, i] = np.log(99.0 + idx)

    for i in range(n - 1):
        idx = i + 1
        value = (
            0.19 * np.sin(0.73 * idx)
            + 0.004 * np.cos(0.41 * idx)
        )
        A[i, i + 1] = value
        A[i + 1, i] = value

    j = np.arange(1, n + 1, dtype=np.float64)
    v1 = np.ones(n, dtype=np.float64) / np.sqrt(n)
    v2 = ((-1.0) ** (j - 1.0)) / np.sqrt(n)
    v3 = np.tile(np.array([1.0, 1.0, -1.0, -1.0], dtype=np.float64), n // 4) / np.sqrt(n)
    V0 = np.column_stack((v1, v2, v3))

    rng = np.random.default_rng(int(seed))
    S = np.zeros((s, n), dtype=np.float64)
    scale = 1.0 / np.sqrt(float(z))

    for col in range(n):
        rows = rng.choice(s, z, replace=False)
        signs = 2 * rng.integers(0, 2, size=z) - 1
        S[rows, col] = signs.astype(np.float64) * scale

    return A, V0, S

import numpy as np


def rcgs_initial(state: tuple) -> tuple:
    """Reference implementation."""

    if not isinstance(state, tuple) or len(state) != 3:
        raise ValueError("state must be the Step 01 tuple")

    A, V0, S = [
        np.asarray(x, dtype=np.float64)
        for x in state
    ]

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")

    n = A.shape[0]

    if V0.ndim != 2 or V0.shape[0] != n:
        raise ValueError("V0 must have shape (n,p)")

    if S.ndim != 2 or S.shape[1] != n:
        raise ValueError("S must have shape (s,n)")

    if (
        not np.all(np.isfinite(A))
        or not np.all(np.isfinite(V0))
        or not np.all(np.isfinite(S))
    ):
        raise ValueError("inputs must be finite")

    T = V0.copy()
    P = S @ T

    for ell in range(T.shape[1]):
        if ell:
            h = P[:, :ell].T @ P[:, ell]
            T[:, ell] -= T[:, :ell] @ h
            P[:, ell] -= P[:, :ell] @ h

        norm_p = float(np.linalg.norm(P[:, ell]))

        if not np.isfinite(norm_p) or norm_p <= 1e-14:
            raise ValueError("rank-deficient initial block")

        T[:, ell] /= norm_p
        P[:, ell] /= norm_p

    if not np.allclose(
        P.T @ P,
        np.eye(P.shape[1]),
        rtol=1e-10,
        atol=1e-12,
    ):
        raise ValueError("initial block is not sketch-orthonormal")

    return A, S, T, P

import numpy as np
from scipy.linalg import eigh


def generalized_ritz(state: tuple, k: int = 3) -> tuple:
    """Reference implementation."""
    if not isinstance(state, tuple) or len(state) != 4:
        raise ValueError("state must be (A, S, Vt, Q)")

    A, S, Vt, Q = [
        np.asarray(x, dtype=np.float64)
        for x in state
    ]

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")

    n = int(A.shape[0])

    if n < 1:
        raise ValueError("A must have positive dimension")

    if not np.all(np.isfinite(A)):
        raise ValueError("A must contain only finite values")

    if S.ndim != 2 or S.shape[1] != n:
        raise ValueError("S must have shape (s,n)")

    if Vt.ndim != 2 or Vt.shape[0] != n or Vt.shape[1] < 1:
        raise ValueError("Vt must have shape (n,j) with j >= 1")

    if Q.ndim != 2:
        raise ValueError("Q must be two-dimensional")

    if Q.shape[0] != S.shape[0] or Q.shape[1] != Vt.shape[1]:
        raise ValueError("inconsistent Vt and Q shapes")

    if not np.all(np.isfinite(S)):
        raise ValueError("S must contain only finite values")

    if not np.all(np.isfinite(Vt)):
        raise ValueError("Vt must contain only finite values")

    if not np.all(np.isfinite(Q)):
        raise ValueError("Q must contain only finite values")

    if not isinstance(k, (int, np.integer)) or isinstance(
        k, (bool, np.bool_)
    ):
        raise ValueError("k must be an integer")

    k = int(k)

    if k < 1 or k > Vt.shape[1]:
        raise ValueError("invalid k")

    # Operator action on the current search basis.
    W = A @ Vt

    if not np.all(np.isfinite(W)):
        raise ValueError("operator-applied basis contains non-finite values")

    # Full-space reduced quantities.
    G = Vt.T @ Vt
    H = Vt.T @ W

    if not np.all(np.isfinite(G)) or not np.all(np.isfinite(H)):
        raise ValueError("reduced matrices contain non-finite values")

    # The reduced problem is solved through the positive-definite Gram matrix.
    try:
        vals, vecs = eigh(
            H,
            G,
            lower=True,
            check_finite=True,
        )
    except np.linalg.LinAlgError as exc:
        raise ValueError("spectral extraction failed") from exc

    theta = vals[:k].copy()
    Y = vecs[:, :k].copy()

    # Construct physical-space vectors and impose a deterministic global
    # sign convention.
    U = Vt @ Y

    for c in range(k):
        nz = np.flatnonzero(np.abs(U[:, c]) > 1e-15)
        if nz.size and U[nz[0], c] < 0.0:
            U[:, c] *= -1.0
            Y[:, c] *= -1.0

    # Residuals in the original space.
    R = W @ Y - U * theta[None, :]

    if (
        not np.all(np.isfinite(theta))
        or not np.all(np.isfinite(Y))
        or not np.all(np.isfinite(U))
        or not np.all(np.isfinite(R))
    ):
        raise ValueError("spectral output contains non-finite values")

    return A, S, Vt, Q, W, theta, Y, U, R

import numpy as np


def davidson_correction(state: tuple) -> tuple:
    """Reference implementation."""
    if not isinstance(state, tuple) or len(state) != 9:
        raise ValueError("state must be the Step 03 tuple")

    A, S, Vt, Q, W, theta, Y, U, R = [
        np.asarray(x, dtype=np.float64) for x in state
    ]

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")

    n = int(A.shape[0])

    if S.ndim != 2 or S.shape[1] != n:
        raise ValueError("S must have shape (s,n)")

    if Vt.ndim != 2 or Vt.shape[0] != n:
        raise ValueError("Vt must have shape (n,j)")

    if Q.ndim != 2 or Q.shape[1] != Vt.shape[1]:
        raise ValueError("Q and Vt dimensions are inconsistent")

    if W.ndim != 2 or W.shape != Vt.shape:
        raise ValueError("W must have the same shape as Vt")

    if theta.ndim != 1:
        raise ValueError("theta must be one-dimensional")

    k = int(theta.size)

    if k < 1:
        raise ValueError("theta must contain at least one value")

    if Y.ndim != 2 or Y.shape != (Vt.shape[1], k):
        raise ValueError("Y has inconsistent shape")

    if U.ndim != 2 or U.shape != (n, k):
        raise ValueError("U has inconsistent shape")

    if R.ndim != 2 or R.shape != (n, k):
        raise ValueError("R has inconsistent shape")

    if not all(
        np.all(np.isfinite(x))
        for x in (A, S, Vt, Q, W, theta, Y, U, R)
    ):
        raise ValueError("state contains non-finite values")

    diagonal = np.diag(A)

    denom = diagonal[:, None] - theta[None, :]
    singular = np.abs(denom) <= 1e-14

    # A zero denominator is admissible only where the corresponding residual
    # component is numerically zero.
    if np.any(singular & (np.abs(R) > 1e-14)):
        raise ValueError(
            "singular correction denominator for a nonzero residual"
        )

    T = np.zeros_like(R, dtype=np.float64)
    valid = ~singular
    T[valid] = R[valid] / denom[valid]

    if not np.all(np.isfinite(T)):
        raise ValueError("correction block contains non-finite values")

    return A, S, Vt, Q, W, theta, Y, U, R, T

import numpy as np


def sketched_block_expansion(state: tuple) -> tuple:
    """Reference implementation."""
    if not isinstance(state, tuple) or len(state) != 6:
        raise ValueError("state must be (A, S, Vt, Q, W, T)")

    A, S, Vt, Q, W, T = [np.asarray(x, dtype=np.float64) for x in state]

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")

    n = A.shape[0]

    if S.ndim != 2 or S.shape[1] != n:
        raise ValueError("S must have shape (s,n)")

    if Vt.ndim != 2 or Vt.shape[0] != n or W.shape != Vt.shape:
        raise ValueError("Vt and W must have shape (n,j)")

    if Q.ndim != 2 or Q.shape != (S.shape[0], Vt.shape[1]):
        raise ValueError("Q must have shape (s,j)")

    if T.ndim != 2 or T.shape[0] != n or T.shape[1] < 1:
        raise ValueError("T must have shape (n,p)")

    if Vt.shape[1] + T.shape[1] > S.shape[0]:
        raise ValueError("sketch has too few rows for the expanded basis")

    if not all(np.all(np.isfinite(x)) for x in (A, S, Vt, Q, W, T)):
        raise ValueError("state contains non-finite values")

    # Stage 1, one pass: coefficients from the stored sketch, removed from both spaces.
    P = S @ T
    C = Q.T @ P
    T_hat = T - Vt @ C
    P = P - Q @ C

    # Stage 2: column loop in the sketched norm.
    for ell in range(T_hat.shape[1]):
        if ell:
            h = P[:, :ell].T @ P[:, ell]
            T_hat[:, ell] -= T_hat[:, :ell] @ h
            P[:, ell] -= P[:, :ell] @ h

        norm_p = float(np.linalg.norm(P[:, ell]))

        if not np.isfinite(norm_p) or norm_p <= 1e-14:
            raise ValueError("rank-deficient sketched correction block")

        T_hat[:, ell] /= norm_p
        P[:, ell] /= norm_p

    return (
        A,
        S,
        np.column_stack((Vt, T_hat)),
        np.column_stack((Q, P)),
        np.column_stack((W, A @ T_hat)),
    )

import numpy as np


def restart_rotation(state: tuple) -> tuple:
    """Reference implementation."""
    if not isinstance(state, tuple) or len(state) != 9:
        raise ValueError("state must be the Step 03 tuple")

    A, S, Vt, Q, W, theta, Y, U, R = [np.asarray(x, dtype=np.float64) for x in state]

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")

    if Vt.ndim != 2 or Vt.shape[0] != A.shape[0] or W.shape != Vt.shape:
        raise ValueError("Vt and W must have shape (n,j)")

    if Q.ndim != 2 or Q.shape[1] != Vt.shape[1]:
        raise ValueError("Q and Vt have inconsistent dimensions")

    if Y.ndim != 2 or Y.shape[0] != Vt.shape[1]:
        raise ValueError("Y has inconsistent shape")

    if not all(np.all(np.isfinite(x)) for x in (A, S, Vt, Q, W, Y)):
        raise ValueError("state contains non-finite values")

    return A, S, Vt @ Y, Q @ Y, W @ Y

import numpy as np


def post_restart_expansion(state: tuple) -> tuple:
    """Reference implementation."""
    return sketched_block_expansion(state)

import numpy as np


def solve_once_restarted_ritz_entry(
    n: int = 96,
    k: int = 3,
    jmax: int = 9,
    sketch_size: int = 45,
    zeta: int = 4,
    seed: int = 2026,
) -> float:
    """Reference implementation integrating the earlier scientific steps."""
    if not isinstance(k, (int, np.integer)) or isinstance(k, (bool, np.bool_)):
        raise ValueError("k must be an integer")
    if not isinstance(jmax, (int, np.integer)) or isinstance(jmax, (bool, np.bool_)):
        raise ValueError("jmax must be an integer")
    if int(k) != 3:
        raise ValueError("this benchmark fixes k=3")
    if int(jmax) < 2 * int(k) or int(jmax) % int(k) != 0:
        raise ValueError("invalid jmax")

    k = int(k)
    jmax = int(jmax)

    A, V0, S = build_instance(
        n, sketch_size=sketch_size, zeta=zeta, seed=seed
    )
    A, S, Vt, Q = rcgs_initial((A, V0, S))
    W = A @ Vt

    # First cycle: two normal sketch-orthonormal expansion transitions.
    extraction = generalized_ritz((A, S, Vt, Q), k=k)
    while extraction[2].shape[1] < jmax:
        correction = davidson_correction(extraction)
        A, S, Vt, Q, W = sketched_block_expansion(
            (A, S, Vt, Q, W, correction[-1])
        )
        extraction = generalized_ritz((A, S, Vt, Q), k=k)

    # One source-defined restart, with no re-sketch-orthonormalization.
    restarted = restart_rotation(extraction)
    A, S, Vt, Q, W = restarted

    # Second cycle: two post-restart/oblique expansion transitions.
    extraction = generalized_ritz((A, S, Vt, Q), k=k)
    while extraction[2].shape[1] < jmax:
        correction = davidson_correction(extraction)
        A, S, Vt, Q, W = post_restart_expansion(
            (A, S, Vt, Q, W, correction[-1])
        )
        extraction = generalized_ritz((A, S, Vt, Q), k=k)

    Vt_final = extraction[2]
    eigs = np.linalg.eigvalsh(Vt_final.T @ Vt_final)
    if eigs.size != Vt_final.shape[1] or np.any(eigs <= 0.0):
        raise ValueError("final Gram matrix is not positive definite")

    result = float(eigs[-1] / eigs[0])
    if not np.isfinite(result):
        raise ValueError("result is not finite")
    return result
SCICODE_GOLD_EOF
