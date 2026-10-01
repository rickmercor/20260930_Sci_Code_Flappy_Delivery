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


def build_augmented_block(

    b: "np.ndarray",

    ell: int,

    seed: int,

) -> "np.ndarray":

    """
    Construct the paper-specific randomized starting block.

    The cited paper defines the alternative initialization for the
    matrix-function setting by using b as the first column and
    Gaussian enrichment directions that are orthonormalized relative
    to b / ||b||_2 and to each other.
    """

    b = np.asarray(b, dtype=float)

    if b.ndim != 1:
        raise ValueError("b must be one-dimensional.")
    if not np.all(np.isfinite(b)):
        raise ValueError("b must contain only finite values.")
    if np.linalg.norm(b) == 0.0:
        raise ValueError("b must have nonzero norm.")
    if not isinstance(ell, (int, np.integer)):
        raise ValueError("ell must be an integer.")
    if ell < 2 or ell > b.size:
        raise ValueError("ell must satisfy 2 <= ell <= len(b).")
    if not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer.")

    n = b.size

    rng = np.random.default_rng(int(seed))
    G = rng.standard_normal((n, ell - 1))

    b_hat = b / np.linalg.norm(b)

    Q = np.empty((n, ell - 1), dtype=float)

    for j in range(ell - 1):

        w = G[:, j].copy()

        w -= b_hat * np.dot(b_hat, w)

        for k in range(j):

            w -= Q[:, k] * np.dot(Q[:, k], w)

        nw = np.linalg.norm(w)

        if nw <= np.finfo(float).eps:
            raise ValueError(
                "Randomized enrichment produced a dependent direction."
            )

        Q[:, j] = w / nw

    return np.column_stack((b, Q))

import numpy as np


def classical_orthonormalize(
    Omega_ell: "np.ndarray",
) -> "np.ndarray":
    """Reference implementation of classical initialization."""
    Omega_ell = np.asarray(Omega_ell, dtype=float)

    if Omega_ell.ndim != 2:
        raise ValueError("Omega_ell must be two-dimensional")

    n, ell = Omega_ell.shape

    if n == 0 or ell == 0:
        raise ValueError("Omega_ell must be non-empty")
    if n < ell:
        raise ValueError("Omega_ell must have at least as many rows as columns")
    if not np.all(np.isfinite(Omega_ell)):
        raise ValueError("Omega_ell must contain only finite values")

    rank = np.linalg.matrix_rank(Omega_ell)
    if rank < ell:
        raise ValueError("Omega_ell must have full column rank")

    V1, R = np.linalg.qr(Omega_ell, mode="reduced")

    # Fix the QR sign convention so the diagonal of R is nonnegative.
    signs = np.sign(np.diag(R))
    signs[signs == 0.0] = 1.0

    V1 = V1 * signs

    return V1

import numpy as np


def global_normalize(
    Omega_ell: "np.ndarray",
) -> "np.ndarray":
    """Reference implementation of global initialization."""
    Omega_ell = np.asarray(Omega_ell, dtype=float)

    if Omega_ell.ndim != 2:
        raise ValueError("Omega_ell must be two-dimensional")
    if Omega_ell.size == 0:
        raise ValueError("Omega_ell must be non-empty")
    if not np.all(np.isfinite(Omega_ell)):
        raise ValueError("Omega_ell must contain only finite values")

    norm_f = np.linalg.norm(Omega_ell, ord="fro")

    if not np.isfinite(norm_f) or norm_f <= 0.0:
        raise ValueError("Omega_ell must have positive Frobenius norm")

    return Omega_ell / norm_f

import numpy as np


def classical_block_arnoldi(
    A: "np.ndarray",
    V1: "np.ndarray",
    ell: int,
    depth: int,
) -> "tuple[np.ndarray, np.ndarray]":
    """Reference implementation of classical block Arnoldi."""
    A = np.asarray(A, dtype=float)
    V1 = np.asarray(V1, dtype=float)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")
    if V1.ndim != 2:
        raise ValueError("V1 must be two-dimensional")
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(V1)):
        raise ValueError("inputs must contain only finite values")

    n = A.shape[0]

    if V1.shape[0] != n:
        raise ValueError("V1 has incompatible row dimension")

    if not isinstance(ell, (int, np.integer)) or isinstance(ell, (bool, np.bool_)):
        raise ValueError("ell must be an integer")
    if not isinstance(depth, (int, np.integer)) or isinstance(depth, (bool, np.bool_)):
        raise ValueError("depth must be an integer")
    if ell <= 0:
        raise ValueError("ell must be positive")
    if depth <= 0:
        raise ValueError("depth must be positive")
    if V1.shape[1] != ell:
        raise ValueError("V1 must have ell columns")

    if np.linalg.matrix_rank(V1) < ell:
        raise ValueError("V1 must have full column rank")

    V = [V1]
    H = np.zeros((depth * ell, depth * ell), dtype=float)

    for k in range(depth):
        W = A @ V[k]

        for j in range(k + 1):
            Hjk = V[j].T @ W
            H[j * ell:(j + 1) * ell,
              k * ell:(k + 1) * ell] = Hjk
            W = W - V[j] @ Hjk

        if k < depth - 1:
            if not np.all(np.isfinite(W)):
                raise ValueError("Arnoldi remainder is non-finite")

            if np.linalg.matrix_rank(W) < ell:
                raise ValueError("Arnoldi breakdown: remainder is rank deficient")

            Vnext, R = np.linalg.qr(W, mode="reduced")

            # Fix the QR sign convention so the diagonal of R is nonnegative,
            # matching the convention used for the starting block. This makes
            # the generated basis independent of the LAPACK driver.
            signs = np.sign(np.diag(R))
            signs[signs == 0.0] = 1.0

            Vnext = Vnext * signs
            R = signs[:, None] * R

            if np.linalg.matrix_rank(R) < ell:
                raise ValueError("Arnoldi breakdown during QR")

            H[(k + 1) * ell:(k + 2) * ell,
              k * ell:(k + 1) * ell] = R

            V.append(Vnext)

    Vq = np.hstack(V)
    Hqq = H[:depth * ell, :depth * ell]

    return Hqq, Vq

import numpy as np


def global_block_arnoldi(
    A: "np.ndarray",
    V1: "np.ndarray",
    ell: int,
    depth: int,
) -> "tuple[np.ndarray, np.ndarray]":
    """Reference implementation of global block Arnoldi."""
    A = np.asarray(A, dtype=float)
    V1 = np.asarray(V1, dtype=float)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")
    if V1.ndim != 2:
        raise ValueError("V1 must be two-dimensional")
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(V1)):
        raise ValueError("inputs must contain only finite values")

    n = A.shape[0]

    if V1.shape[0] != n:
        raise ValueError("V1 has incompatible row dimension")

    if not isinstance(ell, (int, np.integer)) or isinstance(ell, (bool, np.bool_)):
        raise ValueError("ell must be an integer")
    if not isinstance(depth, (int, np.integer)) or isinstance(depth, (bool, np.bool_)):
        raise ValueError("depth must be an integer")
    if ell <= 0:
        raise ValueError("ell must be positive")
    if depth <= 0:
        raise ValueError("depth must be positive")
    if V1.shape[1] != ell:
        raise ValueError("V1 must have ell columns")

    norm_v1 = np.linalg.norm(V1, ord="fro")
    if not np.isfinite(norm_v1) or norm_v1 <= 0.0:
        raise ValueError("V1 must have positive Frobenius norm")

    V = [V1]
    H = np.zeros((depth * ell, depth * ell), dtype=float)

    for k in range(depth):
        W = A @ V[k]

        for j in range(k + 1):
            scalar = np.trace(V[j].T @ W)
            Hjk = scalar * np.eye(ell)

            H[j * ell:(j + 1) * ell,
              k * ell:(k + 1) * ell] = Hjk

            W = W - V[j] @ Hjk

        if k < depth - 1:
            beta = np.linalg.norm(W, ord="fro")

            if not np.isfinite(beta) or beta <= 0.0:
                raise ValueError("Arnoldi breakdown: zero or non-finite remainder")

            H[(k + 1) * ell:(k + 2) * ell,
              k * ell:(k + 1) * ell] = beta * np.eye(ell)

            V.append(W / beta)

    Vq = np.hstack(V)
    Hqq = H[:depth * ell, :depth * ell]

    return Hqq, Vq

import numpy as np
from scipy.linalg import expm


def classical_fAb_approx(
    Vq: "np.ndarray",
    Hqq: "np.ndarray",
    b: "np.ndarray",
) -> "np.ndarray":
    """Reference implementation of the classical matrix-function approximation."""
    Vq = np.asarray(Vq, dtype=float)
    Hqq = np.asarray(Hqq, dtype=float)
    b = np.asarray(b, dtype=float)

    if Vq.ndim != 2:
        raise ValueError("Vq must be two-dimensional")
    if Hqq.ndim != 2 or Hqq.shape[0] != Hqq.shape[1]:
        raise ValueError("Hqq must be square")
    if b.ndim != 1:
        raise ValueError("b must be one-dimensional")
    if not np.all(np.isfinite(Vq)):
        raise ValueError("Vq must contain only finite values")
    if not np.all(np.isfinite(Hqq)):
        raise ValueError("Hqq must contain only finite values")
    if not np.all(np.isfinite(b)):
        raise ValueError("b must contain only finite values")

    if Vq.shape[1] != Hqq.shape[0]:
        raise ValueError("Vq and Hqq dimensions are incompatible")
    if Vq.shape[0] != b.size:
        raise ValueError("Vq and b dimensions are incompatible")
    if Hqq.shape[0] == 0:
        raise ValueError("Hqq must be non-empty")

    xi = np.linalg.norm(b, ord=2)

    if not np.isfinite(xi):
        raise ValueError("b norm must be finite")

    e1 = np.zeros(Hqq.shape[0], dtype=float)
    e1[0] = 1.0

    fH = expm(-Hqq)

    return Vq @ (fH @ e1) * xi

import numpy as np
from scipy.linalg import expm


def global_fAb_approx(
    Vq: "np.ndarray",
    Hqq: "np.ndarray",
    Omega_ell: "np.ndarray",
) -> "np.ndarray":
    """Reference implementation of the global matrix-function approximation."""
    Vq = np.asarray(Vq, dtype=float)
    Hqq = np.asarray(Hqq, dtype=float)
    Omega_ell = np.asarray(Omega_ell, dtype=float)

    if Vq.ndim != 2:
        raise ValueError("Vq must be two-dimensional")
    if Hqq.ndim != 2 or Hqq.shape[0] != Hqq.shape[1]:
        raise ValueError("Hqq must be square")
    if Omega_ell.ndim != 2:
        raise ValueError("Omega_ell must be two-dimensional")

    if not np.all(np.isfinite(Vq)):
        raise ValueError("Vq must contain only finite values")
    if not np.all(np.isfinite(Hqq)):
        raise ValueError("Hqq must contain only finite values")
    if not np.all(np.isfinite(Omega_ell)):
        raise ValueError("Omega_ell must contain only finite values")

    if Vq.shape[1] != Hqq.shape[0]:
        raise ValueError("Vq and Hqq dimensions are incompatible")
    if Omega_ell.shape[0] != Vq.shape[0]:
        raise ValueError("Omega_ell and Vq dimensions are incompatible")
    if Omega_ell.shape[1] == 0:
        raise ValueError("Omega_ell must be non-empty")
    if Hqq.shape[0] == 0:
        raise ValueError("Hqq must be non-empty")

    xi_G = np.linalg.norm(Omega_ell, ord="fro")

    if not np.isfinite(xi_G) or xi_G <= 0.0:
        raise ValueError("Omega_ell must have positive Frobenius norm")

    e1 = np.zeros(Hqq.shape[0], dtype=float)
    e1[0] = 1.0

    fH = expm(-Hqq)

    return Vq @ (fH @ e1) * xi_G

import numpy as np


def block_krylov_strategy_gap(
    A: "np.ndarray",
    b: "np.ndarray",
    ell: int,
    depth: int,
    seed: int,
) -> float:
    """Reference solution for the complete block Krylov comparison."""

    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")

    if b.ndim != 1:
        raise ValueError("b must be one-dimensional")

    if b.size != A.shape[0]:
        raise ValueError("b has incompatible dimension")

    if not np.all(np.isfinite(A)):
        raise ValueError("A must contain only finite values")

    if not np.all(np.isfinite(b)):
        raise ValueError("b must contain only finite values")

    if not isinstance(ell, (int, np.integer)) or isinstance(
        ell, (bool, np.bool_)
    ):
        raise ValueError("ell must be an integer")

    if not isinstance(depth, (int, np.integer)) or isinstance(
        depth, (bool, np.bool_)
    ):
        raise ValueError("depth must be an integer")

    if not isinstance(seed, (int, np.integer)) or isinstance(
        seed, (bool, np.bool_)
    ):
        raise ValueError("seed must be an integer")

    if ell < 2 or ell > A.shape[0]:
        raise ValueError("ell must satisfy 2 <= ell <= n")

    if depth < 1:
        raise ValueError("depth must be positive")

    # Step 01: randomized augmented block
    # This is the original, unnormalized paper-specific block.
    Omega_ell = build_augmented_block(
        b,
        ell,
        seed,
    )

    # Step 02: classical starting block
    V1_C = classical_orthonormalize(
        Omega_ell
    )

    # Step 03: global starting block
    # This is normalized for the global Arnoldi recurrence.
    V1_G = global_normalize(
        Omega_ell
    )

    # Step 04: classical block Arnoldi
    Hqq_C, Vq_C = classical_block_arnoldi(
        A,
        V1_C,
        ell,
        depth,
    )

    # Step 05: global block Arnoldi
    Hqq_G, Vq_G = global_block_arnoldi(
        A,
        V1_G,
        ell,
        depth,
    )

    # Step 06: classical matrix-function approximation
    y_C = classical_fAb_approx(
        Vq_C,
        Hqq_C,
        b,
    )

    # Step 07: global matrix-function approximation
    # The scaling comes from the unnormalized Omega_ell, not from V1_G,
    # whose Frobenius norm is 1.
    y_G = global_fAb_approx(
        Vq_G,
        Hqq_G,
        Omega_ell,
    )

    # Final comparison
    return float(
        np.linalg.norm(y_C - y_G, ord=2)
    )
SCICODE_GOLD_EOF
