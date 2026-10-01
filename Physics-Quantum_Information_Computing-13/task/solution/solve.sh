#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def matrix_supports(M: np.ndarray) -> tuple[list[int], list[int]]:
    import numpy as np

    M = np.asarray(M)
    if M.ndim != 2 or M.size == 0:
        raise ValueError("M must be a nonempty 2D matrix")
    if not np.all((M == 0) | (M == 1)):
        raise ValueError("M must be binary")

    row_support = np.flatnonzero(np.any(M, axis=1)).astype(int).tolist()
    col_support = np.flatnonzero(np.any(M, axis=0)).astype(int).tolist()
    return row_support, col_support

def beta_neighborhood(
    H: np.ndarray,
    vertices: list[int],
    source_side: str,
    beta: float,
) -> list[int]:
    import numpy as np

    H = np.asarray(H)
    if H.ndim != 2 or H.size == 0:
        raise ValueError("H must be a nonempty 2D matrix")
    if not np.all((H == 0) | (H == 1)):
        raise ValueError("H must be binary")
    if source_side not in ("A", "B"):
        raise ValueError("source_side must be 'A' or 'B'")
    if not (0.0 <= float(beta) < 1.0):
        raise ValueError("beta must lie in [0,1)")
    if np.any(H.sum(axis=0) == 0) or np.any(H.sum(axis=1) == 0):
        raise ValueError("isolated graph vertices are not supported")

    vertices = [int(v) for v in vertices]
    if len(vertices) != len(set(vertices)):
        raise ValueError("vertices must be unique")

    if source_side == "A":
        if any(v < 0 or v >= H.shape[1] for v in vertices):
            raise ValueError("A vertex out of range")
        counts = H[:, vertices].sum(axis=1) if vertices else np.zeros(H.shape[0], dtype=int)
        degrees = H.sum(axis=1)
    else:
        if any(v < 0 or v >= H.shape[0] for v in vertices):
            raise ValueError("B vertex out of range")
        counts = H[vertices, :].sum(axis=0) if vertices else np.zeros(H.shape[1], dtype=int)
        degrees = H.sum(axis=0)

    return np.flatnonzero(counts > float(beta) * degrees).astype(int).tolist()

def grow_beta_region(
    H: np.ndarray,
    seed: list[int],
    seed_side: str,
    beta: float,
    rounds: int,
) -> tuple[list[int], list[int]]:
    import numpy as np

    H = np.asarray(H)
    if H.ndim != 2 or H.size == 0:
        raise ValueError("H must be a nonempty 2D matrix")
    if not np.all((H == 0) | (H == 1)):
        raise ValueError("H must be binary")
    if seed_side not in ("A", "B"):
        raise ValueError("seed_side must be 'A' or 'B'")
    if not (0.0 <= float(beta) < 1.0):
        raise ValueError("beta must lie in [0,1)")
    if not isinstance(rounds, (int, np.integer)) or rounds < 0:
        raise ValueError("rounds must be a nonnegative integer")

    seed = [int(v) for v in seed]
    if len(seed) != len(set(seed)):
        raise ValueError("seed indices must be unique")

    if seed_side == "A":
        if any(v < 0 or v >= H.shape[1] for v in seed):
            raise ValueError("A seed out of range")
        A_region = set(seed)
        B_region = set()

        for _ in range(rounds):
            B_region = set(
                beta_neighborhood(H, sorted(A_region), "A", beta)
            )
            if B_region:
                A_region |= set(
                    np.flatnonzero(
                        np.any(H[sorted(B_region), :], axis=0)
                    ).astype(int).tolist()
                )
    else:
        if any(v < 0 or v >= H.shape[0] for v in seed):
            raise ValueError("B seed out of range")
        B_region = set(seed)
        A_region = set()

        for _ in range(rounds):
            A_region = set(
                beta_neighborhood(H, sorted(B_region), "B", beta)
            )
            if A_region:
                B_region |= set(
                    np.flatnonzero(
                        np.any(H[:, sorted(A_region)], axis=1)
                    ).astype(int).tolist()
                )

    return sorted(A_region), sorted(B_region)

def deterministic_peeling(
    H: np.ndarray,
    active: list[int],
    active_side: str,
) -> tuple[list[tuple[int, int]], list[int]]:
    import numpy as np

    H = np.asarray(H)
    if H.ndim != 2 or H.size == 0:
        raise ValueError("H must be a nonempty 2D matrix")
    if not np.all((H == 0) | (H == 1)):
        raise ValueError("H must be binary")
    if active_side not in ("A", "B"):
        raise ValueError("active_side must be 'A' or 'B'")

    active = [int(v) for v in active]
    if len(active) != len(set(active)):
        raise ValueError("active indices must be unique")

    limit = H.shape[1] if active_side == "A" else H.shape[0]
    if any(v < 0 or v >= limit for v in active):
        raise ValueError("active vertex out of range")

    remaining = set(active)
    pairs = []

    while remaining:
        if active_side == "A":
            candidates = []
            for b in range(H.shape[0]):
                nbrs = [a for a in remaining if H[b, a] == 1]
                if len(nbrs) == 1:
                    candidates.append((nbrs[0], b))

            if not candidates:
                raise ValueError("active A-set is not peelable")

            a, b = min(candidates)
            remaining.remove(a)

        else:
            candidates = []
            for a in range(H.shape[1]):
                nbrs = [b for b in remaining if H[b, a] == 1]
                if len(nbrs) == 1:
                    candidates.append((nbrs[0], a))

            if not candidates:
                raise ValueError("active B-set is not peelable")

            b, a = min(candidates)
            remaining.remove(b)

        pairs.append((int(a), int(b)))

    if active_side == "A":
        partners = sorted(b for a, b in pairs)
    else:
        partners = sorted(a for a, b in pairs)

    return pairs, partners

def gf2_inverse(M: np.ndarray) -> np.ndarray:
    import numpy as np

    M = np.asarray(M)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError("M must be square")
    if not np.all((M == 0) | (M == 1)):
        raise ValueError("M must be binary")

    n = M.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=np.uint8)

    aug = np.concatenate(
        [M.astype(np.uint8), np.eye(n, dtype=np.uint8)],
        axis=1,
    )

    row = 0
    for col in range(n):
        pivot = next(
            (r for r in range(row, n) if aug[r, col] == 1),
            None,
        )
        if pivot is None:
            raise ValueError("matrix is singular over GF(2)")

        if pivot != row:
            aug[[row, pivot]] = aug[[pivot, row]]

        for r in range(n):
            if r != row and aug[r, col] == 1:
                aug[r] ^= aug[row]

        row += 1

    return aug[:, n:].astype(np.uint8)

def embedded_inverse(
    H: np.ndarray,
    a_indices: list[int],
    b_indices: list[int],
) -> np.ndarray:
    import numpy as np

    H = np.asarray(H)
    if H.ndim != 2 or H.size == 0:
        raise ValueError("H must be a nonempty 2D matrix")
    if not np.all((H == 0) | (H == 1)):
        raise ValueError("H must be binary")

    a_indices = sorted(int(v) for v in a_indices)
    b_indices = sorted(int(v) for v in b_indices)

    if len(a_indices) != len(set(a_indices)):
        raise ValueError("a_indices must be unique")
    if len(b_indices) != len(set(b_indices)):
        raise ValueError("b_indices must be unique")
    if len(a_indices) != len(b_indices):
        raise ValueError("selected submatrix must be square")
    if any(v < 0 or v >= H.shape[1] for v in a_indices):
        raise ValueError("A index out of range")
    if any(v < 0 or v >= H.shape[0] for v in b_indices):
        raise ValueError("B index out of range")

    embedded = np.zeros((H.shape[1], H.shape[0]), dtype=np.uint8)

    if a_indices:
        submatrix = H[np.ix_(b_indices, a_indices)]
        inverse = gf2_inverse(submatrix)
        embedded[np.ix_(a_indices, b_indices)] = inverse

    return embedded

def decode_x_sector(
    H: np.ndarray,
    W: np.ndarray,
    beta: float,
    rounds: int,
) -> tuple[np.ndarray, np.ndarray]:
    import numpy as np

    H = np.asarray(H)
    W = np.asarray(W)

    if H.ndim != 2 or H.size == 0:
        raise ValueError("H must be a nonempty 2D matrix")
    if W.shape != H.shape:
        raise ValueError("W must have the same shape as H")
    if not np.all((H == 0) | (H == 1)):
        raise ValueError("H must be binary")
    if not np.all((W == 0) | (W == 1)):
        raise ValueError("W must be binary")

    H = H.astype(np.uint8)
    W = W.astype(np.uint8)

    row_support, col_support = matrix_supports(W)

    S, _ = grow_beta_region(
        H, row_support, "B", beta, rounds
    )
    _, T = grow_beta_region(
        H, col_support, "A", beta, rounds
    )

    _, Bp = deterministic_peeling(H, S, "A")
    _, Ap = deterministic_peeling(H, T, "B")

    left_inverse = embedded_inverse(H, S, Bp)
    X_A = (left_inverse @ W) % 2

    remaining = (H @ X_A + W) % 2

    right_inverse = embedded_inverse(H, Ap, T)
    X_B = (remaining @ right_inverse) % 2

    return X_A.astype(np.uint8), X_B.astype(np.uint8)

def decode_z_sector(
    H: np.ndarray,
    U: np.ndarray,
    beta: float,
    rounds: int,
) -> tuple[np.ndarray, np.ndarray]:
    import numpy as np

    H = np.asarray(H)
    U = np.asarray(U)

    if H.ndim != 2 or H.size == 0:
        raise ValueError("H must be a nonempty 2D matrix")
    if U.shape != (H.shape[1], H.shape[0]):
        raise ValueError("U must have shape (A, B)")
    if not np.all((H == 0) | (H == 1)):
        raise ValueError("H must be binary")
    if not np.all((U == 0) | (U == 1)):
        raise ValueError("U must be binary")

    H = H.astype(np.uint8)
    U = U.astype(np.uint8)

    row_support, col_support = matrix_supports(U)

    S, _ = grow_beta_region(
        H, col_support, "B", beta, rounds
    )
    _, T = grow_beta_region(
        H, row_support, "A", beta, rounds
    )

    _, Bp = deterministic_peeling(H, S, "A")
    _, Ap = deterministic_peeling(H, T, "B")

    left_inverse = embedded_inverse(H, S, Bp)
    Z_A = (U @ left_inverse.T) % 2

    remaining = (U + Z_A @ H.T) % 2

    right_inverse = embedded_inverse(H, Ap, T)
    Z_B = (right_inverse.T @ remaining) % 2

    return Z_A.astype(np.uint8), Z_B.astype(np.uint8)

def run_quantum_expander_decoder(
    H: np.ndarray,
    W: np.ndarray,
    U: np.ndarray,
    beta: float,
    rounds: int,
) -> float:
    import numpy as np

    H = np.asarray(H)

    X_A, X_B = decode_x_sector(H, W, beta, rounds)
    Z_A, Z_B = decode_z_sector(H, U, beta, rounds)

    total_weight = (
        int(np.sum(X_A))
        + int(np.sum(X_B))
        + int(np.sum(Z_A))
        + int(np.sum(Z_B))
    )

    blocklength = H.shape[1] ** 2 + H.shape[0] ** 2
    return float(total_weight / (2 * blocklength))
SCICODE_GOLD_EOF
