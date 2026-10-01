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

def prepare_weight_matrix(A: np.ndarray) -> np.ndarray:
    """Deterministic stable reference implementation."""
    A = np.asarray(A, dtype=np.float64)

    if A.ndim != 2:
        raise ValueError("A must be two-dimensional")

    m, n = A.shape

    if m < n or n < 1:
        raise ValueError("A must satisfy m >= n >= 1")

    if not np.all(np.isfinite(A)):
        raise ValueError("A must contain finite values")

    try:
        A_plus, _, rank, _ = np.linalg.lstsq(
            A,
            np.eye(m, dtype=np.float64),
            rcond=None,
        )
    except np.linalg.LinAlgError as exc:
        raise ValueError("least-squares solve failed") from exc

    if rank < n:
        raise ValueError("A must have full column rank")

    if A_plus.shape != (n, m):
        raise ValueError("least-squares operator has incompatible shape")

    if not np.all(np.isfinite(A_plus)):
        raise ValueError("least-squares operator is non-finite")

    return np.asarray(A_plus, dtype=np.float64)

import numpy as np
def generate_column_blocks(
    n_cols: int,
    block_size: int,
    n_iters: int,
    seed: int = 271828,
) -> np.ndarray:
    """Reference implementation."""
    import numpy as np

    if not isinstance(n_cols, (int, np.integer)) or n_cols < 1:
        raise ValueError("n_cols must be positive")

    if not isinstance(block_size, (int, np.integer)):
        raise ValueError("block_size must be an integer")

    if block_size < 1 or block_size > n_cols:
        raise ValueError("invalid block_size")

    if not isinstance(n_iters, (int, np.integer)) or n_iters < 1:
        raise ValueError("n_iters must be positive")

    if not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")

    rng = np.random.default_rng(int(seed))

    blocks = np.empty(
        (int(n_iters), int(block_size)),
        dtype=np.int64,
    )

    for k in range(int(n_iters)):
        blocks[k] = rng.choice(
            int(n_cols),
            size=int(block_size),
            replace=False,
        )

    return blocks

import numpy as np
def form_residual_sketch(
    A: np.ndarray,
    b: np.ndarray,
    x: np.ndarray,
    tau: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reference implementation."""
    import numpy as np

    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    tau = np.asarray(tau)

    if A.ndim != 2:
        raise ValueError("A must be two-dimensional")

    m, n = A.shape

    if b.shape != (m,):
        raise ValueError("b has incompatible shape")

    if x.shape != (n,):
        raise ValueError("x has incompatible shape")

    if tau.ndim != 1 or tau.size < 1:
        raise ValueError("tau must be one-dimensional and nonempty")

    if not np.issubdtype(tau.dtype, np.integer):
        raise ValueError("tau must contain integer indices")

    if np.any(tau < 0) or np.any(tau >= n):
        raise ValueError("tau contains an invalid index")

    if np.unique(tau).size != tau.size:
        raise ValueError("tau must contain distinct indices")

    if not (
        np.all(np.isfinite(A))
        and np.all(np.isfinite(b))
        and np.all(np.isfinite(x))
    ):
        raise ValueError("inputs must be finite")

    e = b - A @ x
    r = A.T @ e
    s = A[:, tau] @ r[tau]

    return (
        np.asarray(e, dtype=np.float64),
        np.asarray(r, dtype=np.float64),
        np.asarray(s, dtype=np.float64),
    )

import numpy as np
def project_historical_direction(
    A: np.ndarray,
    A_plus: np.ndarray,
    P: np.ndarray,
    s: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Deterministic stable reference implementation."""
    import numpy as np

    A = np.asarray(A, dtype=np.float64)
    A_plus = np.asarray(A_plus, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)

    if A.ndim != 2 or A_plus.ndim != 2 or P.ndim != 2 or s.ndim != 1:
        raise ValueError("inputs have incompatible dimensions")

    m, n = A.shape

    if A_plus.shape != (n, m):
        raise ValueError("A_plus has incompatible shape")

    if P.shape[0] != n:
        raise ValueError("P has incompatible shape")

    if s.shape != (m,):
        raise ValueError("s has incompatible shape")

    if not (
        np.all(np.isfinite(A))
        and np.all(np.isfinite(A_plus))
        and np.all(np.isfinite(P))
        and np.all(np.isfinite(s))
    ):
        raise ValueError("inputs must be finite")

    try:
        _, singular_values, _ = np.linalg.svd(A, full_matrices=False)
    except np.linalg.LinAlgError as exc:
        raise ValueError("SVD failed while checking A") from exc

    if singular_values.size != n or not np.all(np.isfinite(singular_values)):
        raise ValueError("A must have full column rank")

    rank_tolerance = (
        max(A.shape)
        * np.finfo(np.float64).eps
        * float(singular_values[0])
    )
    if singular_values[-1] <= rank_tolerance:
        raise ValueError("A must have full column rank")

    z = A_plus @ s
    if not np.all(np.isfinite(z)):
        raise ValueError("least-squares direction is non-finite")

    # One residual refinement keeps the operator-based path numerically aligned
    # with a direct least-squares solve while still consuming Step 01's output.
    refinement = A_plus @ (s - A @ z)
    if not np.all(np.isfinite(refinement)):
        raise ValueError("least-squares refinement is non-finite")
    z = z + refinement

    if P.shape[1] == 0:
        return np.asarray(z, dtype=np.float64), 0.0

    if P.shape[1] >= n:
        raise ValueError("historical correction space has exhausted the available dimension")

    AP = A @ P
    if not np.all(np.isfinite(AP)):
        raise ValueError("A @ P is non-finite")

    c, _, rank_p, _ = np.linalg.lstsq(AP, s, rcond=None)
    if rank_p < P.shape[1]:
        raise ValueError("historical correction space is rank-deficient")
    if not np.all(np.isfinite(c)):
        raise ValueError("historical projection coefficients are non-finite")

    historical_component = AP @ c
    if not np.all(np.isfinite(historical_component)):
        raise ValueError("historical projection is non-finite")

    projected = z - P @ c
    if not np.all(np.isfinite(projected)):
        raise ValueError("projected direction is non-finite")

    delta = float(historical_component @ historical_component)
    if not np.isfinite(delta):
        raise ValueError("historical correction scalar is non-finite")

    return np.asarray(projected, dtype=np.float64), float(delta)

import numpy as np
def compute_projected_update(
    A: np.ndarray,
    b: np.ndarray,
    x: np.ndarray,
    s: np.ndarray,
    projected_direction: np.ndarray,
    delta: float,
) -> tuple[np.ndarray, float]:
    """Deterministic stable reference implementation."""
    import numpy as np

    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    projected_direction = np.asarray(projected_direction, dtype=np.float64)

    if A.ndim != 2:
        raise ValueError("A must be a matrix")
    m, n = A.shape
    if b.ndim != 1 or x.ndim != 1 or s.ndim != 1:
        raise ValueError("b, x, and s must be one-dimensional")
    if b.shape != (m,):
        raise ValueError("b has incompatible shape")
    if x.shape != (n,):
        raise ValueError("x has incompatible shape")
    if s.shape != (m,):
        raise ValueError("s has incompatible shape")
    if projected_direction.shape != (n,):
        raise ValueError("projected direction has incompatible shape")
    if not np.isfinite(delta):
        raise ValueError("delta must be finite")
    if not (np.all(np.isfinite(A)) and np.all(np.isfinite(b)) and np.all(np.isfinite(x)) and np.all(np.isfinite(s)) and np.all(np.isfinite(projected_direction))):
        raise ValueError("inputs must be finite")

    residual = b - A @ x
    if not np.all(np.isfinite(residual)):
        raise ValueError("residual is non-finite")

    numerator = float(s @ residual)
    if not np.isfinite(numerator):
        raise ValueError("update numerator is non-finite")

    projected_image = A @ projected_direction
    if not np.all(np.isfinite(projected_image)):
        raise ValueError("projected direction image is non-finite")

    denominator = float(projected_image @ projected_image)
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("update denominator is zero or numerically invalid")

    gamma = numerator / denominator
    if not np.isfinite(gamma):
        raise ValueError("update multiplier is non-finite")

    p = gamma * projected_direction
    if not np.all(np.isfinite(p)):
        raise ValueError("correction vector is non-finite")

    return np.asarray(p, dtype=np.float64), float(gamma)

import numpy as np
def update_iterate_state(
    x: np.ndarray,
    P: np.ndarray,
    p: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic reference implementation."""
    import numpy as np

    x = np.asarray(x, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)

    # -------------------------
    # Input validation
    # -------------------------
    if x.ndim != 1 or p.ndim != 1:
        raise ValueError("x and p must be one-dimensional")

    if P.ndim != 2:
        raise ValueError("P must be two-dimensional")

    if p.shape != x.shape:
        raise ValueError("p must have the same shape as x")

    if P.shape[0] != x.size:
        raise ValueError("P has incompatible row count")

    if not (
        np.all(np.isfinite(x))
        and np.all(np.isfinite(P))
        and np.all(np.isfinite(p))
    ):
        raise ValueError("inputs must be finite")

    n = x.size

    # -------------------------
    # Validate new correction
    # -------------------------
    p_norm = float(np.linalg.norm(p))

    if not np.isfinite(p_norm) or p_norm == 0.0:
        raise ValueError(
            "correction direction is zero or non-finite"
        )

    # -------------------------
    # Prevent exhaustion
    # -------------------------
    if P.shape[1] >= n:
        raise ValueError(
            "historical correction space has exhausted the "
            "available dimension"
        )

    # -------------------------
    # Check numerical independence
    # -------------------------
    if P.shape[1] > 0:
        try:
            coefficients, _, _, _ = np.linalg.lstsq(
                P,
                p,
                rcond=None,
            )
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "historical independence check failed"
            ) from exc

        residual = p - P @ coefficients
        residual_norm = float(np.linalg.norm(residual))

        if not np.isfinite(residual_norm):
            raise ValueError(
                "historical independence check is non-finite"
            )

        # Reject corrections whose component outside the historical
        # subspace is only at floating-point noise level.
        tolerance = (
            1e-10 * max(1.0, p_norm)
        )

        if residual_norm <= tolerance:
            raise ValueError(
                "correction direction is numerically contained "
                "in the historical subspace"
            )

    x_new = x + p

    if not np.all(np.isfinite(x_new)):
        raise ValueError("updated iterate is non-finite")

    P_new = np.column_stack((P, p))

    if not np.all(np.isfinite(P_new)):
        raise ValueError("updated historical matrix is non-finite")

    return (
        np.asarray(x_new, dtype=np.float64),
        np.asarray(P_new, dtype=np.float64),
    )

import numpy as np
def run_rplss_gcd(
    A: np.ndarray,
    x_ref: np.ndarray,
    eta: np.ndarray,
    x0: np.ndarray,
    seed: int = 271828,
    n_iters: int = 4,
    block_size: int = 2,
) -> float:
    """Deterministic stable reference orchestrator."""
    import numpy as np

    A = np.asarray(A, dtype=np.float64)
    x_ref = np.asarray(x_ref, dtype=np.float64)
    eta = np.asarray(eta, dtype=np.float64)
    x = np.asarray(x0, dtype=np.float64)

    if A.ndim != 2:
        raise ValueError("A must be two-dimensional")
    m, n = A.shape
    if m < n or n < 1:
        raise ValueError("A must satisfy m >= n >= 1")
    if x_ref.shape != (n,):
        raise ValueError("x_ref has incompatible shape")
    if eta.shape != (m,):
        raise ValueError("eta has incompatible shape")
    if x.shape != (n,):
        raise ValueError("x0 has incompatible shape")
    if not (np.all(np.isfinite(A)) and np.all(np.isfinite(x_ref)) and np.all(np.isfinite(eta)) and np.all(np.isfinite(x))):
        raise ValueError("inputs must be finite")
    if not isinstance(n_iters, (int, np.integer)) or n_iters < 1:
        raise ValueError("n_iters must be a positive integer")
    if not isinstance(block_size, (int, np.integer)) or block_size < 1 or block_size > n:
        raise ValueError("invalid block_size")
    if not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")

    max_history_iters = 1 if n == 1 else n - 1
    if n_iters > max_history_iters:
        raise ValueError("n_iters exceeds the available correction-space dimension")

    b = A @ x_ref + eta
    if not np.all(np.isfinite(b)):
        raise ValueError("constructed right-hand side is non-finite")

    # Step 01: prepare the stable least-squares operator consumed by Step 04.
    A_plus = prepare_weight_matrix(A)
    if A_plus.shape != (n, m) or not np.all(np.isfinite(A_plus)):
        raise ValueError("invalid least-squares operator")

    blocks = generate_column_blocks(
        n_cols=n,
        block_size=int(block_size),
        n_iters=int(n_iters),
        seed=int(seed),
    )
    if blocks.shape != (n_iters, block_size):
        raise ValueError("sampled column blocks have incompatible shape")

    P = np.empty((n, 0), dtype=np.float64)

    for k in range(n_iters):
        tau = blocks[k]
        _, _, s = form_residual_sketch(A=A, b=b, x=x, tau=tau)

        projected_direction, delta = project_historical_direction(
            A=A,
            A_plus=A_plus,
            P=P,
            s=s,
        )

        p, _ = compute_projected_update(
            A=A,
            b=b,
            x=x,
            s=s,
            projected_direction=projected_direction,
            delta=delta,
        )

        x, P = update_iterate_state(x=x, P=P, p=p)

    residual = A @ x - b
    if not np.all(np.isfinite(residual)):
        raise ValueError("final residual is non-finite")
    result = float(residual @ residual)
    if not np.isfinite(result):
        raise ValueError("final residual is non-finite")
    return result
SCICODE_GOLD_EOF
