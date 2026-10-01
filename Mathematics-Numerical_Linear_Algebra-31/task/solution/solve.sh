#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def compute_relative_backward_error(
    A: np.ndarray,
    b: np.ndarray,
    x: np.ndarray,
) -> float:
    """Return the deterministic relative backward error."""
    np = __import__("numpy")

    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    x = np.asarray(x, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape[0] == 0:
        raise ValueError("A must be a nonempty square matrix")
    n = A.shape[0]
    if b.shape != (n,) or x.shape != (n,):
        raise ValueError("b and x must have shape (n,)")
    if not (np.all(np.isfinite(A)) and np.all(np.isfinite(b)) and np.all(np.isfinite(x))):
        raise ValueError("all inputs must be finite")
    norm_A = float(np.linalg.norm(A, 2))
    norm_x = float(np.linalg.norm(x))
    if norm_A == 0.0 or norm_x == 0.0:
        raise ValueError("A and x must have nonzero 2-norm")
    return float(np.linalg.norm(A @ x - b) / (norm_A * norm_x))

def construct_controlled_psd_system(
    n: int,
    condition_number: float,
) -> np.ndarray:
    """Return the deterministic augmented spectral system."""
    np = __import__("numpy")

    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 4:
        raise ValueError("n must be an integer at least 4")
    condition_number = float(condition_number)
    if not np.isfinite(condition_number) or condition_number <= 1.0:
        raise ValueError("condition_number must be finite and greater than 1")

    rows = np.arange(n, dtype=float)[:, None]
    cols = np.arange(n, dtype=float)[None, :]
    U = np.sqrt(2.0 / n) * np.cos(np.pi * (rows + 0.5) * cols / n)
    U[:, 0] = 1.0 / np.sqrt(n)
    indices = np.arange(n, dtype=float)
    eigenvalues = condition_number ** (-indices / (n - 1))
    coefficients = 1.0 + 0.25 * np.cos(np.pi * (indices + 1.0) / (n + 1.0))
    A = (U * eigenvalues) @ U.T
    A = 0.5 * (A + A.T)
    b = U @ coefficients
    b = b / np.linalg.norm(b)
    return np.column_stack((A, b))

def build_orthogonal_recurrence(
    A: np.ndarray,
    b: np.ndarray,
    iterations: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the deterministic reorthogonalized recurrence."""
    np = __import__("numpy")

    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape[0] == 0:
        raise ValueError("A must be a nonempty square matrix")
    n = A.shape[0]
    if b.shape != (n,) or not np.all(np.isfinite(A)) or not np.all(np.isfinite(b)):
        raise ValueError("inputs must be finite with compatible shapes")
    if not np.allclose(A, A.T, rtol=0.0, atol=1e-12):
        raise ValueError("A must be symmetric")
    if (
        isinstance(iterations, (bool, np.bool_))
        or not isinstance(iterations, (int, np.integer))
        or not 1 <= iterations < n
    ):
        raise ValueError("iterations must satisfy 1 <= iterations < n")
    norm_b = float(np.linalg.norm(b))
    if norm_b == 0.0:
        raise ValueError("b must be nonzero")

    Q = np.zeros((n, iterations + 1), dtype=float)
    T = np.zeros((iterations + 1, iterations), dtype=float)
    Q[:, 0] = b / norm_b
    q_prev = np.zeros(n, dtype=float)
    beta = 0.0
    scale = max(1.0, float(np.linalg.norm(A, 2)))

    for j in range(iterations):
        q = Q[:, j]
        w = A @ q - beta * q_prev
        alpha = float(q @ w)
        w = w - alpha * q
        active_basis = Q[:, : j + 1]
        w = w - active_basis @ (active_basis.T @ w)
        beta_next = float(np.linalg.norm(w))
        if beta_next <= 64.0 * np.finfo(float).eps * scale:
            raise ValueError("the recurrence broke down before the requested iteration")
        T[j, j] = alpha
        if j > 0:
            T[j - 1, j] = beta
        T[j + 1, j] = beta_next
        Q[:, j + 1] = w / beta_next
        q_prev = q
        beta = beta_next
    return Q, T

def extract_reduced_operator(
    T: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the reduced operator and structured convergence certificate."""
    np = __import__("numpy")

    T = np.asarray(T, dtype=float)
    if T.ndim != 2 or T.shape[1] < 1 or T.shape[0] != T.shape[1] + 1:
        raise ValueError("T must have shape (k+1, k) with k >= 1")
    if not np.all(np.isfinite(T)):
        raise ValueError("T must be finite")
    if isinstance(threshold, (bool, np.bool_)):
        raise ValueError("threshold must be finite and nonnegative")
    try:
        threshold = float(threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError("threshold must be finite and nonnegative") from exc
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("threshold must be finite and nonnegative")

    reduced = T[1:, :].copy()
    k = reduced.shape[0]
    scale = max(1.0, float(np.linalg.norm(reduced, ord=np.inf)))
    structure_tolerance = 64.0 * np.finfo(float).eps * scale
    if np.any(np.abs(np.tril(reduced, k=-1)) > structure_tolerance):
        raise ValueError("the reduced operator must be upper triangular")
    if np.any(np.abs(np.triu(reduced, k=3)) > structure_tolerance):
        raise ValueError("the reduced operator must have upper bandwidth two")

    pivot_tolerance = 64.0 * np.finfo(float).eps * max(
        1.0,
        scale**2,
        threshold**2,
    )
    R = np.zeros((k, k), dtype=float)
    rejected = k
    terminal_pivot = 0.0

    for j in range(k):
        for i in range(max(0, j - 2), j):
            gram_ij = float(reduced[:, i] @ reduced[:, j])
            numerator = gram_ij
            for m in range(max(0, i - 2, j - 2), i):
                numerator -= R[m, i] * R[m, j]
            numerator_scale = max(1.0, abs(gram_ij))
            if abs(R[i, i]) <= pivot_tolerance / numerator_scale:
                rejected = i
                terminal_pivot = float(R[i, i] ** 2)
                break
            R[i, j] = numerator / R[i, i]
        if rejected != k:
            break

        pivot = float(reduced[:, j] @ reduced[:, j]) - threshold**2
        for m in range(max(0, j - 2), j):
            pivot -= R[m, j] ** 2
        terminal_pivot = float(pivot)
        if not np.isfinite(pivot) or pivot <= pivot_tolerance:
            rejected = j
            break
        R[j, j] = np.sqrt(pivot)

    certificate = np.concatenate(
        ([float(rejected), terminal_pivot], np.diag(R))
    )
    return reduced, certificate, R

def approximate_smallest_singular_pair(
    reduced: np.ndarray,
    inverse_steps: int,
    seed: int,
) -> np.ndarray:
    """Return the fixed-iteration smallest-singular-pair approximation."""
    np = __import__("numpy")

    reduced = np.asarray(reduced, dtype=float)
    if reduced.ndim != 2 or reduced.shape[0] != reduced.shape[1] or reduced.shape[0] == 0:
        raise ValueError("reduced must be a nonempty square matrix")
    if not np.all(np.isfinite(reduced)):
        raise ValueError("reduced must be finite")
    if (
        isinstance(inverse_steps, (bool, np.bool_))
        or not isinstance(inverse_steps, (int, np.integer))
        or inverse_steps < 1
    ):
        raise ValueError("inverse_steps must be a positive integer")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    scale = max(1.0, float(np.linalg.norm(reduced, ord=np.inf)))
    structure_tolerance = 64.0 * np.finfo(float).eps * scale
    if np.any(np.abs(np.tril(reduced, k=-1)) > structure_tolerance):
        raise ValueError("reduced must be upper triangular")
    if np.any(np.abs(np.triu(reduced, k=3)) > structure_tolerance):
        raise ValueError("reduced must have upper bandwidth two")
    diagonal = np.diag(reduced)
    if np.any(np.abs(diagonal) <= structure_tolerance):
        raise ValueError("reduced must be nonsingular")

    rng = np.random.default_rng(int(seed))
    v = rng.standard_normal(reduced.shape[0])
    v = v / np.linalg.norm(v)
    for _ in range(int(inverse_steps)):
        z = np.empty_like(v)
        for i in range(reduced.shape[0]):
            rhs = v[i]
            if i >= 1:
                rhs -= reduced[i - 1, i] * z[i - 1]
            if i >= 2:
                rhs -= reduced[i - 2, i] * z[i - 2]
            z[i] = rhs / diagonal[i]

        next_v = np.empty_like(v)
        for i in range(reduced.shape[0] - 1, -1, -1):
            rhs = z[i]
            if i + 1 < reduced.shape[0]:
                rhs -= reduced[i, i + 1] * next_v[i + 1]
            if i + 2 < reduced.shape[0]:
                rhs -= reduced[i, i + 2] * next_v[i + 2]
            next_v[i] = rhs / diagonal[i]
        v = next_v
        norm_v = float(np.linalg.norm(v))
        if not np.isfinite(norm_v) or norm_v == 0.0:
            raise ValueError("inverse iteration produced an invalid vector")
        v = v / norm_v

    nonzero = np.flatnonzero(np.abs(v) > 1e-14)
    if nonzero.size and v[nonzero[0]] < 0.0:
        v = -v
    sigma = float(np.linalg.norm(reduced @ v))
    return np.concatenate(([sigma], v))

def compute_normal_equation_history(
    A: np.ndarray,
    b: np.ndarray,
    iterations: int,
    inverse_steps: int,
    seed: int,
) -> np.ndarray:
    """Return the deterministic MINBERR-NE history."""
    np = __import__("numpy")

    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape[0] == 0:
        raise ValueError("A must be a nonempty square matrix")
    n = A.shape[0]
    if b.shape != (n,) or not np.all(np.isfinite(A)) or not np.all(np.isfinite(b)):
        raise ValueError("inputs must be finite with compatible shapes")
    if (
        isinstance(iterations, (bool, np.bool_))
        or not isinstance(iterations, (int, np.integer))
        or not 1 <= iterations < n
    ):
        raise ValueError("iterations must satisfy 1 <= iterations < n")
    if (
        isinstance(inverse_steps, (bool, np.bool_))
        or not isinstance(inverse_steps, (int, np.integer))
        or inverse_steps < 1
    ):
        raise ValueError("inverse_steps must be a positive integer")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")

    norm_A = float(np.linalg.norm(A, 2))
    norm_b = float(np.linalg.norm(b))
    if norm_A == 0.0 or norm_b == 0.0:
        raise ValueError("A and b must be nonzero")
    breakdown_tolerance = 64.0 * np.finfo(float).eps * max(1.0, norm_A)

    u = b / norm_b
    v_raw = A.T @ u
    alpha = float(np.linalg.norm(v_raw))
    if alpha <= breakdown_tolerance:
        raise ValueError("Golub--Kahan broke down at the initial right vector")
    v = v_raw / alpha

    u_next_raw = A @ v - alpha * u
    u_next_raw -= u * (u @ u_next_raw)
    beta_next = float(np.linalg.norm(u_next_raw))
    if beta_next <= breakdown_tolerance:
        raise ValueError("Golub--Kahan broke down at the initial left update")
    u_next = u_next_raw / beta_next

    V = np.empty((n, iterations + 1), dtype=float)
    V[:, 0] = v
    alphas = [alpha]
    reduced_diagonal = [beta_next]
    history = []
    U_basis = [u.copy(), u_next.copy()]

    for j in range(1, iterations + 1):
        v_next_raw = A.T @ u_next - beta_next * v
        for q in V[:, :j].T:
            v_next_raw -= q * (q @ v_next_raw)
        alpha_next = float(np.linalg.norm(v_next_raw))
        if alpha_next <= breakdown_tolerance:
            raise ValueError("Golub--Kahan broke down in a right update")
        v_next = v_next_raw / alpha_next
        V[:, j] = v_next
        alphas.append(alpha_next)

        following_u_raw = A @ v_next - alpha_next * u_next
        for q in U_basis:
            following_u_raw -= q * (q @ following_u_raw)
        following_beta = float(np.linalg.norm(following_u_raw))
        if following_beta <= breakdown_tolerance:
            raise ValueError("Golub--Kahan broke down in a left update")
        following_u = following_u_raw / following_beta
        reduced_diagonal.append(following_beta)
        U_basis.append(following_u.copy())

        dimension = j + 1
        reduced = np.diag(np.asarray(reduced_diagonal[:dimension]))
        reduced += np.diag(np.asarray(alphas[1:dimension]), k=1)
        pair = approximate_smallest_singular_pair(
            reduced,
            int(inverse_steps),
            int(seed) + j,
        )
        y = pair[1:]
        direction = V[:, :dimension] @ y
        denominator = float((b @ A) @ direction)
        denominator_tolerance = (
            64.0
            * np.finfo(float).eps
            * max(1.0, float(np.linalg.norm(b @ A)))
        )
        if abs(denominator) <= denominator_tolerance:
            raise ValueError("the MINBERR-NE direction cannot be scaled")
        x = direction * (norm_b**2 / denominator)
        history.append(compute_relative_backward_error(A, b, x))

        u = u_next
        u_next = following_u
        v = v_next
        alpha = alpha_next
        beta_next = following_beta

    return np.asarray(history)

def compute_error_histories(
    A: np.ndarray,
    b: np.ndarray,
    iterations: int,
    inverse_steps: int,
    seed: int,
) -> np.ndarray:
    """Return the three deterministic backward-error histories."""
    np = __import__("numpy")

    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape[0] == 0:
        raise ValueError("A must be a nonempty square matrix")
    n = A.shape[0]
    if b.shape != (n,) or not np.all(np.isfinite(A)) or not np.all(np.isfinite(b)):
        raise ValueError("inputs must be finite with compatible shapes")
    if not np.allclose(A, A.T, rtol=0.0, atol=1e-12):
        raise ValueError("A must be symmetric")
    if (
        isinstance(iterations, (bool, np.bool_))
        or not isinstance(iterations, (int, np.integer))
        or not 1 <= iterations < n
    ):
        raise ValueError("iterations must satisfy 1 <= iterations < n")
    if (
        isinstance(inverse_steps, (bool, np.bool_))
        or not isinstance(inverse_steps, (int, np.integer))
        or inverse_steps < 1
    ):
        raise ValueError("inverse_steps must be a positive integer")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    norm_A = float(np.linalg.norm(A, 2))
    norm_b = float(np.linalg.norm(b))
    if norm_A == 0.0 or norm_b == 0.0:
        raise ValueError("A and b must be nonzero")

    Q, T = build_orthogonal_recurrence(A, b, int(iterations))
    fixed_errors = []
    direct_errors = []
    x_fixed = np.zeros(n, dtype=float)

    for j in range(1, iterations + 1):
        x_fixed = x_fixed + (b - A @ x_fixed) / norm_A
        fixed_errors.append(compute_relative_backward_error(A, b, x_fixed))

        Qj = Q[:, : j + 1]
        Tj = T[: j + 1, :j]
        reduced, certificate, factor = extract_reduced_operator(Tj, 1.0e-5)
        if certificate.shape != (j + 2,) or not np.all(np.isfinite(certificate)):
            raise ValueError("invalid shifted-Cholesky certificate")
        if factor.shape != (j, j) or not np.all(np.isfinite(factor)):
            raise ValueError("invalid shifted-Cholesky factor state")
        pair = approximate_smallest_singular_pair(
            reduced,
            int(inverse_steps),
            int(seed) + j,
        )
        v = pair[1:]
        denominator = float(Tj[0, :] @ v)
        denominator_tolerance = (
            64.0
            * np.finfo(float).eps
            * max(1.0, float(np.linalg.norm(Tj[0, :])))
        )
        if abs(denominator) <= denominator_tolerance:
            raise ValueError("the direct MINBERR direction cannot be scaled")
        x_direct = Qj[:, :j] @ ((norm_b / denominator) * v)
        direct_errors.append(compute_relative_backward_error(A, b, x_direct))
        if int(certificate[0]) < j:
            break

    retained = len(direct_errors)
    normal_errors = compute_normal_equation_history(
        A,
        b,
        retained,
        int(inverse_steps),
        int(seed) + 10000,
    )
    return np.vstack(
        (
            np.asarray(fixed_errors),
            np.asarray(direct_errors),
            np.asarray(normal_errors),
        )
    )

def run_backward_error_benchmark(
    n: int,
    condition_number: float,
    iterations: int,
    inverse_steps: int,
    seed: int,
) -> float:
    """Return the deterministic end-to-end improvement factor."""
    np = __import__("numpy")

    packed = construct_controlled_psd_system(n, condition_number)
    A = packed[:, :-1]
    b = packed[:, -1]

    norm_A = float(np.linalg.norm(A, 2))
    norm_b = float(np.linalg.norm(b))

    Q, T = build_orthogonal_recurrence(A, b, int(iterations))

    fixed_errors = []
    direct_errors = []
    x_fixed = np.zeros(A.shape[0], dtype=float)

    for j in range(1, iterations + 1):
        x_fixed = x_fixed + (b - A @ x_fixed) / norm_A
        fixed_errors.append(compute_relative_backward_error(A, b, x_fixed))

        Qj = Q[:, : j + 1]
        Tj = T[: j + 1, :j]
        reduced, certificate, factor = extract_reduced_operator(Tj, 1.0e-5)
        if certificate.shape != (j + 2,) or not np.all(np.isfinite(certificate)):
            raise ValueError("invalid shifted-Cholesky certificate")
        if factor.shape != (j, j) or not np.all(np.isfinite(factor)):
            raise ValueError("invalid shifted-Cholesky factor state")
        pair = approximate_smallest_singular_pair(
            reduced,
            int(inverse_steps),
            int(seed) + j,
        )
        v = pair[1:]
        denom_dir = float(Tj[0, :] @ v)
        denom_tolerance = (
            64.0
            * np.finfo(float).eps
            * max(1.0, float(np.linalg.norm(Tj[0, :])))
        )
        if abs(denom_dir) <= denom_tolerance:
            raise ValueError("the direct MINBERR direction cannot be scaled")
        x_direct = Qj[:, :j] @ ((norm_b / denom_dir) * v)
        direct_errors.append(compute_relative_backward_error(A, b, x_direct))
        if int(certificate[0]) < j:
            break

    retained = len(direct_errors)
    normal_errors = compute_normal_equation_history(
        A,
        b,
        retained,
        int(inverse_steps),
        int(seed) + 10000,
    )

    histories = np.vstack(
        (
            np.asarray(fixed_errors),
            np.asarray(direct_errors),
            np.asarray(normal_errors),
        )
    )
    # Keep reference results independent of candidate function bindings.
    histories_check = compute_error_histories(
        A, b, int(iterations), int(inverse_steps), int(seed)
    )
    if not np.allclose(histories, histories_check, rtol=1e-9, atol=1e-9):
        raise ValueError("compute_error_histories is inconsistent with the per-step chain")

    denominator = float(histories[1, -1])
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("the final minimum backward error must be positive")
    numerator = float(histories[2, -1])
    if not np.isfinite(numerator) or numerator <= 0.0:
        raise ValueError("the final normal-equation backward error must be positive")
    return float(numerator / denominator)
SCICODE_GOLD_EOF
