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
def build_skew_inner_product(q: float, n_sites: int) -> np.ndarray:
    """Reference implementation."""
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    def _is_integer(value):
        return not isinstance(value, bool) and isinstance(value, (int, np.integer))

    if not (_is_number(q) and 0.0 < q < 1.0):
        raise ValueError("q must be a finite real number in (0, 1)")
    if not (_is_integer(n_sites) and n_sites >= 2):
        raise ValueError("n_sites must be an integer of at least 2")
    sites = np.arange(int(n_sites), dtype=float)
    # Symmetric RSK: a matrix of shape lambda carries prod_{i<j} q^{w_ij}
    # prod_i q^{w_ii / 2} = q^{|lambda| / 2}, and the number of symmetric
    # matrices of shape lambda is s_lambda(1^N), proportional to
    # prod_{i<j} (h_i - h_j). With |lambda| = sum_i h_i - N(N - 1)/2 the
    # per-site weight is q^{h / 2}, normalised to 1 at h = 0.
    weight = float(q) ** (0.5 * sites)
    orientation = np.sign(sites[None, :] - sites[:, None])
    return 0.5 * orientation * np.outer(weight, weight)

import numpy as np
def skew_orthonormalize(
    basis: np.ndarray,
    vector: np.ndarray,
    gram: np.ndarray,
    eta: float = 0.75,
) -> tuple[np.ndarray, np.ndarray]:
    """Reference implementation (modified symplectic Gram-Schmidt, iterated reorthogonalization, ESR3m)."""
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    columns = np.asarray(basis, dtype=float)
    work = np.array(vector, dtype=float)
    form = np.asarray(gram, dtype=float)
    if columns.ndim != 2 or work.ndim != 1 or columns.shape[0] != work.shape[0]:
        raise ValueError("basis must be (L, n) and vector (L,)")
    size, count = columns.shape
    if form.shape != (size, size):
        raise ValueError("gram must have shape (L, L)")
    if not (np.all(np.isfinite(columns)) and np.all(np.isfinite(work))
            and np.all(np.isfinite(form))):
        raise ValueError("inputs must be finite")
    if not (_is_number(eta) and 0.0 < eta <= 1.0):
        raise ValueError("eta must satisfy 0 < eta <= 1")
    coefficients = np.zeros(count + 1)
    previous = np.inf
    passes = 0
    while np.linalg.norm(work) < eta * previous and passes < 100:
        previous = np.linalg.norm(work)
        passes += 1
        for k in range(count // 2):
            first, second = columns[:, 2 * k], columns[:, 2 * k + 1]
            along_first = -(second @ form @ work)
            along_second = first @ form @ work
            coefficients[2 * k] += along_first
            coefficients[2 * k + 1] += along_second
            work = work - along_first * first - along_second * second
    # ESR3m: r11 = 1 opens a pair unscaled; r12 = 0 keeps the partner component.
    if count % 2 == 1:
        scale = columns[:, count - 1] @ form @ work
        if scale == 0.0 or not np.isfinite(scale):
            raise ValueError("the unpaired column cannot be completed by this vector")
    else:
        scale = 1.0
    coefficients[count] = scale
    return work / scale, coefficients

import numpy as np
def run_symplectic_arnoldi(
    gram: np.ndarray,
    nodes: np.ndarray,
    n_vectors: int,
    eta: float = 0.75,
) -> tuple[np.ndarray, np.ndarray]:
    """Reference implementation (symplectic Arnoldi iteration)."""
    import numpy as np

    form = np.asarray(gram, dtype=float)
    points = np.asarray(nodes, dtype=float)
    if form.ndim != 2 or form.shape[0] != form.shape[1] or not np.all(np.isfinite(form)):
        raise ValueError("gram must be a finite square array")
    size = form.shape[0]
    if points.shape != (size,) or not np.all(np.isfinite(points)):
        raise ValueError("nodes must be a finite array of shape (L,)")
    if (isinstance(n_vectors, bool) or not isinstance(n_vectors, (int, np.integer))
            or not 1 <= n_vectors <= size):
        raise ValueError("n_vectors must be an integer with 1 <= n_vectors <= L")
    count = int(n_vectors)
    basis = np.zeros((size, count))
    hessenberg = np.zeros((count, count - 1))
    # ESR3m start: r11 = 1, so the first vector is the unscaled constant.
    basis[:, 0] = 1.0
    for j in range(1, count):
        candidate = points * basis[:, j - 1]
        column, coefficients = skew_orthonormalize(
            basis[:, :j], candidate, form, eta
        )
        basis[:, j] = column
        hessenberg[: j + 1, j - 1] = coefficients
    return basis, hessenberg

import numpy as np
def assemble_pfaffian_kernel(polys: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Reference implementation (the paper's convention: Pfaffian, not quaternion determinant)."""
    import numpy as np

    family = np.asarray(polys, dtype=float)
    weight = np.asarray(weights, dtype=float)
    if family.ndim != 2 or family.shape[1] == 0 or family.shape[1] % 2 == 1:
        raise ValueError("polys must have shape (L, N) with N even and positive")
    if not np.all(np.isfinite(family)):
        raise ValueError("polys must be finite")
    size = family.shape[0]
    if weight.shape != (size,) or not np.all(np.isfinite(weight)) or np.any(weight <= 0.0):
        raise ValueError("weights must be finite, positive and of shape (L,)")
    sites = np.arange(size, dtype=float)
    sign = np.sign(sites[:, None] - sites[None, :])
    # psi_k(x) = 1/2 sum_y R_k(y) sign(x - y) w(y)
    transforms = 0.5 * sign @ (family * weight[:, None])
    even, odd = family[:, 0::2], family[:, 1::2]
    psi_even, psi_odd = transforms[:, 0::2], transforms[:, 1::2]
    # S(x, y) = w(x) sum_k [R_{2k+1}(x) psi_{2k}(y) - R_{2k}(x) psi_{2k+1}(y)]
    mixed = weight[:, None] * (odd @ psi_even.T - even @ psi_odd.T)
    # D(x, y) = w(x) w(y) sum_k [R_{2k}(x) R_{2k+1}(y) - R_{2k+1}(x) R_{2k}(y)]
    derivative = np.outer(weight, weight) * (even @ odd.T - odd @ even.T)
    # J(x, y) = sum_k [psi_{2k+1}(x) psi_{2k}(y) - psi_{2k}(x) psi_{2k+1}(y)] - sign(x - y)/2
    integral = psi_odd @ psi_even.T - psi_even @ psi_odd.T - 0.5 * sign
    kernel = np.zeros((2 * size, 2 * size))
    kernel[0::2, 0::2] = integral
    kernel[0::2, 1::2] = mixed.T
    kernel[1::2, 0::2] = -mixed
    kernel[1::2, 1::2] = -derivative
    return kernel

import numpy as np
def condition_kernel(kernel: np.ndarray, sites: list[int], occupied: bool) -> np.ndarray:
    """Reference implementation (Schur complement with or without the symplectic unit)."""
    import numpy as np

    matrix = np.asarray(kernel, dtype=float)
    if (matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] % 2
            or matrix.shape[0] == 0 or not np.all(np.isfinite(matrix))):
        raise ValueError("kernel must be a finite square array of even order")
    flag_is_bool = isinstance(occupied, (bool, np.bool_))
    if not flag_is_bool:
        raise ValueError("occupied must be a bool")
    size = matrix.shape[0] // 2
    labels = list(sites)
    if (not labels or len(labels) >= size
            or any(isinstance(s, bool) or not isinstance(s, (int, np.integer)) for s in labels)
            or len({int(s) for s in labels}) != len(labels)
            or any(not 0 <= int(s) < size for s in labels)):
        raise ValueError("sites must be distinct in-range integers, not all sites")
    chosen = [int(s) for s in labels]
    rest = [s for s in range(size) if s not in set(chosen)]
    inner = np.ravel([[2 * s, 2 * s + 1] for s in chosen])
    outer = np.ravel([[2 * s, 2 * s + 1] for s in rest])
    block = matrix[np.ix_(inner, inner)].copy()
    if not occupied:
        # Emptiness enters through K_Y - J (J the symplectic unit on the listed sites).
        block -= np.kron(np.eye(len(chosen)), np.array([[0.0, 1.0], [-1.0, 0.0]]))
    try:
        correction = np.linalg.solve(block, matrix[np.ix_(inner, outer)])
    except np.linalg.LinAlgError as error:
        raise ValueError("the conditioning event has probability zero") from error
    conditioned = matrix[np.ix_(outer, outer)] - matrix[np.ix_(outer, inner)] @ correction
    if not np.all(np.isfinite(conditioned)):
        raise ValueError("the conditioning event has probability zero")
    return 0.5 * (conditioned - conditioned.T)

import numpy as np
def compute_gap_probability(kernel: np.ndarray, sites: list[int]) -> float:
    """Reference implementation: the rejection branch of the sequential Pfaffian sampler."""
    import numpy as np

    matrix = np.asarray(kernel, dtype=float)
    if (matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] % 2
            or not np.all(np.isfinite(matrix))):
        raise ValueError("kernel must be a finite square array of even order")
    size = matrix.shape[0] // 2
    labels = list(sites)
    if (any(isinstance(s, bool) or not isinstance(s, (int, np.integer)) for s in labels)
            or len({int(s) for s in labels}) != len(labels)
            or any(not 0 <= int(s) < size for s in labels)):
        raise ValueError("sites must be distinct in-range integers")
    if not labels:
        return 1.0
    index = np.ravel([[2 * int(s), 2 * int(s) + 1] for s in labels])
    current = matrix[np.ix_(index, index)]
    probability = 1.0
    # Visit the sites one at a time, always taking the "empty" branch: the
    # conditional occupation is the (0, 1) entry, and the remaining kernel is
    # conditioned on the visited site being empty.
    while current.shape[0] > 2:
        occupation = current[0, 1]
        probability *= 1.0 - occupation
        if occupation == 1.0:
            return 0.0
        current = condition_kernel(current, [0], False)
    probability *= 1.0 - current[0, 1]
    return float(probability)

import numpy as np
def compute_single_occupancy_probability(kernel: np.ndarray, sites: list[int]) -> float:
    """Reference implementation: skew-trace of J K (J - K)^{-1} times the gap probability."""
    import numpy as np

    matrix = np.asarray(kernel, dtype=float)
    if (matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] % 2
            or not np.all(np.isfinite(matrix))):
        raise ValueError("kernel must be a finite square array of even order")
    size = matrix.shape[0] // 2
    labels = list(sites)
    if (any(isinstance(s, bool) or not isinstance(s, (int, np.integer)) for s in labels)
            or len({int(s) for s in labels}) != len(labels)
            or any(not 0 <= int(s) < size for s in labels)):
        raise ValueError("sites must be distinct in-range integers")
    if not labels:
        return 0.0
    count = len(labels)
    index = np.ravel([[2 * int(s), 2 * int(s) + 1] for s in labels])
    restricted = matrix[np.ix_(index, index)]
    unit = np.kron(np.eye(count), np.array([[0.0, 1.0], [-1.0, 0.0]]))
    gap = compute_gap_probability(matrix, [int(s) for s in labels])
    # Generating function Pf(J - z K_A): the linear coefficient in (1 - z) at
    # z = 1 is skewtr(J K_A (J - K_A)^{-1}) Pf(J - K_A); the inverse is needed.
    resolvent = unit @ restricted @ np.linalg.inv(unit - restricted)
    skew_trace = float(np.sum(resolvent[0::2, 1::2].diagonal()))
    return float(skew_trace * gap)

import numpy as np
def compute_second_row_pmf(
    q: float = 0.2,
    n_rows: int = 20,
    row_length: int = 21,
    n_sites: int = 120,
) -> float:
    """Reference orchestrator using only the reference function chain."""
    import numpy as np

    def _is_integer(value):
        return not isinstance(value, bool) and isinstance(value, (int, np.integer))

    if not (_is_integer(n_rows) and n_rows >= 2 and n_rows % 2 == 0):
        raise ValueError("n_rows must be an even integer of at least 2")
    if not (_is_integer(row_length) and row_length >= 1):
        raise ValueError("row_length must be a positive integer")
    if not (_is_integer(n_sites) and row_length + n_rows - 1 < n_sites):
        raise ValueError("n_sites must exceed row_length + n_rows - 1")
    gram = build_skew_inner_product(q, n_sites)
    nodes = np.arange(int(n_sites), dtype=float)
    polys, _ = run_symplectic_arnoldi(gram, nodes, int(n_rows))
    # gram[0, x] = w(0) w(x) / 2 with w(0) = 1 recovers the site weights.
    weights = 2.0 * gram[0].copy()
    weights[0] = 1.0
    kernel = assemble_pfaffian_kernel(polys, weights)
    # lambda_2 = row_length  <=>  the second-largest shifted row sits at `site`.
    site = int(row_length) + int(n_rows) - 2
    density = kernel[2 * site, 2 * site + 1]
    if not (np.isfinite(density) and density > 0.0):
        raise ValueError("the target site carries no probability")
    conditioned = condition_kernel(kernel, [site], True)
    # After removing `site`, the sites above it are relabelled site, ..., n_sites - 2.
    above = list(range(site, int(n_sites) - 1))
    return float(density * compute_single_occupancy_probability(conditioned, above))
SCICODE_GOLD_EOF
