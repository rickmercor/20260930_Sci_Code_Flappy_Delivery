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


def _check_nonnegative_square(matrix, name):
    """Validate a finite real nonnegative square matrix and return it as a float array."""
    a = np.asarray(matrix)
    if np.iscomplexobj(a):
        raise ValueError(name + " must be real")
    a = a.astype(float)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or a.shape[0] < 1:
        raise ValueError(name + " must be a square array of size at least one")
    if not np.all(np.isfinite(a)):
        raise ValueError(name + " must be finite")
    if np.any(a < 0.0):
        raise ValueError(name + " must be nonnegative")
    return a


def _is_irreducible(a):
    """Reachability of every index from every other along the nonzero pattern."""
    size = a.shape[0]
    reach = (a > 0.0) | np.eye(size, dtype=bool)
    for _ in range(int(np.ceil(np.log2(max(size, 2)))) + 1):
        reach = (reach.astype(int) @ reach.astype(int)) > 0
    return bool(np.all(reach))


def _positive_eigenvector(vector):
    """Remove the arbitrary complex phase of a Perron eigenvector and make it positive."""
    x = np.asarray(vector)
    x = x / x[int(np.argmax(np.abs(x)))]
    x = x.real
    if np.min(x) <= 0.0:
        raise ValueError("the Perron eigenvector is not strictly positive; the matrix is ill conditioned")
    return x


def perron_triplet(matrix: np.ndarray) -> dict:
    """Reference implementation."""
    a = _check_nonnegative_square(matrix, "matrix")
    if not _is_irreducible(a):
        raise ValueError("matrix must be irreducible")

    values, vectors = np.linalg.eig(a)
    index = int(np.argmax(values.real))
    root = float(values[index].real)
    right = _positive_eigenvector(vectors[:, index])

    left_values, left_vectors = np.linalg.eig(a.T)
    left_index = int(np.argmax(left_values.real))
    left = _positive_eigenvector(left_vectors[:, left_index])

    stable = right / right.sum()
    reproductive = left / float(left @ stable)
    peripheral = int(np.sum(np.abs(values) >= root * (1.0 - 1e-8)))
    return {"growth_rate": root,
            "stable_distribution": stable,
            "reproductive_values": reproductive,
            "peripheral_count": peripheral}

import numbers

import numpy as np


def _check_stage_matrix(stage_matrix):
    """Validate a stage matrix with Leslie pattern and a plus-group and return it as floats."""
    a = np.asarray(stage_matrix)
    if np.iscomplexobj(a):
        raise ValueError("stage_matrix must be real")
    a = a.astype(float)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or a.shape[0] < 2:
        raise ValueError("stage_matrix must be square with at least two stages")
    if not np.all(np.isfinite(a)) or np.any(a < 0.0):
        raise ValueError("stage_matrix must be finite and nonnegative")
    s = a.shape[0]
    pattern = np.zeros((s, s), dtype=bool)
    pattern[0, :] = True
    pattern[np.arange(1, s), np.arange(s - 1)] = True
    pattern[s - 1, s - 1] = True
    if np.any(a[~pattern] != 0.0):
        raise ValueError("stage_matrix must be zero outside its first row, subdiagonal and corner")
    below = np.array([a[j + 1, j] for j in range(s - 1)])
    if np.any(below <= 0.0) or np.any(below > 1.0):
        raise ValueError("subdiagonal survival probabilities must lie in (0, 1]")
    if a[0, s - 1] <= 0.0:
        raise ValueError("the last stage must be fertile")
    if not 0.0 < a[s - 1, s - 1] < 1.0:
        raise ValueError("the stasis probability of the last stage must lie in (0, 1)")
    return a


def _expansion(a, n):
    """Fertilities and survival probabilities of the age expansion of length n."""
    s = a.shape[0]
    fertility = np.array([a[0, i] if i < s else a[0, s - 1] for i in range(n)])
    survival = np.array([a[j + 1, j] if j < s - 1 else a[s - 1, s - 1] for j in range(n - 1)])
    return fertility, survival


def _leslie(fertility, survival):
    """The Leslie matrix with the given fertilities and survival probabilities."""
    n = fertility.size
    matrix = np.zeros((n, n))
    matrix[0, :] = fertility
    matrix[np.arange(1, n), np.arange(n - 1)] = survival
    return matrix


def age_expand_stage_model(
    stage_matrix: np.ndarray,
    tolerance: float = 1e-3,
    max_classes: int = 500,
) -> dict:
    """Reference implementation."""
    a = _check_stage_matrix(stage_matrix)
    s = a.shape[0]
    if isinstance(tolerance, bool) or not isinstance(tolerance, numbers.Real):
        raise ValueError("tolerance must be a real number")
    tol = float(tolerance)
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("tolerance must be finite and above zero")
    if isinstance(max_classes, bool) or not isinstance(max_classes, numbers.Integral):
        raise ValueError("max_classes must be an integer")
    if int(max_classes) <= s:
        raise ValueError("max_classes must exceed the number of stages")

    target = perron_triplet(a)["growth_rate"]  # noqa: F821
    previous = np.nan
    for n in range(s + 1, int(max_classes) + 1):
        fertility, survival = _expansion(a, n)
        rate = perron_triplet(_leslie(fertility, survival))["growth_rate"]  # noqa: F821
        error = abs(rate - target) / target
        if error < tol:
            return {"fertility": fertility,
                    "survival": survival,
                    "age_classes": n,
                    "stage_growth_rate": target,
                    "age_growth_rate": rate,
                    "relative_error": error,
                    "previous_error": previous}
        previous = error
    raise ValueError("no expansion up to max_classes meets the tolerance")

import numbers

import numpy as np


def _check_vital_rates(fertility, survival):
    """Validate the vital rates of an irreducible Leslie model and return them as float arrays."""
    f = np.asarray(fertility)
    p = np.asarray(survival)
    if np.iscomplexobj(f) or np.iscomplexobj(p):
        raise ValueError("vital rates must be real")
    f = f.astype(float)
    p = p.astype(float)
    if f.ndim != 1 or f.size < 1:
        raise ValueError("fertility must be a one-dimensional array of length at least one")
    if p.ndim != 1 or p.size != f.size - 1:
        raise ValueError("survival must be a one-dimensional array one shorter than fertility")
    if not (np.all(np.isfinite(f)) and np.all(np.isfinite(p))):
        raise ValueError("vital rates must be finite")
    if np.any(f < 0.0) or f[-1] <= 0.0:
        raise ValueError("fertilities must be nonnegative with the oldest class fertile")
    if np.any(p <= 0.0) or np.any(p > 1.0):
        raise ValueError("survival probabilities must lie in (0, 1]")
    return f, p


def disaggregate_leslie(
    fertility: np.ndarray,
    survival: np.ndarray,
    subdivisions: int,
) -> np.ndarray:
    """Reference implementation."""
    f, p = _check_vital_rates(fertility, survival)
    if isinstance(subdivisions, bool) or not isinstance(subdivisions, numbers.Integral):
        raise ValueError("subdivisions must be an integer")
    m = int(subdivisions)
    if m < 1:
        raise ValueError("subdivisions must be at least one")

    n = f.size
    size = n * m
    resolved = np.zeros((size, size))
    for l in range(1, n + 1):
        resolved[0, l * m - 1] = f[l - 1]
    for i in range(1, size):
        resolved[i, i - 1] = p[i // m - 1] if i % m == 0 else 1.0
    return resolved

import numbers

import numpy as np


def _check_count(value, name, lower):
    """Validate an integer count not below a lower bound and return it as an int."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(name + " must be an integer")
    if int(value) < lower:
        raise ValueError(name + " must be at least " + str(lower))
    return int(value)


def _partition_matrix(size, groups):
    """The groups-by-size matrix that sums consecutive blocks of size // groups classes."""
    return np.kron(np.eye(groups), np.ones((1, size // groups)))


def interstage_flow_aggregate(
    matrix: np.ndarray,
    groups: int,
    steps: int,
) -> dict:
    """Reference implementation."""
    triplet = perron_triplet(matrix)  # noqa: F821
    a = np.asarray(matrix, dtype=float)
    size = a.shape[0]
    m = _check_count(groups, "groups", 1)
    s = _check_count(steps, "steps", 1)
    if m > size or size % m != 0:
        raise ValueError("groups must divide the size of the matrix")

    w = triplet["stable_distribution"]
    g = _partition_matrix(size, m)
    power = np.linalg.matrix_power(a, s)
    weight = np.diag(w)
    q = weight @ g.T @ np.linalg.inv(g @ weight @ g.T)
    reduced = g @ power @ q

    root = np.diag(np.sqrt(w))
    fitted = float(np.sum((reduced @ g @ root) ** 2))
    actual = float(np.sum((g @ power @ root) ** 2))
    return {"reduced_matrix": reduced,
            "effectiveness": fitted / actual,
            "grouped_distribution": g @ w}

import numpy as np


def elasticity_matrix(matrix: np.ndarray) -> np.ndarray:
    """Reference implementation."""
    triplet = perron_triplet(matrix)  # noqa: F821
    root = triplet["growth_rate"]
    if root <= 0.0:
        raise ValueError("the Perron root must be positive for elasticities to exist")
    a = np.asarray(matrix, dtype=float)
    v = triplet["reproductive_values"]
    w = triplet["stable_distribution"]
    return np.outer(v, w) * a / (root * float(v @ w))

import numbers

import numpy as np


def _check_count(value, name, lower):
    """Validate an integer count not below a lower bound and return it as an int."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(name + " must be an integer")
    if int(value) < lower:
        raise ValueError(name + " must be at least " + str(lower))
    return int(value)


def _partition_matrix(size, groups):
    """The groups-by-size matrix that sums consecutive blocks of size // groups classes."""
    return np.kron(np.eye(groups), np.ones((1, size // groups)))


def elasticity_consistent_aggregate(
    matrix: np.ndarray,
    groups: int,
    steps: int,
) -> dict:
    """Reference implementation."""
    triplet = perron_triplet(matrix)  # noqa: F821
    a = np.asarray(matrix, dtype=float)
    size = a.shape[0]
    m = _check_count(groups, "groups", 1)
    s = _check_count(steps, "steps", 1)
    if m > size or size % m != 0:
        raise ValueError("groups must divide the size of the matrix")
    w = triplet["stable_distribution"]
    v = triplet["reproductive_values"]

    balanced = (v[:, None] * a) / v[None, :]
    collapsed = interstage_flow_aggregate(balanced, m, s)  # noqa: F821

    g = _partition_matrix(size, m)
    reduced_v = (g @ (v * w)) / (g @ w)
    reduced = collapsed["reduced_matrix"] * reduced_v[None, :] / reduced_v[:, None]
    return {"reduced_matrix": reduced,
            "reduced_reproductive_values": reduced_v / float(reduced_v @ (g @ w)),
            "effectiveness": collapsed["effectiveness"]}

import numbers

import numpy as np


def _check_count(value, name, lower):
    """Validate an integer count not below a lower bound and return it as an int."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(name + " must be an integer")
    if int(value) < lower:
        raise ValueError(name + " must be at least " + str(lower))
    return int(value)


def _partition_matrix(size, groups):
    """The groups-by-size matrix that sums consecutive blocks of size // groups classes."""
    return np.kron(np.eye(groups), np.ones((1, size // groups)))


def consistency_residuals(
    matrix: np.ndarray,
    groups: int,
    steps: int,
    reduced_matrix: np.ndarray,
) -> dict:
    """Reference implementation."""
    fine = perron_triplet(matrix)  # noqa: F821
    a = np.asarray(matrix, dtype=float)
    size = a.shape[0]
    m = _check_count(groups, "groups", 1)
    s = _check_count(steps, "steps", 1)
    if m > size or size % m != 0:
        raise ValueError("groups must divide the size of the matrix")
    b = np.asarray(reduced_matrix)
    if b.ndim != 2 or b.shape != (m, m):
        raise ValueError("reduced_matrix must have shape (groups, groups)")
    coarse = perron_triplet(b)  # noqa: F821

    g = _partition_matrix(size, m)
    w = fine["stable_distribution"]
    v = fine["reproductive_values"]
    target_rate = fine["growth_rate"] ** s
    grouped_w = g @ w
    averaged_v = (g @ (v * w)) / grouped_w
    averaged_v = averaged_v / float(averaged_v @ grouped_w)
    reduced_v = coarse["reproductive_values"] / float(coarse["reproductive_values"] @ grouped_w)

    fine_elasticity = np.outer(v, w) * np.linalg.matrix_power(a, s) / (target_rate * float(v @ w))
    coarse_elasticity = elasticity_matrix(b)  # noqa: F821
    return {
        "growth_rate_residual": abs(coarse["growth_rate"] - target_rate) / target_rate,
        "stable_structure_residual": float(np.max(np.abs(coarse["stable_distribution"] - grouped_w))),
        "reproductive_value_residual": float(np.max(np.abs(reduced_v - averaged_v)) / np.max(averaged_v)),
        "elasticity_residual": float(np.max(np.abs(coarse_elasticity - g @ fine_elasticity @ g.T))),
    }

import numbers

import numpy as np


def demographic_parameters(leslie_matrix: np.ndarray, interval: float) -> dict:
    """Reference implementation."""
    triplet = perron_triplet(leslie_matrix)  # noqa: F821
    if isinstance(interval, bool) or not isinstance(interval, numbers.Real):
        raise ValueError("interval must be a real number")
    dt = float(interval)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("interval must be finite and above zero")

    a = np.asarray(leslie_matrix, dtype=float)
    size = a.shape[0]
    pattern = np.zeros((size, size), dtype=bool)
    pattern[0, :] = True
    pattern[np.arange(1, size), np.arange(size - 1)] = True
    if np.any(np.abs(a[~pattern]) > 1e-10 * np.max(np.abs(a))):
        raise ValueError("leslie_matrix must be zero outside its first row and subdiagonal")

    lam = triplet["growth_rate"]
    fertility = a[0, :]
    survival = np.array([a[i + 1, i] for i in range(size - 1)])
    survivorship = np.concatenate([[1.0], np.cumprod(survival)])
    maternity = fertility * survivorship
    parents = maternity * lam ** (-np.arange(1.0, size + 1.0))

    v = triplet["reproductive_values"]
    w = triplet["stable_distribution"]
    fertility_matrix = np.zeros((size, size))
    fertility_matrix[0, :] = fertility
    generation = dt * lam * float(v @ w) / float(v @ fertility_matrix @ w)

    positive = parents[parents > 0.0]
    return {"growth_rate": lam,
            "net_reproductive_rate": float(maternity.sum()),
            "generation_time": generation,
            "entropy": float(-np.sum(positive * np.log(positive))) + 0.0}

import numbers

import numpy as np


def _check_count(value, name, lower):
    """Validate an integer count not below a lower bound and return it as an int."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(name + " must be an integer")
    if int(value) < lower:
        raise ValueError(name + " must be at least " + str(lower))
    return int(value)


def _check_positive(value, name):
    """Validate a finite real number above zero and return it as a float."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(name + " must be a real number")
    x = float(value)
    if not np.isfinite(x) or x <= 0.0:
        raise ValueError(name + " must be finite and above zero")
    return x


def reduce_stage_model(
    stage_matrix: np.ndarray,
    reduced_classes: int,
    interval: float = 1.0,
    tolerance: float = 1e-3,
    certificate_tol: float = 1e-9,
) -> dict:
    """Reference implementation."""
    expansion = age_expand_stage_model(stage_matrix, tolerance)  # noqa: F821
    fertility = expansion["fertility"]
    survival = expansion["survival"]
    original = disaggregate_leslie(fertility, survival, 1)  # noqa: F821
    n = original.shape[0]
    m = _check_count(reduced_classes, "reduced_classes", 1)
    if m > n:
        raise ValueError("reduced_classes must not exceed the number of age classes")
    dt = _check_positive(interval, "interval")
    tol = _check_positive(certificate_tol, "certificate_tol")

    if n % m == 0:
        resolution = 1
        fine = original
        steps = n // m
    else:
        resolution = m
        fine = disaggregate_leslie(fertility, survival, m)  # noqa: F821
        steps = n

    original_triplet = perron_triplet(original)  # noqa: F821
    fine_triplet = perron_triplet(fine)  # noqa: F821
    consistent = elasticity_consistent_aggregate(fine, m, steps)  # noqa: F821
    standard = interstage_flow_aggregate(fine, m, steps)  # noqa: F821
    b = consistent["reduced_matrix"]
    bs = standard["reduced_matrix"]

    own = consistency_residuals(fine, m, steps, b)  # noqa: F821
    other = consistency_residuals(fine, m, steps, bs)  # noqa: F821
    survival_gap = max([0.0] + [abs(b[i + 1, i] - bs[i + 1, i]) for i in range(m - 1)])
    rate_gap = abs(fine_triplet["growth_rate"] ** resolution - original_triplet["growth_rate"]) \
        / original_triplet["growth_rate"]
    certificate = max(max(own.values()), survival_gap, rate_gap)
    if certificate > tol:
        raise RuntimeError("the reduced model fails the consistency certificate by %.3e" % certificate)

    elasticities = elasticity_matrix(b)  # noqa: F821
    reduced_interval = dt * n / m
    mine = demographic_parameters(b, reduced_interval)  # noqa: F821
    theirs = demographic_parameters(bs, reduced_interval)  # noqa: F821
    base = demographic_parameters(original, dt)  # noqa: F821

    values = consistent["reduced_reproductive_values"]
    return {
        "oldest_fertility": float(b[0, m - 1]),
        "oldest_fertility_elasticity": float(elasticities[0, m - 1]),
        "age_classes": n,
        "stage_growth_rate": expansion["stage_growth_rate"],
        "expansion_error": expansion["relative_error"],
        "reduced_fertility": b[0, :].copy(),
        "reduced_survival": np.array([b[i + 1, i] for i in range(m - 1)]),
        "reduced_growth_rate": mine["growth_rate"],
        "reduced_interval": reduced_interval,
        "reduced_reproductive_values": values / values[0],
        "growth_rate": original_triplet["growth_rate"],
        "resolution": resolution,
        "peripheral_count": fine_triplet["peripheral_count"],
        "standard_fertility": bs[0, :].copy(),
        "balanced_effectiveness": consistent["effectiveness"],
        "standard_effectiveness": standard["effectiveness"],
        "net_reproductive_rate": mine["net_reproductive_rate"],
        "generation_time": mine["generation_time"],
        "entropy": mine["entropy"],
        "original_net_reproductive_rate": base["net_reproductive_rate"],
        "original_generation_time": base["generation_time"],
        "original_entropy": base["entropy"],
        "standard_net_reproductive_rate": theirs["net_reproductive_rate"],
        "standard_generation_time": theirs["generation_time"],
        "standard_entropy": theirs["entropy"],
        "standard_reproductive_value_residual": other["reproductive_value_residual"],
        "standard_elasticity_residual": other["elasticity_residual"],
        "certificate_residual": float(certificate),
    }
SCICODE_GOLD_EOF
