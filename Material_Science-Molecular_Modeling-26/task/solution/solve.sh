#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def _is_real_scalar(value) -> bool:
    """Return True for a real numeric scalar, excluding booleans."""
    import numpy as np

    if isinstance(value, (bool, np.bool_)):
        return False
    return isinstance(value, (int, float, np.integer, np.floating))


def _is_boolean(value) -> bool:
    """Return True for a boolean scalar or a numpy boolean."""
    import numpy as np

    return isinstance(value, (bool, np.bool_))


def compute_umbrella_mean_force(center: np.ndarray, kappa: np.ndarray,
                                        gamma: float = 2.5, dmu: float = 0.4,
                                        offset: float = 8.0) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    for name, value in (("gamma", gamma), ("dmu", dmu), ("offset", offset)):
        if not _is_real_scalar(value):
            raise ValueError(f"{name} must be a real number")
        if not np.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")

    gamma = float(gamma)
    dmu = float(dmu)
    offset = float(offset)

    if gamma <= 0.0:
        raise ValueError("gamma must be > 0")
    if offset <= 0.0:
        raise ValueError("offset must be > 0")

    def _vector_argument(name, value):
        if _is_boolean(value):
            raise ValueError(f"{name} must contain real numbers")
        try:
            array = np.asarray(value, dtype=float)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must contain real numbers")
        if array.ndim > 1:
            raise ValueError(f"{name} must be scalar or one-dimensional")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite entries")
        return array

    center_array = _vector_argument("center", center)
    kappa_array = _vector_argument("kappa", kappa)
    try:
        center_array, kappa_array = np.broadcast_arrays(center_array, kappa_array)
    except ValueError:
        raise ValueError("center and kappa must be broadcast-compatible")
    if np.any(kappa_array <= 0.0):
        raise ValueError("every kappa must be > 0")
    if np.any(center_array <= -offset):
        raise ValueError("every center must lie above -offset")

    def _gradient_at(s):
        return (2.0 / 3.0) * gamma * (s + offset) ** (-1.0 / 3.0) - dmu

    def _curvature_at(s):
        return -(2.0 / 9.0) * gamma * (s + offset) ** (-4.0 / 3.0)

    result = np.empty(center_array.shape, dtype=float)
    for index in np.ndindex(center_array.shape):
        one_center = float(center_array[index])
        one_kappa = float(kappa_array[index])
        mean_cv = one_center
        for _ in range(100):
            denominator = _curvature_at(mean_cv) + one_kappa
            if denominator <= 0.0:
                raise ValueError("the restraint is too weak to stabilise a window")
            step = (_gradient_at(mean_cv) + one_kappa * (mean_cv - one_center)) / denominator
            mean_cv -= step
            if mean_cv <= -offset:
                raise ValueError("a restrained mean left the domain of the model surface")
            if abs(step) <= 1.0e-15 * max(1.0, abs(mean_cv)):
                break
        else:
            raise ValueError("a restrained mean did not converge")
        result[index] = -one_kappa * (mean_cv - one_center)

    if result.ndim == 0:
        return float(result)
    return result

def _is_real_scalar(value) -> bool:
    """Return True for a real numeric scalar, excluding booleans."""
    import numpy as np

    if isinstance(value, (bool, np.bool_)):
        return False
    return isinstance(value, (int, float, np.integer, np.floating))


def _is_boolean(value) -> bool:
    """Return True for a boolean scalar or a numpy boolean."""
    import numpy as np

    return isinstance(value, (bool, np.bool_))


def compute_kernel_embedding(query: np.ndarray, lower: float = 0.0,
                                     upper: float = 288.0, variance: float = 0.25,
                                     lengthscale: float = 20.0,
                                     intervals: np.ndarray = None) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    for name, value in (("lower", lower), ("upper", upper),
                        ("variance", variance), ("lengthscale", lengthscale)):
        if not _is_real_scalar(value):
            raise ValueError(f"{name} must be a real number")
        if not np.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")

    lower = float(lower)
    upper = float(upper)
    variance = float(variance)
    lengthscale = float(lengthscale)

    if upper <= lower:
        raise ValueError("upper must be greater than lower")
    if variance <= 0.0:
        raise ValueError("variance must be > 0")
    if lengthscale <= 0.0:
        raise ValueError("lengthscale must be > 0")
    if _is_boolean(query):
        raise ValueError("query must contain real numbers")
    try:
        queries = np.asarray(query, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("query must contain real numbers")
    scalar_query = queries.ndim == 0
    if queries.ndim > 1 or queries.size == 0:
        raise ValueError("query must be scalar or a non-empty one-dimensional array")
    queries = queries.reshape(-1)
    if not np.all(np.isfinite(queries)):
        raise ValueError("query must contain only finite entries")
    if np.any(queries < lower) or np.any(queries > upper):
        raise ValueError("every query must lie inside the integration range")

    explicit_intervals = intervals is not None
    if intervals is None:
        bounds = np.array([[lower, upper]], dtype=float)
    else:
        try:
            bounds = np.asarray(intervals, dtype=float)
        except (TypeError, ValueError):
            raise ValueError("intervals must contain real numbers")
        if bounds.ndim != 2 or bounds.shape[1] != 2 or bounds.shape[0] == 0:
            raise ValueError("intervals must have shape (m, 2) with m >= 1")
        if not np.all(np.isfinite(bounds)):
            raise ValueError("intervals must contain only finite entries")
        if np.any(bounds[:, 0] >= bounds[:, 1]):
            raise ValueError("every interval must have a lower bound below its upper bound")
        if np.any(bounds[:, 0] < lower) or np.any(bounds[:, 1] > upper):
            raise ValueError("every interval must lie inside the integration range")

    a = bounds[:, 0, None]
    b = bounds[:, 1, None]
    q = queries[None, :]
    span_factor = -np.expm1(-(b - a) / lengthscale)
    left_value = (np.exp(-(a - q) / lengthscale) * span_factor)
    right_value = (np.exp(-(q - b) / lengthscale) * span_factor)
    inside_value = (-np.expm1(-(q - a) / lengthscale)
                    - np.expm1(-(b - q) / lengthscale))
    dimensionless = np.where(q < a, left_value,
                             np.where(q > b, right_value, inside_value))
    result = variance * lengthscale * dimensionless

    if not explicit_intervals:
        result = result[0]
        if scalar_query:
            return float(result[0])
        return result
    if scalar_query:
        return result[:, 0]
    return result

def _is_real_scalar(value) -> bool:
    """Return True for a real numeric scalar, excluding booleans."""
    import numpy as np

    if isinstance(value, (bool, np.bool_)):
        return False
    return isinstance(value, (int, float, np.integer, np.floating))


def compute_initial_integral_variance(lower: float = 0.0, upper: float = 288.0,
                                              variance: float = 0.25,
                                              lengthscale: float = 20.0,
                                              intervals: np.ndarray = None) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    for name, value in (("lower", lower), ("upper", upper),
                        ("variance", variance), ("lengthscale", lengthscale)):
        if not _is_real_scalar(value):
            raise ValueError(f"{name} must be a real number")
        if not np.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")

    lower = float(lower)
    upper = float(upper)
    variance = float(variance)
    lengthscale = float(lengthscale)

    if upper <= lower:
        raise ValueError("upper must be greater than lower")
    if variance <= 0.0:
        raise ValueError("variance must be > 0")
    if lengthscale <= 0.0:
        raise ValueError("lengthscale must be > 0")

    if intervals is None:
        span = upper - lower
        return float(2.0 * variance * lengthscale
                     * (span + lengthscale * np.expm1(-span / lengthscale)))

    try:
        bounds = np.asarray(intervals, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("intervals must contain real numbers")
    if bounds.ndim != 2 or bounds.shape[1] != 2 or bounds.shape[0] == 0:
        raise ValueError("intervals must have shape (m, 2) with m >= 1")
    if not np.all(np.isfinite(bounds)):
        raise ValueError("intervals must contain only finite entries")
    if np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError("every interval must have a lower bound below its upper bound")
    if np.any(bounds[:, 0] < lower) or np.any(bounds[:, 1] > upper):
        raise ValueError("every interval must lie inside the integration range")

    a = bounds[:, 0]
    b = bounds[:, 1]

    def _second_primitive(delta):
        distance = np.abs(np.asarray(delta, dtype=np.longdouble))
        ell = np.longdouble(lengthscale)
        amplitude = np.longdouble(variance)
        return amplitude * ell * ell * (distance / ell + np.exp(-distance / ell))

    result = (_second_primitive(b[:, None] - a[None, :])
              - _second_primitive(a[:, None] - a[None, :])
              - _second_primitive(b[:, None] - b[None, :])
              + _second_primitive(a[:, None] - b[None, :]))
    result = np.asarray(result, dtype=float)
    return 0.5 * (result + result.T)

def _is_real_scalar(value) -> bool:
    """Return True for a real numeric scalar, excluding booleans."""
    import numpy as np

    if isinstance(value, (bool, np.bool_)):
        return False
    return isinstance(value, (int, float, np.integer, np.floating))


def _is_boolean(value) -> bool:
    """Return True for a boolean scalar or a numpy boolean."""
    import numpy as np

    return isinstance(value, (bool, np.bool_))


def compute_posterior_mean_gradient(centers: np.ndarray, forces: np.ndarray,
                                            query: np.ndarray,
                                            variance: float = 0.25, lengthscale: float = 20.0,
                                            noise: np.ndarray = 1.0e-3) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    for name, value in (("variance", variance), ("lengthscale", lengthscale)):
        if not _is_real_scalar(value):
            raise ValueError(f"{name} must be a real number")
        if not np.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    if float(variance) <= 0.0:
        raise ValueError("variance must be > 0")
    if float(lengthscale) <= 0.0:
        raise ValueError("lengthscale must be > 0")
    centers = np.asarray(centers, dtype=float).ravel()
    forces = np.asarray(forces, dtype=float).ravel()
    if centers.size < 1:
        raise ValueError("at least one window is required")
    if forces.shape != centers.shape:
        raise ValueError("centers and forces must have the same length")
    if not (np.all(np.isfinite(centers)) and np.all(np.isfinite(forces))):
        raise ValueError("centers and forces must contain only finite entries")

    if _is_boolean(query):
        raise ValueError("query must contain real numbers")
    try:
        queries = np.asarray(query, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("query must contain real numbers")
    scalar_query = queries.ndim == 0
    if queries.ndim > 1 or queries.size == 0:
        raise ValueError("query must be scalar or a non-empty one-dimensional array")
    queries = queries.reshape(-1)
    if not np.all(np.isfinite(queries)):
        raise ValueError("query must contain only finite entries")

    try:
        noise_array = np.asarray(noise, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("noise must contain real numbers")
    if noise_array.ndim == 0:
        noise_array = np.full(centers.size, float(noise_array))
    elif noise_array.shape != centers.shape:
        raise ValueError("noise must be scalar or have shape (n,)")
    if not np.all(np.isfinite(noise_array)) or np.any(noise_array < 0.0):
        raise ValueError("noise variances must be finite and non-negative")

    gram = float(variance) * np.exp(-np.abs(centers[:, None] - centers[None, :]) / float(lengthscale))
    gram = gram + np.diag(noise_array)
    cross = float(variance) * np.exp(
        -np.abs(queries[:, None] - centers[None, :]) / float(lengthscale))

    try:
        weights = np.linalg.solve(gram, forces)
    except np.linalg.LinAlgError:
        raise ValueError("the covariance matrix of the observations is singular")

    result = cross @ weights
    if scalar_query:
        return float(result[0])
    return np.asarray(result, dtype=float)

def _is_real_scalar(value) -> bool:
    """Return True for a real numeric scalar, excluding booleans."""
    import numpy as np

    if isinstance(value, (bool, np.bool_)):
        return False
    return isinstance(value, (int, float, np.integer, np.floating))


def compute_integral_posterior_mean(centers: np.ndarray, forces: np.ndarray,
                                            lower: float = 0.0, upper: float = 288.0,
                                            variance: float = 0.25, lengthscale: float = 20.0,
                                            noise: np.ndarray = 1.0e-3,
                                            intervals: np.ndarray = None) -> np.ndarray:
    # Local import keeps the oracle self-contained when the harness executes it
    # in isolation. Earlier steps are reached by name in the concatenated
    # namespace the harness builds.
    import numpy as np

    for name, value in (("lower", lower), ("upper", upper), ("variance", variance),
                        ("lengthscale", lengthscale)):
        if not _is_real_scalar(value):
            raise ValueError(f"{name} must be a real number")
        if not np.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    if float(upper) <= float(lower):
        raise ValueError("upper must be greater than lower")
    if float(variance) <= 0.0:
        raise ValueError("variance must be > 0")
    if float(lengthscale) <= 0.0:
        raise ValueError("lengthscale must be > 0")
    centers = np.asarray(centers, dtype=float).ravel()
    forces = np.asarray(forces, dtype=float).ravel()
    if centers.size < 1:
        raise ValueError("at least one window is required")
    if forces.shape != centers.shape:
        raise ValueError("centers and forces must have the same length")
    if not (np.all(np.isfinite(centers)) and np.all(np.isfinite(forces))):
        raise ValueError("centers and forces must contain only finite entries")
    if np.any(centers < float(lower)) or np.any(centers > float(upper)):
        raise ValueError("every window centre must lie inside the integration range")

    try:
        noise_array = np.asarray(noise, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("noise must contain real numbers")
    if noise_array.ndim == 0:
        noise_array = np.full(centers.size, float(noise_array))
    elif noise_array.shape != centers.shape:
        raise ValueError("noise must be scalar or have shape (n,)")
    if not np.all(np.isfinite(noise_array)) or np.any(noise_array < 0.0):
        raise ValueError("noise variances must be finite and non-negative")

    embedding = np.asarray(compute_kernel_embedding(
        centers, float(lower), float(upper), float(variance),
        float(lengthscale), intervals), dtype=float)
    expected_shape = centers.shape if intervals is None else (np.asarray(intervals).shape[0], centers.size)
    if embedding.shape != expected_shape or not np.all(np.isfinite(embedding)):
        raise ValueError("the delegated kernel embedding has the wrong shape")

    gram = float(variance) * np.exp(-np.abs(centers[:, None] - centers[None, :]) / float(lengthscale))
    gram = gram + np.diag(noise_array)

    try:
        weights = np.linalg.solve(gram, forces)
    except np.linalg.LinAlgError:
        raise ValueError("the covariance matrix of the observations is singular")

    result = embedding @ weights
    if intervals is None:
        return float(result)
    return np.asarray(result, dtype=float)

def _is_real_scalar(value) -> bool:
    """Return True for a real numeric scalar, excluding booleans."""
    import numpy as np

    if isinstance(value, (bool, np.bool_)):
        return False
    return isinstance(value, (int, float, np.integer, np.floating))


def compute_integral_posterior_variance(centers: np.ndarray, lower: float = 0.0,
                                                upper: float = 288.0, variance: float = 0.25,
                                                lengthscale: float = 20.0,
                                                noise: np.ndarray = 1.0e-3,
                                                intervals: np.ndarray = None) -> np.ndarray:
    # Local import keeps the oracle self-contained when the harness executes it
    # in isolation. Earlier steps are reached by name in the concatenated
    # namespace the harness builds.
    import numpy as np

    for name, value in (("lower", lower), ("upper", upper), ("variance", variance),
                        ("lengthscale", lengthscale)):
        if not _is_real_scalar(value):
            raise ValueError(f"{name} must be a real number")
        if not np.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    if float(upper) <= float(lower):
        raise ValueError("upper must be greater than lower")
    if float(variance) <= 0.0:
        raise ValueError("variance must be > 0")
    if float(lengthscale) <= 0.0:
        raise ValueError("lengthscale must be > 0")
    centers = np.asarray(centers, dtype=float).ravel()
    if centers.size < 1:
        raise ValueError("at least one window is required")
    if not np.all(np.isfinite(centers)):
        raise ValueError("centers must contain only finite entries")
    if np.any(centers < float(lower)) or np.any(centers > float(upper)):
        raise ValueError("every window centre must lie inside the integration range")

    try:
        noise_array = np.asarray(noise, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("noise must contain real numbers")
    if noise_array.ndim == 0:
        noise_array = np.full(centers.size, float(noise_array))
    elif noise_array.shape != centers.shape:
        raise ValueError("noise must be scalar or have shape (n,)")
    if not np.all(np.isfinite(noise_array)) or np.any(noise_array < 0.0):
        raise ValueError("noise variances must be finite and non-negative")

    embedding = np.asarray(compute_kernel_embedding(
        centers, float(lower), float(upper), float(variance),
        float(lengthscale), intervals), dtype=float)
    prior = np.asarray(compute_initial_integral_variance(
        float(lower), float(upper), float(variance),
        float(lengthscale), intervals), dtype=float)

    if intervals is None:
        if embedding.shape != centers.shape or prior.ndim != 0:
            raise ValueError("a delegated scalar integral result has the wrong shape")
    else:
        try:
            interval_count = np.asarray(intervals).shape[0]
        except (AttributeError, IndexError):
            raise ValueError("intervals is invalid")
        if (embedding.shape != (interval_count, centers.size)
                or prior.shape != (interval_count, interval_count)):
            raise ValueError("a delegated interval result has the wrong shape")
    if not np.all(np.isfinite(embedding)) or not np.all(np.isfinite(prior)):
        raise ValueError("a delegated integral result is non-finite")

    gram = float(variance) * np.exp(-np.abs(centers[:, None] - centers[None, :]) / float(lengthscale))
    gram = gram + np.diag(noise_array)

    try:
        if intervals is None:
            explained = float(embedding @ np.linalg.solve(gram, embedding))
        else:
            explained = embedding @ np.linalg.solve(gram, embedding.T)
    except np.linalg.LinAlgError:
        raise ValueError("the covariance matrix of the observations is singular")

    result = prior - explained
    if intervals is None:
        return float(result)
    result = np.asarray(result, dtype=float)
    return 0.5 * (result + result.T)

def _is_real_scalar(value) -> bool:
    """Return True for a real numeric scalar, excluding booleans."""
    import numpy as np

    if isinstance(value, (bool, np.bool_)):
        return False
    return isinstance(value, (int, float, np.integer, np.floating))


def compute_ivr_acquisition(centers: np.ndarray, candidate: np.ndarray,
                                    lower: float = 0.0, upper: float = 288.0,
                                    variance: float = 0.25, lengthscale: float = 20.0,
                                    noise: np.ndarray = 1.0e-3,
                                    candidate_noise: np.ndarray = None) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    for name, value in (("lower", lower), ("upper", upper),
                        ("variance", variance), ("lengthscale", lengthscale)):
        if not _is_real_scalar(value):
            raise ValueError(f"{name} must be a real number")
        if not np.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    if float(upper) <= float(lower):
        raise ValueError("upper must be greater than lower")
    if float(variance) <= 0.0:
        raise ValueError("variance must be > 0")
    if float(lengthscale) <= 0.0:
        raise ValueError("lengthscale must be > 0")
    try:
        centers = np.asarray(centers, dtype=float)
        candidates = np.asarray(candidate, dtype=float)
        noise_array = np.asarray(noise, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("centers, candidate, and noise must contain real numbers")
    if centers.ndim != 1 or centers.size < 1:
        raise ValueError("at least one window is required")
    if candidates.ndim > 1 or candidates.size < 1:
        raise ValueError("candidate must be a scalar or non-empty one-dimensional array")
    if noise_array.ndim > 1:
        raise ValueError("noise must be scalar or one-dimensional")
    if not np.all(np.isfinite(centers)):
        raise ValueError("centers must contain only finite entries")
    if not np.all(np.isfinite(candidates)):
        raise ValueError("candidate must contain only finite entries")
    if np.any(centers < float(lower)) or np.any(centers > float(upper)):
        raise ValueError("every window centre must lie inside the integration range")
    if np.any(candidates < float(lower)) or np.any(candidates > float(upper)):
        raise ValueError("every candidate must lie inside the integration range")

    noise_was_scalar = noise_array.ndim == 0
    if noise_was_scalar:
        noise_vector = np.full(centers.size, float(noise_array))
    elif noise_array.shape == centers.shape:
        noise_vector = noise_array.astype(float, copy=False)
    else:
        raise ValueError("heteroscedastic noise must have shape (n,)")
    if np.any(~np.isfinite(noise_vector)) or np.any(noise_vector < 0.0):
        raise ValueError("noise variances must be finite and non-negative")

    if candidate_noise is None:
        if not noise_was_scalar:
            raise ValueError("candidate_noise is required with heteroscedastic noise")
        candidate_noise_array = np.asarray(float(noise_array))
    else:
        try:
            candidate_noise_array = np.asarray(candidate_noise, dtype=float)
        except (TypeError, ValueError):
            raise ValueError("candidate_noise must contain real numbers")
        if candidate_noise_array.ndim > 1:
            raise ValueError("candidate_noise must be scalar or one-dimensional")
    try:
        candidate_noise_vector = np.broadcast_to(candidate_noise_array, candidates.shape)
    except ValueError:
        raise ValueError("candidate_noise must be broadcast-compatible with candidate")
    if (np.any(~np.isfinite(candidate_noise_vector)) or
            np.any(candidate_noise_vector < 0.0)):
        raise ValueError("candidate noise variances must be finite and non-negative")

    def _embed(points):
        return float(variance) * float(lengthscale) * (
            2.0 - np.exp(-(points - float(lower)) / float(lengthscale))
            - np.exp(-(float(upper) - points) / float(lengthscale)))

    embedding = _embed(centers)
    flat_candidates = candidates.reshape(-1)
    cross = float(variance) * np.exp(
        -np.abs(flat_candidates[:, None] - centers[None, :]) / float(lengthscale))
    gram = float(variance) * np.exp(-np.abs(centers[:, None] - centers[None, :]) / float(lengthscale))
    gram = gram + np.diag(noise_vector)

    try:
        solved_embedding = np.linalg.solve(gram, embedding)
        solved_cross = np.linalg.solve(gram, cross.T)
    except np.linalg.LinAlgError:
        raise ValueError("the covariance matrix of the observations is singular")

    numerator = _embed(flat_candidates) - cross @ solved_embedding
    candidate_noise_flat = np.broadcast_to(
        candidate_noise_vector, candidates.shape).reshape(-1)
    denominator = (float(variance) + candidate_noise_flat
                   - np.einsum("ij,ji->i", cross, solved_cross))
    scores = np.zeros_like(numerator)
    safe = denominator > 0.0
    scores[safe] = numerator[safe] ** 2 / denominator[safe]
    scores = scores.reshape(candidates.shape)
    if scores.ndim == 0:
        return float(scores)
    return scores

def _is_real_scalar(value) -> bool:
    """Return True for a real numeric scalar, excluding booleans."""
    import numpy as np

    if isinstance(value, (bool, np.bool_)):
        return False
    return isinstance(value, (int, float, np.integer, np.floating))


def _is_integer_scalar(value) -> bool:
    """Return True for an integer scalar, excluding booleans."""
    import numpy as np

    if isinstance(value, (bool, np.bool_)):
        return False
    return isinstance(value, (int, np.integer))


def compute_free_energy_profile_value(centers: np.ndarray, forces: np.ndarray,
                                              node: np.ndarray,
                                              n_grid: int = 100, lower: float = 0.0,
                                              upper: float = 288.0, variance: float = 0.25,
                                              lengthscale: float = 20.0,
                                              noise: np.ndarray = 1.0e-3) -> np.ndarray:
    # Local import keeps the oracle self-contained when the harness executes it
    # in isolation. Earlier steps are reached by name in the concatenated
    # namespace the harness builds.
    import numpy as np

    for name, value in (("lower", lower), ("upper", upper), ("variance", variance),
                        ("lengthscale", lengthscale)):
        if not _is_real_scalar(value):
            raise ValueError(f"{name} must be a real number")
        if not np.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    if not _is_integer_scalar(n_grid):
        raise ValueError("n_grid must be an integer")
    try:
        nodes = np.asarray(node)
    except (TypeError, ValueError):
        raise ValueError("node must contain integers")
    if (nodes.ndim > 1 or nodes.size < 1 or
            np.issubdtype(nodes.dtype, np.bool_) or
            not np.issubdtype(nodes.dtype, np.integer)):
        raise ValueError("node must be a scalar or non-empty one-dimensional integer array")
    if float(upper) <= float(lower):
        raise ValueError("upper must be greater than lower")
    if float(variance) <= 0.0:
        raise ValueError("variance must be > 0")
    if float(lengthscale) <= 0.0:
        raise ValueError("lengthscale must be > 0")
    if int(n_grid) < 2:
        raise ValueError("n_grid must be at least 2")
    if np.any(nodes < 0) or np.any(nodes >= int(n_grid)):
        raise ValueError("every node must index a grid point")

    centers = np.asarray(centers, dtype=float).ravel()
    forces = np.asarray(forces, dtype=float).ravel()
    if centers.size < 1:
        raise ValueError("at least one window is required")
    if forces.shape != centers.shape:
        raise ValueError("centers and forces must have the same length")
    if not (np.all(np.isfinite(centers)) and np.all(np.isfinite(forces))):
        raise ValueError("centers and forces must contain only finite entries")

    grid = np.linspace(float(lower), float(upper), int(n_grid))
    predicted = np.asarray(compute_posterior_mean_gradient(
        centers, forces, grid, float(variance), float(lengthscale), noise),
        dtype=float)
    if predicted.shape != grid.shape or not np.all(np.isfinite(predicted)):
        raise ValueError("the delegated grid prediction has the wrong shape")
    spacing = grid[1] - grid[0]
    raw = np.concatenate(([0.0], np.cumsum(0.5 * (predicted[1:] + predicted[:-1]) * spacing)))
    profile = raw - raw.min()

    values = profile[nodes]
    if values.ndim == 0:
        return float(values)
    return values

def _is_real_scalar(value) -> bool:
    """Return True for a real numeric scalar, excluding booleans."""
    import numpy as np

    if isinstance(value, (bool, np.bool_)):
        return False
    return isinstance(value, (int, float, np.integer, np.floating))


def _is_integer_scalar(value) -> bool:
    """Return True for an integer scalar, excluding booleans."""
    import numpy as np

    if isinstance(value, (bool, np.bool_)):
        return False
    return isinstance(value, (int, np.integer))


def compute_combined_acquisition(centers: np.ndarray, forces: np.ndarray,
                                         node: np.ndarray,
                                         n_grid: int = 100, weight: float = 0.45,
                                         lower: float = 0.0, upper: float = 288.0,
                                         variance: float = 0.25, lengthscale: float = 20.0,
                                         noise: np.ndarray = 1.0e-3,
                                         candidate_noise: np.ndarray = None,
                                         candidate_cost: np.ndarray = None) -> np.ndarray:
    # Local import keeps the oracle self-contained when the harness executes it
    # in isolation. Earlier steps are reached by name in the concatenated
    # namespace the harness builds.
    import numpy as np

    for name, value in (("weight", weight), ("lower", lower), ("upper", upper),
                        ("variance", variance), ("lengthscale", lengthscale)):
        if not _is_real_scalar(value):
            raise ValueError(f"{name} must be a real number")
        if not np.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    if not _is_integer_scalar(n_grid):
        raise ValueError("n_grid must be an integer")
    try:
        nodes = np.asarray(node)
    except (TypeError, ValueError):
        raise ValueError("node must contain integers")
    if (nodes.ndim > 1 or nodes.size < 1 or
            np.issubdtype(nodes.dtype, np.bool_) or
            not np.issubdtype(nodes.dtype, np.integer)):
        raise ValueError("node must be a scalar or non-empty one-dimensional integer array")
    if float(weight) < 0.0 or float(weight) > 1.0:
        raise ValueError("weight must lie in [0, 1]")
    if float(upper) <= float(lower):
        raise ValueError("upper must be greater than lower")
    if float(variance) <= 0.0:
        raise ValueError("variance must be > 0")
    if float(lengthscale) <= 0.0:
        raise ValueError("lengthscale must be > 0")
    if int(n_grid) < 2:
        raise ValueError("n_grid must be at least 2")
    if np.any(nodes < 0) or np.any(nodes >= int(n_grid)):
        raise ValueError("every node must index a grid point")

    centers = np.asarray(centers, dtype=float).ravel()
    forces = np.asarray(forces, dtype=float).ravel()
    if centers.size < 1:
        raise ValueError("at least one window is required")
    if forces.shape != centers.shape:
        raise ValueError("centers and forces must have the same length")
    if not (np.all(np.isfinite(centers)) and np.all(np.isfinite(forces))):
        raise ValueError("centers and forces must contain only finite entries")
    if np.any(centers < float(lower)) or np.any(centers > float(upper)):
        raise ValueError("every window centre must lie inside the integration range")

    try:
        noise_array = np.asarray(noise, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("noise must contain real numbers")
    noise_was_scalar = noise_array.ndim == 0
    if noise_was_scalar:
        observation_noise = np.full(centers.size, float(noise_array))
    elif noise_array.shape == centers.shape:
        observation_noise = noise_array.astype(float, copy=False)
    else:
        raise ValueError("noise must be scalar or have shape (n,)")
    if (not np.all(np.isfinite(observation_noise)) or
            np.any(observation_noise < 0.0)):
        raise ValueError("noise variances must be finite and non-negative")

    if candidate_noise is None:
        if not noise_was_scalar:
            raise ValueError("candidate_noise is required with window-specific noise")
        candidate_noise_array = np.asarray(float(noise_array))
    else:
        try:
            raw_candidate_noise = np.asarray(candidate_noise)
            if np.issubdtype(raw_candidate_noise.dtype, np.bool_):
                raise ValueError("candidate_noise must contain real numbers")
            candidate_noise_array = np.asarray(candidate_noise, dtype=float)
        except (TypeError, ValueError):
            raise ValueError("candidate_noise must contain real numbers")
        if candidate_noise_array.ndim > 1:
            raise ValueError("candidate_noise must be scalar or one-dimensional")
    if candidate_noise_array.ndim == 0:
        candidate_noise_grid = np.full(int(n_grid), float(candidate_noise_array))
    elif candidate_noise_array.shape == (int(n_grid),):
        candidate_noise_grid = candidate_noise_array.astype(float, copy=False)
    else:
        raise ValueError("candidate_noise must be scalar or have shape (n_grid,)")
    if (not np.all(np.isfinite(candidate_noise_grid)) or
            np.any(candidate_noise_grid < 0.0)):
        raise ValueError("candidate noise variances must be finite and non-negative")

    if candidate_cost is None:
        candidate_cost_array = np.asarray(1.0)
    else:
        try:
            raw_candidate_cost = np.asarray(candidate_cost)
            if np.issubdtype(raw_candidate_cost.dtype, np.bool_):
                raise ValueError("candidate_cost must contain real numbers")
            candidate_cost_array = np.asarray(candidate_cost, dtype=float)
        except (TypeError, ValueError):
            raise ValueError("candidate_cost must contain real numbers")
        if candidate_cost_array.ndim > 1:
            raise ValueError("candidate_cost must be scalar or one-dimensional")
    if candidate_cost_array.ndim == 0:
        candidate_cost_grid = np.full(int(n_grid), float(candidate_cost_array))
    elif candidate_cost_array.shape == (int(n_grid),):
        candidate_cost_grid = candidate_cost_array.astype(float, copy=False)
    else:
        raise ValueError("candidate_cost must be scalar or have shape (n_grid,)")
    if (not np.all(np.isfinite(candidate_cost_grid)) or
            np.any(candidate_cost_grid <= 0.0)):
        raise ValueError("candidate costs must be finite and strictly positive")

    def _normalise(values):
        low, high = float(values.min()), float(values.max())
        if high <= low:
            return np.zeros_like(values)
        return (values - low) / (high - low)

    grid = np.linspace(float(lower), float(upper), int(n_grid))
    ivr = np.asarray(compute_ivr_acquisition(
        centers, grid, lower, upper, variance, lengthscale,
        observation_noise, candidate_noise_grid), dtype=float)
    profile = np.asarray(compute_free_energy_profile_value(
        centers, forces, np.arange(int(n_grid), dtype=int), int(n_grid),
        lower, upper, variance, lengthscale, observation_noise), dtype=float)
    if (ivr.shape != grid.shape or profile.shape != grid.shape or
            not np.all(np.isfinite(ivr)) or not np.all(np.isfinite(profile))):
        raise ValueError("the delegated batched acquisition inputs are invalid")

    cost_sensitive_ivr = ivr / candidate_cost_grid
    combined = (-float(weight) * _normalise(profile)
                + (1.0 - float(weight)) * _normalise(cost_sensitive_ivr))

    values = combined[nodes]
    if values.ndim == 0:
        return float(values)
    return values

def _is_real_scalar(value) -> bool:
    """Return True for a real numeric scalar, excluding booleans."""
    import numpy as np

    if isinstance(value, (bool, np.bool_)):
        return False
    return isinstance(value, (int, float, np.integer, np.floating))


def _is_integer_scalar(value) -> bool:
    """Return True for an integer scalar, excluding booleans."""
    import numpy as np

    if isinstance(value, (bool, np.bool_)):
        return False
    return isinstance(value, (int, np.integer))


def select_next_umbrella_center(centers: np.ndarray, forces: np.ndarray, n_grid: int = 100,
                                        weight: float = 0.45, lower: float = 0.0,
                                        upper: float = 288.0, variance: float = 0.25,
                                        lengthscale: float = 20.0,
                                        noise: np.ndarray = 1.0e-3,
                                        candidate_noise: np.ndarray = None,
                                        candidate_cost: np.ndarray = None,
                                        max_cost: float = None) -> float:
    # Local import keeps the oracle self-contained when the harness executes it
    # in isolation. Earlier steps are reached by name in the concatenated
    # namespace the harness builds.
    import numpy as np

    for name, value in (("weight", weight), ("lower", lower), ("upper", upper),
                        ("variance", variance), ("lengthscale", lengthscale)):
        if not _is_real_scalar(value):
            raise ValueError(f"{name} must be a real number")
        if not np.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    if not _is_integer_scalar(n_grid):
        raise ValueError("n_grid must be an integer")
    if float(weight) < 0.0 or float(weight) > 1.0:
        raise ValueError("weight must lie in [0, 1]")
    if float(upper) <= float(lower):
        raise ValueError("upper must be greater than lower")
    if float(variance) <= 0.0:
        raise ValueError("variance must be > 0")
    if float(lengthscale) <= 0.0:
        raise ValueError("lengthscale must be > 0")
    if int(n_grid) < 2:
        raise ValueError("n_grid must be at least 2")

    centers = np.asarray(centers, dtype=float).ravel()
    forces = np.asarray(forces, dtype=float).ravel()
    if centers.size < 1:
        raise ValueError("at least one window is required")
    if forces.shape != centers.shape:
        raise ValueError("centers and forces must have the same length")
    if not (np.all(np.isfinite(centers)) and np.all(np.isfinite(forces))):
        raise ValueError("centers and forces must contain only finite entries")
    if np.any(centers < float(lower)) or np.any(centers > float(upper)):
        raise ValueError("every window centre must lie inside the integration range")

    try:
        noise_array = np.asarray(noise, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("noise must contain real numbers")
    noise_was_scalar = noise_array.ndim == 0
    if noise_was_scalar:
        observation_noise = np.full(centers.size, float(noise_array))
    elif noise_array.shape == centers.shape:
        observation_noise = noise_array.astype(float, copy=False)
    else:
        raise ValueError("noise must be scalar or have shape (n,)")
    if (not np.all(np.isfinite(observation_noise)) or
            np.any(observation_noise < 0.0)):
        raise ValueError("noise variances must be finite and non-negative")

    if candidate_noise is None:
        if not noise_was_scalar:
            raise ValueError("candidate_noise is required with window-specific noise")
        candidate_noise_array = np.asarray(float(noise_array))
    else:
        try:
            raw_candidate_noise = np.asarray(candidate_noise)
            if np.issubdtype(raw_candidate_noise.dtype, np.bool_):
                raise ValueError("candidate_noise must contain real numbers")
            candidate_noise_array = np.asarray(candidate_noise, dtype=float)
        except (TypeError, ValueError):
            raise ValueError("candidate_noise must contain real numbers")
        if candidate_noise_array.ndim > 1:
            raise ValueError("candidate_noise must be scalar or one-dimensional")
    if candidate_noise_array.ndim == 0:
        candidate_noise_grid = np.full(int(n_grid), float(candidate_noise_array))
    elif candidate_noise_array.shape == (int(n_grid),):
        candidate_noise_grid = candidate_noise_array.astype(float, copy=False)
    else:
        raise ValueError("candidate_noise must be scalar or have shape (n_grid,)")
    if (not np.all(np.isfinite(candidate_noise_grid)) or
            np.any(candidate_noise_grid < 0.0)):
        raise ValueError("candidate noise variances must be finite and non-negative")

    if candidate_cost is None:
        candidate_cost_array = np.asarray(1.0)
    else:
        try:
            raw_candidate_cost = np.asarray(candidate_cost)
            if np.issubdtype(raw_candidate_cost.dtype, np.bool_):
                raise ValueError("candidate_cost must contain real numbers")
            candidate_cost_array = np.asarray(candidate_cost, dtype=float)
        except (TypeError, ValueError):
            raise ValueError("candidate_cost must contain real numbers")
        if candidate_cost_array.ndim > 1:
            raise ValueError("candidate_cost must be scalar or one-dimensional")
    if candidate_cost_array.ndim == 0:
        candidate_cost_grid = np.full(int(n_grid), float(candidate_cost_array))
    elif candidate_cost_array.shape == (int(n_grid),):
        candidate_cost_grid = candidate_cost_array.astype(float, copy=False)
    else:
        raise ValueError("candidate_cost must be scalar or have shape (n_grid,)")
    if (not np.all(np.isfinite(candidate_cost_grid)) or
            np.any(candidate_cost_grid <= 0.0)):
        raise ValueError("candidate costs must be finite and strictly positive")
    if max_cost is not None:
        if (not _is_real_scalar(max_cost) or not np.isfinite(float(max_cost))
                or float(max_cost) < 0.0):
            raise ValueError("max_cost must be a finite non-negative real number")

    grid = np.linspace(float(lower), float(upper), int(n_grid))
    score = np.asarray(compute_combined_acquisition(
        centers, forces, np.arange(int(n_grid), dtype=int), int(n_grid),
        weight, lower, upper, variance, lengthscale, observation_noise,
        candidate_noise_grid, candidate_cost_grid), dtype=float)
    if score.shape != grid.shape or not np.all(np.isfinite(score)):
        raise ValueError("the batched combined acquisition is invalid")
    for center in centers:
        score[np.isclose(grid, center, rtol=0.0, atol=1.0e-9)] = -np.inf
    if max_cost is not None:
        score[candidate_cost_grid > float(max_cost) + 1.0e-12] = -np.inf
    if not np.isfinite(score).any():
        raise ValueError("no unsampled affordable candidate remains")

    return float(grid[int(np.argmax(score))])

def run_bayesian_umbrella_quadrature(initial_centers: np.ndarray = None, n_queries: int = 15,
                                             kappa: float = 1.0, surface: tuple = None,
                                             domain: tuple = None, n_grid: int = 100,
                                             weight: float = 0.45, variance: float = 0.25,
                                             lengthscale: float = 20.0,
                                             noise: np.ndarray = 1.0e-3,
                                             rel_tol: float = 0.02, var_floor: float = 0.01,
                                             gain_tol: float = 0.02,
                                             interp_tol: float = 0.05,
                                             candidate_noise: np.ndarray = None,
                                             interval_edges: np.ndarray = None,
                                             total_cost_budget: float = 17.0) -> float:
    # Local import keeps the oracle self-contained when the harness executes it
    # in isolation. Sub-problems 01-10 are reached by their oracle names in the
    # concatenated namespace the harness builds; the public names are never
    # used, because in a shared namespace those belong to the candidate and the
    # gold side must not execute candidate code.
    import numpy as np

    umbrella_mean_force = compute_umbrella_mean_force
    kernel_embedding = compute_kernel_embedding
    initial_integral_variance = compute_initial_integral_variance
    posterior_mean_gradient = compute_posterior_mean_gradient
    integral_posterior_mean = compute_integral_posterior_mean
    integral_posterior_variance = compute_integral_posterior_variance
    ivr_acquisition = compute_ivr_acquisition
    free_energy_profile_value = compute_free_energy_profile_value
    combined_acquisition = compute_combined_acquisition
    next_umbrella_center = select_next_umbrella_center

    # -- Benchmark configuration of the task.
    if initial_centers is None:
        initial_centers = np.array([1.6, 3.0, 284.16, 285.0])
    if surface is None:
        surface = (2.5, 0.4, 8.0)
    if domain is None:
        domain = (0.0, 288.0)

    try:
        initial_centers = np.asarray(initial_centers, dtype=float).ravel()
    except (TypeError, ValueError):
        raise ValueError("initial_centers must contain real numbers")
    if initial_centers.size < 1:
        raise ValueError("at least one initialisation window is required")
    if not np.all(np.isfinite(initial_centers)):
        raise ValueError("initial_centers must contain finite real numbers")
    if isinstance(n_queries, bool) or not isinstance(n_queries, (int, np.integer)) or int(n_queries) < 0:
        raise ValueError("n_queries must be a non-negative integer")
    if (isinstance(n_grid, bool) or
            not isinstance(n_grid, (int, np.integer)) or int(n_grid) < 2):
        raise ValueError("n_grid must be an integer of at least 2")
    try:
        surface = tuple(surface)
    except (TypeError, ValueError):
        raise ValueError("surface must hold the three coefficients (gamma, dmu, offset)")
    try:
        domain = tuple(domain)
    except (TypeError, ValueError):
        raise ValueError("domain must hold the pair (lower, upper)")
    if len(surface) != 3:
        raise ValueError("surface must hold the three coefficients (gamma, dmu, offset)")
    if len(domain) != 2:
        raise ValueError("domain must hold the pair (lower, upper)")
    scalar_types = (int, float, np.integer, np.floating)
    for name, value in (("rel_tol", rel_tol), ("var_floor", var_floor),
                        ("gain_tol", gain_tol), ("interp_tol", interp_tol),
                        ("total_cost_budget", total_cost_budget)):
        if (isinstance(value, (bool, np.bool_)) or
                not isinstance(value, scalar_types) or
                not np.isfinite(float(value))):
            raise ValueError(f"{name} must be a finite real scalar")
    for name, value in (("rel_tol", rel_tol), ("var_floor", var_floor), ("gain_tol", gain_tol)):
        if value < 0.0:
            raise ValueError(f"{name} must be >= 0")
    if interp_tol <= 0.0:
        raise ValueError("interp_tol must be > 0")
    if total_cost_budget <= 0.0:
        raise ValueError("total_cost_budget must be > 0")

    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, scalar_types)
           for v in surface + domain):
        raise ValueError("surface and domain must contain real scalars")
    gamma, dmu, offset = (float(v) for v in surface)
    lower, upper = (float(v) for v in domain)
    if not np.all(np.isfinite((gamma, dmu, offset, lower, upper))):
        raise ValueError("surface and domain must contain finite real scalars")
    if upper <= lower:
        raise ValueError("upper must be greater than lower")
    if np.any(initial_centers < lower) or np.any(initial_centers > upper):
        raise ValueError("every initialisation window must lie inside the range")

    try:
        initial_noise_array = np.asarray(noise, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("noise must contain real numbers")
    noise_was_scalar = initial_noise_array.ndim == 0
    if noise_was_scalar:
        initial_noise_vector = np.full(initial_centers.size, float(initial_noise_array))
    elif initial_noise_array.shape == initial_centers.shape:
        initial_noise_vector = initial_noise_array.astype(float, copy=False)
    else:
        raise ValueError("noise must be scalar or match initial_centers")
    if (not np.all(np.isfinite(initial_noise_vector)) or
            np.any(initial_noise_vector < 0.0)):
        raise ValueError("noise variances must be finite and non-negative")

    if candidate_noise is None:
        if not noise_was_scalar:
            raise ValueError("candidate_noise is required with window-specific noise")
        candidate_noise_array = np.asarray(float(initial_noise_array))
    else:
        try:
            raw_candidate_noise = np.asarray(candidate_noise)
            if np.issubdtype(raw_candidate_noise.dtype, np.bool_):
                raise ValueError("candidate_noise must contain real numbers")
            candidate_noise_array = np.asarray(candidate_noise, dtype=float)
        except (TypeError, ValueError):
            raise ValueError("candidate_noise must contain real numbers")
        if candidate_noise_array.ndim > 1:
            raise ValueError("candidate_noise must be scalar or one-dimensional")
    candidate_noise_was_scalar = candidate_noise_array.ndim == 0
    if candidate_noise_array.ndim == 0:
        candidate_noise_grid = np.full(int(n_grid), float(candidate_noise_array))
    elif candidate_noise_array.shape == (int(n_grid),):
        candidate_noise_grid = candidate_noise_array.astype(float, copy=False)
    else:
        raise ValueError("candidate_noise must be scalar or have shape (n_grid,)")
    if (not np.all(np.isfinite(candidate_noise_grid)) or
            np.any(candidate_noise_grid < 0.0)):
        raise ValueError("candidate noise variances must be finite and non-negative")

    if interval_edges is None:
        partition_edges = np.linspace(lower, upper, 5)
    else:
        try:
            partition_edges = np.asarray(interval_edges, dtype=float)
        except (TypeError, ValueError):
            raise ValueError("interval_edges must contain real numbers")
        if partition_edges.ndim != 1 or partition_edges.size < 2:
            raise ValueError("interval_edges must be one-dimensional with at least two entries")
        if not np.all(np.isfinite(partition_edges)):
            raise ValueError("interval_edges must contain only finite entries")
        if np.any(np.diff(partition_edges) <= 0.0):
            raise ValueError("interval_edges must be strictly increasing")
        endpoint_tolerance = 1.0e-12 * max(1.0, abs(lower), abs(upper))
        if (not np.isclose(partition_edges[0], lower, rtol=0.0,
                           atol=endpoint_tolerance) or
                not np.isclose(partition_edges[-1], upper, rtol=0.0,
                               atol=endpoint_tolerance)):
            raise ValueError("interval_edges must span the complete domain")
        partition_edges = partition_edges.astype(float, copy=True)
        partition_edges[0] = lower
        partition_edges[-1] = upper

    grid = np.linspace(lower, upper, int(n_grid))
    width = upper - lower

    def _sampling_cost(location):
        """Dimensionless location-dependent effort used by both designs."""
        location = np.asarray(location, dtype=float)
        midpoint = 0.5 * (lower + upper)
        return (1.0
                + 0.6 * np.exp(-((location - midpoint) / (5.0 * width / 32.0)) ** 2)
                + 0.35 * np.sin(4.0 * np.pi * (location - lower) / width) ** 2)

    candidate_cost_grid = np.asarray(_sampling_cost(grid), dtype=float)
    if (candidate_cost_grid.shape != grid.shape
            or not np.all(np.isfinite(candidate_cost_grid))
            or np.any(candidate_cost_grid <= 0.0)):
        raise ValueError("the candidate cost schedule is invalid")
    integration_intervals = np.column_stack(
        (partition_edges[:-1], partition_edges[1:]))
    interval_count = integration_intervals.shape[0]

    # -- Sub-problems 02-03: the two kernel embeddings, and the scale-free
    #    reference they define for the acquisition. The doubly integrated
    #    covariance cannot exceed the range width times the largest kernel
    #    mean on the range; a violation means the two are inconsistent.
    prior_covariance = np.asarray(initial_integral_variance(
        lower, upper, variance, lengthscale, integration_intervals), dtype=float)
    if (prior_covariance.shape != (interval_count, interval_count)
            or not np.all(np.isfinite(prior_covariance))):
        raise ValueError("the prior interval covariance has the wrong shape")
    prior_variance = float(prior_covariance.sum())
    peak_embedding = float(kernel_embedding(0.5 * (lower + upper), lower, upper,
                                            variance, lengthscale))
    if prior_variance > (upper - lower) * peak_embedding * (1.0 + 1.0e-9):
        raise ValueError("the two kernel embeddings are mutually inconsistent")
    if candidate_noise_was_scalar:
        largest_gain = peak_embedding ** 2 / (
            float(variance) + float(candidate_noise_array))
    else:
        grid_embedding = np.asarray(kernel_embedding(
            grid, lower, upper, variance, lengthscale), dtype=float)
        if (grid_embedding.shape != grid.shape or
                not np.all(np.isfinite(grid_embedding))):
            raise ValueError("the batched kernel embedding has the wrong shape")
        largest_gain = float(np.max(
            grid_embedding ** 2 / (float(variance) + candidate_noise_grid)))

    # -- Sub-problem 01: evaluate the initial design through its batched
    #    interface; later acquisitions use the same function on a scalar.
    centers = [float(c) for c in initial_centers]
    initial_forces = np.asarray(
        umbrella_mean_force(np.asarray(centers), kappa, gamma, dmu, offset),
        dtype=float)
    if initial_forces.shape != (len(centers),):
        raise ValueError("the batched mean-force result has the wrong shape")
    forces = [float(value) for value in initial_forces]
    noise_values = [float(value) for value in initial_noise_vector]
    spent_cost = float(np.sum(_sampling_cost(np.asarray(centers, dtype=float))))
    if spent_cost > float(total_cost_budget) + 1.0e-12:
        raise ValueError("total_cost_budget cannot fund the initial adaptive design")

    for _ in range(int(n_queries)):
        # -- Sub-problems 05-06: the two convergence diagnostics.
        observation_noise = np.asarray(noise_values, dtype=float)
        interval_means = np.asarray(integral_posterior_mean(
            np.array(centers), np.array(forces), lower, upper,
            variance, lengthscale, observation_noise,
            integration_intervals), dtype=float)
        interval_covariance = np.asarray(integral_posterior_variance(
            np.array(centers), lower, upper, variance, lengthscale,
            observation_noise, integration_intervals), dtype=float)
        if (interval_means.shape != (interval_count,)
                or interval_covariance.shape != (interval_count, interval_count)
                or not np.all(np.isfinite(interval_means))
                or not np.all(np.isfinite(interval_covariance))):
            raise ValueError("the posterior interval diagnostics have the wrong shape")
        estimate = float(interval_means.sum())
        spread = float(interval_covariance.sum())
        if estimate != 0.0 and np.sqrt(max(spread, 0.0)) / abs(estimate) <= float(rel_tol):
            break
        if spread <= float(var_floor) * prior_variance:
            break

        remaining_cost = float(total_cost_budget) - spent_cost
        sampled = np.isclose(grid[:, None], np.asarray(centers)[None, :],
                             rtol=0.0, atol=1.0e-9).any(axis=1)
        affordable = candidate_cost_grid <= remaining_cost + 1.0e-12
        if not np.any((~sampled) & affordable):
            break

        # -- Sub-problem 10: the winning candidate.
        nxt = float(next_umbrella_center(np.array(centers), np.array(forces), int(n_grid),
                                         weight, lower, upper, variance,
                                         lengthscale, observation_noise,
                                         candidate_noise_grid,
                                         candidate_cost_grid, remaining_cost))
        node = int(np.argmin(np.abs(grid - nxt)))
        selected_noise = float(candidate_noise_grid[node])

        # -- Sub-problem 09: its combined score, which a degenerate menu makes
        #    non-finite, and sub-problem 07: the raw variance reduction behind
        #    that score, compared against the best any one observation can buy.
        if not np.isfinite(combined_acquisition(np.array(centers), np.array(forces), node,
                                                int(n_grid), weight, lower, upper,
                                                variance, lengthscale,
                                                observation_noise,
                                                candidate_noise_grid,
                                                candidate_cost_grid)):
            break
        gain = float(ivr_acquisition(np.array(centers), nxt, lower, upper,
                                     variance, lengthscale, observation_noise,
                                     selected_noise))
        if gain <= float(gain_tol) * largest_gain:
            break

        # -- Sub-problem 04: the surrogate must interpolate its own
        #    observations to within the white-noise level before it is trusted
        #    to predict anywhere else.
        for index in (0, len(centers) - 1):
            predicted = float(posterior_mean_gradient(np.array(centers), np.array(forces),
                                                      centers[index], variance,
                                                      lengthscale,
                                                      observation_noise))
            if abs(predicted - forces[index]) > float(interp_tol):
                raise ValueError("the surrogate does not interpolate its own observations")

        centers.append(nxt)
        forces.append(float(umbrella_mean_force(nxt, kappa, gamma, dmu, offset)))
        noise_values.append(selected_noise)
        spent_cost += float(candidate_cost_grid[node])

    # -- Sub-problem 08: the reconstructed profile on the grid.
    reconstructed = np.asarray(
        free_energy_profile_value(np.array(centers), np.array(forces),
                                  np.arange(int(n_grid), dtype=int),
                                  int(n_grid), lower, upper, variance,
                                  lengthscale, np.asarray(noise_values,
                                                         dtype=float)),
        dtype=float)
    if reconstructed.shape != grid.shape:
        raise ValueError("the batched reconstructed profile has the wrong shape")

    reference = gamma * (grid + offset) ** (2.0 / 3.0) - dmu * grid
    reference = reference - reference.min()

    adaptive_rmsd = float(np.sqrt(np.mean((reconstructed - reference) ** 2)))

    # Cost-matched control: choose the largest equal-bin midpoint design whose
    # complete cost fits the same total budget. Searching up to n_grid keeps
    # the control finite and deterministic for any valid benchmark grid.
    uniform_centers = None
    for count in range(1, int(n_grid) + 1):
        trial = lower + (np.arange(count, dtype=float) + 0.5) * width / count
        if float(np.sum(_sampling_cost(trial))) <= float(total_cost_budget) + 1.0e-12:
            uniform_centers = trial
    if uniform_centers is None:
        raise ValueError("total_cost_budget cannot fund one uniform-control window")
    uniform_forces = np.asarray(
        umbrella_mean_force(uniform_centers, kappa, gamma, dmu, offset), dtype=float)
    if uniform_forces.shape != uniform_centers.shape:
        raise ValueError("the uniform-control mean-force result has the wrong shape")
    if candidate_noise_was_scalar:
        uniform_noise = np.full(uniform_centers.size, float(candidate_noise_array))
    else:
        uniform_noise = np.interp(uniform_centers, grid, candidate_noise_grid)
    uniform_reconstructed = np.asarray(
        free_energy_profile_value(uniform_centers, uniform_forces,
                                  np.arange(int(n_grid), dtype=int), int(n_grid),
                                  lower, upper, variance, lengthscale,
                                  uniform_noise), dtype=float)
    if uniform_reconstructed.shape != grid.shape:
        raise ValueError("the uniform-control profile has the wrong shape")
    uniform_rmsd = float(np.sqrt(np.mean((uniform_reconstructed - reference) ** 2)))

    return float(uniform_rmsd - adaptive_rmsd)
SCICODE_GOLD_EOF
