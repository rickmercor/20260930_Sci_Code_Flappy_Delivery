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
def solve_kernel_range(rest_distance: float) -> float:
    """Reference implementation (bisection on the non-trivial stationarity root)."""
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    if not _is_number(rest_distance):
        raise ValueError("rest_distance must be a finite real number")
    distance = float(rest_distance)
    # U'(r) = 0 reads beta * exp(-r) = exp(-r / beta). With x = r / beta this is
    # x - ln(x) = r - ln(r): one root is x = r (beta = 1, U identically zero),
    # the other lies on the far side of x = 1. U''(r) has the sign of
    # 1 - 1 / beta there, so a strict minimum needs beta > 1, i.e. x < 1,
    # which exists only when r > 1.
    if distance <= 1.0:
        raise ValueError("no kernel range gives a strict minimum at this distance")
    level = distance - np.log(distance)

    def _excess(x):
        return x - np.log(x) - level

    low = 0.5 * np.exp(-level)
    high = 1.0
    # _excess is strictly decreasing on (0, 1), positive at low, negative at 1.
    for _ in range(400):
        middle = 0.5 * (low + high)
        if middle <= low or middle >= high:
            break
        if _excess(middle) > 0.0:
            low = middle
        else:
            high = middle
    root = 0.5 * (low + high)
    for _ in range(3):
        slope = 1.0 - 1.0 / root
        if slope == 0.0:
            break
        step = _excess(root) / slope
        if not np.isfinite(step) or not (0.0 < root - step < 1.0):
            break
        root -= step
    beta = distance / root
    if not (np.isfinite(beta) and beta > 1.0):
        raise ValueError("no kernel range gives a strict minimum at this distance")
    return float(beta)

import numpy as np
def compute_adhesion_velocities(
    positions: np.ndarray,
    angles: np.ndarray,
    magnitudes: np.ndarray,
    kernel_range: float,
    cutoff: float,
    tau_v: float,
) -> np.ndarray:
    """Reference implementation (vectorised analytic gradient)."""
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    points = np.asarray(positions, dtype=float)
    theta = np.asarray(angles, dtype=float)
    strength = np.asarray(magnitudes, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] == 0:
        raise ValueError("positions must have shape (n, 2) with n >= 1")
    count = points.shape[0]
    if theta.shape != (count,) or strength.shape != (count,):
        raise ValueError("angles and magnitudes must have shape (n,)")
    if not (np.all(np.isfinite(points)) and np.all(np.isfinite(theta))
            and np.all(np.isfinite(strength))):
        raise ValueError("positions, angles and magnitudes must be finite")
    if np.any(strength < 0.0):
        raise ValueError("magnitudes must be nonnegative")
    if not (_is_number(kernel_range) and kernel_range > 0.0):
        raise ValueError("kernel_range must be a finite positive number")
    if not (_is_number(tau_v) and tau_v > 0.0):
        raise ValueError("tau_v must be a finite positive number")
    if not ((cutoff == np.inf or _is_number(cutoff)) and cutoff > 0.0):
        raise ValueError("cutoff must be a positive number or np.inf")

    beta = float(kernel_range)
    offset = points[None, :, :] - points[:, None, :]          # r_j - r_i
    distance = np.sqrt(np.sum(offset * offset, axis=2))
    distinct = ~np.eye(count, dtype=bool)
    if np.any(distance[distinct] == 0.0):
        raise ValueError("two cells occupy the same position")
    partner = distinct & (distance < cutoff)
    safe = np.where(partner, distance, 1.0)
    unit = offset / safe[:, :, None]
    # p_hat x e = normal . e with normal = (-sin, cos) of the polarity angle.
    normal = np.stack([-np.sin(theta), np.cos(theta)], axis=1)
    cross_i = strength[:, None] * np.einsum("ik,ijk->ij", normal, unit)
    cross_j = strength[None, :] * np.einsum("jk,ijk->ij", normal, unit)
    factor = cross_i * cross_j + 1.0
    kernel = np.exp(-safe) - np.exp(-safe / beta)
    slope = -np.exp(-safe) + np.exp(-safe / beta) / beta
    # d e / d r_i = -(I - e e^T) / r, hence d(m n . e) / d r_i below.
    d_cross_i = -(strength[:, None, None] * normal[:, None, :]
                  - cross_i[:, :, None] * unit) / safe[:, :, None]
    d_cross_j = -(strength[None, :, None] * normal[None, :, :]
                  - cross_j[:, :, None] * unit) / safe[:, :, None]
    d_factor = d_cross_i * cross_j[:, :, None] + cross_i[:, :, None] * d_cross_j
    gradient = -(factor * slope)[:, :, None] * unit + kernel[:, :, None] * d_factor
    gradient = np.where(partner[:, :, None], gradient, 0.0)
    return -float(tau_v) * gradient.sum(axis=1)

import numpy as np
def compute_polarity_rotation_rates(
    positions: np.ndarray,
    angles: np.ndarray,
    magnitudes: np.ndarray,
    regulation_times: np.ndarray,
    kernel_range: float,
    cutoff: float,
    tau_v: float,
) -> np.ndarray:
    """Reference implementation (vectorised analytic angular derivatives)."""
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    points = np.asarray(positions, dtype=float)
    theta = np.asarray(angles, dtype=float)
    strength = np.asarray(magnitudes, dtype=float)
    regulation = np.asarray(regulation_times, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] == 0:
        raise ValueError("positions must have shape (n, 2) with n >= 1")
    count = points.shape[0]
    for values in (theta, strength, regulation):
        if values.shape != (count,):
            raise ValueError("angles, magnitudes and regulation_times must have shape (n,)")
    for values in (points, theta, strength, regulation):
        if not np.all(np.isfinite(values)):
            raise ValueError("inputs must be finite")
    if np.any(strength < 0.0) or np.any(regulation < 0.0):
        raise ValueError("magnitudes and regulation_times must be nonnegative")
    if not (_is_number(kernel_range) and kernel_range > 0.0):
        raise ValueError("kernel_range must be a finite positive number")
    if not (_is_number(tau_v) and tau_v > 0.0):
        raise ValueError("tau_v must be a finite positive number")
    if not ((cutoff == np.inf or _is_number(cutoff)) and cutoff > 0.0):
        raise ValueError("cutoff must be a positive number or np.inf")

    beta = float(kernel_range)
    offset = points[None, :, :] - points[:, None, :]          # r_j - r_i
    distance = np.sqrt(np.sum(offset * offset, axis=2))
    distinct = ~np.eye(count, dtype=bool)
    if np.any(distance[distinct] == 0.0):
        raise ValueError("two cells occupy the same position")
    partner = distinct & (distance < cutoff)
    safe = np.where(partner, distance, 1.0)
    unit = offset / safe[:, :, None]
    polar = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    normal = np.stack([-np.sin(theta), np.cos(theta)], axis=1)
    along_i = np.einsum("ik,ijk->ij", polar, unit)            # u_i . e_ij
    across_i = np.einsum("ik,ijk->ij", normal, unit)          # u_i x e_ij
    cross_j = strength[None, :] * np.einsum("jk,ijk->ij", normal, unit)
    kernel = np.exp(-safe) - np.exp(-safe / beta)
    # d(u_i x e)/d theta_i = -(u_i . e) and d(u_i . e)/d theta_i = u_i x e.
    d_factor = -strength[:, None] * along_i * cross_j
    torque = np.where(partner, kernel * d_factor, 0.0).sum(axis=1)
    alignment = np.where(partner, across_i, 0.0).sum(axis=1)
    return -float(tau_v) * torque - regulation * alignment

import numpy as np
def solve_stacked_triad(
    magnitude: float,
    kernel_range: float,
    cutoff: float,
    tau_v: float,
    velocity_fn: "Callable[..., np.ndarray]",
) -> np.ndarray:
    """Reference implementation (continuation in magnitude with Newton steps)."""
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    def _is_function(value):
        return callable(value)

    if not (_is_number(magnitude) and magnitude >= 0.0):
        raise ValueError("magnitude must be a finite nonnegative number")
    if not (_is_number(kernel_range) and kernel_range > 1.0):
        raise ValueError("kernel_range must be a finite number above 1")
    if not (_is_number(tau_v) and tau_v > 0.0):
        raise ValueError("tau_v must be a finite positive number")
    if not _is_function(velocity_fn):
        raise ValueError("velocity_fn must be callable")
    beta = float(kernel_range)
    rest = beta * np.log(beta) / (beta - 1.0)
    if not (_is_number(cutoff) and cutoff > rest):
        raise ValueError("cutoff must be a finite number above the rest distance")
    angles = np.full(3, 0.5 * np.pi)

    def _layout(state):
        base, height = state
        return np.array([[-0.5 * base, 0.0], [0.5 * base, 0.0], [0.0, height]])

    def _velocities(state, strength, reach):
        out = velocity_fn(_layout(state), angles, np.full(3, strength), beta, reach, tau_v)
        out = np.asarray(out, dtype=float)
        if out.shape != (3, 2) or not np.all(np.isfinite(out)):
            raise ValueError("velocity_fn must return a finite (3, 2) array")
        return out

    def _residual(state, strength):
        # Rates of the base length and of the apex height above the base
        # midpoint on the smooth branch (every pair interacting).
        v = _velocities(state, strength, np.inf)
        return np.array([v[1, 0] - v[0, 0], v[2, 1] - 0.5 * (v[0, 1] + v[1, 1])])

    def _jacobian(state, strength):
        matrix = np.empty((2, 2))
        for k in range(2):
            step = np.zeros(2)
            step[k] = 1e-6 * max(1.0, abs(state[k]))
            matrix[:, k] = (_residual(state + step, strength)
                            - _residual(state - step, strength)) / (2.0 * step[k])
        return matrix

    def _newton(start, strength):
        # Stay on the stacked branch: the apex keeps a clear height and no
        # continuation step may jump to a distant (for example collinear) root.
        state = start.copy()
        for _ in range(60):
            try:
                delta = np.linalg.solve(_jacobian(state, strength), -_residual(state, strength))
            except np.linalg.LinAlgError:
                return None
            state = state + delta
            if not (np.all(np.isfinite(state)) and state[0] > 0.0 and state[1] > 0.1 * rest):
                return None
            if np.max(np.abs(state - start)) > 0.5 * rest:
                return None
            if np.max(np.abs(delta)) <= 1e-14 * (1.0 + np.max(np.abs(state))):
                return state
        return None

    state = np.array([rest, 0.5 * np.sqrt(3.0) * rest])
    target = float(magnitude)
    for k in range(1, 25):
        state = _newton(state, target * np.sqrt(k / 24.0))
        if state is None:
            raise ValueError("no stacked equilibrium at this magnitude")
    base = float(state[0])
    side = float(np.hypot(0.5 * state[0], state[1]))
    if not (base < cutoff and side < cutoff):
        raise ValueError("no stacked equilibrium with every pair interacting")
    matrix = _jacobian(state, target)
    if not (np.trace(matrix) < 0.0 and np.linalg.det(matrix) > 0.0):
        raise ValueError("the stacked equilibrium is not stable")
    check = _velocities(state, target, float(cutoff))
    if np.max(np.abs(check)) > 1e-9 * max(1.0, float(tau_v)):
        raise ValueError("the stacked configuration is not at rest under the cutoff")
    return np.array([base, side])

import numpy as np
def locate_monolayer_threshold(
    triad_fn: "Callable[[float], np.ndarray]",
    magnitude_bracket: tuple = (0.0, 1.0),
    tolerance: float = 1e-12,
) -> float:
    """Reference implementation (bisection on the existence of the equilibrium)."""
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    def _is_function(value):
        return callable(value)

    if not _is_function(triad_fn):
        raise ValueError("triad_fn must be callable")
    try:
        low, high = magnitude_bracket
    except (TypeError, ValueError):
        raise ValueError("magnitude_bracket must hold two numbers") from None
    if not (_is_number(low) and _is_number(high) and 0.0 <= low < high):
        raise ValueError("magnitude_bracket must satisfy 0 <= low < high")
    if not (_is_number(tolerance) and tolerance > 0.0):
        raise ValueError("tolerance must be a finite positive number")
    low, high = float(low), float(high)

    def _exists(strength):
        try:
            separations = triad_fn(strength)
        except ValueError:
            return False
        separations = np.asarray(separations, dtype=float)
        if separations.shape != (2,) or not np.all(np.isfinite(separations)):
            raise ValueError("triad_fn must return two finite separations")
        return True

    if not _exists(low):
        raise ValueError("the stacked equilibrium must exist at the lower bracket end")
    if _exists(high):
        raise ValueError("the stacked equilibrium must not exist at the upper bracket end")
    while high - low > tolerance:
        middle = 0.5 * (low + high)
        if middle <= low or middle >= high:
            break
        if _exists(middle):
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)

import numpy as np
def compute_pair_splay_angle(
    magnitude: float,
    regulation_time: float,
    kernel_range: float,
    cutoff: float,
    tau_v: float,
    rotation_fn: "Callable[..., np.ndarray]",
) -> float:
    """Reference implementation (bisection on the rotation rate of cell 1)."""
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    def _is_function(value):
        return callable(value)

    for value in (magnitude, regulation_time, tau_v):
        if not (_is_number(value) and value > 0.0):
            raise ValueError("magnitude, regulation_time and tau_v must be finite and positive")
    if not (_is_number(kernel_range) and kernel_range > 1.0):
        raise ValueError("kernel_range must be a finite number above 1")
    if not _is_function(rotation_fn):
        raise ValueError("rotation_fn must be callable")
    beta = float(kernel_range)
    rest = beta * np.log(beta) / (beta - 1.0)
    if not ((cutoff == np.inf or _is_number(cutoff)) and cutoff > rest):
        raise ValueError("cutoff must exceed the rest separation of the pair")
    positions = np.array([[0.0, 0.0], [rest, 0.0]])
    strengths = np.full(2, float(magnitude))
    regulation = np.full(2, float(regulation_time))

    def _rate(psi):
        angles = np.array([psi, np.pi - psi])
        out = rotation_fn(positions, angles, strengths, regulation, beta, cutoff, tau_v)
        out = np.asarray(out, dtype=float)
        if out.shape != (2,) or not np.all(np.isfinite(out)):
            raise ValueError("rotation_fn must return two finite rates")
        return float(out[0])

    # The rate of cell 1 is positive just past pi / 2 and must turn negative
    # before pi for a stable splayed rest angle to exist.
    low, high = 0.5 * np.pi, np.pi - 1e-9
    if not (_rate(low) > 0.0 and _rate(high) < 0.0):
        raise ValueError("the pair has no stable splayed rest angle")
    for _ in range(200):
        middle = 0.5 * (low + high)
        if middle <= low or middle >= high:
            break
        if _rate(middle) > 0.0:
            low = middle
        else:
            high = middle
    return float(0.5 * (low + high))

import numpy as np
def solve_wraparound_boundary(
    magnitude: float,
    kernel_range: float,
    cutoff: float,
    splay_fn: "Callable[[float, float], float]",
    regulation_bracket: tuple = (1e-6, 10.0),
    tolerance: float = 1e-12,
) -> float:
    """Reference implementation (bisection on the arc end separation)."""
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    def _is_function(value):
        return callable(value)

    if not (_is_number(magnitude) and magnitude > 0.0):
        raise ValueError("magnitude must be a finite positive number")
    if not (_is_number(kernel_range) and kernel_range > 1.0):
        raise ValueError("kernel_range must be a finite number above 1")
    if not _is_function(splay_fn):
        raise ValueError("splay_fn must be callable")
    beta = float(kernel_range)
    rest = beta * np.log(beta) / (beta - 1.0)
    if not (_is_number(cutoff) and cutoff > rest):
        raise ValueError("cutoff must be a finite number above the rest separation")
    try:
        low, high = regulation_bracket
    except (TypeError, ValueError):
        raise ValueError("regulation_bracket must hold two numbers") from None
    if not (_is_number(low) and _is_number(high) and 0.0 < low < high):
        raise ValueError("regulation_bracket must satisfy 0 < low < high")
    if not (_is_number(tolerance) and tolerance > 0.0):
        raise ValueError("tolerance must be a finite positive number")
    low, high = float(low), float(high)

    def _ends_apart(regulation):
        try:
            psi = splay_fn(float(magnitude), regulation)
        except ValueError:
            return False
        if not (_is_number(psi) and 0.5 * np.pi <= psi < np.pi):
            raise ValueError("splay_fn must return an angle in [pi / 2, pi)")
        # Cell 2's polarity makes pi - psi with the bond from cell 1 and psi
        # with the bond to cell 3, so the bonds turn by 2 psi - pi and the
        # end cells are 2 * rest * sin(psi) apart.
        return 2.0 * rest * np.sin(psi) >= cutoff

    if not _ends_apart(low):
        raise ValueError("the end cells already interact at the lower bracket end")
    if _ends_apart(high):
        raise ValueError("the end cells still do not interact at the upper bracket end")
    while high - low > tolerance:
        middle = 0.5 * (low + high)
        if middle <= low or middle >= high:
            break
        if _ends_apart(middle):
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)

import numpy as np
def estimate_inflation_onset(
    rest_distance: float = 2.0,
    cutoff: float = 2.5,
    tau_v: float = 10.0,
    tolerance: float = 1e-12,
) -> float:
    """Reference orchestrator using only the reference function chain."""
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    if not (_is_number(tau_v) and tau_v > 0.0):
        raise ValueError("tau_v must be a finite positive number")
    if not (_is_number(tolerance) and tolerance > 0.0):
        raise ValueError("tolerance must be a finite positive number")
    beta = solve_kernel_range(rest_distance)

    def _velocity_fn(positions, angles, magnitudes, kernel_range, reach, mobility):
        return compute_adhesion_velocities(
            positions, angles, magnitudes, kernel_range, reach, mobility
        )

    def _rotation_fn(positions, angles, magnitudes, regulation, kernel_range, reach, mobility):
        return compute_polarity_rotation_rates(
            positions, angles, magnitudes, regulation, kernel_range, reach, mobility
        )

    def _triad_fn(magnitude):
        return solve_stacked_triad(magnitude, beta, cutoff, tau_v, _velocity_fn)

    def _splay_fn(magnitude, regulation):
        return compute_pair_splay_angle(
            magnitude, regulation, beta, cutoff, tau_v, _rotation_fn
        )

    threshold = locate_monolayer_threshold(_triad_fn, (0.0, 1.0), tolerance)
    onset = solve_wraparound_boundary(
        threshold, beta, cutoff, _splay_fn, (1e-6, float(tau_v)), tolerance
    )
    if not (np.isfinite(onset) and onset > 0.0):
        raise ValueError("the boundary must be a finite positive regulation constant")
    return float(onset)
SCICODE_GOLD_EOF
