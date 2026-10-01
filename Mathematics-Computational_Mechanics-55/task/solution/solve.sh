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


def _array(value, ndim, name):
    value = np.asarray(value, dtype=float)
    if value.ndim != ndim or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be a finite rank-{ndim} array")
    return value


def _geometry(geometry):
    geometry = _array(geometry, 3, "geometry")
    if geometry.shape[0] != 2 or geometry.shape[1] < 3:
        raise ValueError("geometry must have shape (2, N, N + 8), N >= 3")
    size = geometry.shape[1]
    if geometry.shape[2] != size + 8:
        raise ValueError("geometry must have shape (2, N, N + 8)")
    K = geometry[0, :, :size]
    if not np.allclose(K, K.T, rtol=0, atol=1e-12):
        raise ValueError("stiffness must be symmetric")
    if not np.allclose(K.sum(axis=1), 0, rtol=0, atol=1e-12):
        raise ValueError("stiffness must have zero row sums")
    if np.any(K - np.diag(np.diag(K)) > 1e-12):
        raise ValueError("stiffness off-diagonals must be nonpositive")
    if np.any(geometry[1, :, :size] != 0):
        raise ValueError("the mesh is fixed, so stiffness derivatives must be zero")
    n, dn = geometry[:, :, size : size + 2]
    if not np.allclose(np.sum(n * n, axis=1), 1, rtol=0, atol=1e-12):
        raise ValueError("magnetic directions must have unit length")
    if not np.allclose(np.sum(n * dn, axis=1), 0, rtol=0, atol=1e-12):
        raise ValueError("direction derivatives must be tangent")
    expected = _frames(n, dn)
    if not np.allclose(geometry[:, :, size:], expected, rtol=1e-12, atol=1e-12):
        raise ValueError("the supplied frames and their derivatives are inconsistent")
    return geometry


def _frames(n, dn):
    t = np.column_stack((-n[:, 1], n[:, 0]))
    dt = np.column_stack((-dn[:, 1], dn[:, 0]))
    nu = np.column_stack((n[:, 0] ** 2 - n[:, 1] ** 2, 2 * n[:, 0] * n[:, 1]))
    dnu = np.column_stack(
        (
            2 * n[:, 0] * dn[:, 0] - 2 * n[:, 1] * dn[:, 1],
            2 * (dn[:, 0] * n[:, 1] + n[:, 0] * dn[:, 1]),
        )
    )
    tau = np.column_stack((-nu[:, 1], nu[:, 0]))
    dtau = np.column_stack((-dnu[:, 1], dnu[:, 0]))
    return np.stack(
        (np.column_stack((n, t, nu, tau)), np.column_stack((dn, dt, dnu, dtau)))
    )


def prepare_geometry_jet(
    vertices: np.ndarray,
    triangles: np.ndarray,
    angles: np.ndarray,
    direction: np.ndarray,
) -> np.ndarray:
    vertices = _array(vertices, 2, "vertices")
    triangles = np.asarray(triangles)
    if vertices.shape[0] < 3 or vertices.shape[1] != 2:
        raise ValueError("vertices must have shape (N, 2), N >= 3")
    size = len(vertices)
    if (
        triangles.ndim != 2
        or triangles.shape[0] < 1
        or triangles.shape[1] != 3
        or not np.issubdtype(triangles.dtype, np.integer)
    ):
        raise ValueError("triangles must be a nonempty integer array of shape (M, 3)")
    if (
        np.any(triangles < 0)
        or np.any(triangles >= size)
        or len(np.unique(triangles)) != size
    ):
        raise ValueError("connectivity must be in range and use every vertex")
    angles = _array(angles, 1, "angles")
    direction = _array(direction, 1, "direction")
    if angles.shape != (size,) or direction.shape != (size,):
        raise ValueError("angles and direction must have shape (N,)")
    K = np.zeros((size, size))
    for indices in triangles:
        if len(np.unique(indices)) != 3:
            raise ValueError("triangle indices must be distinct")
        xy = vertices[indices]
        area = abs(np.linalg.det(np.column_stack((xy[1] - xy[0], xy[2] - xy[0])))) / 2
        if area == 0 or not np.isfinite(area):
            raise ValueError("triangle area must be finite and positive")
        gradients = np.linalg.inv(np.column_stack((np.ones(3), xy)))[1:, :]
        K[np.ix_(indices, indices)] += area * gradients.T @ gradients
    n = np.column_stack((np.cos(angles), np.sin(angles)))
    dn = direction[:, None] * np.column_stack((-n[:, 1], n[:, 0]))
    result = np.zeros((2, size, size + 8))
    result[0, :, :size] = K
    result[:, :, size:] = _frames(n, dn)
    return _geometry(result)

import numpy as np


def _array(value, ndim, name):
    value = np.asarray(value, dtype=float)
    if value.ndim != ndim or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be a finite rank-{ndim} array")
    return value


def _positive(value, name):
    if np.ndim(value) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return float(value)


def _geometry(geometry):
    geometry = _array(geometry, 3, "geometry")
    if geometry.shape[0] != 2 or geometry.shape[1] < 3:
        raise ValueError("geometry must have shape (2, N, N + 8), N >= 3")
    size = geometry.shape[1]
    if geometry.shape[2] != size + 8:
        raise ValueError("geometry must have shape (2, N, N + 8)")
    K = geometry[0, :, :size]
    if not np.allclose(K, K.T, rtol=0, atol=1e-12):
        raise ValueError("stiffness must be symmetric")
    if not np.allclose(K.sum(axis=1), 0, rtol=0, atol=1e-12):
        raise ValueError("stiffness must have zero row sums")
    if np.any(K - np.diag(np.diag(K)) > 1e-12):
        raise ValueError("stiffness off-diagonals must be nonpositive")
    if np.any(geometry[1, :, :size] != 0):
        raise ValueError("the mesh is fixed, so stiffness derivatives must be zero")
    n, dn = geometry[:, :, size : size + 2]
    if not np.allclose(np.sum(n * n, axis=1), 1, rtol=0, atol=1e-12):
        raise ValueError("magnetic directions must have unit length")
    if not np.allclose(np.sum(n * dn, axis=1), 0, rtol=0, atol=1e-12):
        raise ValueError("direction derivatives must be tangent")
    expected = _frames(n, dn)
    if not np.allclose(geometry[:, :, size:], expected, rtol=1e-12, atol=1e-12):
        raise ValueError("the supplied frames and their derivatives are inconsistent")
    return geometry


def _frames(n, dn):
    t = np.column_stack((-n[:, 1], n[:, 0]))
    dt = np.column_stack((-dn[:, 1], dn[:, 0]))
    nu = np.column_stack((n[:, 0] ** 2 - n[:, 1] ** 2, 2 * n[:, 0] * n[:, 1]))
    dnu = np.column_stack(
        (
            2 * n[:, 0] * dn[:, 0] - 2 * n[:, 1] * dn[:, 1],
            2 * (dn[:, 0] * n[:, 1] + n[:, 0] * dn[:, 1]),
        )
    )
    tau = np.column_stack((-nu[:, 1], nu[:, 0]))
    dtau = np.column_stack((-dnu[:, 1], dnu[:, 0]))
    return np.stack(
        (np.column_stack((n, t, nu, tau)), np.column_stack((dn, dt, dnu, dtau)))
    )


def build_quadratic_jet(
    geometry: np.ndarray, Qc: float, Mc: float
) -> np.ndarray:
    geometry = _geometry(geometry)
    Qc, Mc = _positive(Qc, "Qc"), _positive(Mc, "Mc")
    size = geometry.shape[1]
    K = geometry[0, :, :size]
    result = np.empty((2, 2, size, size + 1))
    for index, (start, scale) in enumerate(((size, Mc), (size + 4, Qc))):
        x, dx = geometry[:, :, start : start + 2]
        t, dt = geometry[:, :, start + 2 : start + 4]
        c = 2 * scale * scale
        result[0, index, :, :size] = c * K * (t @ t.T)
        result[1, index, :, :size] = c * K * (dt @ t.T + t @ dt.T)
        result[0, index, :, size] = c * np.sum(K * (t @ x.T), axis=1)
        result[1, index, :, size] = c * np.sum(K * (dt @ x.T + t @ dx.T), axis=1)
    if not np.all(np.isfinite(result)):
        raise ValueError("quadratic jet is nonfinite")
    return result

import numpy as np


def _array(value, ndim, name):
    value = np.asarray(value, dtype=float)
    if value.ndim != ndim or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be a finite rank-{ndim} array")
    return value


def _positive(value, name):
    if np.ndim(value) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return float(value)


def _mask(boundary, size):
    boundary = np.asarray(boundary)
    if boundary.dtype != bool or boundary.shape != (size,):
        raise ValueError("boundary must be a Boolean vector of length N")
    return boundary


def _systems(systems):
    systems = _array(systems, 4, "systems")
    if systems.shape[:2] != (2, 2) or systems.shape[2] < 1:
        raise ValueError("systems must have shape (2, 2, N, N + 1)")
    size = systems.shape[2]
    if systems.shape[3] != size + 1:
        raise ValueError("systems must have shape (2, 2, N, N + 1)")
    for layer in range(2):
        for block in systems[layer, :, :, :size]:
            if not np.allclose(block, block.T, rtol=0, atol=1e-12):
                raise ValueError("Hessians and their derivatives must be symmetric")
            if layer == 0 and np.linalg.eigvalsh(block)[0] < -1e-12:
                raise ValueError("base Hessians must be positive semidefinite")
    return systems


def _state(state, size):
    state = _array(state, 3, "state")
    if state.shape != (2, 3, size):
        raise ValueError("state must have shape (2, 3, N)")
    if np.any(np.abs(state[0, 0]) >= 1):
        raise ValueError("base magnetic coefficients must lie in (-1, 1)")
    return state


def _phi(r):
    return 2 * r / (1 - r * r)


def _dphi(r):
    return 2 * (1 + r * r) / (1 - r * r) ** 2


def _ddphi(r):
    return 4 * r * (3 + r * r) / (1 - r * r) ** 3


def _linear_jet(A, dA, b, db):
    try:
        q, upper = np.linalg.qr(A)
        x = np.linalg.solve(upper, q.T @ b)
        dx = np.linalg.solve(upper, q.T @ (db - dA @ x))
    except np.linalg.LinAlgError as exc:
        raise ValueError("linear system is singular") from exc
    result = np.stack((x, dx))
    if not np.all(np.isfinite(result)):
        raise ValueError("linear solve produced a nonfinite jet")
    return result


def solve_magnetic_jet(
    systems: np.ndarray, boundary: np.ndarray, state: np.ndarray, zeta: float
) -> np.ndarray:
    systems = _systems(systems)
    size = systems.shape[2]
    boundary = _mask(boundary, size)
    state = _state(state, size)
    if np.any(state[:, 0, boundary] != 0):
        raise ValueError("both magnetic boundary layers must be zero")
    zeta = _positive(zeta, "zeta")
    r, p, lam = state[0]
    dr, dp, dlam = state[1]
    a = _dphi(r)
    da = _ddphi(r) * dr
    b = _phi(r) - a * r
    db = -da * r
    A = systems[0, 0, :, :size] + zeta * np.diag(a * a)
    dA = systems[1, 0, :, :size] + 2 * zeta * np.diag(a * da)
    rhs = -systems[0, 0, :, size] + a * lam + zeta * a * (p - b)
    drhs = -systems[1, 0, :, size] + da * lam + a * dlam
    drhs += zeta * (da * (p - b) + a * (dp - db))
    free = np.flatnonzero(~boundary)
    result = np.zeros((2, size))
    if free.size:
        result[:, free] = _linear_jet(
            A[np.ix_(free, free)], dA[np.ix_(free, free)], rhs[free], drhs[free]
        )
    if np.any(np.abs(result[0]) >= 1):
        raise ValueError("magnetic iterate left (-1, 1)")
    return result

import numpy as np


def _array(value, ndim, name):
    value = np.asarray(value, dtype=float)
    if value.ndim != ndim or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be a finite rank-{ndim} array")
    return value


def _positive(value, name):
    if np.ndim(value) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return float(value)


def _systems(systems):
    systems = _array(systems, 4, "systems")
    if systems.shape[:2] != (2, 2) or systems.shape[2] < 1:
        raise ValueError("systems must have shape (2, 2, N, N + 1)")
    size = systems.shape[2]
    if systems.shape[3] != size + 1:
        raise ValueError("systems must have shape (2, 2, N, N + 1)")
    for layer in range(2):
        for block in systems[layer, :, :, :size]:
            if not np.allclose(block, block.T, rtol=0, atol=1e-12):
                raise ValueError("Hessians and their derivatives must be symmetric")
            if layer == 0 and np.linalg.eigvalsh(block)[0] < -1e-12:
                raise ValueError("base Hessians must be positive semidefinite")
    return systems


def _phi(r):
    return 2 * r / (1 - r * r)


def _dphi(r):
    return 2 * (1 + r * r) / (1 - r * r) ** 2


def _linear_jet(A, dA, b, db):
    try:
        q, upper = np.linalg.qr(A)
        x = np.linalg.solve(upper, q.T @ b)
        dx = np.linalg.solve(upper, q.T @ (db - dA @ x))
    except np.linalg.LinAlgError as exc:
        raise ValueError("linear system is singular") from exc
    result = np.stack((x, dx))
    if not np.all(np.isfinite(result)):
        raise ValueError("linear solve produced a nonfinite jet")
    return result


def solve_auxiliary_jet(
    systems: np.ndarray,
    r_jet: np.ndarray,
    multiplier_jet: np.ndarray,
    zeta: float,
    rho: float,
) -> np.ndarray:
    systems = _systems(systems)
    size = systems.shape[2]
    r_jet = _array(r_jet, 2, "r_jet")
    multiplier_jet = _array(multiplier_jet, 2, "multiplier_jet")
    if (
        r_jet.shape != (2, size)
        or multiplier_jet.shape != (2, size)
        or np.any(np.abs(r_jet[0]) >= 1)
    ):
        raise ValueError("vector jets must have shape (2, N) and base r in (-1, 1)")
    zeta, rho = _positive(zeta, "zeta"), _positive(rho, "rho")
    r, dr = r_jet
    lam, dlam = multiplier_jet
    f, df = _phi(r), _dphi(r) * dr
    B = systems[0, 1, :, :size] + zeta * np.eye(size)
    dB = systems[1, 1, :, :size]
    rhs = -systems[0, 1, :, size] - lam + zeta * f
    drhs = -systems[1, 1, :, size] - dlam + zeta * df
    p, dp = _linear_jet(B, dB, rhs, drhs)
    updated = np.stack((p, lam + rho * (p - f)))
    variation = np.stack((dp, dlam + rho * (dp - df)))
    return np.stack((updated, variation))

import numpy as np


def _array(value, ndim, name):
    value = np.asarray(value, dtype=float)
    if value.ndim != ndim or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be a finite rank-{ndim} array")
    return value


def _positive(value, name):
    if np.ndim(value) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return float(value)


def _count(value, name):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < 1
    ):
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _mask(boundary, size):
    boundary = np.asarray(boundary)
    if boundary.dtype != bool or boundary.shape != (size,):
        raise ValueError("boundary must be a Boolean vector of length N")
    return boundary


def _systems(systems):
    systems = _array(systems, 4, "systems")
    if systems.shape[:2] != (2, 2) or systems.shape[2] < 1:
        raise ValueError("systems must have shape (2, 2, N, N + 1)")
    size = systems.shape[2]
    if systems.shape[3] != size + 1:
        raise ValueError("systems must have shape (2, 2, N, N + 1)")
    for layer in range(2):
        for block in systems[layer, :, :, :size]:
            if not np.allclose(block, block.T, rtol=0, atol=1e-12):
                raise ValueError("Hessians and their derivatives must be symmetric")
            if layer == 0 and np.linalg.eigvalsh(block)[0] < -1e-12:
                raise ValueError("base Hessians must be positive semidefinite")
    return systems


def _state(state, size):
    state = _array(state, 3, "state")
    if state.shape != (2, 3, size):
        raise ValueError("state must have shape (2, 3, N)")
    if np.any(np.abs(state[0, 0]) >= 1):
        raise ValueError("base magnetic coefficients must lie in (-1, 1)")
    return state


def _phi(r):
    return 2 * r / (1 - r * r)


def _dphi(r):
    return 2 * (1 + r * r) / (1 - r * r) ** 2


def _ddphi(r):
    return 4 * r * (3 + r * r) / (1 - r * r) ** 3


def _linear_jet(A, dA, b, db):
    try:
        q, upper = np.linalg.qr(A)
        x = np.linalg.solve(upper, q.T @ b)
        dx = np.linalg.solve(upper, q.T @ (db - dA @ x))
    except np.linalg.LinAlgError as exc:
        raise ValueError("linear system is singular") from exc
    result = np.stack((x, dx))
    if not np.all(np.isfinite(result)):
        raise ValueError("linear solve produced a nonfinite jet")
    return result


def _solve_magnetic_jet(
    systems: np.ndarray, boundary: np.ndarray, state: np.ndarray, zeta: float
) -> np.ndarray:
    systems = _systems(systems)
    size = systems.shape[2]
    boundary = _mask(boundary, size)
    state = _state(state, size)
    if np.any(state[:, 0, boundary] != 0):
        raise ValueError("both magnetic boundary layers must be zero")
    zeta = _positive(zeta, "zeta")
    r, p, lam = state[0]
    dr, dp, dlam = state[1]
    a = _dphi(r)
    da = _ddphi(r) * dr
    b = _phi(r) - a * r
    db = -da * r
    A = systems[0, 0, :, :size] + zeta * np.diag(a * a)
    dA = systems[1, 0, :, :size] + 2 * zeta * np.diag(a * da)
    rhs = -systems[0, 0, :, size] + a * lam + zeta * a * (p - b)
    drhs = -systems[1, 0, :, size] + da * lam + a * dlam
    drhs += zeta * (da * (p - b) + a * (dp - db))
    free = np.flatnonzero(~boundary)
    result = np.zeros((2, size))
    if free.size:
        result[:, free] = _linear_jet(
            A[np.ix_(free, free)], dA[np.ix_(free, free)], rhs[free], drhs[free]
        )
    if np.any(np.abs(result[0]) >= 1):
        raise ValueError("magnetic iterate left (-1, 1)")
    return result


def _solve_auxiliary_jet(
    systems: np.ndarray,
    r_jet: np.ndarray,
    multiplier_jet: np.ndarray,
    zeta: float,
    rho: float,
) -> np.ndarray:
    systems = _systems(systems)
    size = systems.shape[2]
    r_jet = _array(r_jet, 2, "r_jet")
    multiplier_jet = _array(multiplier_jet, 2, "multiplier_jet")
    if (
        r_jet.shape != (2, size)
        or multiplier_jet.shape != (2, size)
        or np.any(np.abs(r_jet[0]) >= 1)
    ):
        raise ValueError("vector jets must have shape (2, N) and base r in (-1, 1)")
    zeta, rho = _positive(zeta, "zeta"), _positive(rho, "rho")
    r, dr = r_jet
    lam, dlam = multiplier_jet
    f, df = _phi(r), _dphi(r) * dr
    B = systems[0, 1, :, :size] + zeta * np.eye(size)
    dB = systems[1, 1, :, :size]
    rhs = -systems[0, 1, :, size] - lam + zeta * f
    drhs = -systems[1, 1, :, size] - dlam + zeta * df
    p, dp = _linear_jet(B, dB, rhs, drhs)
    updated = np.stack((p, lam + rho * (p - f)))
    variation = np.stack((dp, dlam + rho * (dp - df)))
    return np.stack((updated, variation))


def relax_tangent_jet(
    systems: np.ndarray,
    boundary: np.ndarray,
    state: np.ndarray,
    zeta: float = 4.0,
    rho: float = 1.0,
    tolerance: float = 1e-10,
    max_iterations: int = 20000,
) -> np.ndarray:
    systems = _systems(systems)
    size = systems.shape[2]
    boundary = _mask(boundary, size)
    state = _state(state, size).copy()
    if np.any(state[:, 0, boundary] != 0):
        raise ValueError("both magnetic boundary layers must be zero")
    zeta, rho = _positive(zeta, "zeta"), _positive(rho, "rho")
    tolerance = _positive(tolerance, "tolerance")
    max_iterations = _count(max_iterations, "max_iterations")
    for _ in range(max_iterations):
        r = _solve_magnetic_jet(systems, boundary, state, zeta)
        auxiliary = _solve_auxiliary_jet(systems, r, state[:, 2], zeta, rho)
        state = np.concatenate((r[:, None, :], auxiliary), axis=1)
        residual = np.sqrt(np.mean((state[0, 1] - _phi(state[0, 0])) ** 2))
        if residual <= tolerance:
            return state
    raise RuntimeError("inner cycle cap reached before compatibility tolerance")

import numpy as np


def _array(value, ndim, name):
    value = np.asarray(value, dtype=float)
    if value.ndim != ndim or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be a finite rank-{ndim} array")
    return value


def _geometry(geometry):
    geometry = _array(geometry, 3, "geometry")
    if geometry.shape[0] != 2 or geometry.shape[1] < 3:
        raise ValueError("geometry must have shape (2, N, N + 8), N >= 3")
    size = geometry.shape[1]
    if geometry.shape[2] != size + 8:
        raise ValueError("geometry must have shape (2, N, N + 8)")
    K = geometry[0, :, :size]
    if not np.allclose(K, K.T, rtol=0, atol=1e-12):
        raise ValueError("stiffness must be symmetric")
    if not np.allclose(K.sum(axis=1), 0, rtol=0, atol=1e-12):
        raise ValueError("stiffness must have zero row sums")
    if np.any(K - np.diag(np.diag(K)) > 1e-12):
        raise ValueError("stiffness off-diagonals must be nonpositive")
    if np.any(geometry[1, :, :size] != 0):
        raise ValueError("the mesh is fixed, so stiffness derivatives must be zero")
    n, dn = geometry[:, :, size : size + 2]
    if not np.allclose(np.sum(n * n, axis=1), 1, rtol=0, atol=1e-12):
        raise ValueError("magnetic directions must have unit length")
    if not np.allclose(np.sum(n * dn, axis=1), 0, rtol=0, atol=1e-12):
        raise ValueError("direction derivatives must be tangent")
    expected = _frames(n, dn)
    if not np.allclose(geometry[:, :, size:], expected, rtol=1e-12, atol=1e-12):
        raise ValueError("the supplied frames and their derivatives are inconsistent")
    return geometry


def _frames(n, dn):
    t = np.column_stack((-n[:, 1], n[:, 0]))
    dt = np.column_stack((-dn[:, 1], dn[:, 0]))
    nu = np.column_stack((n[:, 0] ** 2 - n[:, 1] ** 2, 2 * n[:, 0] * n[:, 1]))
    dnu = np.column_stack(
        (
            2 * n[:, 0] * dn[:, 0] - 2 * n[:, 1] * dn[:, 1],
            2 * (dn[:, 0] * n[:, 1] + n[:, 0] * dn[:, 1]),
        )
    )
    tau = np.column_stack((-nu[:, 1], nu[:, 0]))
    dtau = np.column_stack((-dnu[:, 1], dnu[:, 0]))
    return np.stack(
        (np.column_stack((n, t, nu, tau)), np.column_stack((dn, dt, dnu, dtau)))
    )


def project_geometry_jet(geometry: np.ndarray, r_jet: np.ndarray) -> np.ndarray:
    geometry = _geometry(geometry)
    size = geometry.shape[1]
    r_jet = _array(r_jet, 2, "r_jet")
    if r_jet.shape != (2, size) or np.any(np.abs(r_jet[0]) >= 1):
        raise ValueError("r_jet must have shape (2, N) and base r in (-1, 1)")
    n, dn = geometry[:, :, size : size + 2]
    t, dt = geometry[:, :, size + 2 : size + 4]
    r, dr = r_jet
    x = n + r[:, None] * t
    dx = dn + dr[:, None] * t + r[:, None] * dt
    lengths = np.linalg.norm(x, axis=1)
    projected = x / lengths[:, None]
    dprojected = (dx - projected * np.sum(projected * dx, axis=1)[:, None]) / lengths[
        :, None
    ]
    unchanged = (r == 0) & (dr == 0)
    projected[unchanged] = n[unchanged]
    dprojected[unchanged] = dn[unchanged]
    result = geometry.copy()
    result[:, :, size:] = _frames(projected, dprojected)
    return result

import numpy as np


def _array(value, ndim, name):
    value = np.asarray(value, dtype=float)
    if value.ndim != ndim or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be a finite rank-{ndim} array")
    return value


def _positive(value, name):
    if np.ndim(value) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return float(value)


def _geometry(geometry):
    geometry = _array(geometry, 3, "geometry")
    if geometry.shape[0] != 2 or geometry.shape[1] < 3:
        raise ValueError("geometry must have shape (2, N, N + 8), N >= 3")
    size = geometry.shape[1]
    if geometry.shape[2] != size + 8:
        raise ValueError("geometry must have shape (2, N, N + 8)")
    K = geometry[0, :, :size]
    if not np.allclose(K, K.T, rtol=0, atol=1e-12):
        raise ValueError("stiffness must be symmetric")
    if not np.allclose(K.sum(axis=1), 0, rtol=0, atol=1e-12):
        raise ValueError("stiffness must have zero row sums")
    if np.any(K - np.diag(np.diag(K)) > 1e-12):
        raise ValueError("stiffness off-diagonals must be nonpositive")
    if np.any(geometry[1, :, :size] != 0):
        raise ValueError("the mesh is fixed, so stiffness derivatives must be zero")
    n, dn = geometry[:, :, size : size + 2]
    if not np.allclose(np.sum(n * n, axis=1), 1, rtol=0, atol=1e-12):
        raise ValueError("magnetic directions must have unit length")
    if not np.allclose(np.sum(n * dn, axis=1), 0, rtol=0, atol=1e-12):
        raise ValueError("direction derivatives must be tangent")
    expected = _frames(n, dn)
    if not np.allclose(geometry[:, :, size:], expected, rtol=1e-12, atol=1e-12):
        raise ValueError("the supplied frames and their derivatives are inconsistent")
    return geometry


def _frames(n, dn):
    t = np.column_stack((-n[:, 1], n[:, 0]))
    dt = np.column_stack((-dn[:, 1], dn[:, 0]))
    nu = np.column_stack((n[:, 0] ** 2 - n[:, 1] ** 2, 2 * n[:, 0] * n[:, 1]))
    dnu = np.column_stack(
        (
            2 * n[:, 0] * dn[:, 0] - 2 * n[:, 1] * dn[:, 1],
            2 * (dn[:, 0] * n[:, 1] + n[:, 0] * dn[:, 1]),
        )
    )
    tau = np.column_stack((-nu[:, 1], nu[:, 0]))
    dtau = np.column_stack((-dnu[:, 1], dnu[:, 0]))
    return np.stack(
        (np.column_stack((n, t, nu, tau)), np.column_stack((dn, dt, dnu, dtau)))
    )


def measure_energy_jet(
    geometry: np.ndarray, Qc: float, Mc: float
) -> np.ndarray:
    geometry = _geometry(geometry)
    size = geometry.shape[1]
    Qc, Mc = _positive(Qc, "Qc"), _positive(Mc, "Mc")
    K = geometry[0, :, :size]
    n, dn = geometry[:, :, size : size + 2]
    nu, dnu = geometry[:, :, size + 4 : size + 6]
    energy = 0.5 * np.sum(K * (Qc * Qc * (nu @ nu.T) + Mc * Mc * (n @ n.T)))
    variation = np.sum(K * (Qc * Qc * (dnu @ nu.T) + Mc * Mc * (dn @ n.T)))
    result = np.array([energy, variation])
    if not np.all(np.isfinite(result)):
        raise ValueError("energy jet is nonfinite")
    return result

import numpy as np


def _array(value, ndim, name):
    value = np.asarray(value, dtype=float)
    if value.ndim != ndim or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be a finite rank-{ndim} array")
    return value


def _positive(value, name):
    if np.ndim(value) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return float(value)


def _count(value, name):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < 1
    ):
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _mask(boundary, size):
    boundary = np.asarray(boundary)
    if boundary.dtype != bool or boundary.shape != (size,):
        raise ValueError("boundary must be a Boolean vector of length N")
    return boundary


def _geometry(geometry):
    geometry = _array(geometry, 3, "geometry")
    if geometry.shape[0] != 2 or geometry.shape[1] < 3:
        raise ValueError("geometry must have shape (2, N, N + 8), N >= 3")
    size = geometry.shape[1]
    if geometry.shape[2] != size + 8:
        raise ValueError("geometry must have shape (2, N, N + 8)")
    K = geometry[0, :, :size]
    if not np.allclose(K, K.T, rtol=0, atol=1e-12):
        raise ValueError("stiffness must be symmetric")
    if not np.allclose(K.sum(axis=1), 0, rtol=0, atol=1e-12):
        raise ValueError("stiffness must have zero row sums")
    if np.any(K - np.diag(np.diag(K)) > 1e-12):
        raise ValueError("stiffness off-diagonals must be nonpositive")
    if np.any(geometry[1, :, :size] != 0):
        raise ValueError("the mesh is fixed, so stiffness derivatives must be zero")
    n, dn = geometry[:, :, size : size + 2]
    if not np.allclose(np.sum(n * n, axis=1), 1, rtol=0, atol=1e-12):
        raise ValueError("magnetic directions must have unit length")
    if not np.allclose(np.sum(n * dn, axis=1), 0, rtol=0, atol=1e-12):
        raise ValueError("direction derivatives must be tangent")
    expected = _frames(n, dn)
    if not np.allclose(geometry[:, :, size:], expected, rtol=1e-12, atol=1e-12):
        raise ValueError("the supplied frames and their derivatives are inconsistent")
    return geometry


def _frames(n, dn):
    t = np.column_stack((-n[:, 1], n[:, 0]))
    dt = np.column_stack((-dn[:, 1], dn[:, 0]))
    nu = np.column_stack((n[:, 0] ** 2 - n[:, 1] ** 2, 2 * n[:, 0] * n[:, 1]))
    dnu = np.column_stack(
        (
            2 * n[:, 0] * dn[:, 0] - 2 * n[:, 1] * dn[:, 1],
            2 * (dn[:, 0] * n[:, 1] + n[:, 0] * dn[:, 1]),
        )
    )
    tau = np.column_stack((-nu[:, 1], nu[:, 0]))
    dtau = np.column_stack((-dnu[:, 1], dnu[:, 0]))
    return np.stack(
        (np.column_stack((n, t, nu, tau)), np.column_stack((dn, dt, dnu, dtau)))
    )


def _systems(systems):
    systems = _array(systems, 4, "systems")
    if systems.shape[:2] != (2, 2) or systems.shape[2] < 1:
        raise ValueError("systems must have shape (2, 2, N, N + 1)")
    size = systems.shape[2]
    if systems.shape[3] != size + 1:
        raise ValueError("systems must have shape (2, 2, N, N + 1)")
    for layer in range(2):
        for block in systems[layer, :, :, :size]:
            if not np.allclose(block, block.T, rtol=0, atol=1e-12):
                raise ValueError("Hessians and their derivatives must be symmetric")
            if layer == 0 and np.linalg.eigvalsh(block)[0] < -1e-12:
                raise ValueError("base Hessians must be positive semidefinite")
    return systems


def _state(state, size):
    state = _array(state, 3, "state")
    if state.shape != (2, 3, size):
        raise ValueError("state must have shape (2, 3, N)")
    if np.any(np.abs(state[0, 0]) >= 1):
        raise ValueError("base magnetic coefficients must lie in (-1, 1)")
    return state


def _phi(r):
    return 2 * r / (1 - r * r)


def _dphi(r):
    return 2 * (1 + r * r) / (1 - r * r) ** 2


def _ddphi(r):
    return 4 * r * (3 + r * r) / (1 - r * r) ** 3


def _linear_jet(A, dA, b, db):
    try:
        q, upper = np.linalg.qr(A)
        x = np.linalg.solve(upper, q.T @ b)
        dx = np.linalg.solve(upper, q.T @ (db - dA @ x))
    except np.linalg.LinAlgError as exc:
        raise ValueError("linear system is singular") from exc
    result = np.stack((x, dx))
    if not np.all(np.isfinite(result)):
        raise ValueError("linear solve produced a nonfinite jet")
    return result


def _prepare_geometry_jet(
    vertices: np.ndarray,
    triangles: np.ndarray,
    angles: np.ndarray,
    direction: np.ndarray,
) -> np.ndarray:
    vertices = _array(vertices, 2, "vertices")
    triangles = np.asarray(triangles)
    if vertices.shape[0] < 3 or vertices.shape[1] != 2:
        raise ValueError("vertices must have shape (N, 2), N >= 3")
    size = len(vertices)
    if (
        triangles.ndim != 2
        or triangles.shape[0] < 1
        or triangles.shape[1] != 3
        or not np.issubdtype(triangles.dtype, np.integer)
    ):
        raise ValueError("triangles must be a nonempty integer array of shape (M, 3)")
    if (
        np.any(triangles < 0)
        or np.any(triangles >= size)
        or len(np.unique(triangles)) != size
    ):
        raise ValueError("connectivity must be in range and use every vertex")
    angles = _array(angles, 1, "angles")
    direction = _array(direction, 1, "direction")
    if angles.shape != (size,) or direction.shape != (size,):
        raise ValueError("angles and direction must have shape (N,)")
    K = np.zeros((size, size))
    for indices in triangles:
        if len(np.unique(indices)) != 3:
            raise ValueError("triangle indices must be distinct")
        xy = vertices[indices]
        area = abs(np.linalg.det(np.column_stack((xy[1] - xy[0], xy[2] - xy[0])))) / 2
        if area == 0 or not np.isfinite(area):
            raise ValueError("triangle area must be finite and positive")
        gradients = np.linalg.inv(np.column_stack((np.ones(3), xy)))[1:, :]
        K[np.ix_(indices, indices)] += area * gradients.T @ gradients
    n = np.column_stack((np.cos(angles), np.sin(angles)))
    dn = direction[:, None] * np.column_stack((-n[:, 1], n[:, 0]))
    result = np.zeros((2, size, size + 8))
    result[0, :, :size] = K
    result[:, :, size:] = _frames(n, dn)
    return _geometry(result)


def _build_quadratic_jet(geometry: np.ndarray, Qc: float, Mc: float) -> np.ndarray:
    geometry = _geometry(geometry)
    Qc, Mc = _positive(Qc, "Qc"), _positive(Mc, "Mc")
    size = geometry.shape[1]
    K = geometry[0, :, :size]
    result = np.empty((2, 2, size, size + 1))
    for index, (start, scale) in enumerate(((size, Mc), (size + 4, Qc))):
        x, dx = geometry[:, :, start : start + 2]
        t, dt = geometry[:, :, start + 2 : start + 4]
        c = 2 * scale * scale
        result[0, index, :, :size] = c * K * (t @ t.T)
        result[1, index, :, :size] = c * K * (dt @ t.T + t @ dt.T)
        result[0, index, :, size] = c * np.sum(K * (t @ x.T), axis=1)
        result[1, index, :, size] = c * np.sum(K * (dt @ x.T + t @ dx.T), axis=1)
    if not np.all(np.isfinite(result)):
        raise ValueError("quadratic jet is nonfinite")
    return result


def _solve_magnetic_jet(
    systems: np.ndarray, boundary: np.ndarray, state: np.ndarray, zeta: float
) -> np.ndarray:
    systems = _systems(systems)
    size = systems.shape[2]
    boundary = _mask(boundary, size)
    state = _state(state, size)
    if np.any(state[:, 0, boundary] != 0):
        raise ValueError("both magnetic boundary layers must be zero")
    zeta = _positive(zeta, "zeta")
    r, p, lam = state[0]
    dr, dp, dlam = state[1]
    a = _dphi(r)
    da = _ddphi(r) * dr
    b = _phi(r) - a * r
    db = -da * r
    A = systems[0, 0, :, :size] + zeta * np.diag(a * a)
    dA = systems[1, 0, :, :size] + 2 * zeta * np.diag(a * da)
    rhs = -systems[0, 0, :, size] + a * lam + zeta * a * (p - b)
    drhs = -systems[1, 0, :, size] + da * lam + a * dlam
    drhs += zeta * (da * (p - b) + a * (dp - db))
    free = np.flatnonzero(~boundary)
    result = np.zeros((2, size))
    if free.size:
        result[:, free] = _linear_jet(
            A[np.ix_(free, free)], dA[np.ix_(free, free)], rhs[free], drhs[free]
        )
    if np.any(np.abs(result[0]) >= 1):
        raise ValueError("magnetic iterate left (-1, 1)")
    return result


def _solve_auxiliary_jet(
    systems: np.ndarray,
    r_jet: np.ndarray,
    multiplier_jet: np.ndarray,
    zeta: float,
    rho: float,
) -> np.ndarray:
    systems = _systems(systems)
    size = systems.shape[2]
    r_jet = _array(r_jet, 2, "r_jet")
    multiplier_jet = _array(multiplier_jet, 2, "multiplier_jet")
    if (
        r_jet.shape != (2, size)
        or multiplier_jet.shape != (2, size)
        or np.any(np.abs(r_jet[0]) >= 1)
    ):
        raise ValueError("vector jets must have shape (2, N) and base r in (-1, 1)")
    zeta, rho = _positive(zeta, "zeta"), _positive(rho, "rho")
    r, dr = r_jet
    lam, dlam = multiplier_jet
    f, df = _phi(r), _dphi(r) * dr
    B = systems[0, 1, :, :size] + zeta * np.eye(size)
    dB = systems[1, 1, :, :size]
    rhs = -systems[0, 1, :, size] - lam + zeta * f
    drhs = -systems[1, 1, :, size] - dlam + zeta * df
    p, dp = _linear_jet(B, dB, rhs, drhs)
    updated = np.stack((p, lam + rho * (p - f)))
    variation = np.stack((dp, dlam + rho * (dp - df)))
    return np.stack((updated, variation))


def _relax_tangent_jet(
    systems: np.ndarray,
    boundary: np.ndarray,
    state: np.ndarray,
    zeta: float = 4.0,
    rho: float = 1.0,
    tolerance: float = 1e-10,
    max_iterations: int = 20000,
) -> np.ndarray:
    systems = _systems(systems)
    size = systems.shape[2]
    boundary = _mask(boundary, size)
    state = _state(state, size).copy()
    if np.any(state[:, 0, boundary] != 0):
        raise ValueError("both magnetic boundary layers must be zero")
    zeta, rho = _positive(zeta, "zeta"), _positive(rho, "rho")
    tolerance = _positive(tolerance, "tolerance")
    max_iterations = _count(max_iterations, "max_iterations")
    for _ in range(max_iterations):
        r = _solve_magnetic_jet(systems, boundary, state, zeta)
        auxiliary = _solve_auxiliary_jet(systems, r, state[:, 2], zeta, rho)
        state = np.concatenate((r[:, None, :], auxiliary), axis=1)
        residual = np.sqrt(np.mean((state[0, 1] - _phi(state[0, 0])) ** 2))
        if residual <= tolerance:
            return state
    raise RuntimeError("inner cycle cap reached before compatibility tolerance")


def _project_geometry_jet(geometry: np.ndarray, r_jet: np.ndarray) -> np.ndarray:
    geometry = _geometry(geometry)
    size = geometry.shape[1]
    r_jet = _array(r_jet, 2, "r_jet")
    if r_jet.shape != (2, size) or np.any(np.abs(r_jet[0]) >= 1):
        raise ValueError("r_jet must have shape (2, N) and base r in (-1, 1)")
    n, dn = geometry[:, :, size : size + 2]
    t, dt = geometry[:, :, size + 2 : size + 4]
    r, dr = r_jet
    x = n + r[:, None] * t
    dx = dn + dr[:, None] * t + r[:, None] * dt
    lengths = np.linalg.norm(x, axis=1)
    projected = x / lengths[:, None]
    dprojected = (dx - projected * np.sum(projected * dx, axis=1)[:, None]) / lengths[
        :, None
    ]
    unchanged = (r == 0) & (dr == 0)
    projected[unchanged] = n[unchanged]
    dprojected[unchanged] = dn[unchanged]
    result = geometry.copy()
    result[:, :, size:] = _frames(projected, dprojected)
    return result


def _measure_energy_jet(geometry: np.ndarray, Qc: float, Mc: float) -> np.ndarray:
    geometry = _geometry(geometry)
    size = geometry.shape[1]
    Qc, Mc = _positive(Qc, "Qc"), _positive(Mc, "Mc")
    K = geometry[0, :, :size]
    n, dn = geometry[:, :, size : size + 2]
    nu, dnu = geometry[:, :, size + 4 : size + 6]
    energy = 0.5 * np.sum(K * (Qc * Qc * (nu @ nu.T) + Mc * Mc * (n @ n.T)))
    variation = np.sum(K * (Qc * Qc * (dnu @ nu.T) + Mc * Mc * (dn @ n.T)))
    result = np.array([energy, variation])
    if not np.all(np.isfinite(result)):
        raise ValueError("energy jet is nonfinite")
    return result


def _compose_steps(operations):
    from types import FunctionType

    context = globals().copy()
    bound = {
        name: FunctionType(
            function.__code__,
            context,
            name,
            function.__defaults__,
            function.__closure__,
        )
        for name, function in operations.items()
    }
    context.update(bound)
    return bound


def compute_boundary_sensitivity(
    vertices: np.ndarray,
    triangles: np.ndarray,
    boundary: np.ndarray,
    angles: np.ndarray,
    direction: np.ndarray,
    Qc: float = 1.0013,
    Mc: float = 1.0025,
    outer_steps: int = 6,
    zeta: float = 4.0,
    rho: float = 1.0,
    tolerance: float = 1e-10,
    max_iterations: int = 20000,
) -> float:
    operations = _compose_steps(
        {
            "_prepare_geometry_jet": globals().get(
                "prepare_geometry_jet", _prepare_geometry_jet
            ),
            "_build_quadratic_jet": globals().get(
                "build_quadratic_jet", _build_quadratic_jet
            ),
            "_solve_magnetic_jet": globals().get(
                "solve_magnetic_jet", _solve_magnetic_jet
            ),
            "_solve_auxiliary_jet": globals().get(
                "solve_auxiliary_jet", _solve_auxiliary_jet
            ),
            "_relax_tangent_jet": globals().get(
                "relax_tangent_jet", _relax_tangent_jet
            ),
            "_project_geometry_jet": globals().get(
                "project_geometry_jet", _project_geometry_jet
            ),
            "_measure_energy_jet": globals().get(
                "measure_energy_jet", _measure_energy_jet
            ),
        }
    )
    geometry = operations["_prepare_geometry_jet"](
        vertices, triangles, angles, direction
    )
    size = geometry.shape[1]
    boundary = _mask(boundary, size)
    Qc, Mc = _positive(Qc, "Qc"), _positive(Mc, "Mc")
    zeta, rho = _positive(zeta, "zeta"), _positive(rho, "rho")
    tolerance = _positive(tolerance, "tolerance")
    outer_steps = _count(outer_steps, "outer_steps")
    max_iterations = _count(max_iterations, "max_iterations")
    initial = operations["_measure_energy_jet"](geometry, Qc, Mc)
    if initial[0] <= 1e-14:
        raise ValueError("initial energy must exceed 1e-14")
    state = np.zeros((2, 3, size))
    for _ in range(outer_steps):
        systems = operations["_build_quadratic_jet"](geometry, Qc, Mc)
        state = operations["_relax_tangent_jet"](
            systems, boundary, state, zeta, rho, tolerance, max_iterations
        )
        geometry = operations["_project_geometry_jet"](geometry, state[:, 0])
    final = operations["_measure_energy_jet"](geometry, Qc, Mc)
    decrease = 1 - final[0] / initial[0]
    derivative = -final[1] / initial[0] + final[0] * initial[1] / initial[0] ** 2
    if abs(decrease) <= 1e-12:
        raise ValueError("fractional energy decrease must exceed 1e-12 in magnitude")
    result = float(derivative / decrease)
    if not np.isfinite(result):
        raise ValueError("relative sensitivity is nonfinite")
    return result
SCICODE_GOLD_EOF
