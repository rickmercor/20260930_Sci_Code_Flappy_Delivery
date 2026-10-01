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

def assemble_virtual_node_volumes(points, boundary_flags, epsilon, domain_measure):
    import numpy as np

    P = np.asarray(points, dtype=float)
    b = np.asarray(boundary_flags, dtype=bool)
    if P.ndim != 2 or P.shape[1] != 2 or b.shape != (len(P),):
        raise ValueError("points must be (N, 2) and boundary_flags (N,)")
    eps = float(epsilon)
    n = len(P)
    E = np.array([(i, j) for i in range(n) for j in range(i + 1, n)
                  if np.linalg.norm(P[j] - P[i]) < eps], dtype=int).reshape(-1, 2)
    r = np.linalg.norm(P[E[:, 1]] - P[E[:, 0]], axis=1)
    phi = np.clip(1.0 - r / eps, 0.0, None) ** 2
    kappa = np.zeros(n)
    np.add.at(kappa, E[:, 0], phi)
    np.add.at(kappa, E[:, 1], phi)
    volumes = np.zeros(n)
    inv = 1.0 / kappa[~b]
    volumes[~b] = inv / inv.sum() * float(domain_measure)
    return volumes

import numpy as np

def assemble_moment_constraint_system(points, boundary_flags, epsilon, node_volumes):
    import numpy as np

    P = np.asarray(points, dtype=float)
    b = np.asarray(boundary_flags, dtype=bool)
    m = np.asarray(node_volumes, dtype=float)
    if P.ndim != 2 or P.shape[1] != 2 or b.shape != (len(P),) or m.shape != (len(P),):
        raise ValueError("inconsistent point cloud, flags or volumes")
    eps = float(epsilon)
    n = len(P)
    E = np.array([(i, j) for i in range(n) for j in range(i + 1, n)
                  if np.linalg.norm(P[j] - P[i]) < eps], dtype=int).reshape(-1, 2)
    incident = [[] for _ in range(n)]
    for t, (p, q) in enumerate(E):
        incident[p].append(t)
        incident[q].append(t)
    interior = np.where(~b)[0]
    pairs = [(0, 0), (0, 1), (1, 1)]
    system = np.zeros((5 * len(interior), len(E) + 1))
    for k, i in enumerate(interior):
        r0 = 5 * k
        for t in incident[i]:
            p, q = E[t]
            eta = P[q if p == i else p] - P[i]
            system[r0, t] = eta[0]
            system[r0 + 1, t] = eta[1]
            for s, (c, d) in enumerate(pairs):
                system[r0 + 2 + s, t] = eta[c] * eta[d]
        for s, (c, d) in enumerate(pairs):
            system[r0 + 2 + s, -1] = 2.0 * m[i] if c == d else 0.0
    return system

import numpy as np

def solve_edge_areas(points, epsilon, moment_system):
    import numpy as np

    P = np.asarray(points, dtype=float)
    A = np.asarray(moment_system, dtype=float)
    eps = float(epsilon)
    n = len(P)
    E = np.array([(i, j) for i in range(n) for j in range(i + 1, n)
                  if np.linalg.norm(P[j] - P[i]) < eps], dtype=int).reshape(-1, 2)
    if A.ndim != 2 or A.shape[1] != len(E) + 1:
        raise ValueError("moment system does not match the edge set of these points")
    r = np.linalg.norm(P[E[:, 1]] - P[E[:, 0]], axis=1)
    phi = np.clip(1.0 - r / eps, 0.0, None) ** 2
    B, c = A[:, :-1], A[:, -1]
    lam = np.linalg.solve((B * phi) @ B.T, c)
    return phi * (B.T @ lam)

import numpy as np

def build_coboundary(points, epsilon):
    import numpy as np

    try:
        P = np.asarray(points, dtype=float)
        eps = float(epsilon)
    except (TypeError, ValueError) as exc:
        raise ValueError("points and epsilon must be real-valued") from exc
    if (P.ndim != 2 or P.shape[1] != 2 or len(P) == 0
            or not np.all(np.isfinite(P))):
        raise ValueError("points must be a nonempty finite array of shape (N, 2)")
    if (isinstance(epsilon, (bool, np.bool_)) or np.ndim(epsilon) != 0
            or not np.isfinite(eps) or eps <= 0.0):
        raise ValueError("epsilon must be a finite positive scalar")
    n = len(P)
    E = np.array([(i, j) for i in range(n) for j in range(i + 1, n)
                  if np.linalg.norm(P[j] - P[i]) < eps], dtype=int).reshape(-1, 2)
    d0 = np.zeros((len(E), n))
    for t, (p, q) in enumerate(E):
        d0[t, p] = -1.0
        d0[t, q] = 1.0
    return d0

import numpy as np

def assemble_hodge_laplacian(coboundary_matrix, edge_areas):
    import numpy as np

    d0 = np.asarray(coboundary_matrix, dtype=float)
    a = np.asarray(edge_areas, dtype=float)
    if d0.ndim != 2 or a.shape != (d0.shape[0],):
        raise ValueError("coboundary and edge areas disagree on the edge count")
    return d0.T @ (a[:, None] * d0)

import numpy as np

def solve_dirichlet_problem(laplacian_matrix, node_volumes, boundary_flags,
                                    forcing, boundary_values):
    import numpy as np

    K = np.asarray(laplacian_matrix, dtype=float)
    m = np.asarray(node_volumes, dtype=float)
    b = np.asarray(boundary_flags, dtype=bool)
    f = np.asarray(forcing, dtype=float)
    ub = np.asarray(boundary_values, dtype=float)
    n = len(m)
    if K.shape != (n, n) or b.shape != (n,) or f.shape != (n,) or ub.shape != (n,):
        raise ValueError("inconsistent shapes")
    u = np.zeros(n)
    u[b] = ub[b]
    rhs = (m * f) - K[:, b] @ u[b]
    u[~b] = np.linalg.solve(K[np.ix_(~b, ~b)], rhs[~b])
    return u

import numpy as np

def volume_weighted_relative_error(solution, reference, node_volumes, boundary_flags):
    import numpy as np

    try:
        u = np.asarray(solution, dtype=float)
        r = np.asarray(reference, dtype=float)
        m = np.asarray(node_volumes, dtype=float)
        raw_flags = np.asarray(boundary_flags)
    except (TypeError, ValueError) as exc:
        raise ValueError("inputs must be real-valued arrays") from exc
    if u.ndim != 1 or len(u) == 0 or r.shape != u.shape or m.shape != u.shape:
        raise ValueError("solution, reference and node_volumes must be same-length 1D arrays")
    if raw_flags.shape != u.shape or raw_flags.dtype.kind != "b":
        raise ValueError("boundary_flags must be a boolean 1D array of the same length")
    if (not np.all(np.isfinite(u)) or not np.all(np.isfinite(r))
            or not np.all(np.isfinite(m))):
        raise ValueError("solution, reference and node_volumes must be finite")
    if np.any(m < 0.0):
        raise ValueError("node volumes must be nonnegative")
    b = raw_flags.astype(bool, copy=False)
    denominator_sq = float(np.sum(m[~b] * r[~b] ** 2))
    if not np.isfinite(denominator_sq) or denominator_sq <= 0.0:
        raise ValueError("weighted reference norm must be finite and nonzero")
    numerator_sq = float(np.sum(m[~b] * (u[~b] - r[~b]) ** 2))
    if not np.isfinite(numerator_sq) or numerator_sq < 0.0:
        raise ValueError("weighted error norm must be finite")
    return float(np.sqrt(numerator_sq / denominator_sq))

import numpy as np

def pointcloud_poisson_error(grid_size, amplitude, epsilon_factor, domain_measure):
    import numpy as np

    if isinstance(grid_size, (bool, np.bool_)) or not isinstance(grid_size, (int, np.integer)):
        raise ValueError("grid_size must be an integer")
    k = int(grid_size)
    if k < 3:
        raise ValueError("grid_size must be at least 3")
    try:
        amp = float(amplitude)
        eps_factor = float(epsilon_factor)
        measure = float(domain_measure)
    except (TypeError, ValueError) as exc:
        raise ValueError("scalar parameters must be real-valued") from exc
    if (np.ndim(amplitude) != 0 or np.ndim(epsilon_factor) != 0
            or np.ndim(domain_measure) != 0):
        raise ValueError("amplitude, epsilon_factor and domain_measure must be scalars")
    if not np.isfinite(amp):
        raise ValueError("amplitude must be finite")
    if not np.isfinite(eps_factor) or eps_factor <= 0.0:
        raise ValueError("epsilon_factor must be finite and positive")
    if not np.isfinite(measure) or measure <= 0.0:
        raise ValueError("domain_measure must be finite and positive")
    xs = np.linspace(0.0, 1.0, k)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    P0 = np.stack([X.ravel(), Y.ravel()], axis=1)
    h = 1.0 / (k - 1)
    flags = (np.isclose(P0[:, 0], 0.0) | np.isclose(P0[:, 0], 1.0) |
             np.isclose(P0[:, 1], 0.0) | np.isclose(P0[:, 1], 1.0))
    P = P0.copy()
    P[~flags, 0] += amp * h * np.sin(6.0 * P0[~flags, 0] + 2.0 * P0[~flags, 1])
    P[~flags, 1] += amp * h * np.cos(2.0 * P0[~flags, 0] + 5.0 * P0[~flags, 1])
    eps = eps_factor * h

    volumes = assemble_virtual_node_volumes(P, flags, eps, measure)
    system = assemble_moment_constraint_system(P, flags, eps, volumes)
    areas = solve_edge_areas(P, eps, system)
    d0 = build_coboundary(P, eps)
    K = assemble_hodge_laplacian(d0, areas)
    exact = np.sin(np.pi * P[:, 0]) * np.sin(np.pi * P[:, 1])
    forcing = 2.0 * np.pi ** 2 * exact
    u = solve_dirichlet_problem(K, volumes, flags, forcing, exact)
    return float(volume_weighted_relative_error(u, exact, volumes, flags))
SCICODE_GOLD_EOF
