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


def semielliptic_dos(omega: "np.ndarray", half_bandwidth: float) -> "np.ndarray":
    """Bare Bethe-lattice density of states rho0(w) = (2 / (pi D^2)) sqrt(D^2 - w^2) on |w| < D."""
    omega = np.asarray(omega, dtype=np.float64)
    if omega.ndim != 1 or omega.size < 1 or not np.all(np.isfinite(omega)):
        raise ValueError("omega must be a non-empty finite 1-D array")
    if not (np.isfinite(half_bandwidth) and half_bandwidth > 0.0):
        raise ValueError("half_bandwidth must be positive and finite")
    D = float(half_bandwidth)
    rho = np.zeros_like(omega)
    inside = np.abs(omega) < D
    rho[inside] = 2.0 / (np.pi * D * D) * np.sqrt(D * D - omega[inside] ** 2)
    return rho

import numpy as np


def half_filled_moments(U: float, half_bandwidth: float, bond_amplitude: float) -> "np.ndarray":
    """Eq (30a-e): the source's approximate moment closure mu0..mu5 for the half-filled Bethe
    Hubbard model (its self-consistent pseudo-moment map, not the exact EOM moments, whose
    second moment would be U^2/4 + D^2/4 with no bond dependence).

    X = t^2 z_NN <c+c> with the scaled Bethe hopping t = D/2, t^2 z_NN = D^2/4.
    mu1 = mu3 = mu5 = 0 by particle-hole symmetry.  mu2 = U^2/4 + X.  mu4 keeps only the
    leading connected correction, mu4 = mu2^2 + <eps^2>_bath X with <eps^2>_bath = D^2/4.
    """
    if not (np.isfinite(U) and U >= 0.0):
        raise ValueError("U must be nonnegative and finite")
    if not (np.isfinite(half_bandwidth) and half_bandwidth > 0.0):
        raise ValueError("half_bandwidth must be positive and finite")
    if not np.isfinite(bond_amplitude):
        raise ValueError("bond_amplitude must be finite")
    D = float(half_bandwidth)
    t2z = D * D / 4.0                 # t^2 z_NN for the Bethe lattice with half-bandwidth D
    eps2 = D * D / 4.0                # second moment of the semielliptic band
    X = t2z * float(bond_amplitude)
    mu2 = U * U / 4.0 + X
    mu4 = mu2 * mu2 + eps2 * X        # (30e): leading connected correction only
    return np.array([1.0, 0.0, mu2, 0.0, mu4, 0.0], dtype=np.float64)

import numpy as np


def resolvable_rank(moments: "np.ndarray", order: int, tau: float) -> int:
    """Eq (14): N* = max{n : sigma_n / sigma_1 > tau} from the SVD of the N x N Hankel matrix."""
    moments = np.asarray(moments, dtype=np.float64)
    if int(order) != order or order < 1:
        raise ValueError("order must be a positive integer")
    order = int(order)
    if moments.ndim != 1 or moments.size < 2 * order - 1 or not np.all(np.isfinite(moments)):
        raise ValueError("moments must be a finite 1-D array with at least 2*order-1 entries")
    if not (np.isfinite(tau) and 0.0 < tau < 1.0):
        raise ValueError("tau must lie in (0, 1)")
    hankel = np.array([[moments[i + j] for j in range(order)] for i in range(order)])
    sigma = np.linalg.svd(hankel, compute_uv=False)
    kept = np.nonzero(sigma / sigma[0] > tau)[0]
    return int(kept.max() + 1)

import numpy as np


def jacobi_matrix(moments: "np.ndarray", rank: int) -> "np.ndarray":
    """Tridiagonal Jacobi matrix of the rank-point Gauss-Christoffel rule via the LDL^T route.

    With M = L L^T the Cholesky factor of the rank x rank Hankel matrix and M' the shifted
    Hankel [mu_{i+j+1}], the Jacobi matrix is J = L^{-1} M' L^{-T}, symmetrised.
    """
    moments = np.asarray(moments, dtype=np.float64)
    if int(rank) != rank or rank < 1:
        raise ValueError("rank must be a positive integer")
    rank = int(rank)
    if moments.ndim != 1 or moments.size < 2 * rank or not np.all(np.isfinite(moments)):
        raise ValueError("moments must be a finite 1-D array with at least 2*rank entries")
    M = np.array([[moments[i + j] for j in range(rank)] for i in range(rank)])
    Ms = np.array([[moments[i + j + 1] for j in range(rank)] for i in range(rank)])
    try:
        L = np.linalg.cholesky(M)
    except np.linalg.LinAlgError:
        raise ValueError("the Hankel matrix is not positive definite at this rank")
    Li = np.linalg.inv(L)
    J = Li @ Ms @ Li.T
    return 0.5 * (J + J.T)

import numpy as np


def quadrature_nodes_weights(jacobi: "np.ndarray", mu0: float) -> "np.ndarray":
    """Nodes (eigenvalues) and weights mu0 * v[0]^2 of the Jacobi matrix, ascending by node."""
    J = np.asarray(jacobi, dtype=np.float64)
    if J.ndim != 2 or J.shape[0] != J.shape[1] or J.shape[0] < 1 or not np.all(np.isfinite(J)):
        raise ValueError("jacobi must be a finite square matrix")
    if not np.allclose(J, J.T, rtol=0.0, atol=1e-12):
        raise ValueError("jacobi must be symmetric")
    if not (np.isfinite(mu0) and mu0 > 0.0):
        raise ValueError("mu0 must be positive and finite")
    nodes, vecs = np.linalg.eigh(J)
    weights = float(mu0) * vecs[0, :] ** 2
    order = np.argsort(nodes)
    return np.vstack([nodes[order], weights[order]])

import numpy as np


def bond_amplitude_closure(nodes: "np.ndarray", weights: "np.ndarray", half_bandwidth: float) -> float:
    """<c+c> = int_{-inf}^0 t rho0(w) A(w) dw on the discrete measure, with particle-hole
    symmetry enforced at the moment update.

    rho0 * A is even at half filling, so the occupied integral is half the full one:
    <c+c> = (1/2) sum_i w_i t rho0(eps_i), which gives a Fermi-level pole half its weight
    instead of leaving it to the sign of a machine-zero node.
    """
    nodes = np.asarray(nodes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if nodes.ndim != 1 or nodes.shape != weights.shape or nodes.size < 1:
        raise ValueError("nodes and weights must be 1-D arrays of equal, nonzero length")
    if not (np.all(np.isfinite(nodes)) and np.all(np.isfinite(weights))):
        raise ValueError("nodes and weights must be finite")
    if np.any(weights < 0.0):
        raise ValueError("weights must be nonnegative")
    if not (np.isfinite(half_bandwidth) and half_bandwidth > 0.0):
        raise ValueError("half_bandwidth must be positive and finite")
    t = float(half_bandwidth) / 2.0
    rho = semielliptic_dos(nodes, half_bandwidth)
    return 0.5 * float(np.sum(weights * t * rho))

import numpy as np


def self_consistent_central_weight(U: float, half_bandwidth: float, order: int, tau: float,
                                delta: float, mixing: float, seed_amplitude: float,
                                max_iter: int) -> float:
    """Inner moment self-consistency loop (source Sec. IV B) and the weight of the central pole.

    Seed the moments from seed_amplitude, then iterate: rank (14) -> Jacobi -> nodes and
    weights -> bond amplitude closure -> new moments, mixed linearly with the previous
    ones, until Eq (17) max_n |mu_n' - mu_n| / |mu_n| < delta over the nonzero moments.
    Returns the weight of the node closest to zero energy at the fixed point; the rule is
    rebuilt from the final (mixed) moment set after the stop test is met.
    """
    if not (np.isfinite(U) and U >= 0.0):
        raise ValueError("U must be nonnegative and finite")
    if not (np.isfinite(half_bandwidth) and half_bandwidth > 0.0):
        raise ValueError("half_bandwidth must be positive and finite")
    if int(order) != order or order < 1:
        raise ValueError("order must be a positive integer")
    if not (np.isfinite(tau) and 0.0 < tau < 1.0):
        raise ValueError("tau must lie in (0, 1)")
    if not (np.isfinite(delta) and delta > 0.0):
        raise ValueError("delta must be positive")
    if not (np.isfinite(mixing) and 0.0 < mixing <= 1.0):
        raise ValueError("mixing must lie in (0, 1]")
    if not np.isfinite(seed_amplitude):
        raise ValueError("seed_amplitude must be finite")
    if int(max_iter) != max_iter or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    order, max_iter = int(order), int(max_iter)

    mu = half_filled_moments(U, half_bandwidth, seed_amplitude)
    for _ in range(max_iter):
        n_star = resolvable_rank(mu, order, tau)
        J = jacobi_matrix(mu, n_star)
        nw = quadrature_nodes_weights(J, mu[0])
        amp = bond_amplitude_closure(nw[0], nw[1], half_bandwidth)
        mu_new = half_filled_moments(U, half_bandwidth, amp)
        mu_mixed = float(mixing) * mu_new + (1.0 - float(mixing)) * mu
        nz = np.abs(mu) > 0.0
        change = float(np.max(np.abs(mu_mixed[nz] - mu[nz]) / np.abs(mu[nz])))
        mu = mu_mixed
        if change < delta:
            break
    else:
        raise ValueError("moment iteration did not converge within max_iter")

    n_star = resolvable_rank(mu, order, tau)
    nw = quadrature_nodes_weights(jacobi_matrix(mu, n_star), mu[0])
    _ = semielliptic_dos(nw[0], half_bandwidth)     # the bare band the closure integrates against
    central = int(np.argmin(np.abs(nw[0])))
    return float(nw[1][central])
SCICODE_GOLD_EOF
