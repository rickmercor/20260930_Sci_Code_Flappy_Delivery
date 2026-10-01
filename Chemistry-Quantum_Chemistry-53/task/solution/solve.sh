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


def ring_one_body(L, t, v_ext):
    if isinstance(L, bool) or not isinstance(L, (int, np.integer)):
        raise ValueError("L must be an integer")
    if L < 3:
        raise ValueError("the ring must have at least three sites")
    L = int(L)
    t = float(t)
    if not np.isfinite(t):
        raise ValueError("t must be finite")
    v = np.asarray(v_ext, dtype=float)
    if v.ndim != 1 or v.shape[0] != L:
        raise ValueError("v_ext must be a one-dimensional array with L entries")
    if not np.all(np.isfinite(v)):
        raise ValueError("v_ext must be finite")
    h = np.diag(v)
    for i in range(L):
        j = (i + 1) % L
        h[i, j] -= t
        h[j, i] -= t
    return h

import numpy as np


def gks_density_matrix(h, U, v_c, n_elec):
    h = np.asarray(h, dtype=float)
    if h.ndim != 2 or h.shape[0] != h.shape[1] or h.shape[0] < 2:
        raise ValueError("h must be a square matrix")
    if not np.all(np.isfinite(h)) or not np.allclose(h, h.T, rtol=0.0, atol=1e-10):
        raise ValueError("h must be finite and symmetric")
    L = h.shape[0]
    U = float(U)
    if not np.isfinite(U):
        raise ValueError("U must be finite")
    v_c = np.asarray(v_c, dtype=float)
    if v_c.ndim != 1 or v_c.shape[0] != L or not np.all(np.isfinite(v_c)):
        raise ValueError("v_c must be a finite vector with L entries")
    if isinstance(n_elec, bool) or not isinstance(n_elec, (int, np.integer)):
        raise ValueError("n_elec must be an integer")
    if n_elec % 2 != 0 or n_elec < 2 or n_elec > 2 * L - 2:
        raise ValueError("n_elec must be even and satisfy 2 <= n_elec <= 2L - 2")
    n_occ = int(n_elec) // 2
    gamma = np.zeros((L, L))
    focks, errors = [], []
    ndiis = 8
    for _ in range(5000):
        fock = h + np.diag(v_c) + np.diag(U * np.diag(gamma))
        err = fock @ gamma - gamma @ fock
        focks.append(fock)
        errors.append(err.ravel())
        if len(focks) > ndiis:
            focks.pop(0)
            errors.pop(0)
        fock_use = fock
        m = len(focks)
        if m > 1:
            bmat = -np.ones((m + 1, m + 1))
            bmat[m, m] = 0.0
            for a in range(m):
                for b in range(m):
                    bmat[a, b] = errors[a] @ errors[b]
            rhs = np.zeros(m + 1)
            rhs[m] = -1.0
            try:
                coef = np.linalg.solve(bmat, rhs)[:m]
                if np.all(np.isfinite(coef)):
                    fock_use = sum(c * f for c, f in zip(coef, focks))
            except np.linalg.LinAlgError:
                pass
        _, cmat = np.linalg.eigh(fock_use)
        g_new = cmat[:, :n_occ] @ cmat[:, :n_occ].T
        if np.max(np.abs(g_new - gamma)) < 1e-12 and np.max(np.abs(err)) < 1e-10:
            gamma = g_new
            break
        gamma = g_new
    else:
        raise RuntimeError("gKS self-consistency did not converge")
    fock = h + np.diag(v_c) + np.diag(U * np.diag(gamma))
    levels = np.linalg.eigvalsh(fock)
    if levels[n_occ] - levels[n_occ - 1] < 1e-8:
        raise ValueError("degenerate Fermi level: closed-shell occupation undefined")
    return gamma

import numpy as np


def bath_orbital(gamma, site):
    gamma = np.asarray(gamma, dtype=float)
    if gamma.ndim != 2 or gamma.shape[0] != gamma.shape[1] or gamma.shape[0] < 2:
        raise ValueError("gamma must be a square matrix")
    if not np.all(np.isfinite(gamma)) or not np.allclose(gamma, gamma.T, rtol=0.0, atol=1e-8):
        raise ValueError("gamma must be finite and symmetric")
    if isinstance(site, bool) or not isinstance(site, (int, np.integer)):
        raise ValueError("site must be an integer")
    L = gamma.shape[0]
    if site < 0 or site >= L:
        raise ValueError("site index out of range")
    b = gamma[int(site), :].copy()
    b[int(site)] = 0.0
    norm = np.linalg.norm(b)
    if norm < 1e-12:
        raise ValueError("the embedded site is decoupled: no bath orbital exists")
    return b / norm

import numpy as np


def cluster_hamiltonian(h, U, gamma, site):
    h = np.asarray(h, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    if h.ndim != 2 or h.shape[0] != h.shape[1] or gamma.shape != h.shape:
        raise ValueError("h and gamma must be square matrices of the same shape")
    if not np.all(np.isfinite(h)) or not np.allclose(h, h.T, rtol=0.0, atol=1e-10):
        raise ValueError("h must be finite and symmetric")
    if not np.all(np.isfinite(gamma)) or not np.allclose(gamma, gamma.T, rtol=0.0, atol=1e-8):
        raise ValueError("gamma must be finite and symmetric")
    if np.max(np.abs(gamma @ gamma - gamma)) > 1e-6:
        raise ValueError("gamma must be idempotent")
    U = float(U)
    if not np.isfinite(U):
        raise ValueError("U must be finite")
    if isinstance(site, bool) or not isinstance(site, (int, np.integer)):
        raise ValueError("site must be an integer")
    L = h.shape[0]
    if site < 0 or site >= L:
        raise ValueError("site index out of range")
    i = int(site)
    b = gamma[i, :].copy()
    b[i] = 0.0
    norm = np.linalg.norm(b)
    if norm < 1e-12:
        raise ValueError("the embedded site is decoupled: no bath orbital exists")
    b = b / norm
    cmat = np.zeros((L, 2))
    cmat[i, 0] = 1.0
    cmat[:, 1] = b
    q = cmat @ cmat.T
    if abs(np.trace(q @ gamma) - 1.0) > 1e-6:
        raise ValueError("the cluster must hold exactly one electron per spin")
    p = np.eye(L) - q
    gamma_core = p @ gamma @ p
    h_eff = h + np.diag(U * np.diag(gamma_core))
    hc = cmat.T @ h_eff @ cmat
    e_i, e_b, tau = hc[0, 0], hc[1, 1], hc[0, 1]
    u_b = U * float(np.sum(b ** 4))
    return np.array([
        [2.0 * e_i + U, tau, tau, 0.0],
        [tau, e_i + e_b, 0.0, tau],
        [tau, 0.0, e_i + e_b, tau],
        [0.0, tau, tau, 2.0 * e_b + u_b],
    ])

import numpy as np


def impurity_chemical_potential(b, v_c):
    b = np.asarray(b, dtype=float)
    v_c = np.asarray(v_c, dtype=float)
    if b.ndim != 1 or v_c.ndim != 1 or b.shape != v_c.shape or b.shape[0] < 1:
        raise ValueError("b and v_c must be vectors of the same length")
    if not np.all(np.isfinite(b)) or not np.all(np.isfinite(v_c)):
        raise ValueError("b and v_c must be finite")
    return float(np.sum(b * b * v_c))

import numpy as np


def cluster_site_density(h_cl, mu):
    h_cl = np.asarray(h_cl, dtype=float)
    if h_cl.shape != (4, 4):
        raise ValueError("h_cl must be a 4 x 4 matrix")
    if not np.all(np.isfinite(h_cl)) or not np.allclose(h_cl, h_cl.T, rtol=0.0, atol=1e-10):
        raise ValueError("h_cl must be finite and symmetric")
    mu = float(mu)
    if not np.isfinite(mu):
        raise ValueError("mu must be finite")
    n_op = np.diag([2.0, 1.0, 1.0, 0.0])
    w, v = np.linalg.eigh(h_cl - mu * n_op)
    if w[1] - w[0] < 1e-10:
        raise ValueError("degenerate cluster ground level: occupation undefined")
    psi = v[:, 0]
    return float(2.0 * psi[0] ** 2 + psi[1] ** 2 + psi[2] ** 2)

import numpy as np


def _gks_reference(h, U, v_c, n_occ):
    L = h.shape[0]
    gamma = np.zeros((L, L))
    focks, errors = [], []
    for _ in range(5000):
        fock = h + np.diag(v_c) + np.diag(U * np.diag(gamma))
        err = fock @ gamma - gamma @ fock
        focks.append(fock)
        errors.append(err.ravel())
        if len(focks) > 8:
            focks.pop(0)
            errors.pop(0)
        fock_use = fock
        m = len(focks)
        if m > 1:
            bmat = -np.ones((m + 1, m + 1))
            bmat[m, m] = 0.0
            for a in range(m):
                for b in range(m):
                    bmat[a, b] = errors[a] @ errors[b]
            rhs = np.zeros(m + 1)
            rhs[m] = -1.0
            try:
                coef = np.linalg.solve(bmat, rhs)[:m]
                if np.all(np.isfinite(coef)):
                    fock_use = sum(c * f for c, f in zip(coef, focks))
            except np.linalg.LinAlgError:
                pass
        _, cmat = np.linalg.eigh(fock_use)
        g_new = cmat[:, :n_occ] @ cmat[:, :n_occ].T
        if np.max(np.abs(g_new - gamma)) < 1e-12 and np.max(np.abs(err)) < 1e-10:
            gamma = g_new
            break
        gamma = g_new
    else:
        raise RuntimeError("gKS self-consistency did not converge")
    fock = h + np.diag(v_c) + np.diag(U * np.diag(gamma))
    levels = np.linalg.eigvalsh(fock)
    if levels[n_occ] - levels[n_occ - 1] < 1e-8:
        raise ValueError("degenerate Fermi level: closed-shell occupation undefined")
    return gamma


def _cluster_density(h, U, gamma, v_c, i):
    L = h.shape[0]
    b = gamma[i, :].copy()
    b[i] = 0.0
    norm = np.linalg.norm(b)
    if norm < 1e-12:
        raise ValueError("the embedded site is decoupled: no bath orbital exists")
    b = b / norm
    cmat = np.zeros((L, 2))
    cmat[i, 0] = 1.0
    cmat[:, 1] = b
    q = cmat @ cmat.T
    gamma_core = (np.eye(L) - q) @ gamma @ (np.eye(L) - q)
    hc = cmat.T @ (h + np.diag(U * np.diag(gamma_core))) @ cmat
    e_i, e_b, tau = hc[0, 0], hc[1, 1], hc[0, 1]
    u_b = U * float(np.sum(b ** 4))
    mu = float(np.sum(b * b * v_c))
    hm = np.array([
        [2.0 * e_i + U - 2.0 * mu, tau, tau, 0.0],
        [tau, e_i + e_b - mu, 0.0, tau],
        [tau, 0.0, e_i + e_b - mu, tau],
        [0.0, tau, tau, 2.0 * e_b + u_b],
    ])
    w, v = np.linalg.eigh(hm)
    if w[1] - w[0] < 1e-10:
        raise ValueError("degenerate cluster ground level: occupation undefined")
    psi = v[:, 0]
    return float(2.0 * psi[0] ** 2 + psi[1] ** 2 + psi[2] ** 2)


def density_mismatch(h, U, v_c, n_elec):
    h = np.asarray(h, dtype=float)
    if h.ndim != 2 or h.shape[0] != h.shape[1] or h.shape[0] < 2:
        raise ValueError("h must be a square matrix")
    if not np.all(np.isfinite(h)) or not np.allclose(h, h.T, rtol=0.0, atol=1e-10):
        raise ValueError("h must be finite and symmetric")
    L = h.shape[0]
    U = float(U)
    if not np.isfinite(U):
        raise ValueError("U must be finite")
    v_c = np.asarray(v_c, dtype=float)
    if v_c.ndim != 1 or v_c.shape[0] != L or not np.all(np.isfinite(v_c)):
        raise ValueError("v_c must be a finite vector with L entries")
    if isinstance(n_elec, bool) or not isinstance(n_elec, (int, np.integer)):
        raise ValueError("n_elec must be an integer")
    if n_elec % 2 != 0 or n_elec < 2 or n_elec > 2 * L - 2:
        raise ValueError("n_elec must be even and satisfy 2 <= n_elec <= 2L - 2")
    gamma = _gks_reference(h, U, v_c, int(n_elec) // 2)
    n_gks = 2.0 * np.diag(gamma)
    n_cl = np.array([_cluster_density(h, U, gamma, v_c, i) for i in range(L)])
    return n_cl - n_gks

import numpy as np


def _gks_reference(h, U, v_c, n_occ):
    L = h.shape[0]
    gamma = np.zeros((L, L))
    focks, errors = [], []
    for _ in range(5000):
        fock = h + np.diag(v_c) + np.diag(U * np.diag(gamma))
        err = fock @ gamma - gamma @ fock
        focks.append(fock)
        errors.append(err.ravel())
        if len(focks) > 8:
            focks.pop(0)
            errors.pop(0)
        fock_use = fock
        m = len(focks)
        if m > 1:
            bmat = -np.ones((m + 1, m + 1))
            bmat[m, m] = 0.0
            for a in range(m):
                for b in range(m):
                    bmat[a, b] = errors[a] @ errors[b]
            rhs = np.zeros(m + 1)
            rhs[m] = -1.0
            try:
                coef = np.linalg.solve(bmat, rhs)[:m]
                if np.all(np.isfinite(coef)):
                    fock_use = sum(c * f for c, f in zip(coef, focks))
            except np.linalg.LinAlgError:
                pass
        _, cmat = np.linalg.eigh(fock_use)
        g_new = cmat[:, :n_occ] @ cmat[:, :n_occ].T
        if np.max(np.abs(g_new - gamma)) < 1e-12 and np.max(np.abs(err)) < 1e-10:
            gamma = g_new
            break
        gamma = g_new
    else:
        raise RuntimeError("gKS self-consistency did not converge")
    fock = h + np.diag(v_c) + np.diag(U * np.diag(gamma))
    levels = np.linalg.eigvalsh(fock)
    if levels[n_occ] - levels[n_occ - 1] < 1e-8:
        raise ValueError("degenerate Fermi level: closed-shell occupation undefined")
    return gamma


def _cluster_density(h, U, gamma, v_c, i):
    L = h.shape[0]
    b = gamma[i, :].copy()
    b[i] = 0.0
    norm = np.linalg.norm(b)
    if norm < 1e-12:
        raise ValueError("the embedded site is decoupled: no bath orbital exists")
    b = b / norm
    cmat = np.zeros((L, 2))
    cmat[i, 0] = 1.0
    cmat[:, 1] = b
    q = cmat @ cmat.T
    gamma_core = (np.eye(L) - q) @ gamma @ (np.eye(L) - q)
    hc = cmat.T @ (h + np.diag(U * np.diag(gamma_core))) @ cmat
    e_i, e_b, tau = hc[0, 0], hc[1, 1], hc[0, 1]
    u_b = U * float(np.sum(b ** 4))
    mu = float(np.sum(b * b * v_c))
    hm = np.array([
        [2.0 * e_i + U - 2.0 * mu, tau, tau, 0.0],
        [tau, e_i + e_b - mu, 0.0, tau],
        [tau, 0.0, e_i + e_b - mu, tau],
        [0.0, tau, tau, 2.0 * e_b + u_b],
    ])
    w, v = np.linalg.eigh(hm)
    if w[1] - w[0] < 1e-10:
        raise ValueError("degenerate cluster ground level: occupation undefined")
    psi = v[:, 0]
    return float(2.0 * psi[0] ** 2 + psi[1] ** 2 + psi[2] ** 2)


def _mismatch(v_c, h, U, n_occ):
    gamma = _gks_reference(h, U, v_c, n_occ)
    n_gks = 2.0 * np.diag(gamma)
    n_cl = np.array([_cluster_density(h, U, gamma, v_c, i) for i in range(h.shape[0])])
    return n_cl - n_gks


def solve_correlation_potential(h, U, n_elec, tol=1e-10):
    from scipy.optimize import root
    h = np.asarray(h, dtype=float)
    if h.ndim != 2 or h.shape[0] != h.shape[1] or h.shape[0] < 2:
        raise ValueError("h must be a square matrix")
    if not np.all(np.isfinite(h)) or not np.allclose(h, h.T, rtol=0.0, atol=1e-10):
        raise ValueError("h must be finite and symmetric")
    L = h.shape[0]
    U = float(U)
    if not np.isfinite(U):
        raise ValueError("U must be finite")
    if isinstance(n_elec, bool) or not isinstance(n_elec, (int, np.integer)):
        raise ValueError("n_elec must be an integer")
    if n_elec % 2 != 0 or n_elec < 2 or n_elec > 2 * L - 2:
        raise ValueError("n_elec must be even and satisfy 2 <= n_elec <= 2L - 2")
    tol = float(tol)
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("tol must be a positive number")
    n_occ = int(n_elec) // 2
    v0 = np.zeros(L)
    if np.linalg.norm(_mismatch(v0, h, U, n_occ)) < tol:
        return v0
    sol = root(_mismatch, v0, args=(h, U, n_occ), method="hybr", tol=1e-13)
    v_c = np.asarray(sol.x, dtype=float)
    if np.linalg.norm(_mismatch(v_c, h, U, n_occ)) > tol:
        raise RuntimeError("the gLPFET density mapping did not converge")
    return v_c

import numpy as np
from itertools import combinations


def hubbard_fci_reference(h, U, n_elec):
    h = np.asarray(h, dtype=float)
    if h.ndim != 2 or h.shape[0] != h.shape[1]:
        raise ValueError("h must be a square matrix")

    L = h.shape[0]
    if L < 2 or L > 6:
        raise ValueError(
            "the exact reference supports 2 <= L <= 6"
        )
    if (
        not np.all(np.isfinite(h))
        or not np.allclose(
            h, h.T, rtol=0.0, atol=1e-10
        )
    ):
        raise ValueError("h must be finite and symmetric")

    U = float(U)
    if not np.isfinite(U):
        raise ValueError("U must be finite")
    if isinstance(n_elec, bool) or not isinstance(
        n_elec, (int, np.integer)
    ):
        raise ValueError("n_elec must be an integer")
    if (
        n_elec % 2 != 0
        or n_elec < 2
        or n_elec > 2 * L - 2
    ):
        raise ValueError(
            "n_elec must be even and satisfy "
            "2 <= n_elec <= 2L - 2"
        )

    n_spin = int(n_elec) // 2
    states = [
        sum(1 << p for p in occupied)
        for occupied in combinations(
            range(L), n_spin
        )
    ]
    index = {
        state: position
        for position, state in enumerate(states)
    }
    ncfg = len(states)

    h_spin = np.zeros(
        (ncfg, ncfg), dtype=float
    )

    for column, state in enumerate(states):
        for q in range(L):
            if not (state >> q) & 1:
                continue

            h_spin[column, column] += h[q, q]
            without_q = state ^ (1 << q)

            sign_q = (
                -1.0
                if (
                    state & ((1 << q) - 1)
                ).bit_count() % 2
                else 1.0
            )

            for p in range(L):
                if (
                    p == q
                    or (without_q >> p) & 1
                    or h[p, q] == 0.0
                ):
                    continue

                sign_p = (
                    -1.0
                    if (
                        without_q & ((1 << p) - 1)
                    ).bit_count() % 2
                    else 1.0
                )

                row_state = without_q | (1 << p)
                row = index[row_state]

                h_spin[row, column] += (
                    sign_q * sign_p * h[p, q]
                )

    identity = np.eye(ncfg)
    h_fci = (
        np.kron(h_spin, identity)
        + np.kron(identity, h_spin)
    )

    interaction = np.empty(
        ncfg * ncfg, dtype=float
    )
    basis_occupations = np.empty(
        (ncfg * ncfg, L), dtype=float
    )

    for ia, alpha in enumerate(states):
        for ib, beta in enumerate(states):
            position = ia * ncfg + ib

            interaction[position] = (
                U * (alpha & beta).bit_count()
            )

            for site in range(L):
                basis_occupations[
                    position, site
                ] = (
                    ((alpha >> site) & 1)
                    + ((beta >> site) & 1)
                )

    h_fci[np.diag_indices_from(h_fci)] += (
        interaction
    )

    levels, vectors = np.linalg.eigh(h_fci)
    if levels[1] - levels[0] < 1e-10:
        raise ValueError(
            "degenerate FCI ground state: "
            "density undefined"
        )

    coefficients = vectors[:, 0].reshape(
        ncfg, ncfg
    )
    probabilities = coefficients ** 2
    rdm = np.zeros((L, L), dtype=float)

    spin_density_matrices = (
        coefficients @ coefficients.T,
        coefficients.T @ coefficients,
    )

    for rho in spin_density_matrices:
        for column, state in enumerate(states):
            for q in range(L):
                if not (state >> q) & 1:
                    continue

                rdm[q, q] += rho[column, column]
                without_q = state ^ (1 << q)

                sign_q = (
                    -1.0
                    if (
                        state & ((1 << q) - 1)
                    ).bit_count() % 2
                    else 1.0
                )

                for p in range(L):
                    if (
                        p == q
                        or (without_q >> p) & 1
                    ):
                        continue

                    sign_p = (
                        -1.0
                        if (
                            without_q
                            & ((1 << p) - 1)
                        ).bit_count() % 2
                        else 1.0
                    )

                    row_state = (
                        without_q | (1 << p)
                    )
                    row = index[row_state]

                    rdm[p, q] += (
                        sign_q
                        * sign_p
                        * rho[row, column]
                    )

    double_occupations = np.zeros(
        L, dtype=float
    )

    for ia, alpha in enumerate(states):
        for ib, beta in enumerate(states):
            shared = alpha & beta
            weight = probabilities[ia, ib]

            for site in range(L):
                if (shared >> site) & 1:
                    double_occupations[site] += (
                        weight
                    )

    energy = float(levels[0])
    reconstructed_energy = float(
        np.sum(h * rdm.T)
        + U * np.sum(double_occupations)
    )

    if not np.isclose(
        reconstructed_energy,
        energy,
        rtol=0.0,
        atol=1e-9,
    ):
        raise RuntimeError(
            "the reduced densities do not "
            "reconstruct the FCI energy"
        )

    ground = vectors[:, 0]
    transition_density = (
        vectors[:, 1:].T
        @ (
            basis_occupations
            * ground[:, None]
        )
    )

    inverse_gaps = (
        1.0 / (levels[0] - levels[1:])
    )
    response = (
        2.0
        * transition_density.T
        @ (
            transition_density
            * inverse_gaps[:, None]
        )
    )
    response = 0.5 * (
        response + response.T
    )

    return np.concatenate(
        (
            [energy],
            rdm.ravel(),
            double_occupations,
            response.ravel(),
        )
    )

def glpfet_impurity_potential(L, t, v_ext, U, n_elec, site):
    import numpy as np

    if isinstance(site, bool) or not isinstance(site, (int, np.integer)):
        raise ValueError("site must be an integer")

    h = ring_one_body(L, t, v_ext)

    if site < 0 or site >= h.shape[0]:
        raise ValueError("site index out of range")

    v_c = solve_correlation_potential(h, U, n_elec)

    r = density_mismatch(h, U, v_c, n_elec)
    if np.linalg.norm(r) > 1e-8:
        raise RuntimeError(
            "the converged potential does not satisfy the density constraints"
        )

    gamma = gks_density_matrix(h, U, v_c, n_elec)

    fci = np.asarray(
        hubbard_fci_reference(h, U, n_elec),
        dtype=float,
    )
    L_ref = h.shape[0]
    expected_size = 1 + 2 * L_ref * L_ref + L_ref

    if fci.shape != (expected_size,) or not np.all(np.isfinite(fci)):
        raise RuntimeError(
            "the exact FCI benchmark is not finite or has the wrong shape"
        )

    rdm_end = 1 + L_ref * L_ref
    double_end = rdm_end + L_ref

    fci_rdm = fci[1:rdm_end].reshape(L_ref, L_ref)
    fci_double = fci[rdm_end:double_end]
    fci_response = fci[double_end:].reshape(L_ref, L_ref)

    if not np.allclose(
        fci_rdm,
        fci_rdm.T,
        rtol=0.0,
        atol=1e-10,
    ):
        raise RuntimeError(
            "the exact one-particle density matrix is not symmetric"
        )

    occupations = np.linalg.eigvalsh(fci_rdm)
    if (
        np.min(occupations) < -1e-10
        or np.max(occupations) > 2.0 + 1e-10
    ):
        raise RuntimeError(
            "the exact natural occupations are not physical"
        )

    fci_density = np.diag(fci_rdm)
    density_is_physical = (
        np.all(fci_density >= -1e-10)
        and np.all(fci_density <= 2.0 + 1e-10)
    )
    if (
        not density_is_physical
        or abs(np.sum(fci_density) - n_elec) > 1e-8
    ):
        raise RuntimeError(
            "the exact FCI density is not physically normalized"
        )

    lower_double = np.maximum(0.0, fci_density - 1.0)
    upper_double = 0.5 * fci_density
    if (
        np.any(fci_double < lower_double - 1e-10)
        or np.any(fci_double > upper_double + 1e-10)
    ):
        raise RuntimeError(
            "the exact local double occupations are not physical"
        )

    if not np.allclose(
        fci_response,
        fci_response.T,
        rtol=0.0,
        atol=1e-10,
    ):
        raise RuntimeError(
            "the exact static density response is not symmetric"
        )

    if np.max(np.abs(np.sum(fci_response, axis=0))) > 1e-8:
        raise RuntimeError(
            "the exact static density response violates its gauge null mode"
        )

    if np.max(np.linalg.eigvalsh(fci_response)) > 1e-9:
        raise RuntimeError(
            "the exact static density response is not negative semidefinite"
        )

    rebuilt_energy = float(
        np.sum(h * fci_rdm.T) + U * np.sum(fci_double)
    )
    if not np.isclose(
        rebuilt_energy,
        fci[0],
        rtol=0.0,
        atol=1e-9,
    ):
        raise RuntimeError(
            "the exact reduced densities do not reconstruct the FCI energy"
        )

    b = bath_orbital(gamma, int(site))
    mu = float(impurity_chemical_potential(b, v_c))

    h_cl = cluster_hamiltonian(
        h,
        U,
        gamma,
        int(site),
    )
    n_cl = cluster_site_density(h_cl, mu)

    if abs(n_cl - 2.0 * gamma[int(site), int(site)]) > 1e-8:
        raise RuntimeError(
            "cluster and reference densities disagree at the requested site"
        )

    return mu
SCICODE_GOLD_EOF
