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


def fock_space_operators(n_modes):
    if isinstance(n_modes, (bool, np.bool_)) or not isinstance(n_modes, (int, np.integer)):
        raise ValueError("n_modes must be an integer")
    if n_modes < 1 or n_modes > 16:
        raise ValueError("n_modes must be between 1 and 16")
    dim = 2 ** n_modes
    c_dag = np.zeros((n_modes, dim, dim))
    c = np.zeros((n_modes, dim, dim))
    for i in range(n_modes):
        for s in range(dim):
            if not (s >> i) & 1:
                new_s = s | (1 << i)
                sign = 0
                for j in range(i):
                    if (s >> j) & 1:
                        sign += 1
                phase = (-1) ** sign
                c_dag[i, new_s, s] = phase
                c[i, s, new_s] = phase
    return c_dag, c

import numpy as np


def anderson_impurity_hamiltonian(U, J, delta, mu, eps_bath, V_bath, n_orb, n_bath_per_orb):
    if isinstance(n_orb, (bool, np.bool_)) or not isinstance(n_orb, (int, np.integer)) or n_orb < 1:
        raise ValueError("n_orb must be a positive integer")
    if isinstance(n_bath_per_orb, (bool, np.bool_)) or not isinstance(n_bath_per_orb, (int, np.integer)) or n_bath_per_orb < 1:
        raise ValueError("n_bath_per_orb must be a positive integer")
    eps_bath = np.asarray(eps_bath)
    V_bath = np.asarray(V_bath)
    expected_shape = (n_orb, n_bath_per_orb)
    if eps_bath.shape != expected_shape or V_bath.shape != expected_shape:
        raise ValueError("bath arrays have incompatible shapes")
    try:
        finite_scalars = all(np.asarray(value).ndim == 0 and bool(np.isfinite(value)) for value in (U, J, delta, mu))
    except (TypeError, ValueError):
        finite_scalars = False
    if not finite_scalars:
        raise ValueError("U, J, delta, and mu must be finite scalars")
    n_spin = 2
    n_sites = 1 + n_bath_per_orb
    n_modes = n_orb * n_spin * n_sites
    c_dag, c = fock_space_operators(n_modes)
    dim = 2 ** n_modes
    H = np.zeros((dim, dim))

    def mode_idx(orb, spin, site):
        return orb * n_spin * n_sites + spin * n_sites + site

    def n_op(orb, spin, site):
        idx = mode_idx(orb, spin, site)
        return c_dag[idx] @ c[idx]

    for orb in range(n_orb):
        eps_orb = (-1)**orb * delta / 2.0 - mu
        for spin in range(n_spin):
            H += eps_orb * n_op(orb, spin, 0)
    for orb in range(n_orb):
        for k in range(n_bath_per_orb):
            for spin in range(n_spin):
                H += eps_bath[orb, k] * n_op(orb, spin, k + 1)
    for orb in range(n_orb):
        for k in range(n_bath_per_orb):
            for spin in range(n_spin):
                imp = mode_idx(orb, spin, 0)
                bath = mode_idx(orb, spin, k + 1)
                H += V_bath[orb, k] * (c_dag[imp] @ c[bath] + c_dag[bath] @ c[imp])
    Up = U - 2.0 * J
    for orb in range(n_orb):
        H += U * n_op(orb, 0, 0) @ n_op(orb, 1, 0)
    for a in range(n_orb):
        for b in range(a + 1, n_orb):
            for s1 in range(n_spin):
                for s2 in range(n_spin):
                    if s1 == s2:
                        H += (Up - J) * n_op(a, s1, 0) @ n_op(b, s2, 0)
                    else:
                        H += Up * n_op(a, s1, 0) @ n_op(b, s2, 0)
    for a in range(n_orb):
        for b in range(n_orb):
            if a != b:
                ia_up = mode_idx(a, 0, 0)
                ia_dn = mode_idx(a, 1, 0)
                ib_up = mode_idx(b, 0, 0)
                ib_dn = mode_idx(b, 1, 0)
                H += -J * (c_dag[ia_up] @ c[ia_dn] @ c_dag[ib_dn] @ c[ib_up])
                H += J * (c_dag[ia_up] @ c_dag[ia_dn] @ c[ib_dn] @ c[ib_up])
    return H

import numpy as np


def thermal_density_matrix(H, beta):
    if not isinstance(H, np.ndarray) or H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError("H must be a two-dimensional square numpy array")
    if not np.allclose(H, H.T):
        raise ValueError("H must be symmetric")
    try:
        valid_beta = np.asarray(beta).ndim == 0 and bool(np.isfinite(beta)) and beta > 0
    except (TypeError, ValueError):
        valid_beta = False
    if not valid_beta:
        raise ValueError("beta must be positive and finite")
    ev, evec = np.linalg.eigh(H)
    w = np.exp(-beta * (ev - ev[0]))
    w = w / w.sum()
    return (evec * w) @ evec.T

import numpy as np


def retained_configuration_weights(rho, n_orb, n_bath_per_orb):
    if isinstance(n_orb, (bool, np.bool_)) or not isinstance(n_orb, (int, np.integer)) or n_orb < 1:
        raise ValueError("n_orb must be a positive integer")
    if isinstance(n_bath_per_orb, (bool, np.bool_)) or not isinstance(n_bath_per_orb, (int, np.integer)) or n_bath_per_orb < 1:
        raise ValueError("n_bath_per_orb must be a positive integer")
    n_spin = 2
    n_sites = 1 + n_bath_per_orb
    n_modes = n_orb * n_spin * n_sites
    dim = 2 ** n_modes
    rho = np.asarray(rho)
    if rho.ndim != 2 or rho.shape != (dim, dim):
        raise ValueError("rho must be a square full-Fock-space density matrix")
    c_dag, c = fock_space_operators(n_modes)
    eye = np.eye(dim)

    def mode_idx(orb, spin, site):
        return orb * n_spin * n_sites + spin * n_sites + site

    def n_op(orb, spin):
        i = mode_idx(orb, spin, 0)
        return c_dag[i] @ c[i]

    def proj(orb, occ_up, occ_dn):
        nu = n_op(orb, 0)
        nd = n_op(orb, 1)
        pu = nu if occ_up else (eye - nu)
        pd = nd if occ_dn else (eye - nd)
        return pu @ pd

    p1 = proj(0, 1, 1) @ proj(1, 0, 0)
    p2 = proj(0, 1, 0) @ proj(1, 0, 1)
    p3 = proj(0, 0, 1) @ proj(1, 1, 0)
    p4 = proj(0, 0, 0) @ proj(1, 1, 1)
    out = np.array([
        float(np.trace(rho @ p1).real),
        float(np.trace(rho @ p2).real),
        float(np.trace(rho @ p3).real),
        float(np.trace(rho @ p4).real),
    ])
    return out

import numpy as np


def spin_exchange_correlator(rho, n_orb, n_bath_per_orb):
    if isinstance(n_orb, (bool, np.bool_)) or not isinstance(n_orb, (int, np.integer)) or n_orb < 1:
        raise ValueError("n_orb must be a positive integer")
    if isinstance(n_bath_per_orb, (bool, np.bool_)) or not isinstance(n_bath_per_orb, (int, np.integer)) or n_bath_per_orb < 1:
        raise ValueError("n_bath_per_orb must be a positive integer")
    n_spin = 2
    n_sites = 1 + n_bath_per_orb
    n_modes = n_orb * n_spin * n_sites
    dim = 2 ** n_modes
    rho = np.asarray(rho)
    if rho.ndim != 2 or rho.shape != (dim, dim):
        raise ValueError("rho must be a square full-Fock-space density matrix")
    c_dag, c = fock_space_operators(n_modes)

    def mode_idx(orb, spin, site):
        return orb * n_spin * n_sites + spin * n_sites + site

    s_plus_0 = c_dag[mode_idx(0, 0, 0)] @ c[mode_idx(0, 1, 0)]
    s_minus_1 = c_dag[mode_idx(1, 1, 0)] @ c[mode_idx(1, 0, 0)]
    z = np.trace(rho @ (s_plus_0 @ s_minus_1))
    return float(abs(z))

import numpy as np


def pair_transfer_coherence(rho, n_orb, n_bath_per_orb):
    if isinstance(n_orb, (bool, np.bool_)) or not isinstance(n_orb, (int, np.integer)) or n_orb < 1:
        raise ValueError("n_orb must be a positive integer")
    if isinstance(n_bath_per_orb, (bool, np.bool_)) or not isinstance(n_bath_per_orb, (int, np.integer)) or n_bath_per_orb < 1:
        raise ValueError("n_bath_per_orb must be a positive integer")
    rho = np.asarray(rho)
    n_spin = 2
    n_sites = 1 + n_bath_per_orb
    n_modes = n_orb * n_spin * n_sites
    dim = 2 ** n_modes
    if rho.ndim != 2 or rho.shape != (dim, dim):
        raise ValueError("rho must be a square density matrix with the Fock-space dimension")
    c_dag, c = fock_space_operators(n_modes)

    def mode_idx(orb, spin, site):
        return orb * n_spin * n_sites + spin * n_sites + site

    transfer = (c_dag[mode_idx(0, 0, 0)] @ c_dag[mode_idx(0, 1, 0)]
                @ c[mode_idx(1, 1, 0)] @ c[mode_idx(1, 0, 0)])
    value = np.trace(rho @ transfer)
    return float(abs(value))

import numpy as np


def orbital_concurrence(weights, z_magnitude):
    try:
        w = np.asarray(weights, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("weights must contain numeric values") from exc
    if w.ndim != 1 or w.size != 4:
        raise ValueError("weights must be a one-dimensional array with four entries")
    if np.any(w < 0.0) or not np.all(np.isfinite(w)):
        raise ValueError("weights must be finite and nonnegative")
    try:
        valid_z = np.asarray(z_magnitude).ndim == 0 and bool(np.isfinite(z_magnitude)) and z_magnitude >= 0.0
    except (TypeError, ValueError):
        valid_z = False
    if not valid_z:
        raise ValueError("z_magnitude must be a finite nonnegative scalar")
    u_plus = w[0]
    u_minus = w[3]
    return float(2.0 * max(0.0, z_magnitude - np.sqrt(u_plus * u_minus)))

import numpy as np


def run_pipeline(U, J, delta, V_1, V_2, beta):
    for name, value in (("U", U), ("J", J), ("delta", delta),
                        ("V_1", V_1), ("V_2", V_2), ("beta", beta)):
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if beta <= 0.0:
        raise ValueError("beta must be positive")

    n_orb = 2
    n_bath_per_orb = 1
    n_modes = n_orb * 2 * (1 + n_bath_per_orb)
    mu = (3.0 * U - 5.0 * J) / 2.0
    eps_bath = np.zeros((n_orb, n_bath_per_orb))
    V_bath = np.array([[float(V_1)], [float(V_2)]])

    # The mode operators fix the Fock-space dimension every later object lives
    # in, so they are built first and their shape is what the Hamiltonian is
    # checked against.
    c_dag, _c = fock_space_operators(n_modes)
    dim = c_dag.shape[1]

    H = anderson_impurity_hamiltonian(U, J, delta, mu, eps_bath, V_bath,
                                            n_orb, n_bath_per_orb)
    if H.shape != (dim, dim):
        raise ValueError("Hamiltonian does not match the Fock-space dimension")
    rho = thermal_density_matrix(H, beta)

    weights = retained_configuration_weights(rho, n_orb, n_bath_per_orb)
    z_magnitude = spin_exchange_correlator(rho, n_orb, n_bath_per_orb)

    # Reducing each orbital to a two-level system restricts the available
    # operations to those that respect local particle number, and that
    # restriction dephases coherences between different local charge sectors.
    # The pair-transfer coherence links local charge (2, 0) to (0, 2), so the
    # dephasing map sends it to zero; applying that map explicitly is what turns
    # the general two-branch concurrence into the single branch evaluated below.
    raw_pair = pair_transfer_coherence(rho, n_orb, n_bath_per_orb)
    if not np.isfinite(raw_pair) or raw_pair < 0.0:
        raise ValueError("pair-transfer coherence must be a finite magnitude")
    # The coherence is a magnitude drawn from a normalized thermal state, so it
    # cannot exceed the geometric mean of the two populations it connects. That
    # bound is checked before the dephasing map discards the value, which is the
    # only way an error in this step can be caught: multiplying by zero first
    # would let any wrong number, including a NaN, pass through unnoticed.
    u_plus, u_minus = float(weights[0]), float(weights[3])
    if raw_pair > np.sqrt(u_plus * u_minus) + 1e-12:
        raise ValueError("pair-transfer coherence exceeds its Cauchy-Schwarz bound")
    dephased_pair = 0.0 * raw_pair
    w_1, w_2 = float(weights[1]), float(weights[2])
    pair_branch = 2.0 * max(0.0, dephased_pair - np.sqrt(w_1 * w_2))

    spin_branch = float(orbital_concurrence(weights, z_magnitude))
    return float(max(spin_branch, pair_branch))
SCICODE_GOLD_EOF
