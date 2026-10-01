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
from scipy.linalg import expm
 
def _antisymmetric(values: "np.ndarray", size: int) -> "np.ndarray":
    values = np.asarray(values)
    generator = np.zeros((size, size), dtype=np.result_type(values, float))
    generator[np.triu_indices(size, 1)] = values
    return generator - generator.T
 
def unrestricted_orbitals(x_params: "np.ndarray", y_params: "np.ndarray", n_levels: int, n_spin_levels: int) -> "np.ndarray":
    m, k = int(n_levels), int(n_spin_levels)
    if m < 1 or not 0 <= k <= m:
        raise ValueError("need n_levels >= 1 and 0 <= n_spin_levels <= n_levels")
    x = np.asarray(x_params).ravel()
    y = np.asarray(y_params).ravel()
    if x.size != m * (m - 1) // 2 or y.size != k * (k - 1) // 2:
        raise ValueError("rotation parameter arrays have the wrong length")
    gen_x = _antisymmetric(x, m)
    gen_y = np.zeros((m, m), dtype=gen_x.dtype)
    if k > 1:
        gen_y[m - k:, m - k:] = _antisymmetric(y, k)
    c_up = expm(gen_x)
    return np.stack([c_up, c_up @ expm(gen_y)])

import numpy as np
 
def seniority_coefficients(hopping: "np.ndarray", onsite_U: float, orbitals: "np.ndarray") -> "np.ndarray":
    h = np.asarray(hopping)
    c = np.asarray(orbitals)
    if h.ndim != 2 or h.shape[0] != h.shape[1] or not np.allclose(h, h.T, atol=1e-12):
        raise ValueError("hopping must be a square symmetric matrix")
    m = h.shape[0]
    if c.shape != (2, m, m):
        raise ValueError("orbitals must have shape (2, M, M)")
    c_up, c_dn = c[0], c[1]
    h_up = c_up.T @ h @ c_up
    h_dn = c_dn.T @ h @ c_dn
    sq_up, sq_dn, mixed = c_up * c_up, c_dn * c_dn, c_up * c_dn
    j_uu = onsite_U * sq_up.T @ sq_up
    j_dd = onsite_U * sq_dn.T @ sq_dn
    j_ud = onsite_U * sq_up.T @ sq_dn
    k_uu, k_dd = j_uu, j_dd
    k_ud = onsite_U * mixed.T @ mixed
    pair = onsite_U * mixed.T @ mixed
    off = 1.0 - np.eye(m)
    eps = 0.5 * (np.diag(h_up) + np.diag(h_dn))
    b_single = np.diag(h_up) - np.diag(h_dn)
    w = 0.5 * j_uu + 0.5 * j_dd + j_ud - 0.5 * k_uu - 0.5 * k_dd
    b_pair = 0.5 * j_uu + 0.5 * j_dd - j_ud - 0.5 * k_uu - 0.5 * k_dd
    x = 0.5 * (j_uu - j_dd) - 0.5 * (k_uu - k_dd) - 0.5 * (j_ud - j_ud.T)
    return np.stack([np.diag(eps), np.diag(b_single), pair, w * off, b_pair * off, k_ud * off, x * off])

import itertools
import numpy as np
 
def seniority_sector_basis(n_pairing_levels: int, n_pairs: int, n_spin_levels: int, n_up: int) -> "np.ndarray":
    p, npair, k, nup = int(n_pairing_levels), int(n_pairs), int(n_spin_levels), int(n_up)
    if min(p, npair, k, nup) < 0 or npair > p or nup > k or p + k == 0:
        raise ValueError("invalid sector specification")
    rows = []
    for occupied in itertools.combinations(range(p), npair):
        for ups in itertools.combinations(range(k), nup):
            row = np.zeros(p + k, dtype=int)
            row[list(occupied)] = 2
            spins = -np.ones(k, dtype=int)
            spins[list(ups)] = 1
            row[p:] = spins
            rows.append(row)
    return np.array(rows, dtype=int).reshape(-1, p + k)

import numpy as np
 
def seci_hamiltonian(coefficients: "np.ndarray", sector_basis: "np.ndarray") -> "np.ndarray":
    coef = np.asarray(coefficients)
    states = np.asarray(sector_basis)
    if states.ndim != 2 or coef.shape != (7, states.shape[1], states.shape[1]):
        raise ValueError("coefficients must have shape (7, M, M) matching the basis width")
    if not np.all(np.isin(states, [2, 0, 1, -1])):
        raise ValueError("basis entries must be 2, 0, 1 or -1")
    eps, b_single = np.diag(coef[0]), np.diag(coef[1])
    pair, w, b_pair, k_ud, x = coef[2], coef[3], coef[4], coef[5], coef[6]
    charge = np.where(states == 2, 2.0, np.where(states == 0, 0.0, 1.0))
    spin = np.where(states == 1, 0.5, np.where(states == -1, -0.5, 0.0))
    diagonal = (charge @ eps + spin @ b_single
                + 0.25 * np.einsum("ip,pq,iq->i", charge, w, charge)
                + np.einsum("ip,pq,iq->i", spin, b_pair, spin)
                + np.einsum("ip,pq,iq->i", charge, x, spin)
                + (states == 2) @ np.diag(pair))
    ham = np.diag(diagonal).astype(np.result_type(coef, float))
    index = {tuple(row): i for i, row in enumerate(states)}
    m = states.shape[1]
    for j, row in enumerate(states):
        for p in range(m):
            for q in range(m):
                if p == q:
                    continue
                if row[p] == 0 and row[q] == 2:
                    target = row.copy()
                    target[p], target[q] = 2, 0
                    if tuple(target) not in index:
                        raise ValueError("pair transfer leaves the basis")
                    ham[index[tuple(target)], j] += pair[p, q]
                if row[p] == -1 and row[q] == 1:
                    target = row.copy()
                    target[p], target[q] = 1, -1
                    if tuple(target) not in index:
                        raise ValueError("spin exchange leaves the basis")
                    ham[index[tuple(target)], j] -= k_ud[p, q]
    return ham

import numpy as np
from scipy.linalg import expm, expm_frechet
 
def _sector_moves(states: "np.ndarray") -> tuple:
    index = {tuple(row): i for i, row in enumerate(states)}
    pair_moves, spin_moves = [], []
    m = states.shape[1]
    for j, row in enumerate(states):
        for p in range(m):
            for q in range(m):
                if p == q:
                    continue
                if row[p] == 0 and row[q] == 2:
                    target = row.copy()
                    target[p], target[q] = 2, 0
                    pair_moves.append((index[tuple(target)], j, p, q))
                if row[p] == -1 and row[q] == 1:
                    target = row.copy()
                    target[p], target[q] = 1, -1
                    spin_moves.append((index[tuple(target)], j, p, q))
    return np.array(pair_moves, dtype=int).reshape(-1, 4), np.array(spin_moves, dtype=int).reshape(-1, 4)
 
def _energy_gradient_core(theta: "np.ndarray", hopping: "np.ndarray", onsite_U: float, states: "np.ndarray", moves: tuple, n_spin_levels: int) -> "np.ndarray":
    m, k = hopping.shape[0], int(n_spin_levels)
    nx = m * (m - 1) // 2
    orb = unrestricted_orbitals(theta[:nx], theta[nx:], m, k)
    c_up, c_dn = orb[0], orb[1]
    gen_x = _antisymmetric(theta[:nx], m)
    gen_y = np.zeros((m, m))
    if k > 1:
        gen_y[m - k:, m - k:] = _antisymmetric(theta[nx:], k)
    coef = seniority_coefficients(hopping, onsite_U, orb)
    ham = seci_hamiltonian(coef, states)
    values, vectors = np.linalg.eigh(ham)
    if len(values) > 1 and values[1] - values[0] < 1e-9:
        raise ValueError("degenerate SECI ground state; gradient undefined")
    psi = vectors[:, 0]
    prob = psi ** 2
    charge = np.where(states == 2, 2.0, np.where(states == 0, 0.0, 1.0))
    spin = np.where(states == 1, 0.5, np.where(states == -1, -0.5, 0.0))
    off = 1.0 - np.eye(m)
    # expectation values multiplying each coefficient (Hellmann-Feynman weights)
    w_eps = prob @ charge
    w_b = prob @ spin
    w_pair = np.diag(prob @ (states == 2))
    pair_moves, spin_moves = moves
    if len(pair_moves):
        np.add.at(w_pair, (pair_moves[:, 2], pair_moves[:, 3]), psi[pair_moves[:, 0]] * psi[pair_moves[:, 1]])
    w_w = 0.25 * np.einsum("i,ip,iq->pq", prob, charge, charge) * off
    w_bz = np.einsum("i,ip,iq->pq", prob, spin, spin) * off
    w_x = np.einsum("i,ip,iq->pq", prob, charge, spin) * off
    w_k = np.zeros((m, m))
    if len(spin_moves):
        np.add.at(w_k, (spin_moves[:, 2], spin_moves[:, 3]), -psi[spin_moves[:, 0]] * psi[spin_moves[:, 1]])
    w_k *= off
    # same-spin Coulomb and exchange cancel for the Hubbard interaction, leaving
    # E = sum (w_eps/2 + w_b) h_up_pp + (w_eps/2 - w_b) h_dn_pp + U [sum Om_pq (uu^T dd)_pq + sum Lam_pq (P^T P)_pq]
    omega = (w_w - w_bz - 0.5 * (w_x - w_x.T)) * off
    lam = w_pair + w_k
    sq_up, sq_dn, mixed = c_up * c_up, c_dn * c_dn, c_up * c_dn
    grad_up = 2.0 * hopping @ c_up * (0.5 * w_eps + w_b)[None, :] + onsite_U * (2.0 * c_up * (sq_dn @ omega.T) + c_dn * (mixed @ (lam + lam.T)))
    grad_dn = 2.0 * hopping @ c_dn * (0.5 * w_eps - w_b)[None, :] + onsite_U * (2.0 * c_dn * (sq_up @ omega) + c_up * (mixed @ (lam + lam.T)))
    exp_y = expm(gen_y)
    gx = expm_frechet(gen_x.T, grad_up + grad_dn @ exp_y.T, compute_expm=False)
    grad_x = (gx - gx.T)[np.triu_indices(m, 1)]
    if k > 1:
        gy = expm_frechet(gen_y.T, expm(gen_x).T @ grad_dn, compute_expm=False)[m - k:, m - k:]
        grad_y = (gy - gy.T)[np.triu_indices(k, 1)]
    else:
        grad_y = np.zeros(0)
    return np.concatenate([[values[0]], grad_x, grad_y])
 
def _sector_setup(hopping: "np.ndarray", n_pairs: int, n_spin_levels: int) -> tuple:
    h = np.asarray(hopping, dtype=float)
    if h.ndim != 2 or h.shape[0] != h.shape[1]:
        raise ValueError("hopping must be square")
    m, k, npair = h.shape[0], int(n_spin_levels), int(n_pairs)
    if k % 2 or not 0 <= k <= m or not 0 <= npair <= m - k:
        raise ValueError("invalid seniority sector")
    states = seniority_sector_basis(m - k, npair, k, k // 2)
    return h, states, _sector_moves(states)
 
def seci_energy_gradient(x_params: "np.ndarray", y_params: "np.ndarray", hopping: "np.ndarray", onsite_U: float, n_pairs: int, n_spin_levels: int) -> "np.ndarray":
    h, states, moves = _sector_setup(hopping, n_pairs, n_spin_levels)
    m, k = h.shape[0], int(n_spin_levels)
    x = np.asarray(x_params, dtype=float).ravel()
    y = np.asarray(y_params, dtype=float).ravel()
    if x.size != m * (m - 1) // 2 or y.size != k * (k - 1) // 2:
        raise ValueError("rotation parameter arrays have the wrong length")
    return _energy_gradient_core(np.concatenate([x, y]), h, float(onsite_U), states, moves, k)

import numpy as np
from scipy.optimize import minimize
 
def optimize_seci_orbitals(hopping: "np.ndarray", onsite_U: float, n_pairs: int, n_spin_levels: int, seed: int = 20260927) -> float:
    h = np.asarray(hopping, dtype=float)
    if h.ndim != 2 or h.shape[0] != h.shape[1]:
        raise ValueError("hopping must be square")
    m, k = h.shape[0], int(n_spin_levels)
    if k % 2 or not 0 <= k <= m or not 0 <= int(n_pairs) <= m - k:
        raise ValueError("invalid seniority sector")
    n_x, n_y = m * (m - 1) // 2, k * (k - 1) // 2
 
    def _objective(theta):
        # energy and gradient from the Step-5 function; at a degenerate ground state the
        # gradient is undefined, so evaluate at a nearby point
        shift = np.cos(np.arange(theta.size) + 1.0)
        for scale in (0.0, 1e-7, 1e-6, 1e-5, 1e-4):
            point = theta + scale * shift
            try:
                out = seci_energy_gradient(point[:n_x], point[n_x:], h, onsite_U, n_pairs, k)
                return out[0], out[1:]
            except ValueError:
                continue
        raise ValueError("degenerate SECI ground state throughout the neighbourhood")
 
    if n_x + n_y == 0:
        return float(_objective(np.zeros(0))[0])
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(12):
        run = minimize(_objective, rng.normal(size=n_x + n_y), jac=True, method="L-BFGS-B",
                       options={"maxiter": 3000, "ftol": 1e-13, "gtol": 1e-8})
        if best is None or run.fun < best.fun:
            best = run
    polish = minimize(_objective, best.x, jac=True, method="BFGS", options={"gtol": 1e-10, "maxiter": 2000})
    return float(min(polish.fun, best.fun))

import itertools
import numpy as np
from scipy.sparse import coo_matrix, diags, identity, kron
from scipy.sparse.linalg import eigsh
 
def _spin_strings(m: int, n: int) -> tuple:
    strings = [sum(1 << i for i in occ) for occ in itertools.combinations(range(m), n)]
    return strings, {s: i for i, s in enumerate(strings)}
 
def _one_body_spin(h: "np.ndarray", strings: list, index: dict) -> "np.ndarray":
    m = h.shape[0]
    rows, cols, vals = [], [], []
    for j, s in enumerate(strings):
        for q in range(m):
            if not s >> q & 1:
                continue
            removed = s ^ (1 << q)
            sign_q = (-1) ** bin(s & ((1 << q) - 1)).count("1")
            for p in range(m):
                if h[p, q] == 0.0 or (removed >> p & 1):
                    continue
                sign_p = (-1) ** bin(removed & ((1 << p) - 1)).count("1")
                rows.append(index[removed | (1 << p)])
                cols.append(j)
                vals.append(sign_q * sign_p * h[p, q])
    return coo_matrix((vals, (rows, cols)), shape=(len(strings), len(strings))).toarray()
 
def hubbard_fci_energy(hopping: "np.ndarray", onsite_U: float, n_up: int, n_down: int) -> float:
    h = np.asarray(hopping, dtype=float)
    if h.ndim != 2 or h.shape[0] != h.shape[1] or not np.allclose(h, h.T, atol=1e-12):
        raise ValueError("hopping must be a square symmetric matrix")
    m = h.shape[0]
    if not (0 <= n_up <= m and 0 <= n_down <= m):
        raise ValueError("electron counts must lie in [0, M]")
    up, up_index = _spin_strings(m, int(n_up))
    dn, dn_index = _spin_strings(m, int(n_down))
    h_up = _one_body_spin(h, up, up_index)
    h_dn = _one_body_spin(h, dn, dn_index)
    occ_up = np.array([[s >> i & 1 for i in range(m)] for s in up], dtype=float)
    occ_dn = np.array([[s >> i & 1 for i in range(m)] for s in dn], dtype=float)
    double = onsite_U * (occ_up @ occ_dn.T).ravel()
    dim = len(up) * len(dn)
    ham = kron(coo_matrix(h_up), identity(len(dn))) + kron(identity(len(up)), coo_matrix(h_dn)) + diags(double)
    if dim <= 400:
        return float(np.linalg.eigvalsh(ham.toarray())[0])
    return float(eigsh(ham.tocsr(), k=1, which="SA", tol=1e-13, maxiter=100000)[0][0])

import numpy as np
 
def seci_correlation_fraction(n_sites: int, hopping_t: float, onsite_U: float, n_electrons: int, n_spin_levels: int, seed: int = 20260927) -> float:
    m, ne, k = int(n_sites), int(n_electrons), int(n_spin_levels)
    if m < 3 or not hopping_t > 0 or not onsite_U > 0 or ne % 2 or not 0 < ne < 2 * m:
        raise ValueError("invalid ring or electron number")
    if k % 2 or not 0 <= k <= m or (ne - k) % 2 or not 0 <= (ne - k) // 2 <= m - k:
        raise ValueError("invalid seniority sector")
    hopping = np.zeros((m, m))
    for i in range(m):
        hopping[i, (i + 1) % m] = hopping[(i + 1) % m, i] = -float(hopping_t)
    n_occ = ne // 2
    levels, vectors = np.linalg.eigh(hopping)
    if n_occ < m and levels[n_occ] - levels[n_occ - 1] <= 1e-9:
        raise ValueError("open-shell reference determinant")
    density = np.sum(vectors[:, :n_occ] ** 2, axis=1)
    e_ref = 2.0 * np.sum(levels[:n_occ]) + onsite_U * np.sum(density ** 2)
    e_seci = optimize_seci_orbitals(hopping, onsite_U, (ne - k) // 2, k, seed)
    e_fci = hubbard_fci_energy(hopping, onsite_U, n_occ, n_occ)
    return float((e_ref - e_seci) / (e_ref - e_fci))
SCICODE_GOLD_EOF
