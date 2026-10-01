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
from scipy.special import gammainc, gamma


def _cartesian_components(l: int) -> list:
    """Cartesian exponent triples of angular momentum l (x, y, z order for l = 1)."""
    return [(lx, ly, l - lx - ly) for lx in range(l, -1, -1) for ly in range(l - lx, -1, -1)]


def _double_factorial(n: int) -> float:
    return 1.0 if n <= 0 else float(np.prod(np.arange(n, 0, -2)))


def _boys(n_max: int, t: "np.ndarray") -> "np.ndarray":
    """Boys functions F_0..F_n_max at the points t (shape (n_max + 1,) + t.shape)."""
    t = np.asarray(t, dtype=float)
    small = t < 1e-10
    t_safe = np.where(small, 1.0, t)
    out = np.empty((n_max + 1,) + t.shape)
    for n in range(n_max + 1):
        exact = gamma(n + 0.5) * gammainc(n + 0.5, t_safe) / (2.0 * t_safe ** (n + 0.5))
        out[n] = np.where(small, 1.0 / (2 * n + 1) - t / (2 * n + 3), exact)
    return out


def _hermite_expansion(i: int, j: int, x_ab: float, a: "np.ndarray", b: "np.ndarray") -> list:
    """Hermite expansion coefficients E^{ij}_t, t = 0..i+j, of a 1-D Gaussian product."""
    p = a + b
    x_pa = -b / p * x_ab
    x_pb = a / p * x_ab
    cache = {(0, 0, 0): np.exp(-a * b / p * x_ab * x_ab)}

    def _coef(ii, jj, t):
        if ii < 0 or jj < 0 or t < 0 or t > ii + jj:
            return 0.0
        key = (ii, jj, t)
        if key not in cache:
            if ii > 0:
                cache[key] = _coef(ii - 1, jj, t - 1) / (2 * p) + x_pa * _coef(ii - 1, jj, t) + (t + 1) * _coef(ii - 1, jj, t + 1)
            else:
                cache[key] = _coef(ii, jj - 1, t - 1) / (2 * p) + x_pb * _coef(ii, jj - 1, t) + (t + 1) * _coef(ii, jj - 1, t + 1)
        return cache[key]

    return [_coef(i, j, t) * np.ones_like(p) for t in range(i + j + 1)]


def _hermite_coulomb(l_max: int, alpha: "np.ndarray", r: "np.ndarray") -> dict:
    """Hermite Coulomb integrals R_{tuv}, t + u + v <= l_max, for separations r (shape (3,) + batch)."""
    x, y, z = r
    boys = _boys(l_max, alpha * (x * x + y * y + z * z))
    cache = {}

    def _rec(t, u, v, n):
        if t < 0 or u < 0 or v < 0:
            return 0.0
        key = (t, u, v, n)
        if key not in cache:
            if t == u == v == 0:
                cache[key] = (-2.0 * alpha) ** n * boys[n]
            elif t > 0:
                cache[key] = (t - 1) * _rec(t - 2, u, v, n + 1) + x * _rec(t - 1, u, v, n + 1)
            elif u > 0:
                cache[key] = (u - 1) * _rec(t, u - 2, v, n + 1) + y * _rec(t, u - 1, v, n + 1)
            else:
                cache[key] = (v - 1) * _rec(t, u, v - 2, n + 1) + z * _rec(t, u, v - 1, n + 1)
        return cache[key]

    return {(t, u, v): _rec(t, u, v, 0)
            for t in range(l_max + 1) for u in range(l_max + 1 - t) for v in range(l_max + 1 - t - u)}


def _basis_functions(coords: "np.ndarray", shells: list) -> list:
    """Contracted Cartesian functions (centre, exponent triple, exponents, normalized weights)."""
    functions = []
    for shell in shells:
        atom, l, exponents, coefficients = shell
        exponents = np.asarray(exponents, dtype=float).ravel()
        coefficients = np.asarray(coefficients, dtype=float).ravel()
        if l not in (0, 1, 2):
            raise ValueError("only s, p and d shells (l = 0, 1, 2) are supported")
        if not (0 <= atom < len(coords)):
            raise ValueError("shell refers to a missing atom")
        if exponents.size == 0 or exponents.size != coefficients.size:
            raise ValueError("exponents and coefficients must be non-empty and of equal length")
        if np.any(exponents <= 0.0):
            raise ValueError("exponents must be positive")
        for lmn in _cartesian_components(l):
            dfac = np.prod([_double_factorial(2 * k - 1) for k in lmn])
            prim_norm = (2 * exponents / np.pi) ** 0.75 * (4 * exponents) ** (l / 2) / np.sqrt(dfac)
            weights = coefficients * prim_norm
            pair = exponents[:, None] + exponents[None, :]
            self_overlap = np.sum(weights[:, None] * weights[None, :] * dfac * (np.pi / pair) ** 1.5 / (2 * pair) ** l)
            functions.append((coords[atom], lmn, exponents, weights / np.sqrt(self_overlap)))
    return functions


def _primitive_pairs(f_a: tuple, f_b: tuple) -> tuple:
    """Flattened primitive-pair data and Hermite coefficients for two contracted functions."""
    centre_a, lmn_a, exp_a, w_a = f_a
    centre_b, lmn_b, exp_b, w_b = f_b
    a = np.repeat(exp_a, exp_b.size)
    b = np.tile(exp_b, exp_a.size)
    weight = np.repeat(w_a, exp_b.size) * np.tile(w_b, exp_a.size)
    p = a + b
    centre_p = (a * centre_a[:, None] + b * centre_b[:, None]) / p
    hermite = [{jj: _hermite_expansion(lmn_a[k], jj, centre_a[k] - centre_b[k], a, b)
                for jj in range(max(0, lmn_b[k] - 2), lmn_b[k] + 3)} for k in range(3)]
    return b, weight, p, centre_p, hermite


def compute_ao_integrals(charges: "np.ndarray", coords: "np.ndarray", shells: list) -> tuple:
    charges = np.asarray(charges, dtype=float).ravel()
    coords = np.asarray(coords, dtype=float).reshape(-1, 3)
    if charges.size != coords.shape[0]:
        raise ValueError("charges and coords must describe the same atoms")
    functions = _basis_functions(coords, shells)
    n_bf = len(functions)
    overlap = np.zeros((n_bf, n_bf))
    kinetic = np.zeros((n_bf, n_bf))
    attraction = np.zeros((n_bf, n_bf))
    densities = {}
    for i in range(n_bf):
        for j in range(i + 1):
            b, weight, p, centre_p, hermite = _primitive_pairs(functions[i], functions[j])
            lmn_i, lmn_j = functions[i][1], functions[j][1]

            def _shifted_overlap(shift):
                value = (np.pi / p) ** 1.5
                for k in range(3):
                    jj = lmn_j[k] + shift[k]
                    if jj < 0:
                        return np.zeros_like(p)
                    value = value * hermite[k][jj][0]
                return value

            s_ij = _shifted_overlap((0, 0, 0))
            overlap[i, j] = overlap[j, i] = np.sum(weight * s_ij)
            t_ij = b * (2 * sum(lmn_j) + 3) * s_ij
            for k in range(3):
                up = [0, 0, 0]
                up[k] = 2
                down = [0, 0, 0]
                down[k] = -2
                t_ij = t_ij - 2 * b * b * _shifted_overlap(up) - 0.5 * lmn_j[k] * (lmn_j[k] - 1) * _shifted_overlap(down)
            kinetic[i, j] = kinetic[j, i] = np.sum(weight * t_ij)
            terms = []
            for t in range(lmn_i[0] + lmn_j[0] + 1):
                for u in range(lmn_i[1] + lmn_j[1] + 1):
                    for v in range(lmn_i[2] + lmn_j[2] + 1):
                        terms.append(((t, u, v), weight * hermite[0][lmn_j[0]][t] * hermite[1][lmn_j[1]][u] * hermite[2][lmn_j[2]][v]))
            l_pair = sum(lmn_i) + sum(lmn_j)
            v_ij = 0.0
            for z_c, centre_c in zip(charges, coords):
                r_tuv = _hermite_coulomb(l_pair, p, centre_p - centre_c[:, None])
                v_ij -= z_c * sum(np.sum(c * 2 * np.pi / p * r_tuv[key]) for key, c in terms)
            attraction[i, j] = attraction[j, i] = v_ij
            densities[(i, j)] = (p, centre_p, terms, l_pair)
    eri = np.zeros((n_bf, n_bf, n_bf, n_bf))
    keys = sorted(densities)
    for n1, bra in enumerate(keys):
        p, centre_p, terms_p, l_p = densities[bra]
        for ket in keys[:n1 + 1]:
            q, centre_q, terms_q, l_q = densities[ket]
            pp, qq = p[:, None], q[None, :]
            alpha = pp * qq / (pp + qq)
            r_tuv = _hermite_coulomb(l_p + l_q, alpha, centre_p[:, :, None] - centre_q[:, None, :])
            prefactor = 2 * np.pi ** 2.5 / (pp * qq * np.sqrt(pp + qq))
            value = 0.0
            for (t1, u1, v1), c1 in terms_p:
                for (t2, u2, v2), c2 in terms_q:
                    value += (-1) ** (t2 + u2 + v2) * np.sum(c1[:, None] * c2[None, :] * prefactor * r_tuv[(t1 + t2, u1 + u2, v1 + v2)])
            (i, j), (k, l) = bra, ket
            for idx in ((i, j, k, l), (j, i, k, l), (i, j, l, k), (j, i, l, k),
                        (k, l, i, j), (l, k, i, j), (k, l, j, i), (l, k, j, i)):
                eri[idx] = value
    e_nuc = 0.0
    for a_idx in range(charges.size):
        for b_idx in range(a_idx):
            distance = np.linalg.norm(coords[a_idx] - coords[b_idx])
            if distance == 0.0:
                raise ValueError("two nuclei coincide")
            e_nuc += charges[a_idx] * charges[b_idx] / distance
    return overlap, kinetic + attraction, eri, float(e_nuc)

import numpy as np


def _fock_matrix(core_hamiltonian: "np.ndarray", eri: "np.ndarray", density: "np.ndarray") -> "np.ndarray":
    """Closed-shell Fock matrix for the spin-summed AO density."""
    coulomb = np.einsum('pqrs,rs->pq', eri, density)
    exchange = np.einsum('prqs,rs->pq', eri, density)
    return core_hamiltonian + coulomb - 0.5 * exchange


def run_rhf(overlap: "np.ndarray", core_hamiltonian: "np.ndarray", eri: "np.ndarray", e_nuc: float, n_occ: int) -> tuple:
    overlap = np.asarray(overlap, dtype=float)
    core_hamiltonian = np.asarray(core_hamiltonian, dtype=float)
    eri = np.asarray(eri, dtype=float)
    n_bf = overlap.shape[0]
    if overlap.shape != (n_bf, n_bf) or core_hamiltonian.shape != (n_bf, n_bf) or eri.shape != (n_bf,) * 4:
        raise ValueError("inconsistent integral shapes")
    if not (1 <= int(n_occ) <= n_bf):
        raise ValueError("n_occ must lie in 1..n_bf")
    n_occ = int(n_occ)
    s_vals, s_vecs = np.linalg.eigh(overlap)
    if s_vals[0] <= 0.0:
        raise ValueError("overlap matrix is not positive definite")
    ortho = s_vecs / np.sqrt(s_vals)
    _, c_prime = np.linalg.eigh(ortho.T @ core_hamiltonian @ ortho)
    coeff = ortho @ c_prime
    density = 2.0 * coeff[:, :n_occ] @ coeff[:, :n_occ].T
    focks, errors = [], []
    e_old = None
    for _ in range(500):
        fock = _fock_matrix(core_hamiltonian, eri, density)
        e_el = 0.5 * np.sum(density * (core_hamiltonian + fock))
        focks.append(fock)
        errors.append(ortho.T @ (fock @ density @ overlap - overlap @ density @ fock) @ ortho)
        focks, errors = focks[-8:], errors[-8:]
        n = len(focks)
        b_mat = -np.ones((n + 1, n + 1))
        b_mat[n, n] = 0.0
        b_mat[:n, :n] = [[np.sum(e1 * e2) for e2 in errors] for e1 in errors]
        rhs = np.zeros(n + 1)
        rhs[n] = -1.0
        weights = np.linalg.lstsq(b_mat, rhs, rcond=None)[0][:n]
        extrapolated = sum(w * f for w, f in zip(weights, focks))
        _, c_prime = np.linalg.eigh(ortho.T @ extrapolated @ ortho)
        coeff = ortho @ c_prime
        new_density = 2.0 * coeff[:, :n_occ] @ coeff[:, :n_occ].T
        converged = (e_old is not None and abs(e_el - e_old) < 1e-13
                     and np.max(np.abs(new_density - density)) < 1e-11)
        density, e_old = new_density, e_el
        if converged:
            break
    else:
        raise ValueError("SCF iterations did not converge")
    fock = _fock_matrix(core_hamiltonian, eri, density)
    mo_energies, c_prime = np.linalg.eigh(ortho.T @ fock @ ortho)
    mo_coeff = ortho @ c_prime
    density = 2.0 * mo_coeff[:, :n_occ] @ mo_coeff[:, :n_occ].T
    e_hf = 0.5 * np.sum(density * (core_hamiltonian + _fock_matrix(core_hamiltonian, eri, density))) + float(e_nuc)
    return float(e_hf), mo_energies, mo_coeff

import numpy as np


def _ovov_integrals(eri: "np.ndarray", mo_coeff: "np.ndarray", n_occ: int) -> "np.ndarray":
    """(ia|jb) in the canonical molecular-orbital basis, index order [i, a, j, b]."""
    occ, vir = mo_coeff[:, :n_occ], mo_coeff[:, n_occ:]
    return np.einsum('pqrs,pi,qa,rj,sb->iajb', eri, occ, vir, occ, vir, optimize=True)


def _pair_denominators(mo_energies: "np.ndarray", n_occ: int) -> "np.ndarray":
    """e_i + e_j - e_a - e_b with index order [i, a, j, b]."""
    e_occ, e_vir = mo_energies[:n_occ], mo_energies[n_occ:]
    return (e_occ[:, None, None, None] - e_vir[None, :, None, None]
            + e_occ[None, None, :, None] - e_vir[None, None, None, :])


def _check_mp2_inputs(eri: "np.ndarray", mo_coeff: "np.ndarray", mo_energies: "np.ndarray", n_occ: int) -> tuple:
    eri = np.asarray(eri, dtype=float)
    mo_coeff = np.asarray(mo_coeff, dtype=float)
    mo_energies = np.asarray(mo_energies, dtype=float).ravel()
    n_bf, n_mo = mo_coeff.shape
    if eri.shape != (n_bf,) * 4 or mo_energies.size != n_mo:
        raise ValueError("inconsistent array shapes")
    n_occ = int(n_occ)
    if not (1 <= n_occ < n_mo):
        raise ValueError("n_occ must leave at least one virtual orbital")
    if np.max(mo_energies[:n_occ]) >= np.min(mo_energies[n_occ:]):
        raise ValueError("occupied orbital energies must lie below the virtual ones")
    return eri, mo_coeff, mo_energies, n_occ


def mp2_spin_components(eri: "np.ndarray", mo_coeff: "np.ndarray", mo_energies: "np.ndarray", n_occ: int) -> "np.ndarray":
    eri, mo_coeff, mo_energies, n_occ = _check_mp2_inputs(eri, mo_coeff, mo_energies, n_occ)
    ovov = _ovov_integrals(eri, mo_coeff, n_occ)
    denom = _pair_denominators(mo_energies, n_occ)
    e_os = np.sum(ovov * ovov / denom)
    e_ss = np.sum(ovov * (ovov - ovov.transpose(0, 3, 2, 1)) / denom)
    return np.array([e_os, e_ss])

import numpy as np


def unrelaxed_mp2_density(eri: "np.ndarray", mo_coeff: "np.ndarray", mo_energies: "np.ndarray", n_occ: int) -> "np.ndarray":
    eri, mo_coeff, mo_energies, n_occ = _check_mp2_inputs(eri, mo_coeff, mo_energies, n_occ)
    ovov = _ovov_integrals(eri, mo_coeff, n_occ)
    amplitudes = (ovov / _pair_denominators(mo_energies, n_occ)).transpose(0, 2, 1, 3)  # t[i, j, a, b]
    spin_adapted = 2.0 * amplitudes - amplitudes.transpose(0, 1, 3, 2)
    occ_block = -2.0 * np.einsum('ikab,jkab->ij', amplitudes, spin_adapted)
    vir_block = 2.0 * np.einsum('ijac,ijbc->ab', amplitudes, spin_adapted)
    n_mo = mo_coeff.shape[1]
    density = np.zeros((n_mo, n_mo))
    density[:n_occ, :n_occ] = 2.0 * np.eye(n_occ) + 0.5 * (occ_block + occ_block.T)
    density[n_occ:, n_occ:] = 0.5 * (vir_block + vir_block.T)
    return density

import numpy as np


def correlation_indices(density: "np.ndarray", n_electrons: int) -> "np.ndarray":
    density = np.asarray(density, dtype=float)
    if density.ndim != 2 or density.shape[0] != density.shape[1]:
        raise ValueError("density must be a square matrix")
    if not np.allclose(density, density.T, rtol=0.0, atol=1e-10):
        raise ValueError("density must be symmetric")
    if int(n_electrons) != n_electrons or n_electrons <= 0 or int(n_electrons) % 2:
        raise ValueError("n_electrons must be a positive even integer")
    spatial = np.linalg.eigvalsh(0.5 * (density + density.T))
    if np.max(spatial) > 2.0 + 1e-10:
        raise ValueError("a natural occupation exceeds 2")
    spin = np.concatenate([spatial, spatial]) / 2.0
    spin = np.clip(spin[spin > 0.0], 0.0, 1.0)
    fluct = spin * (1.0 - spin)
    i_nd = np.sum(fluct) / n_electrons
    i_t = np.sum(np.sqrt(fluct)) / (2.0 * n_electrons)
    return np.array([i_t - i_nd, i_nd, i_t])

import numpy as np


def correlation_driven_weights(indices: "np.ndarray") -> "np.ndarray":
    indices = np.asarray(indices, dtype=float).ravel()
    if indices.size != 3 or not np.all(np.isfinite(indices)):
        raise ValueError("indices must hold three finite values")
    i_d, i_nd, i_t = indices
    if i_d < 0.0 or i_nd < 0.0 or i_t <= 0.0:
        raise ValueError("indices must be non-negative with a positive total")
    if abs(i_t - (i_d + i_nd)) > 1e-8 * i_t:
        raise ValueError("I_T must equal I_D + I_ND")
    return np.array([1.38 * i_d / i_t, 2.89 * i_nd / i_t])

import numpy as np


def cd_scs_mp2_energy(charges: "np.ndarray", coords: "np.ndarray", shells: list, n_electrons: int) -> float:
    if int(n_electrons) != n_electrons or n_electrons <= 0 or int(n_electrons) % 2:
        raise ValueError("n_electrons must be a positive even integer")
    n_occ = int(n_electrons) // 2
    overlap, core, eri, e_nuc = compute_ao_integrals(charges, coords, shells)
    e_hf, mo_energies, mo_coeff = run_rhf(overlap, core, eri, e_nuc, n_occ)
    e_os, e_ss = mp2_spin_components(eri, mo_coeff, mo_energies, n_occ)
    density = unrelaxed_mp2_density(eri, mo_coeff, mo_energies, n_occ)
    c_os, c_ss = correlation_driven_weights(correlation_indices(density, n_electrons))
    return float(e_hf + c_os * e_os + c_ss * e_ss)

import numpy as np


def stretch_energy(charges: "np.ndarray", coords_initial: "np.ndarray", coords_final: "np.ndarray", shells: list, n_electrons: int) -> float:
    charges = np.asarray(charges, dtype=float).ravel()
    coords_initial = np.asarray(coords_initial, dtype=float)
    coords_final = np.asarray(coords_final, dtype=float)
    if coords_initial.shape != (charges.size, 3) or coords_final.shape != (charges.size, 3):
        raise ValueError("both geometries must have shape (n_atoms, 3)")
    e_initial = cd_scs_mp2_energy(charges, coords_initial, shells, n_electrons)
    e_final = cd_scs_mp2_energy(charges, coords_final, shells, n_electrons)
    return float(e_final - e_initial)
SCICODE_GOLD_EOF
