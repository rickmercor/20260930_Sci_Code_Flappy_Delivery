#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

import functools
import numpy as np

import functools

import numpy as np


def _lattice():
    """Primitive vectors and the buckled two-site Cr basis of bulk CrSBr."""
    a, b, c, buckling = 3.50, 4.76, 7.96, 1.96
    a1 = np.array([a, 0.0, 0.0])
    a2 = np.array([0.0, b, 0.0])
    a3 = np.array([0.0, 0.0, c])
    basis = [np.array([0.0, 0.0, 0.5 * buckling]),
             np.array([0.5 * a, 0.5 * b, -0.5 * buckling])]
    return a1, a2, a3, basis


@functools.lru_cache(maxsize=None)
def _bulk_bonds(nmax=3):
    """Every bulk Cr-Cr bond, ranked by length over the 3D structure (interlayer
    included). Depends on nmax alone, so the cache cannot make the ranking call-order
    dependent."""
    a1, a2, a3, basis = _lattice()
    raw = []
    for a in range(2):
        for b in range(2):
            for n1 in range(-nmax, nmax + 1):
                for n2 in range(-nmax, nmax + 1):
                    for n3 in (-1, 0, 1):
                        r_ij = n1 * a1 + n2 * a2 + n3 * a3
                        r = r_ij + basis[b] - basis[a]
                        d = float(np.linalg.norm(r))
                        if d > 1e-9:
                            raw.append((d, a, b, n3, r_ij, r))
    raw.sort(key=lambda t: t[0])
    radii = []
    for t in raw:
        if not radii or t[0] - radii[-1] > 1e-3:
            radii.append(t[0])
    out = []
    for (d, a, b, n3, r_ij, r) in raw:
        shell = next(k + 1 for k, rad in enumerate(radii) if abs(d - rad) < 1e-3)
        out.append((shell, a, b, n3, r_ij, r, d))
    return out


def _shell_bonds(shell, a, b):
    """In-plane bonds of one bulk shell from sublattice a to b, angle-sorted."""
    sel = [t for t in _bulk_bonds()
           if t[0] == shell and t[1] == a and t[2] == b and t[3] == 0]
    sel.sort(key=lambda t: np.arctan2(t[5][1], t[5][0]) % (2.0 * np.pi))
    return sel


def neighbour_bond_vectors(shell: int, a: int, b: int) -> "np.ndarray":
    """Assemble the in-plane bond vectors of one neighbour shell of monolayer CrSBr."""
    bonds = [t[5] for t in _shell_bonds(shell, a, b)]
    return np.array(bonds, dtype=float).reshape(-1, 3)

import functools

import numpy as np


def _lattice():
    """Primitive vectors and the buckled two-site Cr basis of bulk CrSBr."""
    a, b, c, buckling = 3.50, 4.76, 7.96, 1.96
    a1 = np.array([a, 0.0, 0.0])
    a2 = np.array([0.0, b, 0.0])
    a3 = np.array([0.0, 0.0, c])
    basis = [np.array([0.0, 0.0, 0.5 * buckling]),
             np.array([0.5 * a, 0.5 * b, -0.5 * buckling])]
    return a1, a2, a3, basis


@functools.lru_cache(maxsize=None)
def _bulk_bonds(nmax=3):
    """Every bulk Cr-Cr bond, ranked by length over the 3D structure (interlayer
    included). Depends on nmax alone, so the cache cannot make the ranking call-order
    dependent."""
    a1, a2, a3, basis = _lattice()
    raw = []
    for a in range(2):
        for b in range(2):
            for n1 in range(-nmax, nmax + 1):
                for n2 in range(-nmax, nmax + 1):
                    for n3 in (-1, 0, 1):
                        r_ij = n1 * a1 + n2 * a2 + n3 * a3
                        r = r_ij + basis[b] - basis[a]
                        d = float(np.linalg.norm(r))
                        if d > 1e-9:
                            raw.append((d, a, b, n3, r_ij, r))
    raw.sort(key=lambda t: t[0])
    radii = []
    for t in raw:
        if not radii or t[0] - radii[-1] > 1e-3:
            radii.append(t[0])
    out = []
    for (d, a, b, n3, r_ij, r) in raw:
        shell = next(k + 1 for k, rad in enumerate(radii) if abs(d - rad) < 1e-3)
        out.append((shell, a, b, n3, r_ij, r, d))
    return out


def _shell_bonds(shell, a, b):
    """In-plane bonds of one bulk shell from sublattice a to b, angle-sorted."""
    sel = [t for t in _bulk_bonds()
           if t[0] == shell and t[1] == a and t[2] == b and t[3] == 0]
    sel.sort(key=lambda t: np.arctan2(t[5][1], t[5][0]) % (2.0 * np.pi))
    return sel


def _gamma(shell, a, b, q):
    """Structure factor of one shell, sum_i exp(i q . r_i) over its bonds."""
    return sum(np.exp(1j * float(np.dot(q, t[4][:2])))
               for t in _shell_bonds(shell, a, b))


def exchange_structure_factor(shell: int, a: int, b: int, qx: float, qy: float) -> "np.ndarray":
    """Sum the Bloch phases of one neighbour shell."""
    g = _gamma(shell, a, b, np.array([qx, qy]))
    return np.array([g.real, g.imag])

import functools

import numpy as np


def _exchange_constants():
    """Isotropic exchange by BULK shell index, meV; the interlayer sixth is absent."""
    return {1: 1.90, 2: 3.38, 3: 1.67, 4: 0.09, 5: 0.09, 7: -0.37, 8: 0.29}


def _lattice():
    """Primitive vectors and the buckled two-site Cr basis of bulk CrSBr."""
    a, b, c, buckling = 3.50, 4.76, 7.96, 1.96
    a1 = np.array([a, 0.0, 0.0])
    a2 = np.array([0.0, b, 0.0])
    a3 = np.array([0.0, 0.0, c])
    basis = [np.array([0.0, 0.0, 0.5 * buckling]),
             np.array([0.5 * a, 0.5 * b, -0.5 * buckling])]
    return a1, a2, a3, basis


@functools.lru_cache(maxsize=None)
def _bulk_bonds(nmax=3):
    """Every bulk Cr-Cr bond, ranked by length over the 3D structure (interlayer
    included). Depends on nmax alone, so the cache cannot make the ranking call-order
    dependent."""
    a1, a2, a3, basis = _lattice()
    raw = []
    for a in range(2):
        for b in range(2):
            for n1 in range(-nmax, nmax + 1):
                for n2 in range(-nmax, nmax + 1):
                    for n3 in (-1, 0, 1):
                        r_ij = n1 * a1 + n2 * a2 + n3 * a3
                        r = r_ij + basis[b] - basis[a]
                        d = float(np.linalg.norm(r))
                        if d > 1e-9:
                            raw.append((d, a, b, n3, r_ij, r))
    raw.sort(key=lambda t: t[0])
    radii = []
    for t in raw:
        if not radii or t[0] - radii[-1] > 1e-3:
            radii.append(t[0])
    out = []
    for (d, a, b, n3, r_ij, r) in raw:
        shell = next(k + 1 for k, rad in enumerate(radii) if abs(d - rad) < 1e-3)
        out.append((shell, a, b, n3, r_ij, r, d))
    return out


def _shell_bonds(shell, a, b):
    """In-plane bonds of one bulk shell from sublattice a to b, angle-sorted."""
    sel = [t for t in _bulk_bonds()
           if t[0] == shell and t[1] == a and t[2] == b and t[3] == 0]
    sel.sort(key=lambda t: np.arctan2(t[5][1], t[5][0]) % (2.0 * np.pi))
    return sel


def _gamma(shell, a, b, q):
    """Structure factor of one shell, sum_i exp(i q . r_i) over its bonds."""
    return sum(np.exp(1j * float(np.dot(q, t[4][:2])))
               for t in _shell_bonds(shell, a, b))


def _J_scalar(a, b, q):
    """J_ab(q): isotropic exchange summed over every shell that carries one."""
    js = _exchange_constants()
    return sum(js[shell] * _gamma(shell, a, b, q) for shell in sorted(js))


def exchange_at_q(a: int, b: int, qx: float, qy: float) -> "np.ndarray":
    """Sum every neighbour shell that carries an exchange constant."""
    g = _J_scalar(a, b, np.array([qx, qy]))
    return np.array([g.real, g.imag])

import functools

import numpy as np


def _exchange_constants():
    """Isotropic exchange by BULK shell index, meV; the interlayer sixth is absent."""
    return {1: 1.90, 2: 3.38, 3: 1.67, 4: 0.09, 5: 0.09, 7: -0.37, 8: 0.29}


def _lattice():
    """Primitive vectors and the buckled two-site Cr basis of bulk CrSBr."""
    a, b, c, buckling = 3.50, 4.76, 7.96, 1.96
    a1 = np.array([a, 0.0, 0.0])
    a2 = np.array([0.0, b, 0.0])
    a3 = np.array([0.0, 0.0, c])
    basis = [np.array([0.0, 0.0, 0.5 * buckling]),
             np.array([0.5 * a, 0.5 * b, -0.5 * buckling])]
    return a1, a2, a3, basis


@functools.lru_cache(maxsize=None)
def _bulk_bonds(nmax=3):
    """Every bulk Cr-Cr bond, ranked by length over the 3D structure (interlayer
    included). Depends on nmax alone, so the cache cannot make the ranking call-order
    dependent."""
    a1, a2, a3, basis = _lattice()
    raw = []
    for a in range(2):
        for b in range(2):
            for n1 in range(-nmax, nmax + 1):
                for n2 in range(-nmax, nmax + 1):
                    for n3 in (-1, 0, 1):
                        r_ij = n1 * a1 + n2 * a2 + n3 * a3
                        r = r_ij + basis[b] - basis[a]
                        d = float(np.linalg.norm(r))
                        if d > 1e-9:
                            raw.append((d, a, b, n3, r_ij, r))
    raw.sort(key=lambda t: t[0])
    radii = []
    for t in raw:
        if not radii or t[0] - radii[-1] > 1e-3:
            radii.append(t[0])
    out = []
    for (d, a, b, n3, r_ij, r) in raw:
        shell = next(k + 1 for k, rad in enumerate(radii) if abs(d - rad) < 1e-3)
        out.append((shell, a, b, n3, r_ij, r, d))
    return out


def _shell_bonds(shell, a, b):
    """In-plane bonds of one bulk shell from sublattice a to b, angle-sorted."""
    sel = [t for t in _bulk_bonds()
           if t[0] == shell and t[1] == a and t[2] == b and t[3] == 0]
    sel.sort(key=lambda t: np.arctan2(t[5][1], t[5][0]) % (2.0 * np.pi))
    return sel


def _gamma(shell, a, b, q):
    """Structure factor of one shell, sum_i exp(i q . r_i) over its bonds."""
    return sum(np.exp(1j * float(np.dot(q, t[4][:2])))
               for t in _shell_bonds(shell, a, b))


def _J_scalar(a, b, q):
    """J_ab(q): isotropic exchange summed over every shell that carries one."""
    js = _exchange_constants()
    return sum(js[shell] * _gamma(shell, a, b, q) for shell in sorted(js))


def _J_matrix(q):
    """The 2x2 matrix of J_ab(q)."""
    return np.array([[_J_scalar(a, b, q) for b in range(2)] for a in range(2)])


def _mol_field(sz):
    """C^RPA_aa = sum_c <Sz_c> v^T J'_ac(0) v, the on-site molecular field."""
    J0 = _J_matrix(np.zeros(2))
    return float(np.real(sz * J0[0].sum()))


def molecular_field(sz: float) -> float:
    """Compute the on-site molecular field of the RPA Hamiltonian."""
    return _mol_field(sz)

import numpy as np


def _uv():
    """Local frame, [u]^mu = R^{mu x} + i R^{mu y} and [v]^mu = R^{mu z}, R in SO(3).

    Moment along the easy axis b = crystal y, so v = y_hat; the right-handed triad
    (e1, e2, v) = (z_hat, x_hat, y_hat) gives u = e1 + i e2. Only u's phase is free.
    """
    return np.array([1j, 0.0, 1.0]), np.array([0.0, 1.0, 0.0])


def _K_tensor():
    """Anisotropy tensor diag(K^xx, K^yy, K^zz) in meV, the RPA+CD fitted triple."""
    return np.diag([0.066, 0.087, 0.0]).astype(complex)


def _anisotropy(sz):
    """(A^RPA, B^RPA, C^RPA) for the single-ion anisotropy."""
    u, v = _uv()
    K = _K_tensor()
    return (float(np.real(sz * (u @ K @ np.conj(u)))),
            float(np.real(sz * (u @ K @ u))),
            float(np.real(2.0 * sz * (v @ K @ v))))


def anisotropy_contractions(sz: float) -> "np.ndarray":
    """Contract the single-ion anisotropy tensor into the local frame."""
    a, b, c = _anisotropy(sz)
    # The sign of the anomalous element follows the phase convention chosen for u and
    # carries no physical content, so this step reports its modulus.
    return np.array([a, abs(b), c])

def _spin():
    """The Cr3+ spin of CrSBr."""
    return 1.5


def _callen(sz, phi):
    """Callen renormalisation of the single-ion anisotropy."""
    S = _spin()
    return 1.0 - sz * (1.0 + 2.0 * phi) / (2.0 * S ** 2)


def callen_factor(sz: float, phi: float) -> float:
    """Compute the Callen-decoupling renormalisation of the single-ion anisotropy."""
    return _callen(sz, phi)

import functools

import numpy as np


def _spin():
    """The Cr3+ spin of CrSBr."""
    return 1.5


def _exchange_constants():
    """Isotropic exchange by BULK shell index, meV; the interlayer sixth is absent."""
    return {1: 1.90, 2: 3.38, 3: 1.67, 4: 0.09, 5: 0.09, 7: -0.37, 8: 0.29}


def _lattice():
    """Primitive vectors and the buckled two-site Cr basis of bulk CrSBr."""
    a, b, c, buckling = 3.50, 4.76, 7.96, 1.96
    a1 = np.array([a, 0.0, 0.0])
    a2 = np.array([0.0, b, 0.0])
    a3 = np.array([0.0, 0.0, c])
    basis = [np.array([0.0, 0.0, 0.5 * buckling]),
             np.array([0.5 * a, 0.5 * b, -0.5 * buckling])]
    return a1, a2, a3, basis


@functools.lru_cache(maxsize=None)
def _bulk_bonds(nmax=3):
    """Every bulk Cr-Cr bond, ranked by length over the 3D structure (interlayer
    included). Depends on nmax alone, so the cache cannot make the ranking call-order
    dependent."""
    a1, a2, a3, basis = _lattice()
    raw = []
    for a in range(2):
        for b in range(2):
            for n1 in range(-nmax, nmax + 1):
                for n2 in range(-nmax, nmax + 1):
                    for n3 in (-1, 0, 1):
                        r_ij = n1 * a1 + n2 * a2 + n3 * a3
                        r = r_ij + basis[b] - basis[a]
                        d = float(np.linalg.norm(r))
                        if d > 1e-9:
                            raw.append((d, a, b, n3, r_ij, r))
    raw.sort(key=lambda t: t[0])
    radii = []
    for t in raw:
        if not radii or t[0] - radii[-1] > 1e-3:
            radii.append(t[0])
    out = []
    for (d, a, b, n3, r_ij, r) in raw:
        shell = next(k + 1 for k, rad in enumerate(radii) if abs(d - rad) < 1e-3)
        out.append((shell, a, b, n3, r_ij, r, d))
    return out


def _shell_bonds(shell, a, b):
    """In-plane bonds of one bulk shell from sublattice a to b, angle-sorted."""
    sel = [t for t in _bulk_bonds()
           if t[0] == shell and t[1] == a and t[2] == b and t[3] == 0]
    sel.sort(key=lambda t: np.arctan2(t[5][1], t[5][0]) % (2.0 * np.pi))
    return sel


def _gamma(shell, a, b, q):
    """Structure factor of one shell, sum_i exp(i q . r_i) over its bonds."""
    return sum(np.exp(1j * float(np.dot(q, t[4][:2])))
               for t in _shell_bonds(shell, a, b))


def _J_scalar(a, b, q):
    """J_ab(q): isotropic exchange summed over every shell that carries one."""
    js = _exchange_constants()
    return sum(js[shell] * _gamma(shell, a, b, q) for shell in sorted(js))


def _J_matrix(q):
    """The 2x2 matrix of J_ab(q)."""
    return np.array([[_J_scalar(a, b, q) for b in range(2)] for a in range(2)])


def _uv():
    """Local frame, [u]^mu = R^{mu x} + i R^{mu y} and [v]^mu = R^{mu z}, R in SO(3).

    Moment along the easy axis b = crystal y, so v = y_hat; the right-handed triad
    (e1, e2, v) = (z_hat, x_hat, y_hat) gives u = e1 + i e2. Only u's phase is free.
    """
    return np.array([1j, 0.0, 1.0]), np.array([0.0, 1.0, 0.0])


def _K_tensor():
    """Anisotropy tensor diag(K^xx, K^yy, K^zz) in meV, the RPA+CD fitted triple."""
    return np.diag([0.066, 0.087, 0.0]).astype(complex)


def _anisotropy(sz):
    """(A^RPA, B^RPA, C^RPA) for the single-ion anisotropy."""
    u, v = _uv()
    K = _K_tensor()
    return (float(np.real(sz * (u @ K @ np.conj(u)))),
            float(np.real(sz * (u @ K @ u))),
            float(np.real(2.0 * sz * (v @ K @ v))))


def _callen(sz, phi):
    """Callen renormalisation of the single-ion anisotropy."""
    S = _spin()
    return 1.0 - sz * (1.0 + 2.0 * phi) / (2.0 * S ** 2)


def _mol_field(sz):
    """C^RPA_aa = sum_c <Sz_c> v^T J'_ac(0) v, the on-site molecular field."""
    J0 = _J_matrix(np.zeros(2))
    return float(np.real(sz * J0[0].sum()))


def _blocks(Jq, Jm, sz, phi):
    """The 4x4 Hamiltonian from the exchange matrices at +q and -q.

    The lower-right block carries A_ab(-q)*: the inter-sublattice Bloch phase is not
    even in q, so it differs from A_ab(q)*.
    """
    u, v = _uv()
    A = np.zeros((2, 2), dtype=complex)
    B = np.zeros((2, 2), dtype=complex)
    Am = np.zeros((2, 2), dtype=complex)
    for a in range(2):
        for b in range(2):
            A[a, b] = 0.5 * sz * (u @ (Jq[a, b] * np.eye(3)) @ np.conj(u))
            B[a, b] = 0.5 * sz * (u @ (Jq[a, b] * np.eye(3)) @ u)
            Am[a, b] = 0.5 * sz * (u @ (Jm[a, b] * np.eye(3)) @ np.conj(u))
    C = np.eye(2) * _mol_field(sz)
    Ak, Bk, Ck = _anisotropy(sz)
    ups = _callen(sz, phi)
    H = np.zeros((4, 4), dtype=complex)
    H[:2, :2] = -A + C + np.eye(2) * (-Ak + Ck) * ups
    H[:2, 2:] = -B - np.eye(2) * Bk * ups
    H[2:, :2] = B.conj().T + np.eye(2) * np.conj(Bk) * ups
    H[2:, 2:] = np.conj(Am) - C + np.eye(2) * (Ak - Ck) * ups
    return H


def _hamiltonian(q, sz, phi):
    """H^RPA_q + H^CD_K at wavevector q."""
    return _blocks(_J_matrix(q), _J_matrix(-q), sz, phi)


def _energies(q, sz, phi):
    """The four magnon energies at q, ascending."""
    return np.sort(np.linalg.eigvals(_hamiltonian(q, sz, phi)).real)


def magnon_energies(qx: float, qy: float, sz: float, phi: float) -> "np.ndarray":
    """Assemble the magnon Hamiltonian and diagonalise it."""
    return _energies(np.array([qx, qy]), sz, phi)

import functools

import numpy as np


def _spin():
    """The Cr3+ spin of CrSBr."""
    return 1.5


def _boltzmann():
    """Boltzmann's constant in meV/K."""
    return 0.08617333262


def _exchange_constants():
    """Isotropic exchange by BULK shell index, meV; the interlayer sixth is absent."""
    return {1: 1.90, 2: 3.38, 3: 1.67, 4: 0.09, 5: 0.09, 7: -0.37, 8: 0.29}


def _lattice():
    """Primitive vectors and the buckled two-site Cr basis of bulk CrSBr."""
    a, b, c, buckling = 3.50, 4.76, 7.96, 1.96
    a1 = np.array([a, 0.0, 0.0])
    a2 = np.array([0.0, b, 0.0])
    a3 = np.array([0.0, 0.0, c])
    basis = [np.array([0.0, 0.0, 0.5 * buckling]),
             np.array([0.5 * a, 0.5 * b, -0.5 * buckling])]
    return a1, a2, a3, basis


@functools.lru_cache(maxsize=None)
def _bulk_bonds(nmax=3):
    """Every bulk Cr-Cr bond, ranked by length over the 3D structure (interlayer
    included). Depends on nmax alone, so the cache cannot make the ranking call-order
    dependent."""
    a1, a2, a3, basis = _lattice()
    raw = []
    for a in range(2):
        for b in range(2):
            for n1 in range(-nmax, nmax + 1):
                for n2 in range(-nmax, nmax + 1):
                    for n3 in (-1, 0, 1):
                        r_ij = n1 * a1 + n2 * a2 + n3 * a3
                        r = r_ij + basis[b] - basis[a]
                        d = float(np.linalg.norm(r))
                        if d > 1e-9:
                            raw.append((d, a, b, n3, r_ij, r))
    raw.sort(key=lambda t: t[0])
    radii = []
    for t in raw:
        if not radii or t[0] - radii[-1] > 1e-3:
            radii.append(t[0])
    out = []
    for (d, a, b, n3, r_ij, r) in raw:
        shell = next(k + 1 for k, rad in enumerate(radii) if abs(d - rad) < 1e-3)
        out.append((shell, a, b, n3, r_ij, r, d))
    return out


def _shell_bonds(shell, a, b):
    """In-plane bonds of one bulk shell from sublattice a to b, angle-sorted."""
    sel = [t for t in _bulk_bonds()
           if t[0] == shell and t[1] == a and t[2] == b and t[3] == 0]
    sel.sort(key=lambda t: np.arctan2(t[5][1], t[5][0]) % (2.0 * np.pi))
    return sel


def _gamma(shell, a, b, q):
    """Structure factor of one shell, sum_i exp(i q . r_i) over its bonds."""
    return sum(np.exp(1j * float(np.dot(q, t[4][:2])))
               for t in _shell_bonds(shell, a, b))


def _J_scalar(a, b, q):
    """J_ab(q): isotropic exchange summed over every shell that carries one."""
    js = _exchange_constants()
    return sum(js[shell] * _gamma(shell, a, b, q) for shell in sorted(js))


def _J_matrix(q):
    """The 2x2 matrix of J_ab(q)."""
    return np.array([[_J_scalar(a, b, q) for b in range(2)] for a in range(2)])


def _uv():
    """Local frame, [u]^mu = R^{mu x} + i R^{mu y} and [v]^mu = R^{mu z}, R in SO(3).

    Moment along the easy axis b = crystal y, so v = y_hat; the right-handed triad
    (e1, e2, v) = (z_hat, x_hat, y_hat) gives u = e1 + i e2. Only u's phase is free.
    """
    return np.array([1j, 0.0, 1.0]), np.array([0.0, 1.0, 0.0])


def _K_tensor():
    """Anisotropy tensor diag(K^xx, K^yy, K^zz) in meV, the RPA+CD fitted triple."""
    return np.diag([0.066, 0.087, 0.0]).astype(complex)


def _anisotropy(sz):
    """(A^RPA, B^RPA, C^RPA) for the single-ion anisotropy."""
    u, v = _uv()
    K = _K_tensor()
    return (float(np.real(sz * (u @ K @ np.conj(u)))),
            float(np.real(sz * (u @ K @ u))),
            float(np.real(2.0 * sz * (v @ K @ v))))


def _callen(sz, phi):
    """Callen renormalisation of the single-ion anisotropy."""
    S = _spin()
    return 1.0 - sz * (1.0 + 2.0 * phi) / (2.0 * S ** 2)


def _mol_field(sz):
    """C^RPA_aa = sum_c <Sz_c> v^T J'_ac(0) v, the on-site molecular field."""
    J0 = _J_matrix(np.zeros(2))
    return float(np.real(sz * J0[0].sum()))


def _mesh_q(n):
    """Gamma-centred n x n sampling of the first Brillouin zone."""
    a1, a2, a3, basis = _lattice()
    b1 = np.array([2.0 * np.pi / a1[0], 0.0])
    b2 = np.array([0.0, 2.0 * np.pi / a2[1]])
    return np.array([i / n * b1 + j / n * b2 for i in range(n) for j in range(n)])


@functools.lru_cache(maxsize=None)
def _mesh_J(n):
    """J_ab(q) on the whole mesh; depends on the mesh size alone, so built once."""
    js = _exchange_constants()
    qs = _mesh_q(n)
    J = np.zeros((len(qs), 2, 2), dtype=complex)
    for (shell, a, b, n3, r_ij, r, d) in _bulk_bonds():
        if n3 != 0 or shell not in js:
            continue
        J[:, a, b] += js[shell] * np.exp(1j * (qs @ r_ij[:2]))
    return J


def _bose(w, T):
    """Bose factor, guarded against overflow in the exponent."""
    x = w / (_boltzmann() * T)
    out = np.empty_like(x)
    big = np.abs(x) > 700.0
    out[big] = np.where(x[big] > 0.0, 0.0, -1.0)
    ok = ~big
    out[ok] = 1.0 / np.expm1(x[ok])
    return out


def _mesh_H(n, sz, phi):
    """The Hamiltonian on the whole mesh, without a Python loop over q.

    Real couplings on an inversion-closed bond set give J_ab(-q) = J_ab(q)*, and
    u^T u = 0 for an isotropic tensor, so the exchange adds nothing anomalous.
    """
    Js = _mesh_J(n)
    eye = np.eye(2)
    A = sz * Js                        # (sz/2) u^T (J I3) u*, with u . u* = 2
    Am = sz * np.conj(Js)              # the same at -q
    C = _mol_field(sz)
    Ak, Bk, Ck = _anisotropy(sz)
    ups = _callen(sz, phi)
    diag = C + (-Ak + Ck) * ups
    H = np.zeros((len(Js), 4, 4), dtype=complex)
    H[:, :2, :2] = -A + eye * diag
    H[:, :2, 2:] = -eye * (Bk * ups)
    H[:, 2:, :2] = eye * (np.conj(Bk) * ups)
    H[:, 2:, 2:] = np.conj(Am) - eye * diag
    return H


def _occupation(T, sz, phi, n):
    """Phi_0, averaged over the zone, summed over all four eigenvalues."""
    Hs = _mesh_H(n, sz, phi)
    w, U = np.linalg.eig(Hs)
    order = np.argsort(w.real, axis=1)
    w = np.take_along_axis(w, order, axis=1).real
    U = np.take_along_axis(U, order[:, None, :], axis=2)
    Ui = np.linalg.inv(U)
    nb = _bose(w, T)
    contrib = U * nb[:, None, :] * np.swapaxes(Ui, 1, 2)
    return float(np.real(contrib.sum(axis=2))[:, 0].mean())


def magnon_occupation(T: float, sz: float, phi: float, n: int) -> float:
    """Average the magnon occupation over the Brillouin zone."""
    return _occupation(T, sz, phi, n)

def _spin():
    """The Cr3+ spin of CrSBr."""
    return 1.5


def _magnetisation(phi):
    """Callen's expression for the sublattice magnetisation."""
    S = _spin()
    e = 2.0 * S + 1.0
    p = max(float(phi), 1e-15)
    return float(((S - p) * (1.0 + p) ** e + (S + 1.0 + p) * p ** e)
                 / ((1.0 + p) ** e - p ** e))


def sublattice_magnetisation(phi: float) -> float:
    """Evaluate Callen's expression for the sublattice magnetisation."""
    return _magnetisation(phi)

import functools

import numpy as np


def _spin():
    """The Cr3+ spin of CrSBr."""
    return 1.5


def _boltzmann():
    """Boltzmann's constant in meV/K."""
    return 0.08617333262


def _exchange_constants():
    """Isotropic exchange by BULK shell index, meV; the interlayer sixth is absent."""
    return {1: 1.90, 2: 3.38, 3: 1.67, 4: 0.09, 5: 0.09, 7: -0.37, 8: 0.29}


def _lattice():
    """Primitive vectors and the buckled two-site Cr basis of bulk CrSBr."""
    a, b, c, buckling = 3.50, 4.76, 7.96, 1.96
    a1 = np.array([a, 0.0, 0.0])
    a2 = np.array([0.0, b, 0.0])
    a3 = np.array([0.0, 0.0, c])
    basis = [np.array([0.0, 0.0, 0.5 * buckling]),
             np.array([0.5 * a, 0.5 * b, -0.5 * buckling])]
    return a1, a2, a3, basis


@functools.lru_cache(maxsize=None)
def _bulk_bonds(nmax=3):
    """Every bulk Cr-Cr bond, ranked by length over the 3D structure (interlayer
    included). Depends on nmax alone, so the cache cannot make the ranking call-order
    dependent."""
    a1, a2, a3, basis = _lattice()
    raw = []
    for a in range(2):
        for b in range(2):
            for n1 in range(-nmax, nmax + 1):
                for n2 in range(-nmax, nmax + 1):
                    for n3 in (-1, 0, 1):
                        r_ij = n1 * a1 + n2 * a2 + n3 * a3
                        r = r_ij + basis[b] - basis[a]
                        d = float(np.linalg.norm(r))
                        if d > 1e-9:
                            raw.append((d, a, b, n3, r_ij, r))
    raw.sort(key=lambda t: t[0])
    radii = []
    for t in raw:
        if not radii or t[0] - radii[-1] > 1e-3:
            radii.append(t[0])
    out = []
    for (d, a, b, n3, r_ij, r) in raw:
        shell = next(k + 1 for k, rad in enumerate(radii) if abs(d - rad) < 1e-3)
        out.append((shell, a, b, n3, r_ij, r, d))
    return out


def _mesh_q(n):
    """Gamma-centred n x n sampling of the first Brillouin zone."""
    a1, a2, a3, basis = _lattice()
    b1 = np.array([2.0 * np.pi / a1[0], 0.0])
    b2 = np.array([0.0, 2.0 * np.pi / a2[1]])
    return np.array([i / n * b1 + j / n * b2 for i in range(n) for j in range(n)])


@functools.lru_cache(maxsize=None)
def _mesh_J(n):
    """J_ab(q) on the whole mesh; depends on the mesh size alone, so built once."""
    js = _exchange_constants()
    qs = _mesh_q(n)
    J = np.zeros((len(qs), 2, 2), dtype=complex)
    for (shell, a, b, n3, r_ij, r, d) in _bulk_bonds():
        if n3 != 0 or shell not in js:
            continue
        J[:, a, b] += js[shell] * np.exp(1j * (qs @ r_ij[:2]))
    return J


def curie_temperature(n: int) -> float:
    """Evaluate the RPA+CD Curie temperature of monolayer CrSBr."""
    # the sixth bulk shell is interlayer, so the monolayer carries none of it
    if neighbour_bond_vectors(6, 0, 0).shape != (0, 3):
        raise ValueError("the sixth bulk shell must contribute no in-plane bond")
    if abs(exchange_structure_factor(2, 0, 1, 0.0, 0.0)[0] - 4.0) > 1e-9:
        raise ValueError("shell 2 must carry four in-plane bonds")

    # the vectorised mesh exchange must reproduce the bond-by-bond J_ab(q)
    qs = _mesh_q(n)
    Js = _mesh_J(n)
    for idx in (0, len(qs) // 3, len(qs) - 1):
        for a in range(2):
            for b in range(2):
                ref = exchange_at_q(a, b, float(qs[idx][0]), float(qs[idx][1]))
                if abs(complex(ref[0], ref[1]) - Js[idx, a, b]) > 1e-9:
                    raise ValueError("the mesh exchange disagrees with J_ab(q)")

    # the Callen argument is pinned at t* = (S+1)/(3S) at the transition, which is
    # Upsilon evaluated at unit magnetisation with an occupation of 3/4
    ups = callen_factor(1.0, 0.75)

    # the block that fixes the temperature scale, at unit magnetisation
    c0 = molecular_field(1.0)
    a1, b1, c1 = anisotropy_contractions(1.0)
    diag = c0 + (-a1 + c1) * ups
    beta = abs(b1) * ups

    M = -Js + np.eye(2)[None, :, :] * diag
    mvals, mvecs = np.linalg.eigh(M)
    if np.any(mvals ** 2 <= beta ** 2):
        raise ValueError("the magnon spectrum is not positive definite")
    w0 = np.abs(mvecs[:, 0, :]) ** 2
    Q = float((w0 * mvals / (mvals ** 2 - beta ** 2)).sum(axis=1).mean())
    S = _spin()
    Tc = S * (S + 1.0) / (3.0 * Q) / _boltzmann()

    # the ordered state below the transition must be reduced but not destroyed
    w = magnon_energies(0.0, 0.0, S, 0.0)
    if not w[2] > 0.0:
        raise ValueError("the zero-temperature gap must be positive")
    sz, phi = S, 0.0
    for _ in range(40):
        phi = magnon_occupation(0.5 * Tc, sz, phi, min(n, 48))
        sz = 0.7 * sz + 0.3 * sublattice_magnetisation(phi)
    if not 0.0 < sz < S:
        raise ValueError("the ordered state below the transition is unphysical")
    return Tc
SCICODE_GOLD_EOF
