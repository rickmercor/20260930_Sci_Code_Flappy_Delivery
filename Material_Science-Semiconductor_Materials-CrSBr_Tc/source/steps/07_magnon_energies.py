"""
Assemble the magnon Hamiltonian and diagonalise it. Returns the four magnon energies in meV, ascending.

Assemble the exchange part of the RPA Hamiltonian as the two-by-two block matrix

    H^RPA_q = [[ -A_ab(q) + C_ab , -B_ab(q) ],
               [  B_ab(q)^dagger ,  A_ab(-q)* - C_ab ]]

with

    A_ab(q) = (<S''z_a>/2) u^T J'_ab(q) u*,
    B_ab(q) = (<S''z_a>/2) u^T J'_ab(q) u,

and C the molecular field on the diagonal. The single-ion anisotropy adds

    H^RPA_K = delta_ab [[ -A^RPA_a + C^RPA_a , -B^RPA_a ],
                        [  (B^RPA_a)^dagger  ,  A^RPA_a - C^RPA_a ]]

with every element multiplied by the Callen factor.

Return all four eigenvalues, ascending.

Returns
-------
numpy.ndarray: shape (4,) array of magnon energies in meV, ascending.
"""

import functools
import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def magnon_energies(qx: float, qy: float, sz: float, phi: float) -> "np.ndarray":
    """Assemble the magnon Hamiltonian and diagonalise it. Returns the four magnon energies in meV, ascending.

    Args:
        qx: Cartesian x component of the wavevector.
        qy: Cartesian y component of the wavevector.
        sz: sublattice magnetisation.
        phi: magnon occupation entering the Callen factor.

    Returns:
        numpy.ndarray: shape (4,) array of magnon energies in meV, ascending.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

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


def _oracle_magnon_energies(qx: float, qy: float, sz: float, phi: float) -> "np.ndarray":
    """Assemble the magnon Hamiltonian and diagonalise it."""
    return _energies(np.array([qx, qy]), sz, phi)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Differential cases: normal, boundary and edge."""
    return [
        {
            # normal: the zone centre at saturation, where the gap opens
            "setup": "",
            "call": 'magnon_energies(0.0, 0.0, 1.5, 0.0)',
            "gold_call": '_oracle_magnon_energies(0.0, 0.0, 1.5, 0.0)',
        },
        {
            # normal: away from any high-symmetry point at saturation
            "setup": "",
            "call": 'magnon_energies(0.9, -0.4, 1.5, 0.0)',
            "gold_call": '_oracle_magnon_energies(0.9, -0.4, 1.5, 0.0)',
        },
        {
            # normal: the zone centre in a thermally renormalised state
            "setup": "",
            "call": 'magnon_energies(0.0, 0.0, 1.0, 0.5)',
            "gold_call": '_oracle_magnon_energies(0.0, 0.0, 1.0, 0.5)',
        },
        {
            # edge: the zone corner close to the transition
            "setup": "",
            "call": 'magnon_energies(0.8975979010256552, 0.6600444097822498, 0.4, 2.5)',
            "gold_call": '_oracle_magnon_energies(0.8975979010256552, 0.6600444097822498, 0.4, 2.5)',
        },
    ]
