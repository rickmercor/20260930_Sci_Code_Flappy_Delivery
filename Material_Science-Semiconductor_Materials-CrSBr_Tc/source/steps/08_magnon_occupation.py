"""
Average the magnon occupation over the Brillouin zone. Returns the dimensionless site-resolved occupation of sublattice 0.

The site-resolved magnon occupation follows from the fluctuation-dissipation
theorem,

    Phi_a = (1/N) sum_{eta,q} U_{a,eta,q} n_B(omega_{eta,q}) U^{-1}_{eta,a,q},

where U is the matrix whose columns are the eigenvectors of the Hamiltonian at q, eta indexes
its eigenvalues omega_{eta,q}, and n_B(w) = 1 / (exp(w / kB T) - 1) is the Bose factor.

Sample the zone on a Gamma-centred n by n grid of the reciprocal vectors b_i, which satisfy
b_i . a_j = 2*pi*delta_ij, that is at fractional coordinates (i/n, j/n) for i, j in [0, n).
Use kB = 0.08617333262 meV/K.

Return Phi for sublattice 0.

Returns
-------
float: the dimensionless magnon occupation of sublattice 0.
"""

import functools
import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def magnon_occupation(T: float, sz: float, phi: float, n: int) -> float:
    """Average the magnon occupation over the Brillouin zone. Returns the dimensionless site-resolved occupation of sublattice 0.

    Args:
        T: temperature in K; must be positive.
        sz: sublattice magnetisation entering the Hamiltonian.
        phi: magnon occupation entering the Hamiltonian's Callen factor.
        n: linear size of the Brillouin-zone mesh.

    Returns:
        float: the dimensionless magnon occupation of sublattice 0.
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


def _oracle_magnon_occupation(T: float, sz: float, phi: float, n: int) -> float:
    """Average the magnon occupation over the Brillouin zone."""
    return _occupation(T, sz, phi, n)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Differential cases: normal, boundary and edge."""
    return [
        {
            # normal: saturation at an intermediate temperature
            "setup": "",
            "call": 'magnon_occupation(100.0, 1.5, 0.0, 48)',
            "gold_call": '_oracle_magnon_occupation(100.0, 1.5, 0.0, 48)',
        },
        {
            # normal: a self-consistent state part way down the curve
            "setup": "",
            "call": 'magnon_occupation(100.0, 1.0, 0.5, 96)',
            "gold_call": '_oracle_magnon_occupation(100.0, 1.0, 0.5, 96)',
        },
        {
            # boundary: low temperature, where only zero-point depletion survives
            "setup": "",
            "call": 'magnon_occupation(5.0, 1.5, 0.0, 48)',
            "gold_call": '_oracle_magnon_occupation(5.0, 1.5, 0.0, 48)',
        },
        {
            # edge: close to the transition, where the occupation is large
            "setup": "",
            "call": 'magnon_occupation(140.0, 0.4, 2.5, 96)',
            "gold_call": '_oracle_magnon_occupation(140.0, 0.4, 2.5, 96)',
        },
    ]
