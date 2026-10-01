"""
Diagonalise the zone-folded moire exciton Hamiltonian on a grid of the moire Brillouin zone and return mini-band energies, eigenvectors and squared group velocities.

The exciton centre-of-mass motion is free with total mass M = m_e + m_h and kinetic energy
hbar^2 |K|^2 / (2 M), hbar^2/(2 m0) = 38.09982 meV nm^2. Folding momenta into the moire Brillouin zone,
a state with crystal momentum Q is a superposition of plane waves Q + g, g = i b1 + j b2, and the moire
potential V(R) = sum_s Theta exp(i s.R) + c.c. over s in {b1, b2, -(b1 + b2)} couples each plane wave
Q + g to Q + g + s with matrix element Theta (and back with its conjugate). The plane-wave set keeps every
(i, j) with max(|i|, |j|, |i - j|) <= n_shell, ordered by i ascending and then j ascending. The grid is
the shifted Monkhorst-Pack set Q = ((u + 1/2)/nk) b1 + ((v + 1/2)/nk) b2, u, v = 0..nk-1, with u as the
outer index; the shift keeps the grid off the high-symmetry points. The group velocity of each mini-band
state is (1/hbar) dE/dQ evaluated exactly at the grid point, hbar = 0.6582119569 meV ps.

Returns
-------
result : tuple -- (q_pts, energies, coeffs, v2, basis):
"""

import numpy as np
from scipy.linalg import expm

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def moire_exciton_bands(b_vectors: "np.ndarray", theta_coupling: "np.ndarray", m_e: float, m_h: float,
                        n_shell: int, nk: int, n_bands: int) -> tuple:
    """Return the moire exciton mini-bands on the shifted nk x nk grid.

    Parameters
    ----------
    b_vectors : np.ndarray
        (2, 2) moire reciprocal vectors, rows b1 and b2, nm^-1.
    theta_coupling : np.ndarray
        [Re Theta, Im Theta] of the exciton moire amplitude, meV.
    m_e, m_h : float
        Electron and hole masses in units of the free-electron mass.
    n_shell : int
        Plane-wave cutoff, max(|i|, |j|, |i - j|) <= n_shell.
    nk : int
        Grid points per reciprocal direction.
    n_bands : int
        Number of lowest mini-bands kept at each grid point.

    Returns
    -------
    result : tuple
        (q_pts, energies, coeffs, v2, basis):
        q_pts (nk*nk, 2) grid momenta in nm^-1, index k = u*nk + v;
        energies (nk*nk, n_bands) ascending eigenvalues in meV, no offset removed;
        coeffs (nk*nk, n_pw, n_bands) complex normalised eigenvectors in the plane-wave order, column n
        belonging to energies[k, n];
        v2 (nk*nk, n_bands) squared group-velocity magnitudes in nm^2/ps^2;
        basis (n_pw, 2) integer (i, j) pairs in the plane-wave order.

    Raises
    ------
    ValueError
        If n_bands exceeds the number of plane waves.
    """
    return q_pts, energies, coeffs, v2, basis

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _shell_basis(n_shell):
    return np.array([(i, j) for i in range(-n_shell, n_shell + 1) for j in range(-n_shell, n_shell + 1)
                     if max(abs(i), abs(j), abs(i - j)) <= n_shell], dtype=int)


def _oracle_moire_exciton_bands(b_vectors: "np.ndarray", theta_coupling: "np.ndarray", m_e: float, m_h: float,
                                n_shell: int, nk: int, n_bands: int) -> tuple:
    hb2_2m0, hbar = 38.09982, 0.6582119569
    b = np.asarray(b_vectors, dtype=float)
    basis = _shell_basis(n_shell)
    if n_bands > len(basis):
        raise ValueError("n_bands exceeds the plane-wave basis size")
    g = basis @ b
    th = complex(theta_coupling[0], theta_coupling[1])
    lut = {tuple(x): n for n, x in enumerate(basis)}
    h_m = np.zeros((len(basis), len(basis)), complex)
    for n, (i, j) in enumerate(basis):
        for s in ((1, 0), (0, 1), (-1, -1)):
            m = lut.get((i + s[0], j + s[1]))
            if m is not None:
                h_m[m, n] += th
                h_m[n, m] += np.conj(th)
    mass = m_e + m_h
    f = (np.arange(nk) + 0.5) / nk
    q_pts = np.array([u * b[0] + v * b[1] for u in f for v in f])
    e_out = np.empty((len(q_pts), n_bands))
    c_out = np.empty((len(q_pts), len(basis), n_bands), complex)
    v2_out = np.empty((len(q_pts), n_bands))
    for k, q in enumerate(q_pts):
        kv = q + g
        e, c = np.linalg.eigh(h_m + np.diag(hb2_2m0 / mass * np.sum(kv ** 2, axis=1)))
        e_out[k], c_out[k] = e[:n_bands], c[:, :n_bands]
        vel = np.einsum("gn,gd->nd", np.abs(c[:, :n_bands]) ** 2, 2.0 * hb2_2m0 / mass * kv) / hbar
        v2_out[k] = np.sum(vel ** 2, axis=1)
    return q_pts, e_out, c_out, v2_out, basis

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():

    """Test all returned bands without fixing eigenvector phases."""

    helper = """import numpy as np

b = np.array([[1.0059742, -0.5808106], [0.0, 1.1616212]])

def flat(r):

    q, e, c, v2, basis = r

    shapes = np.array([z for x in r for z in (x.ndim, *x.shape)], dtype=float)

    th_complex = complex(th[0], th[1])

    delta = basis[:, None, :] - basis[None, :, :]

    potential = np.zeros((len(basis), len(basis)), dtype=complex)

    for shift in ((1, 0), (0, 1), (-1, -1)):

        potential[np.all(delta == shift, axis=-1)] = th_complex

        potential[np.all(delta == -np.asarray(shift), axis=-1)] = th_complex.conjugate()

    momentum = q[:, None, :] + (basis @ b)[None, :, :]

    kinetic = 38.09982 / mass * np.sum(momentum ** 2, axis=-1)

    h = potential[None, :, :] + kinetic[:, :, None] * np.eye(len(basis))[None, :, :]

    residual = h @ c - c * e[:, None, :]

    orthogonality = c.conj().transpose(0, 2, 1) @ c - np.eye(e.shape[1])[None, :, :]

    return np.concatenate([shapes, q.ravel(), e.ravel(), v2.ravel(), basis.ravel(),

                           residual.real.ravel(), residual.imag.ravel(),

                           orthogonality.real.ravel(), orthogonality.imag.ravel()])

"""

    return [

        {'setup': helper + 'th = np.array([-2.5055, -9.4741]); mass = 1.15',

         'call': 'flat(moire_exciton_bands(b, th, 0.64, 0.51, 2, 3, 4))',

         'gold_call': 'flat(_oracle_moire_exciton_bands(b, th, 0.64, 0.51, 2, 3, 4))'},

        {'setup': helper + 'th = np.array([0.0, 0.0]); mass = 1.15',

         'call': 'flat(moire_exciton_bands(b, th, 0.64, 0.51, 1, 2, 3))',

         'gold_call': 'flat(_oracle_moire_exciton_bands(b, th, 0.64, 0.51, 1, 2, 3))'},

        {'setup': helper + 'th = np.array([6.0, 0.0]); mass = 2.0',

         'call': 'flat(moire_exciton_bands(b, th, 1.1, 0.9, 3, 4, 5))',

         'gold_call': 'flat(_oracle_moire_exciton_bands(b, th, 1.1, 0.9, 3, 4, 5))'},

        {'setup': helper + """th = np.array([1.0, 2.0]); mass = 1.15

def run(fn):

    try:

        fn(b, th, 0.64, 0.51, 1, 2, 8)

        return 0

    except ValueError:

        return 1

""", 'call': 'run(moire_exciton_bands)', 'gold_call': 'run(_oracle_moire_exciton_bands)'}

    ]
