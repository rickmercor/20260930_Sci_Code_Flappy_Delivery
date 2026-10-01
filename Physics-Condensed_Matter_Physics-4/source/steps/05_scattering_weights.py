"""
Assemble the broadening-independent weights of phonon-assisted scattering between moire exciton states, resolved by umklapp vector, layer and emission or absorption.

A moire exciton state (k, n) is a superposition of plane waves Q_k + g with coefficients C_g(k, n). A phonon
of momentum q takes the plane wave Q_k + g to Q_k' + g + d, so q = Q_k' - Q_k + d, where the umklapp vector
d = i b1 + j b2 runs over max(|i|, |j|, |i - j|) <= 2 in the plane-wave ordering convention (i then j
ascending). Phonons of different q do not interfere, so for each d the transition probability carries the
squared overlap |sum_g C*_g(k, n) C_{g+d}(k', m)|^2, the sum running over the g for which both g and g + d
are in the plane-wave set. The electron layer (MoSe2, layer 0) and the hole layer (WSe2, layer 1) have
independent phonon baths; each couples through its own carrier's form factor (the electron sits at
mass_ratio m_h/M along the relative coordinate, the hole at m_e/M). Material constants per layer
(v_LA nm/ps, D1 meV, D0 meV/nm, E_op meV, a nm, molar mass g/mol): layer 0 (4.1, 3.4e3, 5.2e4, 36.6, 0.327,
253.86); layer 1 (3.3, 2.1e3, 3.1e4, 30.8, 0.325, 341.76). The exciton radius comes from e_b as in the
moire-coupling step. Fermi's golden rule gives a rate (2 pi/hbar) times the area-scaled squared coupling,
times n_B for absorption or n_B + 1 for emission (Bose-Einstein occupation of the phonon at the given
temperature, zero for an acoustic phonon of q = 0), times the squared overlap, times the energy-conserving
kernel; the sum over the grid is converted to an integral with measure A_mBZ / (N_k (2 pi)^2), A_mBZ the
area of the moire Brillouin zone and N_k the number of grid points. The weights are everything except the
kernel. Optical-phonon energies do not depend on q, so optical weights are summed over d.

Returns
-------
weights : tuple -- (w_ac, omega, w_op, e_op):
"""

import numpy as np
from scipy.linalg import expm

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def scattering_weights(q_pts: "np.ndarray", coeffs: "np.ndarray", basis: "np.ndarray", b_vectors: "np.ndarray",
                       m_e: float, m_h: float, e_b: float, temperature: float) -> tuple:
    """Return the scattering weights of every ordered pair of moire exciton states.

    States are flattened as s = k * n_bands + n, with k the grid index of q_pts.

    Parameters
    ----------
    q_pts : np.ndarray
        (N_k, 2) grid momenta, nm^-1.
    coeffs : np.ndarray
        (N_k, n_pw, n_bands) complex eigenvector coefficients in the order of basis.
    basis : np.ndarray
        (n_pw, 2) integer plane-wave labels (i, j).
    b_vectors : np.ndarray
        (2, 2) moire reciprocal vectors, rows b1, b2, nm^-1.
    m_e, m_h : float
        Electron and hole masses, units of the free-electron mass.
    e_b : float
        1s interlayer binding energy, meV.
    temperature : float
        Lattice temperature, K; k_B = 0.08617333262 meV/K.

    Returns
    -------
    weights : tuple
        (w_ac, omega, w_op, e_op):
        w_ac (S, S, 19, 2, 2), index [s_initial, s_final, d, layer, channel], channel 0 = emission
        (kernel centred at E_final = E_initial - hbar Omega), channel 1 = absorption, in meV/ps;
        omega (N_k, N_k, 19, 2), index [k_initial, k_final, d, layer], acoustic phonon energy in meV;
        w_op (S, S, 2, 2), index [s_initial, s_final, layer, channel], optical weights summed over d, meV/ps;
        e_op (2,) optical phonon energies of layers 0 and 1, meV.
    """
    return w_ac, omega, w_op, e_op

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _material_layers():
    # (carrier, v_LA nm/ps, D1 meV, D0 meV/nm, E_op meV, a nm, molar mass g/mol)
    return (("e", 4.1, 3.4e3, 5.2e4, 36.6, 0.327, 253.86), ("h", 3.3, 2.1e3, 3.1e4, 30.8, 0.325, 341.76))


def _oracle_scattering_weights(q_pts: "np.ndarray", coeffs: "np.ndarray", basis: "np.ndarray",
                               b_vectors: "np.ndarray", m_e: float, m_h: float, e_b: float,
                               temperature: float) -> tuple:
    hbar, kb = 0.6582119569, 0.08617333262
    b = np.asarray(b_vectors, dtype=float)
    nk_tot, _, nb = coeffs.shape
    a_b = _bohr_radius(m_e, m_h, e_b)
    kt = kb * temperature
    lut = {tuple(x): n for n, x in enumerate(basis)}
    shifts = _shell_basis(2)
    pref = (2.0 * np.pi / hbar) * abs(np.linalg.det(b)) / nk_tot / (2.0 * np.pi) ** 2
    n_d = len(shifts)
    w_ac = np.zeros((nk_tot, nb, nk_tot, nb, n_d, 2, 2))
    om = np.zeros((nk_tot, nk_tot, n_d, 2))
    w_op = np.zeros((nk_tot, nb, nk_tot, nb, 2, 2))
    layers = _material_layers()
    for di, d in enumerate(shifts):
        src = [n for n, x in enumerate(basis) if (x[0] + d[0], x[1] + d[1]) in lut]
        dst = [lut[(basis[n][0] + d[0], basis[n][1] + d[1])] for n in src]
        ov = np.abs(np.einsum("ign,fgm->infm", coeffs[:, src, :].conj(), coeffs[:, dst, :])) ** 2
        qn = np.linalg.norm(q_pts[None, :, :] - q_pts[:, None, :] + d @ b, axis=2)
        for li, (owner, v_la, d1, d0, eo, a_l, mm) in enumerate(layers):
            ratio = (m_h if owner == "e" else m_e) / (m_e + m_h)
            m_ac, omega, m_op = _oracle_phonon_matrix_elements(qn, ratio, a_b, v_la, d1, d0, eo, a_l, mm)
            with np.errstate(divide="ignore", invalid="ignore"):
                n_ac = np.where(qn > 0.0, 1.0 / np.expm1(omega / kt), 0.0)
            n_op = 1.0 / np.expm1(eo / kt)
            om[:, :, di, li] = omega
            w_ac[:, :, :, :, di, li, 0] = pref * (m_ac * (n_ac + 1.0))[:, None, :, None] * ov
            w_ac[:, :, :, :, di, li, 1] = pref * (m_ac * n_ac)[:, None, :, None] * ov
            w_op[:, :, :, :, li, 0] += pref * (m_op * (n_op + 1.0))[:, None, :, None] * ov
            w_op[:, :, :, :, li, 1] += pref * (m_op * n_op)[:, None, :, None] * ov
    s = nk_tot * nb
    e_op = np.array([lay[4] for lay in layers])
    return w_ac.reshape(s, s, n_d, 2, 2), om, w_op.reshape(s, s, 2, 2), e_op

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():

    """Return list of test case specifications."""

    base = """import numpy as np

b = np.array([[1.0059742, -0.5808106], [0.0, 1.1616212]])

def make(nk, n_shell, nb, seed):

    basis = np.array([(i, j) for i in range(-n_shell, n_shell + 1) for j in range(-n_shell, n_shell + 1)

                      if max(abs(i), abs(j), abs(i - j)) <= n_shell], dtype=int)

    f = (np.arange(nk) + 0.5) / nk

    q = np.array([u * b[0] + v * b[1] for u in f for v in f])

    rng = np.random.default_rng(seed)

    c = rng.normal(size=(nk * nk, len(basis), nb)) + 1j * rng.normal(size=(nk * nk, len(basis), nb))

    c = np.linalg.qr(c)[0]

    return q, c, basis

def flat(r):

    return np.concatenate([np.concatenate([np.asarray([x.ndim, *x.shape], dtype=float), x.ravel()]) for x in r])

"""

    return [

        {"setup": base + "q, c, basis = make(2, 1, 2, 3)",

         "call": "flat(scattering_weights(q, c, basis, b, 0.64, 0.51, 173.0, 10.0))",

         "gold_call": "flat(_oracle_scattering_weights(q, c, basis, b, 0.64, 0.51, 173.0, 10.0))"},

        {"setup": base + "q, c, basis = make(2, 2, 3, 11)",

         "call": "flat(scattering_weights(q, c, basis, b, 0.5, 0.7, 120.0, 70.0))",

         "gold_call": "flat(_oracle_scattering_weights(q, c, basis, b, 0.5, 0.7, 120.0, 70.0))"},

        # a single grid point: only q = d transfers, including the q = 0 acoustic zero

        {"setup": base + "q, c, basis = make(1, 1, 2, 5)",

         "call": "flat(scattering_weights(q, c, basis, b, 0.64, 0.51, 173.0, 4.0))",

         "gold_call": "flat(_oracle_scattering_weights(q, c, basis, b, 0.64, 0.51, 173.0, 4.0))"},

    ]
