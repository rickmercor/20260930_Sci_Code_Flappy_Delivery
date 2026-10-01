"""
Step 05 - Lower and upper polariton dispersions and Hopfield amplitudes.

In a planar microcavity the bright exciton and the cavity photon of the same in-plane momentum Q mix through the light-matter coupling g. At each Q the 2x2 Hamiltonian with diagonal entries E_X(Q) and E_C(Q) and off-diagonal entries g, taken real and positive, gives two eigenvalues, the lower (LP) and upper (UP) polariton, and normalised eigenvectors (X, C) whose squares are the exciton and photon fractions of each branch.

The exciton dispersion is E_X(Q) = E_X(0) + hbar^2 Q^2/(2 M), with M the total exciton mass in units of m0, and the cavity mode follows E_C(Q) = sqrt(E_C(0)^2 + (hbar c Q/n_c)^2) with E_C(0) = E_X(0) plus the cavity detuning. The photon is so light that the polariton dispersion is sharply curved inside a narrow cone of small Q and becomes exciton-like just outside it. The overall sign of each eigenvector is fixed so that its photon amplitude C is non-negative; the relative sign of X and C in each branch then follows from the Hamiltonian. Constants: hbar^2/(2 m0) = 0.0380998 eV nm^2 and hbar c = 197.32698 eV nm.

Returns
-------
numpy.ndarray of shape (6, nQ): E_LP and E_UP in eV, then X_LP, C_LP, X_UP, C_UP
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
import numpy.typing as npt


def polariton_branches(momenta: npt.ArrayLike, exciton_energy: float, total_mass: float, cavity_detuning: float, cavity_index: float, coupling: float) -> np.ndarray:
    '''Polariton energies and Hopfield amplitudes of the 2x2 exciton-photon problem.

    Parameters
    ----------
    momenta : array_like
        One-dimensional array of in-plane momenta Q >= 0, nm^-1.
    exciton_energy : float
        Exciton energy at Q = 0, E_X(0), in eV, > 0.
    total_mass : float
        Total exciton mass M = m_e + m_h in units of m0, > 0.
    cavity_detuning : float
        E_C(0) - E_X(0) in eV (negative when the cavity lies below the exciton);
        E_C(0) must be positive.
    cavity_index : float
        Effective refractive index n_c of the cavity mode, > 0.
    coupling : float
        Real exciton-photon coupling g in eV, > 0.

    Returns
    -------
    branches : numpy.ndarray
        Array of shape (6, len(momenta)) with rows E_LP and E_UP in eV, then the
        amplitudes X_LP, C_LP, X_UP, C_UP of the normalised eigenvectors, each
        signed so that its photon amplitude is non-negative.

    Raises
    ------
    ValueError
        If momenta is not a non-empty one-dimensional array of finite
        non-negative values, if a scalar input is not finite, or if E_X(0),
        E_C(0), total_mass, cavity_index or coupling is not positive.
    '''
    return branches

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
import numpy.typing as npt


def _oracle_polariton_branches(momenta: npt.ArrayLike, exciton_energy: float, total_mass: float, cavity_detuning: float, cavity_index: float, coupling: float) -> np.ndarray:
    """Lower and upper polariton energies and Hopfield amplitudes of the 2x2 exciton-photon problem."""
    import numpy as np
    hb2m0 = 0.0380998                  # hbar^2/(2 m0), eV nm^2
    hbarc = 197.32698                  # eV nm
    q = np.asarray(momenta, dtype=float)
    if q.ndim != 1 or q.size == 0 or not np.all(np.isfinite(q)) or np.any(q < 0.0):
        raise ValueError("momenta must be a non-empty 1D array of finite non-negative values")
    vals = [float(exciton_energy), float(total_mass), float(cavity_detuning), float(cavity_index), float(coupling)]
    if not all(np.isfinite(v) for v in vals):
        raise ValueError("all scalar inputs must be finite")
    ex0, mass, det, nc, g = vals
    if ex0 <= 0.0 or mass <= 0.0 or nc <= 0.0 or g <= 0.0 or ex0 + det <= 0.0:
        raise ValueError("energies, mass, index and coupling must be positive")
    ex = ex0 + hb2m0 * q ** 2 / mass
    ec = np.sqrt((ex0 + det) ** 2 + (hbarc * q / nc) ** 2)
    half = 0.5 * np.sqrt((ex - ec) ** 2 + 4.0 * g * g)
    elp = 0.5 * (ex + ec) - half
    eup = 0.5 * (ex + ec) + half
    rows = [elp, eup]
    for e in (elp, eup):
        xa = np.full_like(e, g)        # eigenvector (g, E - E_X) of [[E_X, g], [g, E_C]]
        ca = e - ex
        norm = np.sqrt(xa * xa + ca * ca)
        sgn = np.where(ca < 0.0, -1.0, 1.0)
        rows.extend([sgn * xa / norm, sgn * ca / norm])
    return np.stack(rows)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: cavity 30 meV below the exciton, from normal incidence into the reservoir ---
        {
            "setup": ('import numpy as np\n'
                      'momenta = np.array([0.0, 1.0e-3, 3.0e-3, 1.0e-2, 0.5])\n'
                      'exciton_energy = 1.9\n'
                      'total_mass = 1.01\n'
                      'cavity_detuning = -0.03\n'
                      'cavity_index = 1.8\n'
                      'coupling = 0.02\n'),
            "call": 'polariton_branches(momenta, exciton_energy, total_mass, cavity_detuning, cavity_index, coupling)',
            "gold_call": '_oracle_polariton_branches(momenta, exciton_energy, total_mass, cavity_detuning, cavity_index, coupling)',
            "tol": 6e-07,
        },
        # --- Boundary: zero detuning, where both branches are half exciton and half photon at Q = 0 ---
        {
            "setup": ('import numpy as np\n'
                      'momenta = np.array([0.0, 2.0e-3])\n'
                      'exciton_energy = 1.9\n'
                      'total_mass = 1.01\n'
                      'cavity_detuning = 0.0\n'
                      'cavity_index = 1.8\n'
                      'coupling = 0.02\n'),
            "call": 'polariton_branches(momenta, exciton_energy, total_mass, cavity_detuning, cavity_index, coupling)',
            "gold_call": '_oracle_polariton_branches(momenta, exciton_energy, total_mass, cavity_detuning, cavity_index, coupling)',
            "tol": 2e-08,
        },
        # --- Edge: cavity far above the exciton and a momentum deep in the photon-like upper branch ---
        {
            "setup": ('import numpy as np\n'
                      'momenta = np.array([0.0, 1.0e-2, 2.0])\n'
                      'exciton_energy = 1.9\n'
                      'total_mass = 1.01\n'
                      'cavity_detuning = 0.08\n'
                      'cavity_index = 1.8\n'
                      'coupling = 0.02\n'),
            "call": 'polariton_branches(momenta, exciton_energy, total_mass, cavity_detuning, cavity_index, coupling)',
            "gold_call": '_oracle_polariton_branches(momenta, exciton_energy, total_mass, cavity_detuning, cavity_index, coupling)',
            "tol": 3e-06,
        },
        # --- Edge: weak coupling near the anticrossing ---
        {
            "setup": ('import numpy as np\n'
                      'momenta = np.array([0.0, 1.0e-3, 5.0e-3])\n'
                      'exciton_energy = 1.9\n'
                      'total_mass = 1.01\n'
                      'cavity_detuning = -0.01\n'
                      'cavity_index = 1.8\n'
                      'coupling = 0.001\n'),
            "call": 'polariton_branches(momenta, exciton_energy, total_mass, cavity_detuning, cavity_index, coupling)',
            "gold_call": '_oracle_polariton_branches(momenta, exciton_energy, total_mass, cavity_detuning, cavity_index, coupling)',
            "tol": 2e-08,
        },
    ]
