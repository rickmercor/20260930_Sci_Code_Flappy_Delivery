"""
Step 02 - Lowest bound state of the screened Wannier problem in a Gaussian basis.

The relative motion of the electron and the hole in a bright exciton obeys the effective-mass Wannier equation: the kinetic energy of a particle with the reduced mass mu = m_e m_h/(m_e + m_h), plus the attractive screened interaction whose Gaussian moments step 01 provides. The 1s state is radially symmetric, and expanding it in real Gaussians psi(r) = sum_i c_i exp(-a_i r^2) turns the Wannier equation into a generalised symmetric eigenvalue problem H c = E S c, built from the overlap, kinetic and potential matrices of the basis.

The lowest eigenvalue is the energy of the 1s state measured from the band gap, so its magnitude is the binding energy, and the eigenvector is the wavefunction that every later step uses. Masses are in units of the free electron mass m0 and hbar^2/(2 m0) = 0.0380998 eV nm^2. The eigenvector is normalised so that the integral of psi^2 over the plane is one, and its overall sign is chosen so that the amplitude at the origin, psi(0) = sum_i c_i, is positive. The returned array carries the energy first and the coefficients after it, in the order of the exponents.

Returns
-------
numpy.ndarray of shape (N+1,), the 1s energy in eV followed by the N normalised basis coefficients
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
import numpy.typing as npt


def exciton_ground_state(exponents: npt.ArrayLike, m_e: float, m_h: float, kappa: float, r0: float) -> np.ndarray:
    '''Lowest eigenstate of the screened Wannier equation in a Gaussian basis.

    Parameters
    ----------
    exponents : array_like
        One-dimensional array of distinct, finite, positive exponents a_i of the
        basis functions exp(-a_i r^2), in nm^-2.
    m_e : float
        Electron effective mass in units of m0, finite and > 0.
    m_h : float
        Hole effective mass in units of m0, finite and > 0.
    kappa : float
        Mean dielectric constant of the surrounding media, finite and > 0.
    r0 : float
        Screening length of the sheet in nm, finite and >= 0.

    Returns
    -------
    state : numpy.ndarray
        Array of length N + 1: the lowest eigenvalue E_1s in eV (negative for a
        bound state), followed by the N coefficients c_i of
        psi(r) = sum_i c_i exp(-a_i r^2), normalised so that the integral of psi^2
        over the plane is one and signed so that sum_i c_i > 0.

    Raises
    ------
    ValueError
        If exponents is not a non-empty one-dimensional array of distinct, finite,
        positive values, if a mass is not finite and positive, if kappa or r0 is
        invalid as in step 01, or if the basis is numerically linearly dependent.
    '''
    return state

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
import numpy.typing as npt


def _oracle_exciton_ground_state(exponents: npt.ArrayLike, m_e: float, m_h: float, kappa: float, r0: float) -> np.ndarray:
    """Lowest Wannier state of the screened electron-hole problem in a Gaussian basis."""
    import numpy as np
    from scipy.linalg import eigh
    hb2m0 = 0.0380998                  # hbar^2/(2 m0), eV nm^2
    a = np.asarray(exponents, dtype=float)
    if a.ndim != 1 or a.size == 0:
        raise ValueError("exponents must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(a)) or np.any(a <= 0.0):
        raise ValueError("exponents must be finite and positive")
    if np.unique(a).size != a.size:
        raise ValueError("exponents must be distinct")
    m_e = float(m_e)
    m_h = float(m_h)
    if not (np.isfinite(m_e) and np.isfinite(m_h)) or m_e <= 0.0 or m_h <= 0.0:
        raise ValueError("carrier masses must be finite and positive")
    mu = m_e * m_h / (m_e + m_h)
    s = a[:, None] + a[None, :]
    overlap = np.pi / s
    kinetic = (hb2m0 / mu) * 4.0 * np.pi * np.outer(a, a) / s ** 2
    potential = -_oracle_screened_gaussian_moments(s.ravel(), kappa, r0).reshape(s.shape)
    try:
        evals, evecs = eigh(kinetic + potential, overlap)
    except np.linalg.LinAlgError:
        raise ValueError("the Gaussian basis is numerically linearly dependent")
    c = evecs[:, 0].copy()
    c = c / np.sqrt(c @ overlap @ c)
    if c.sum() < 0.0:
        c = -c
    return np.concatenate([[evals[0]], c])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: monolayer MoS2 in hBN with a five-function basis ---
        {
            "setup": ('import numpy as np\n'
                      'exponents = np.array([0.02, 0.1, 0.5, 2.5, 12.5])\n'
                      'm_e = 0.47\n'
                      'm_h = 0.54\n'
                      'kappa = 4.5\n'
                      'r0 = 4.15\n'),
            "call": 'exciton_ground_state(exponents, m_e, m_h, kappa, r0)',
            "gold_call": '_oracle_exciton_ground_state(exponents, m_e, m_h, kappa, r0)',
            "tol": 1e-08,
        },
        # --- Boundary: r0 = 0, the two-dimensional hydrogen problem screened by kappa ---
        {
            "setup": ('import numpy as np\n'
                      'exponents = np.array([0.02, 0.1, 0.5, 2.5, 12.5])\n'
                      'm_e = 0.47\n'
                      'm_h = 0.54\n'
                      'kappa = 4.5\n'
                      'r0 = 0.0\n'),
            "call": 'exciton_ground_state(exponents, m_e, m_h, kappa, r0)',
            "gold_call": '_oracle_exciton_ground_state(exponents, m_e, m_h, kappa, r0)',
            "tol": 1e-08,
        },
        # --- Edge: a single Gaussian, where the eigenproblem is one-dimensional ---
        {
            "setup": ('import numpy as np\n'
                      'exponents = np.array([0.3])\n'
                      'm_e = 0.47\n'
                      'm_h = 0.54\n'
                      'kappa = 4.5\n'
                      'r0 = 4.15\n'),
            "call": 'exciton_ground_state(exponents, m_e, m_h, kappa, r0)',
            "gold_call": '_oracle_exciton_ground_state(exponents, m_e, m_h, kappa, r0)',
            "tol": 1e-08,
        },
        # --- Edge: heavy carriers and weak screening, a deeply bound compact exciton ---
        {
            "setup": ('import numpy as np\n'
                      'exponents = np.array([0.05, 0.4, 3.2, 25.6])\n'
                      'm_e = 1.0\n'
                      'm_h = 1.2\n'
                      'kappa = 1.0\n'
                      'r0 = 2.0\n'),
            "call": 'exciton_ground_state(exponents, m_e, m_h, kappa, r0)',
            "gold_call": '_oracle_exciton_ground_state(exponents, m_e, m_h, kappa, r0)',
            "tol": 1e-08,
        },
    ]
