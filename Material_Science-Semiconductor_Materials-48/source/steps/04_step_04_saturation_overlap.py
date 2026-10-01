"""
Step 04 - Phase-space-filling overlap of a zero-momentum pair state with an exciton at momentum Q.

An exciton population blocks the electron and hole states it occupies, and that reduces the strength with which the light field can create a further electron-hole pair. Consider a pair created by a photon at zero centre-of-mass momentum and an occupied 1s exciton of centre-of-mass momentum Q. The electron of the occupied exciton carries the share alpha Q of its momentum, alpha = m_e/(m_e + m_h), and its hole the share beta Q, beta = m_h/(m_e + m_h), so each blocks the probed relative motion at a displaced argument.

The overlap that controls the blocking is I(Q) = integral of d^2k/(2 pi)^2 phi(k)^2 [phi(|k + alpha Q|) + phi(|k + beta Q|)], the first term from electron blocking and the second from hole blocking, with phi the normalised momentum amplitude of step 03. At Q = 0 the ratio I(0)/psi(0) is the inverse of the saturation density at which the linearised bleaching would remove the light-matter coupling altogether. I(Q) falls once Q times the exciton radius exceeds about one, so hot excitons block less than cold ones.

Returns
-------
numpy.ndarray, the phase-space-filling overlap I(Q) in nm at each momentum
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
import numpy.typing as npt


def saturation_overlap(exponents: npt.ArrayLike, coefficients: npt.ArrayLike, m_e: float, m_h: float, momenta: npt.ArrayLike) -> np.ndarray:
    '''Phase-space-filling overlap I(Q) for a zero-momentum probe and an exciton at Q.

    Parameters
    ----------
    exponents : array_like
        One-dimensional array of finite positive Gaussian exponents a_i, nm^-2.
    coefficients : array_like
        Coefficients c_i of the real amplitude psi(r) = sum_i c_i exp(-a_i r^2),
        same length as exponents, taken as given (not renormalised).
    m_e : float
        Electron effective mass in units of m0, finite and > 0.
    m_h : float
        Hole effective mass in units of m0, finite and > 0.
    momenta : array_like
        One-dimensional array of centre-of-mass momenta Q >= 0 of the occupied
        exciton, nm^-1.

    Returns
    -------
    overlap : numpy.ndarray
        I(Q) in nm for each momentum, same shape as momenta.

    Raises
    ------
    ValueError
        If exponents and coefficients are not non-empty one-dimensional arrays of
        equal length with finite entries and positive exponents, if a mass is not
        finite and positive, or if momenta is not a non-empty one-dimensional
        array of finite non-negative values.
    '''
    return overlap

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
import numpy.typing as npt


def _oracle_saturation_overlap(exponents: npt.ArrayLike, coefficients: npt.ArrayLike, m_e: float, m_h: float, momenta: npt.ArrayLike) -> np.ndarray:
    """Phase-space-filling overlap I(Q) of a zero-momentum pair state with an exciton at Q, nm."""
    import numpy as np
    a = np.asarray(exponents, dtype=float)
    c = np.asarray(coefficients, dtype=float)
    q = np.asarray(momenta, dtype=float)
    if a.ndim != 1 or a.size == 0 or c.shape != a.shape:
        raise ValueError("exponents and coefficients must be non-empty 1D arrays of equal length")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(c))) or np.any(a <= 0.0):
        raise ValueError("exponents must be finite and positive and coefficients finite")
    if q.ndim != 1 or q.size == 0 or not np.all(np.isfinite(q)) or np.any(q < 0.0):
        raise ValueError("momenta must be a non-empty 1D array of finite non-negative values")
    m_e = float(m_e)
    m_h = float(m_h)
    if not (np.isfinite(m_e) and np.isfinite(m_h)) or m_e <= 0.0 or m_h <= 0.0:
        raise ValueError("carrier masses must be finite and positive")
    alpha = m_e / (m_e + m_h)
    beta = m_h / (m_e + m_h)
    w = c * np.pi / a
    t = 1.0 / (4.0 * a)
    p = (t[:, None] + t[None, :]).ravel()
    wp = (w[:, None] * w[None, :]).ravel()
    pref = (wp[:, None] * w[None, :] / (4.0 * np.pi * (p[:, None] + t[None, :]))).ravel()
    red = (p[:, None] * t[None, :] / (p[:, None] + t[None, :])).ravel()
    out = np.empty_like(q)
    for start in range(0, q.size, 128):
        qq = q[start:start + 128]
        out[start:start + 128] = (pref[None, :] * (np.exp(-np.outer((alpha * qq) ** 2, red))
                                                   + np.exp(-np.outer((beta * qq) ** 2, red)))).sum(axis=1)
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: MoS2 ground state from zero momentum to the thermal tail ---
        {
            "setup": ('import numpy as np\n'
                      'exponents = np.array([0.02, 0.1, 0.5, 2.5, 12.5])\n'
                      'coefficients = np.array([-0.00484781934334, 0.131037470592, 0.300175291225, 0.0798788920878, 0.0234948262371])\n'
                      'm_e = 0.47\n'
                      'm_h = 0.54\n'
                      'momenta = np.array([0.0, 0.2, 0.8, 2.0, 5.0])\n'),
            "call": 'saturation_overlap(exponents, coefficients, m_e, m_h, momenta)',
            "gold_call": '_oracle_saturation_overlap(exponents, coefficients, m_e, m_h, momenta)',
            "tol": 1e-06,
        },
        # --- Boundary: zero momentum only, the saturation-density overlap ---
        {
            "setup": ('import numpy as np\n'
                      'exponents = np.array([0.05, 0.4, 3.2, 25.6])\n'
                      'coefficients = np.array([-0.00951763743232, 0.169820649119, 1.06859210545, 0.17307684644])\n'
                      'm_e = 1.0\n'
                      'm_h = 1.2\n'
                      'momenta = np.array([0.0])\n'),
            "call": 'saturation_overlap(exponents, coefficients, m_e, m_h, momenta)',
            "gold_call": '_oracle_saturation_overlap(exponents, coefficients, m_e, m_h, momenta)',
            "tol": 1e-06,
        },
        # --- Edge: very unequal masses at large momentum, where the two blocking terms separate ---
        {
            "setup": ('import numpy as np\n'
                      'exponents = np.array([0.02, 0.1, 0.5, 2.5, 12.5])\n'
                      'coefficients = np.array([-0.00484781934334, 0.131037470592, 0.300175291225, 0.0798788920878, 0.0234948262371])\n'
                      'm_e = 0.2\n'
                      'm_h = 1.8\n'
                      'momenta = np.array([0.5, 3.0, 10.0])\n'),
            "call": 'saturation_overlap(exponents, coefficients, m_e, m_h, momenta)',
            "gold_call": '_oracle_saturation_overlap(exponents, coefficients, m_e, m_h, momenta)',
            "tol": 1e-06,
        },
        # --- Edge: single Gaussian amplitude ---
        {
            "setup": ('import numpy as np\n'
                      'exponents = np.array([0.3])\n'
                      'coefficients = np.array([np.sqrt(0.6 / np.pi)])\n'
                      'm_e = 0.47\n'
                      'm_h = 0.54\n'
                      'momenta = np.array([0.0, 1.0, 4.0])\n'),
            "call": 'saturation_overlap(exponents, coefficients, m_e, m_h, momenta)',
            "gold_call": '_oracle_saturation_overlap(exponents, coefficients, m_e, m_h, momenta)',
            "tol": 1e-06,
        },
    ]
