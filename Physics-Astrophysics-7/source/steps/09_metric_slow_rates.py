"""
Return the proper-time derivatives of the two slow-mode synchronous metric perturbations at comoving wavenumber k, for a universe whose non-axion species carry no perturbations. Write P for conj(psi_s)*dpsi_s + psi_s*conj(dpsi_s) and M for conj(psi_s)*dpsi_s - psi_s*conj(dpsi_s). The first output is ((2*m_pl**2*k**2/(a_s**2*hubble_s))*eta_s + (m/hubble_s)*Re(P)) / m_pl**2. The second output is the real part of ((1j*m/4)*M - (1/16)*abs(psi_s)**2*hdot_s - (3/8)*hubble_s*P - (1j*k**2/(16*m*a_s**2))*M) / m_pl**2, where hdot_s is the first output. The two are stacked along a new leading axis of length 2 in that order, both real.

The two Einstein constraint equations that close the perturbed system are the time-time and the time-space ones. The first fixes the rate of the trace metric perturbation from the total density perturbation and the curvature perturbation; the second fixes the rate of the curvature perturbation from the total velocity perturbation. Because the velocity perturbation of the axion itself contains the trace rate, the second equation cannot be formed before the first has been.

Returns
-------
ndarray of shape (2,) + np.shape(psi_s), float64: the time derivative of the trace metric perturbation followed by that of the curvature perturbation.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def metric_slow_rates(psi_s: 'np.ndarray', dpsi_s: 'np.ndarray', eta_s: 'np.ndarray', hubble_s: 'np.ndarray', a_s: 'np.ndarray', m: float, m_pl: float, k: float) -> 'np.ndarray':
    """Return the proper-time derivatives of the two slow-mode synchronous metric perturbations at comoving wavenumber k, for a universe whose non-axion species carry no perturbations. Write P for conj(psi_s)*dpsi_s + psi_s*conj(dpsi_s) and M for conj(psi_s)*dpsi_s - psi_s*conj(dpsi_s). The first output is ((2*m_pl**2*k**2/(a_s**2*hubble_s))*eta_s + (m/hubble_s)*Re(P)) / m_pl**2. The second output is the real part of ((1j*m/4)*M - (1/16)*abs(psi_s)**2*hdot_s - (3/8)*hubble_s*P - (1j*k**2/(16*m*a_s**2))*M) / m_pl**2, where hdot_s is the first output. The two are stacked along a new leading axis of length 2 in that order, both real.

    Returns
    -------
    ndarray of shape (2,) + np.shape(psi_s), float64: the time derivative of the trace metric perturbation followed by that of the curvature perturbation.

    Raises
    ------
    ValueError
        If m or m_pl is not strictly positive, or if any entry of a_s or hubble_s is not strictly positive.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_metric_slow_rates(psi_s: 'np.ndarray', dpsi_s: 'np.ndarray', eta_s: 'np.ndarray', hubble_s: 'np.ndarray', a_s: 'np.ndarray', m: float, m_pl: float, k: float) -> 'np.ndarray':
    psi_s = np.asarray(psi_s, dtype=complex)
    dpsi_s = np.asarray(dpsi_s, dtype=complex)
    eta_s = np.asarray(eta_s, dtype=float)
    hubble_s = np.asarray(hubble_s, dtype=float)
    a_s = np.asarray(a_s, dtype=float)
    m = float(m); m_pl = float(m_pl); k = float(k)
    if not (m > 0.0 and m_pl > 0.0):
        raise ValueError("m and m_pl must be strictly positive")
    if np.any(a_s <= 0.0) or np.any(hubble_s <= 0.0):
        raise ValueError("a_s and hubble_s must be strictly positive")
    plus = np.conj(psi_s) * dpsi_s + psi_s * np.conj(dpsi_s)
    minus = np.conj(psi_s) * dpsi_s - psi_s * np.conj(dpsi_s)
    # Eq (10).  The second entry needs the first, so the order is forced.
    hdot_s = ((2.0 * m_pl ** 2 * k ** 2 / (a_s ** 2 * hubble_s)) * eta_s
              + (m / hubble_s) * np.real(plus)) / m_pl ** 2
    # Eq (11).
    etadot_s = ((1j * m / 4.0) * minus
                - (1.0 / 16.0) * np.abs(psi_s) ** 2 * hdot_s
                - (3.0 / 8.0) * hubble_s * plus
                - (1j * k ** 2 / (16.0 * m * a_s ** 2)) * minus) / m_pl ** 2
    return np.stack(np.broadcast_arrays(np.real(hdot_s), np.real(etadot_s)), axis=0)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return independent test specifications."""
    return [
        {
            "setup": (
                "import numpy as np\n"
            ),
            "call": (
                "metric_slow_rates(\n"
                "    0.05 + 0.01j,\n"
                "    0.001 + 0.0002j,\n"
                "    0.001,\n"
                "    0.03,\n"
                "    2.0,\n"
                "    1.0,\n"
                "    1.0,\n"
                "    0.5,\n"
                ")"
            ),
            "gold_call": (
                "_oracle_metric_slow_rates(\n"
                "    0.05 + 0.01j,\n"
                "    0.001 + 0.0002j,\n"
                "    0.001,\n"
                "    0.03,\n"
                "    2.0,\n"
                "    1.0,\n"
                "    1.0,\n"
                "    0.5,\n"
                ")"
            ),
        },
        {
            "setup": (
                "import numpy as np\n"
                "\n"
                "z = np.array([0.02 + 0j, 0.0 + 0.03j])\n"
                "dz = np.array([1e-3 + 0j, 0.0 + 2e-3j])\n"
                "et = np.array([1e-3, 2e-3])\n"
                "h = np.array([0.01, 0.02])\n"
                "aa = np.array([1.5, 2.5])\n"
            ),
            "call": (
                "metric_slow_rates(\n"
                "    z.copy(),\n"
                "    dz.copy(),\n"
                "    et.copy(),\n"
                "    h.copy(),\n"
                "    aa.copy(),\n"
                "    2.0,\n"
                "    1.0,\n"
                "    0.8,\n"
                ")"
            ),
            "gold_call": (
                "_oracle_metric_slow_rates(\n"
                "    z.copy(),\n"
                "    dz.copy(),\n"
                "    et.copy(),\n"
                "    h.copy(),\n"
                "    aa.copy(),\n"
                "    2.0,\n"
                "    1.0,\n"
                "    0.8,\n"
                ")"
            ),
        },
        {
            "setup": (
                "import numpy as np\n"
            ),
            "call": (
                "metric_slow_rates(\n"
                "    0.05 + 0.01j, 0.0 + 0j, 0.0, 0.03, 1.0, 1.0, 1.0, 0.5\n"
                ")"
            ),
            "gold_call": (
                "_oracle_metric_slow_rates(\n"
                "    0.05 + 0.01j, 0.0 + 0j, 0.0, 0.03, 1.0, 1.0, 1.0, 0.5\n"
                ")"
            ),
        },
        {
            "setup": (
                "import numpy as np\n"
                "\n"
                "\n"
                "def _raises(fn):\n"
                "    try:\n"
                "        fn(\n"
                "            0.05 + 0.01j,\n"
                "            1e-3 + 0j,\n"
                "            1e-3,\n"
                "            -0.03,\n"
                "            2.0,\n"
                "            1.0,\n"
                "            1.0,\n"
                "            0.5,\n"
                "        )\n"
                "        return False\n"
                "    except ValueError:\n"
                "        return True\n"
            ),
            "call": (
                "_raises(metric_slow_rates)"
            ),
            "gold_call": (
                "_raises(_oracle_metric_slow_rates)"
            ),
        },
    ]
