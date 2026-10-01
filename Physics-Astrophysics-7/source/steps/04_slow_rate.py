"""
Return the proper-time derivative of the slow-mode wavefunction psi_s under the effective theory, given the slow-mode Hubble rate hubble_s and the slow-mode energy density and pressure of all other species. Write A for abs(psi_s)**2. The derivative is the sum of three terms: -1.5*hubble_s*psi_s, plus (3j/(16*m*m_pl**2)) * psi_s * (3*m*A + 2*rho_other_s), minus (9/(32*m**2*m_pl**2)) * hubble_s * psi_s * (m*A + rho_other_s + p_other_s). The result is complex and has the shape of psi_s. It carries no explicit dependence on time.

Removing the explicit oscillation from the equation of motion is what makes the effective system non-stiff; what is left are relativistic corrections built from the wavefunction amplitude and the other species.

Returns
-------
ndarray with the shape of psi_s, complex128: the time derivative of psi_s.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def slow_rate(psi_s: 'np.ndarray', hubble_s: 'np.ndarray', rho_other_s: 'np.ndarray', p_other_s: 'np.ndarray', m: float, m_pl: float) -> 'np.ndarray':
    """Return the proper-time derivative of the slow-mode wavefunction psi_s under the effective theory, given the slow-mode Hubble rate hubble_s and the slow-mode energy density and pressure of all other species. Write A for abs(psi_s)**2. The derivative is the sum of three terms: -1.5*hubble_s*psi_s, plus (3j/(16*m*m_pl**2)) * psi_s * (3*m*A + 2*rho_other_s), minus (9/(32*m**2*m_pl**2)) * hubble_s * psi_s * (m*A + rho_other_s + p_other_s). The result is complex and has the shape of psi_s. It carries no explicit dependence on time.

    Returns
    -------
    ndarray with the shape of psi_s, complex128: the time derivative of psi_s.

    Raises
    ------
    ValueError
        If m or m_pl is not strictly positive.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_slow_rate(psi_s: 'np.ndarray', hubble_s: 'np.ndarray', rho_other_s: 'np.ndarray', p_other_s: 'np.ndarray', m: float, m_pl: float) -> 'np.ndarray':
    psi_s = np.asarray(psi_s, dtype=complex)
    hubble_s = np.asarray(hubble_s, dtype=float)
    rho_other_s = np.asarray(rho_other_s, dtype=float)
    p_other_s = np.asarray(p_other_s, dtype=float)
    m = float(m); m_pl = float(m_pl)
    if not (m > 0.0 and m_pl > 0.0):
        raise ValueError("m and m_pl must be strictly positive")
    a2 = np.abs(psi_s) ** 2
    # Eq (7): a damping term, a purely imaginary term, and a real damping correction.
    term1 = -1.5 * hubble_s * psi_s
    term2 = (3j / (16.0 * m * m_pl ** 2)) * psi_s * (3.0 * m * a2 + 2.0 * rho_other_s)
    term3 = -(9.0 / (32.0 * m ** 2 * m_pl ** 2)) * hubble_s * psi_s * (
        m * a2 + rho_other_s + p_other_s)
    return term1 + term2 + term3

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
                "slow_rate(0.05 + 0.01j, 0.03, 0.004, 0.0005, 1.0, 1.0)"
            ),
            "gold_call": (
                "_oracle_slow_rate(\n"
                "    0.05 + 0.01j, 0.03, 0.004, 0.0005, 1.0, 1.0\n"
                ")"
            ),
        },
        {
            "setup": (
                "import numpy as np\n"
                "\n"
                "z = np.array([0.02 + 0j, 0.0 + 0.03j])\n"
                "h = np.array([0.01, 0.02])\n"
                "r = np.array([1e-3, 2e-3])\n"
                "p = np.array([1e-4, 2e-4])\n"
            ),
            "call": (
                "slow_rate(z.copy(), h.copy(), r.copy(), p.copy(), 2.0, 1.0)"
            ),
            "gold_call": (
                "_oracle_slow_rate(\n"
                "    z.copy(), h.copy(), r.copy(), p.copy(), 2.0, 1.0\n"
                ")"
            ),
        },
        {
            "setup": (
                "import numpy as np\n"
            ),
            "call": (
                "slow_rate(0.05 + 0.01j, 0.0, 0.0, 0.0, 1.0, 1.0)"
            ),
            "gold_call": (
                "_oracle_slow_rate(0.05 + 0.01j, 0.0, 0.0, 0.0, 1.0, 1.0)"
            ),
        },
        {
            "setup": (
                "import numpy as np\n"
                "\n"
                "\n"
                "def _raises(fn):\n"
                "    try:\n"
                "        fn(0.05 + 0.01j, 0.05, 0.004, 0.001, -1.0, 1.0)\n"
                "        return False\n"
                "    except ValueError:\n"
                "        return True\n"
            ),
            "call": (
                "_raises(slow_rate)"
            ),
            "gold_call": (
                "_raises(_oracle_slow_rate)"
            ),
        },
    ]
