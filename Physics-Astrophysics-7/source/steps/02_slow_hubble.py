"""
Return the Hubble rate that the effective theory assigns to the slow-mode wavefunction psi_s, with rho_other_s the slow-mode energy density of all other species. Write A for abs(psi_s)**2. The effective total density is m*A + rho_other_s + (3*A/(32*m*m_pl**2)) * (m*A + 2*rho_other_s), and the slow-mode Hubble rate is sqrt of that total divided by 3*m_pl**2. Note that this is not the exact constraint of the previous step with slow variables substituted: the effective theory carries a relativistic correction of its own, the term with coefficient 3/32 above. Raise ValueError if the total comes out negative.

The effective theory reorganises the Friedmann constraint as well as the equation of motion, so the relation between the slow-mode wavefunction and the slow-mode expansion rate carries its own relativistic term. It defines the slow-mode expansion rate rather than following from the exact constraint of the previous step, which is why substituting slow variables into that constraint gives a different and wrong answer.

Returns
-------
ndarray with the shape of psi_s, float64: the slow-mode Hubble rate.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def slow_hubble(psi_s: 'np.ndarray', rho_other_s: 'np.ndarray', m: float, m_pl: float) -> 'np.ndarray':
    """Return the Hubble rate that the effective theory assigns to the slow-mode wavefunction psi_s, with rho_other_s the slow-mode energy density of all other species. Write A for abs(psi_s)**2. The effective total density is m*A + rho_other_s + (3*A/(32*m*m_pl**2)) * (m*A + 2*rho_other_s), and the slow-mode Hubble rate is sqrt of that total divided by 3*m_pl**2. Note that this is not the exact constraint of the previous step with slow variables substituted: the effective theory carries a relativistic correction of its own, the term with coefficient 3/32 above. Raise ValueError if the total comes out negative.

    Returns
    -------
    ndarray with the shape of psi_s, float64: the slow-mode Hubble rate.

    Raises
    ------
    ValueError
        If m or m_pl is not strictly positive, or if the total slow-mode density is negative.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_slow_hubble(psi_s: 'np.ndarray', rho_other_s: 'np.ndarray', m: float, m_pl: float) -> 'np.ndarray':
    psi_s = np.asarray(psi_s, dtype=complex)
    rho_other_s = np.asarray(rho_other_s, dtype=float)
    m = float(m); m_pl = float(m_pl)
    if not (m > 0.0 and m_pl > 0.0):
        raise ValueError("m and m_pl must be strictly positive")
    a2 = np.abs(psi_s) ** 2
    # Eq (9).  The slow-mode Friedmann is NOT the exact one with slow variables
    # substituted: it carries a relativistic correction of its own.
    tot = (m * a2 + rho_other_s
           + (3.0 * a2 / (32.0 * m * m_pl ** 2)) * (m * a2 + 2.0 * rho_other_s))
    if np.any(tot < 0.0):
        raise ValueError("total slow-mode density is negative")
    return np.sqrt(tot / (3.0 * m_pl ** 2))

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
                "slow_hubble(0.05 + 0.01j, 0.004, 1.0, 1.0)"
            ),
            "gold_call": (
                "_oracle_slow_hubble(0.05 + 0.01j, 0.004, 1.0, 1.0)"
            ),
        },
        {
            "setup": (
                "import numpy as np\n"
                "\n"
                "z = np.array([0.02 + 0j, 0.0 + 0.03j])\n"
                "r = np.array([1e-3, 2e-3])\n"
            ),
            "call": (
                "slow_hubble(z.copy(), r.copy(), 2.0, 1.0)"
            ),
            "gold_call": (
                "_oracle_slow_hubble(z.copy(), r.copy(), 2.0, 1.0)"
            ),
        },
        {
            "setup": (
                "import numpy as np\n"
            ),
            "call": (
                "slow_hubble(0.0 + 0j, 1e-08, 1.0, 1.0)"
            ),
            "gold_call": (
                "_oracle_slow_hubble(0.0 + 0j, 1e-08, 1.0, 1.0)"
            ),
        },
        {
            "setup": (
                "import numpy as np\n"
                "\n"
                "\n"
                "def _raises(fn):\n"
                "    try:\n"
                "        fn(0.05 + 0.01j, -1.0, 1.0, 1.0)\n"
                "        return False\n"
                "    except ValueError:\n"
                "        return True\n"
            ),
            "call": (
                "_raises(slow_hubble)"
            ),
            "gold_call": (
                "_raises(_oracle_slow_hubble)"
            ),
        },
    ]
