"""
Return the Hubble rate of a flat universe containing the axion described by the complex wavefunction psi together with all other species, whose combined energy density is rho_other. m is the axion mass and m_pl the reduced Planck mass. Raise ValueError if the total density comes out negative.

This is the ordinary first Friedmann equation applied to the exact, oscillating axion density rather than to any averaged version of it.

Returns
-------
ndarray with the shape of psi, float64: the Hubble rate.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def exact_hubble(psi: 'np.ndarray', rho_other: 'np.ndarray', m: float, m_pl: float) -> 'np.ndarray':
    """Return the Hubble rate of a flat universe containing the axion described by the complex wavefunction psi together with all other species, whose combined energy density is rho_other. m is the axion mass and m_pl the reduced Planck mass. Raise ValueError if the total density comes out negative.

    Returns
    -------
    ndarray with the shape of psi, float64: the Hubble rate.

    Raises
    ------
    ValueError
        If m or m_pl is not strictly positive, or if the total density is negative.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_exact_hubble(psi: 'np.ndarray', rho_other: 'np.ndarray', m: float, m_pl: float) -> 'np.ndarray':
    psi = np.asarray(psi, dtype=complex)
    rho_other = np.asarray(rho_other, dtype=float)
    m = float(m); m_pl = float(m_pl)
    if not (m > 0.0 and m_pl > 0.0):
        raise ValueError("m and m_pl must be strictly positive")
    # First Friedmann equation with the exact axion density of Eq (21).
    tot = m * np.abs(psi) ** 2 + rho_other
    if np.any(tot < 0.0):
        raise ValueError("total background density is negative")
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
                "exact_hubble(0.05 + 0.01j, 0.004, 1.0, 1.0)"
            ),
            "gold_call": (
                "_oracle_exact_hubble(0.05 + 0.01j, 0.004, 1.0, 1.0)"
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
                "exact_hubble(z.copy(), r.copy(), 2.0, 1.0)"
            ),
            "gold_call": (
                "_oracle_exact_hubble(z.copy(), r.copy(), 2.0, 1.0)"
            ),
        },
        {
            "setup": (
                "import numpy as np\n"
            ),
            "call": (
                "exact_hubble(0.0 + 0j, 1e-08, 1.0, 1.0)"
            ),
            "gold_call": (
                "_oracle_exact_hubble(0.0 + 0j, 1e-08, 1.0, 1.0)"
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
                "_raises(exact_hubble)"
            ),
            "gold_call": (
                "_raises(_oracle_exact_hubble)"
            ),
        },
    ]
