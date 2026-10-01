"""
Return the proper-time derivative of the exact complex wavefunction psi at time t, given the Hubble rate hubble and the axion mass m. The result is complex and has the shape of psi. t may be a scalar or an array, and broadcasts against the other array inputs.

Written for the wavefunction rather than the field, the background equation of motion is first order and its explicit time dependence is what makes it stiff once the oscillation is faster than the expansion.

Returns
-------
ndarray with the shape of psi, complex128: the time derivative of psi.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def exact_rate(psi: 'np.ndarray', hubble: 'np.ndarray', m: float, t: 'np.ndarray') -> 'np.ndarray':
    """Return the proper-time derivative of the exact complex wavefunction psi at time t, given the Hubble rate hubble and the axion mass m. The result is complex and has the shape of psi. t may be a scalar or an array, and broadcasts against the other array inputs.

    Returns
    -------
    ndarray with the shape of psi, complex128: the time derivative of psi.

    Raises
    ------
    ValueError
        If m is not strictly positive.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_exact_rate(psi: 'np.ndarray', hubble: 'np.ndarray', m: float, t: 'np.ndarray') -> 'np.ndarray':
    psi = np.asarray(psi, dtype=complex)
    hubble = np.asarray(hubble, dtype=float)
    m = float(m); t = float(t) if isinstance(t, (int, float)) else np.asarray(t, dtype=float)
    if not (m > 0.0):
        raise ValueError("m must be strictly positive")
    # Eq (4).  Both terms carry 3/2; the second is conjugated and phase shifted.
    return -1.5 * hubble * psi + 1.5 * hubble * np.conj(psi) * np.exp(2j * m * t)

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
                "exact_rate(0.05 + 0.01j, 0.03, 1.0, 0.7)"
            ),
            "gold_call": (
                "_oracle_exact_rate(0.05 + 0.01j, 0.03, 1.0, 0.7)"
            ),
        },
        {
            "setup": (
                "import numpy as np\n"
                "\n"
                "z = np.array([0.02 + 0j, 0.0 + 0.03j])\n"
                "h = np.array([0.01, 0.02])\n"
            ),
            "call": (
                "exact_rate(z.copy(), h.copy(), 2.0, 1.3)"
            ),
            "gold_call": (
                "_oracle_exact_rate(z.copy(), h.copy(), 2.0, 1.3)"
            ),
        },
        {
            "setup": (
                "import numpy as np\n"
            ),
            "call": (
                "exact_rate(0.05 + 0.01j, 0.0, 1.0, 0.7)"
            ),
            "gold_call": (
                "_oracle_exact_rate(0.05 + 0.01j, 0.0, 1.0, 0.7)"
            ),
        },
        {
            "setup": (
                "import numpy as np\n"
                "\n"
                "z = np.array([0.02 + 0j, 0.0 + 0.03j])\n"
                "h = np.array([0.01, 0.02])\n"
                "tv = np.array([0.4, 1.9])\n"
            ),
            "call": (
                "exact_rate(z.copy(), h.copy(), 1.0, tv.copy())"
            ),
            "gold_call": (
                "_oracle_exact_rate(z.copy(), h.copy(), 1.0, tv.copy())"
            ),
        },
        {
            "setup": (
                "import numpy as np\n"
                "\n"
                "\n"
                "def _raises(fn):\n"
                "    try:\n"
                "        fn(0.05 + 0.01j, 0.05, -1.0, 0.7)\n"
                "        return False\n"
                "    except ValueError:\n"
                "        return True\n"
            ),
            "call": (
                "_raises(exact_rate)"
            ),
            "gold_call": (
                "_raises(_oracle_exact_rate)"
            ),
        },
    ]
