"""
Evaluate the non-interacting density of states of the Bethe lattice with half-bandwidth D on a 1-D grid of energies, normalised to unit integral and identically zero outside the band.

Every closure of the iterated moment-quadrature scheme integrates the reconstructed spectral function against the bare band, so the bare density of states is the kernel that couples the local moments to the lattice.

Returns
-------
numpy.ndarray, Density of states at each energy, same shape as omega (float64).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def semielliptic_dos(omega: "np.ndarray", half_bandwidth: float) -> "np.ndarray":
    """Evaluate the non-interacting density of states of the Bethe lattice with half-bandwidth D on a 1-D grid of energies, normalised to unit integral and identically zero outside the band.

    Parameters
    ----------
    omega : numpy.ndarray
        Non-empty finite 1-D array of energies.
    half_bandwidth : float
        Positive half-bandwidth D of the semielliptic band.

    Returns
    -------
    rho : numpy.ndarray
        Density of states at each energy, same shape as omega (float64).

    Raises
    ------
    ValueError
        If omega is not a non-empty finite 1-D array, or half_bandwidth is not positive and finite.
    """
    return rho

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_semielliptic_dos(omega: "np.ndarray", half_bandwidth: float) -> "np.ndarray":
    """Bare Bethe-lattice density of states rho0(w) = (2 / (pi D^2)) sqrt(D^2 - w^2) on |w| < D."""
    omega = np.asarray(omega, dtype=np.float64)
    if omega.ndim != 1 or omega.size < 1 or not np.all(np.isfinite(omega)):
        raise ValueError("omega must be a non-empty finite 1-D array")
    if not (np.isfinite(half_bandwidth) and half_bandwidth > 0.0):
        raise ValueError("half_bandwidth must be positive and finite")
    D = float(half_bandwidth)
    rho = np.zeros_like(omega)
    inside = np.abs(omega) < D
    rho[inside] = 2.0 / (np.pi * D * D) * np.sqrt(D * D - omega[inside] ** 2)
    return rho

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nomega = np.linspace(-1.5, 1.5, 13)\nhalf_bandwidth = 1.0\n",
            "call": "semielliptic_dos(omega, half_bandwidth)",
            "gold_call": "_oracle_semielliptic_dos(omega, half_bandwidth)",
        },
        {
            "setup": "import numpy as np\nomega = np.array([-1.0, 0.0, 1.0])\nhalf_bandwidth = 1.0\n",
            "call": "semielliptic_dos(omega, half_bandwidth)",
            "gold_call": "_oracle_semielliptic_dos(omega, half_bandwidth)",
        },
        {
            "setup": "import numpy as np\nomega = np.array([-0.25, 0.75])\nhalf_bandwidth = 2.5\n",
            "call": "semielliptic_dos(omega, half_bandwidth)",
            "gold_call": "_oracle_semielliptic_dos(omega, half_bandwidth)",
        },
        {
            "setup": "import numpy as np\nomega = np.array([0.0, 0.5])\nhalf_bandwidth = 0.0\ndef run_model():\n    try:\n        semielliptic_dos(omega, half_bandwidth)\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_semielliptic_dos(omega, half_bandwidth)\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
