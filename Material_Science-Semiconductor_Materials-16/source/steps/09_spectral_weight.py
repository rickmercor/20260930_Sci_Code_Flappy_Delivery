"""
Return the integral of the real part of the optical conductivity divided by photon energy, taken by the trapezoidal rule over the photon energies supplied and the conductivity sampled at them. The source fixes a convention here that the natural reading does not; follow the source.

This integral is the optical side of a sum rule whose other side is the quantum metric, so the two should agree once the broadening is small. The caller supplies the window rather than the step choosing it, because the broadened spectrum does not vanish where the true one does, and the integrand is weighted by the reciprocal of the photon energy.

Returns
-------
float, the integrated spectral weight in Angstrom^2.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def spectral_weight(omega: "np.ndarray", conductivity: "np.ndarray") -> float:
    """Return the integral of the real part of the optical conductivity divided by photon energy, taken by the trapezoidal rule over the photon energies supplied and the conductivity sampled at them. The source fixes a convention here that the natural reading does not; follow the source.

    Returns
    -------
    float, the integrated spectral weight in Angstrom^2.

    Raises
    ------
    ValueError: if omega and conductivity are not matching one-dimensional arrays of at least two points, or if any photon energy is not positive.
    """
    return 0.0  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

HBAR2_OVER_ME = 7.6199682     # hbar^2 / m_e in eV Angstrom^2


def _oracle_spectral_weight(omega: "np.ndarray", conductivity: "np.ndarray") -> float:
    w = np.asarray(omega, dtype=float)
    s = np.asarray(conductivity, dtype=float)
    if w.ndim != 1 or w.shape != s.shape or w.shape[0] < 2:
        raise ValueError("omega and conductivity must be matching 1-D arrays of at least two points")
    if np.any(w <= 0.0):
        raise ValueError("every photon energy must be positive")
    # CONVENTION: the window starts at a POSITIVE omega_min. A Lorentzian does not vanish at
    # zero frequency, so Re sigma / omega has a 1/omega tail and an integral from zero
    # diverges logarithmically with the grid; cutting below the onset removes the artefact.
    y = s / w
    # trapezoidal rule written out: np.trapezoid exists only from numpy 2.0 and np.trapz was
    # removed in that release, so neither name runs on both sides of the boundary.
    return float(np.sum(np.diff(w) * (y[1:] + y[:-1]) / 2.0))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "import numpy as np\nW=np.linspace(1.0,14.0,401)",
         "call": "spectral_weight(W, np.exp(-(W-3.1)**2)*2.0+0.05)",
         "gold_call": "_oracle_spectral_weight(W, np.exp(-(W-3.1)**2)*2.0+0.05)"},   # normal
        {"setup": "import numpy as np",
         "call": "spectral_weight(np.array([1.0,2.0]), np.array([0.5,0.25]))",
         "gold_call": "_oracle_spectral_weight(np.array([1.0,2.0]), np.array([0.5,0.25]))"},   # boundary
        {"setup": "import numpy as np\nW=np.linspace(1.0,14.0,401)",
         "call": "spectral_weight(W*0.01+0.2, np.sin(W)**2*1e-3)",
         "gold_call": "_oracle_spectral_weight(W*0.01+0.2, np.sin(W)**2*1e-3)"},   # edge
        {"setup": "import numpy as np\ndef _c():\n    try:\n        spectral_weight(np.array([0.0,2.0]), np.array([0.5,0.25]))\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef _g():\n    try:\n        _oracle_spectral_weight(np.array([0.0,2.0]), np.array([0.5,0.25]))\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2",
         "call": "_c()",
         "gold_call": "_g()"},   # invalid input
    ]
