"""
Deterministic construction of the oblique Alfvenic mode spectrum.

Deterministic construction of the oblique Alfvenic mode spectrum.

The benchmark spectrum consists of $num_modes$ modes that share one propagation
direction in the x-z plane.  The mode frequencies are the equally spaced values
covering a fixed band, the amplitudes follow a power law in frequency and are
renormalised to a prescribed total wave power, and every mode carries the
wavevector implied by its own frequency and the common propagation angle.

Returns
-------
Stacked array of the four per-mode quantities used by every later step.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import math

import numpy as np


def mode_spectrum(
    num_modes: int = 11,
    bw2: float = 0.19,
    omega1: float = 0.11,
    q: float = 1.667,
    tan_alpha: float = 4.4,
    band_width: float = 0.08,
) -> np.ndarray:
    """Return the per-mode spectrum of the benchmark wave field.

    Parameters
    ----------
    num_modes : int
        Number of wave modes in the spectrum. Must be at least 1.
    bw2 : float
        Total dimensionless wave power, the sum of the squared mode amplitudes.
        Must be positive.
    omega1 : float
        Lowest dimensionless mode frequency. Must be positive.
    q : float
        Exponent of the amplitude power law in frequency.
    tan_alpha : float
        Tangent of the common propagation angle measured from the background
        field direction. Must be positive.
    band_width : float
        Width of the dimensionless frequency band above ``omega1``.
        Must be positive.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(4, num_modes)`` holding, in order, the mode
        frequencies, the mode amplitudes, the perpendicular wavenumber
        components and the parallel wavenumber components.

    Raises
    ------
    ValueError
        If ``num_modes`` is below one, or if ``bw2``, ``omega1``, ``tan_alpha``
        or ``band_width`` is not strictly positive.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math

import numpy as np


def _oracle_mode_spectrum(
    num_modes: int = 11,
    bw2: float = 0.19,
    omega1: float = 0.11,
    q: float = 1.667,
    tan_alpha: float = 4.4,
    band_width: float = 0.08,
) -> np.ndarray:
    if int(num_modes) < 1:
        raise ValueError("num_modes must be at least 1")
    if float(bw2) <= 0.0:
        raise ValueError("bw2 must be positive")
    if float(omega1) <= 0.0:
        raise ValueError("omega1 must be positive")
    if float(tan_alpha) <= 0.0:
        raise ValueError("tan_alpha must be positive")
    if float(band_width) <= 0.0:
        raise ValueError("band_width must be positive")
    omega = np.linspace(float(omega1), float(omega1) + float(band_width), int(num_modes))
    weights = (omega / float(omega1)) ** (-float(q))
    amplitude = np.sqrt(float(bw2) * weights / weights.sum())
    k_par = omega.copy()
    k_perp = k_par * float(tan_alpha)
    return np.vstack((omega, amplitude, k_perp, k_par))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return deterministic normal, boundary and edge cases."""
    return [
        {
            "setup": "import numpy as np\n",
            "call": "mode_spectrum()",
            "gold_call": "_oracle_mode_spectrum()",
        },
        {
            "setup": "import numpy as np\n",
            "call": "mode_spectrum(num_modes=1, bw2=0.02, omega1=0.05, q=0.0, tan_alpha=1.0)",
            "gold_call": "_oracle_mode_spectrum(num_modes=1, bw2=0.02, omega1=0.05, q=0.0, tan_alpha=1.0)",
        },
        {
            "setup": (
                "import numpy as np\n"
                "def _probe(fn):\n"
                "    try:\n"
                "        fn(num_modes=0)\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "    return 0\n"
            ),
            "call": "_probe(mode_spectrum)",
            "gold_call": "_probe(_oracle_mode_spectrum)",
        },
        {
            "setup": "import numpy as np\n",
            "call": "mode_spectrum(num_modes=15, bw2=0.30, omega1=0.02, q=2.5, tan_alpha=9.0, band_width=0.2)",
            "gold_call": "_oracle_mode_spectrum(num_modes=15, bw2=0.30, omega1=0.02, q=2.5, tan_alpha=9.0, band_width=0.2)",
        },
    ]
