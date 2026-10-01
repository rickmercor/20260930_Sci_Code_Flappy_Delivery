"""
Step 07: Threshold intensity for a given confidence within a given time (orchestrator).

Smallest laser intensity at which a chosen confidence of overtone participation is reached within a given time (orchestrator).

For the resonantly driven Morse molecule of the overtone step, raising the intensity makes the overtone take part in
the flow of probability sooner, but not monotonically: the stronger field also shifts the levels, so the fixed laser
frequency drifts off the multiphoton resonance, and at some intensities the confidence reached by a fixed time falls
again as the intensity grows. The design question is the smallest intensity at which one is b percent confident, t_end
atomic time units after the field is switched on, that the overtone v = N has participated, that is, at which the
excitation delay tau_b equals t_end.

The search runs over the intensity grid I_j = I_min + j (I_max - I_min) / (n_scan - 1), j = 0 .. n_scan - 1, in
increasing order. At each grid intensity the non-participation history is computed and its excitation delay tau_b is
evaluated with sample times t_k = k. The first grid intensity with tau_b <= t_end decides the answer: if it is I_min,
the answer is I_min; otherwise the answer is the intensity between the previous grid point and this one at which
P_not(t_end) = 1 - b/100 exactly, located to an absolute precision of 1e-9 in TW/cm^2 (P_not(t_end) varies continuously
with the intensity). If no grid intensity reaches tau_b <= t_end, the confidence is not attainable on the grid.

Returns
-------
float, threshold intensity in TW/cm^2 at which the confidence of overtone participation reaches b percent at t_end
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def threshold_intensity(t_end: int, confidence: float, intensity_min: float, intensity_max: float, n_scan: int,
                        n_levels: int, target: int, mass: float, depth: float, alpha: float,
                        dipole_coeffs: "np.ndarray") -> float:
    '''Smallest intensity (TW/cm^2) on the scan at which the excitation delay tau_b of the overtone equals t_end.

    Parameters
    ----------
    t_end : int
        Time after switch-on, in atomic time units, at which the confidence is required; a positive integer.
    confidence : float
        Confidence level b in percent, 0 < b < 100.
    intensity_min : float
        Lower end of the intensity scan in TW/cm^2, positive.
    intensity_max : float
        Upper end of the intensity scan in TW/cm^2, larger than intensity_min.
    n_scan : int
        Number of equally spaced scan intensities, at least 2.
    n_levels : int
        Number of lowest bound Morse levels kept.
    target : int
        Overtone N driven at its N-photon resonance, 1 <= N <= n_levels - 1.
    mass : float
        Reduced mass in electron masses.
    depth : float
        Morse well depth in hartree.
    alpha : float
        Morse range parameter in inverse bohr.
    dipole_coeffs : np.ndarray
        Coefficients [c_0, c_1, ...] of the dipole polynomial in atomic units.

    Returns
    -------
    result : float
        The threshold intensity in TW/cm^2.

    Raises
    ------
    ValueError
        If intensity_min >= intensity_max, n_scan < 2, or no scan intensity reaches tau_b <= t_end, and for the invalid
        inputs of the overtone and excitation-delay steps.
    '''
    return result  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import brentq


def _oracle_threshold_intensity(t_end: int, confidence: float, intensity_min: float, intensity_max: float,
                                n_scan: int, n_levels: int, target: int, mass: float, depth: float, alpha: float,
                                dipole_coeffs: "np.ndarray") -> float:
    """Reference implementation."""
    import numpy as np
    from scipy.optimize import brentq
    if not intensity_min < intensity_max:
        raise ValueError("intensity_min must be smaller than intensity_max")
    if int(n_scan) < 2:
        raise ValueError("n_scan must be at least 2")
    times = np.arange(int(t_end) + 1, dtype=float)
    level = 1.0 - confidence / 100.0

    history = lambda intensity: _oracle_overtone_nonparticipation(intensity, t_end, n_levels, target, mass, depth,
                                                                  alpha, dipole_coeffs)
    previous = None
    for intensity in np.linspace(intensity_min, intensity_max, int(n_scan)):
        p_not = history(intensity)
        if _oracle_excitation_delay(times, p_not, confidence) <= t_end:
            if previous is None:
                return float(intensity)
            return float(brentq(lambda x: history(x)[-1] - level, previous, intensity, xtol=1e-12, rtol=1e-12))
        previous = float(intensity)
    raise ValueError("the confidence level is not reached on the intensity scan")

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: one-photon excitation of v = 1 in a two-level truncation, 90 % confidence after 5000 a.u. ---
        {
            "setup": "import numpy as np\n"
                     "c = np.array([0.7091, 0.3162, -0.0165])\n",
            "call": "threshold_intensity(5000, 90.0, 1.0, 8.0, 8, 2, 1, 1741.312, 0.225019, 1.174145, c)",
            "gold_call": "_oracle_threshold_intensity(5000, 90.0, 1.0, 8.0, 8, 2, 1, 1741.312, 0.225019, 1.174145, c)",
            "tol": 1e-6,
        },
        # --- Normal: two-photon excitation of v = 2 with five levels, 70 % confidence after 8000 a.u. ---
        {
            "setup": "import numpy as np\n"
                     "c = np.array([0.7091, 0.3162, -0.0165])\n",
            "call": "threshold_intensity(8000, 70.0, 2.0, 10.0, 5, 5, 2, 1741.312, 0.225019, 1.174145, c)",
            "gold_call": "_oracle_threshold_intensity(8000, 70.0, 2.0, 10.0, 5, 5, 2, 1741.312, 0.225019, 1.174145, c)",
            "tol": 1e-6,
        },
        # --- Boundary: three-level truncation where the confidence is reached and then lost again at higher
        #     intensity, so the scan must stop at the first crossing ---
        {
            "setup": "import numpy as np\n"
                     "c = np.array([0.7091, 0.3162, -0.0165])\n",
            "call": "threshold_intensity(12000, 82.0, 3.0, 9.0, 7, 3, 2, 1741.312, 0.225019, 1.174145, c)",
            "gold_call": "_oracle_threshold_intensity(12000, 82.0, 3.0, 9.0, 7, 3, 2, 1741.312, 0.225019, 1.174145, c)",
            "tol": 1e-6,
        },
        # --- Edge: the confidence is already reached at the lowest scan intensity ---
        {
            "setup": "import numpy as np\n"
                     "c = np.array([0.7091, 0.3162, -0.0165])\n",
            "call": "threshold_intensity(5000, 50.0, 3.0, 6.0, 4, 2, 1, 1741.312, 0.225019, 1.174145, c)",
            "gold_call": "_oracle_threshold_intensity(5000, 50.0, 3.0, 6.0, 4, 2, 1, 1741.312, 0.225019, 1.174145, c)",
            "tol": 1e-9,
        },
        # --- Normal: five-level two-photon drive, 93 % confidence after 12000 a.u., crossing between the last grid points ---
        {
            "setup": "import numpy as np\n"
                     "c = np.array([0.7091, 0.3162, -0.0165])\n",
            "call": "threshold_intensity(12000, 93.0, 5.0, 9.0, 5, 5, 2, 1741.312, 0.225019, 1.174145, c)",
            "gold_call": "_oracle_threshold_intensity(12000, 93.0, 5.0, 9.0, 5, 5, 2, 1741.312, 0.225019, 1.174145, c)",
            "tol": 1e-6,
        },
        # --- Error: a confidence that the scan never reaches must raise ValueError ---
        {
            "setup": "import numpy as np\n"
                     "def _probe(fn):\n"
                     "    try:\n"
                     "        fn(1000, 99.0, 0.1, 0.5, 3, 3, 2, 1741.312, 0.225019, 1.174145, np.array([0.7091, 0.3162]))\n"
                     "    except ValueError:\n"
                     "        return 1\n"
                     "    except Exception:\n"
                     "        return 2\n"
                     "    return 0\n",
            "call": "_probe(threshold_intensity)",
            "gold_call": "_probe(_oracle_threshold_intensity)",
        },
    ]
