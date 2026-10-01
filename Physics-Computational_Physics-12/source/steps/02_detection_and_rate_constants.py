"""
Registration probabilities and rate constants of the two resolved junctions.

The four short-time coefficients of a junction carry exactly enough information to separate how often its transitions are registered from how fast that junction actually runs. Eliminate the two rate constants from the four short-time relations of step 01, which leaves each registration probability of the junction as a closed expression in the two zero-time densities $c_{L+}$, $c_{L-}$ and the two zero-time slopes $a_{L++}$, $a_{L--}$ of that junction. Each rate constant then follows from the zero-time density of its own direction.

Returns
-------
A NumPy array of shape `(2, 4)` whose row `L` holds `[eta_plus, eta_minus, k_plus, k_minus]`, the first two dimensionless and the last two in inverse microseconds. Raise `ValueError` if a zero-time density is not positive, if a zero-time slope is negative, or if any input is non-finite.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

"""Registration probabilities and rate constants of the two resolved junctions."""

import numpy as np


def detection_and_rate_constants(short_time):
    """Invert the short-time relations of both incompletely registered junctions.

    Parameters
    ----------
    short_time : array_like, shape (2, 4)
        Row ``L`` holds ``[c_plus, c_minus, a_pp, a_mm]`` for junction ``L`` as
        returned by the short-time stage, the first two entries in inverse
        microseconds and the last two in inverse microseconds squared.

    Returns
    -------
    numpy.ndarray, shape (2, 4)
        Row ``L`` holds ``[eta_plus, eta_minus, k_plus, k_minus]``: the two
        registration probabilities of that junction, dimensionless, followed by
        its two junction rate constants in inverse microseconds.
    """
    return np.zeros((2, 4))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_detection_and_rate_constants(short_time):
    import numpy as np
    values = np.asarray(short_time, dtype=float)
    if values.shape != (2, 4) or not np.all(np.isfinite(values)):
        raise ValueError("short_time must be a finite array of shape (2, 4)")
    out = np.zeros((2, 4))
    for link in range(2):
        c_plus, c_minus, a_pp, a_mm = values[link]
        if c_plus <= 0.0 or c_minus <= 0.0:
            raise ValueError("both zero-time densities of each junction must be positive")
        if a_pp < 0.0 or a_mm < 0.0:
            raise ValueError("both zero-time slopes of each junction must be non-negative")
        product = c_plus * c_minus
        eta_plus = product / (product + a_mm)
        eta_minus = product / (product + a_pp)
        out[link] = [eta_plus, eta_minus, c_plus / eta_plus, c_minus / eta_minus]
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Differential cases for the registration-probability inversion."""
    asymmetric = (
        "import numpy as np\n"
        "spec = [(0.35, 0.88, 2.9, 6.1), (0.78, 0.42, 6.2, 3.2)]\n"
        "short_time = np.zeros((2, 4))\n"
        "for link, (ep, em, kp, km) in enumerate(spec):\n"
        "    cp, cm = ep * kp, em * km\n"
        "    short_time[link] = [cp, cm, (1.0 - em) * km * cp, (1.0 - ep) * kp * cm]\n"
    )
    symmetric = (
        "import numpy as np\n"
        "spec = [(0.5, 0.5, 3.0, 3.0), (0.64, 0.64, 4.5, 1.8)]\n"
        "short_time = np.zeros((2, 4))\n"
        "for link, (ep, em, kp, km) in enumerate(spec):\n"
        "    cp, cm = ep * kp, em * km\n"
        "    short_time[link] = [cp, cm, (1.0 - em) * km * cp, (1.0 - ep) * kp * cm]\n"
    )
    complete = (
        "import numpy as np\n"
        "short_time = np.array([[4.25, 1.75, 0.0, 0.0], [2.60, 3.40, 0.0, 0.0]])\n"
    )
    return [
        {
            "setup": asymmetric,
            "call": "detection_and_rate_constants(short_time)",
            "gold_call": "_oracle_detection_and_rate_constants(short_time)",
        },
        {
            "setup": symmetric,
            "call": "detection_and_rate_constants(short_time)",
            "gold_call": "_oracle_detection_and_rate_constants(short_time)",
        },
        {
            "setup": complete,
            "call": "detection_and_rate_constants(short_time)",
            "gold_call": "_oracle_detection_and_rate_constants(short_time)",
        },
    ]
