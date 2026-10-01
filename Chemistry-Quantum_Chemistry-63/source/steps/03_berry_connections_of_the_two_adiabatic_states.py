"""
Return the Berry connections of the lower and the upper adiabatic state, in the orthonormal polar frame of the nuclear plane.

With the phase of each adiabatic state fixed as in the previous step, the Berry connection of a state is the expectation value of minus i times the nuclear gradient taken in that state. Both connections are reported in the orthonormal polar frame attached to the nuclear plane, so that a gradient has the radial component d/dQ and the angular component (1/Q) d/d theta. The connection of the lower state is what the geometric phase of the next step integrates around the loop. The connection of the upper state is returned alongside it because later steps need both.

Returns
-------
np.ndarray of shape (2, 2): Berry connection of the lower state in row 0 and of the upper state in row 1, radial then angular component
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def berry_connections(Q: float, theta: float, g: float, K: float,
                      Delta: float) -> "np.ndarray":
    '''Berry connections of the lower and the upper adiabatic state.

    Parameters
    ----------
    Q : float
        Radial nuclear coordinate, strictly positive.
    theta : float
        Polar angle of the nuclear coordinate, in radians.
    g : float
        Linear vibronic coupling strength, strictly positive.
    K : float
        Harmonic force constant of the two normal modes.
    Delta : float
        Strength of the constant term that opens the gap, strictly positive.

    Returns
    -------
    connections : np.ndarray
        Real array of shape (2, 2). Row 0 is the connection of the lower state
        and row 1 the connection of the upper state; within a row column 0 is
        the radial component and column 1 the angular component, both in the
        orthonormal polar frame.

    Raises
    ------
    ValueError
        If any argument is not a finite scalar, or if Q, g or Delta is not
        strictly positive.
    '''
    return connections

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _check_point(Q, theta, g, K, Delta):
    vals = [Q, theta, g, K, Delta]

    if any(np.ndim(v) != 0 for v in vals):
        raise ValueError("Q, theta, g, K and Delta must all be scalars")

    Q, theta, g, K, Delta = (float(v) for v in vals)

    if not np.all(np.isfinite([Q, theta, g, K, Delta])):
        raise ValueError("every model parameter must be finite")

    if Q <= 0.0:
        raise ValueError(
            "Q must be strictly positive: the loop may not pass through the origin"
        )

    if g <= 0.0:
        raise ValueError("g must be strictly positive")

    if Delta <= 0.0:
        raise ValueError(
            "Delta must be strictly positive: at Delta = 0 the lower state is "
            "antiperiodic and the phase convention below cannot be imposed"
        )

    return Q, theta, g, K, Delta


def _grad_alpha(Q, theta, g, Delta):
    """(d alpha / dQ, (1/Q) d alpha / d theta) of the azimuth of the field direction."""
    S2 = g * g * Q * Q * np.sin(theta) ** 2 + Delta * Delta

    return np.array([
        Delta * g * np.sin(theta) / S2,
        Delta * g * np.cos(theta) / S2,
    ])


def _oracle_berry_connections(
    Q: float,
    theta: float,
    g: float,
    K: float,
    Delta: float,
) -> "np.ndarray":
    Q, theta, g, K, Delta = _check_point(
        Q,
        theta,
        g,
        K,
        Delta,
    )

    stacked = _oracle_adiabatic_states(
        Q,
        theta,
        g,
        K,
        Delta,
    )

    lower_second = stacked[0][1, 0]
    upper_second = stacked[0][1, 1]
    da = _grad_alpha(Q, theta, g, Delta)

    return np.stack([
        -(1.0 - lower_second**2) * da,
        -(1.0 - upper_second**2) * da,
    ])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {"setup": "import numpy as np\n",
         "call": "np.round(berry_connections(0.20, 0.7, 0.60, 1.00, 0.12), 9)",
         "gold_call": "np.round(_oracle_berry_connections(0.20, 0.7, 0.60, 1.00, 0.12), 9)"},
        {"setup": "import numpy as np\n",
         "call": "np.round(berry_connections(0.20, 0.0, 0.60, 1.00, 0.12), 9)",
         "gold_call": "np.round(_oracle_berry_connections(0.20, 0.0, 0.60, 1.00, 0.12), 9)"},
        {"setup": "import numpy as np\n",
         "call": "np.round(berry_connections(0.20, np.pi / 2, 0.60, 1.00, 0.12), 9)",
         "gold_call": "np.round(_oracle_berry_connections(0.20, np.pi / 2, 0.60, 1.00, 0.12), 9)"},
        {"setup": "import numpy as np\n",
         "call": "np.round(berry_connections(0.20, 3.90, 0.60, 1.00, 0.12), 9)",
         "gold_call": "np.round(_oracle_berry_connections(0.20, 3.90, 0.60, 1.00, 0.12), 9)"},
        {"setup": "import numpy as np\n",
         "call": "np.round(berry_connections(1.60, 2.20, 0.45, 0.80, 0.05), 9)",
         "gold_call": "np.round(_oracle_berry_connections(1.60, 2.20, 0.45, 0.80, 0.05), 9)"},
        {"setup": "import numpy as np\n",
         "call": "np.round(berry_connections(0.03, 1.10, 0.70, 1.00, 0.90), 9)",
         "gold_call": "np.round(_oracle_berry_connections(0.03, 1.10, 0.70, 1.00, 0.90), 9)"},
    ]
