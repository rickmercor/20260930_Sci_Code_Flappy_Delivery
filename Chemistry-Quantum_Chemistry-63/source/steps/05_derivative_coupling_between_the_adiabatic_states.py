"""
Return the matrix element of the nuclear gradient taken between the upper adiabatic state on the left and the lower one on the right, in the orthonormal polar frame.

The object needed everywhere below is the matrix element of the nuclear gradient taken between the upper adiabatic state on the left and the lower one on the right, in the same orthonormal polar frame. It is complex, and it is tied to the phase convention of the previous step: a result computed in any other gauge will not compose with the quantities built from it later. How the matrix element is obtained is left open, but note that whichever route is taken has to survive being evaluated at a few hundred points around a loop.

Returns
-------
np.ndarray of shape (2, 2): real part in row 0 and imaginary part in row 1 of the derivative coupling, radial then angular component
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def interstate_coupling(Q: float, theta: float, g: float, K: float,
                        Delta: float) -> "np.ndarray":
    '''Nuclear derivative coupling from the lower to the upper adiabatic state.

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
    coupling : np.ndarray
        Real array of shape (2, 2). Row 0 holds the real part and row 1 the
        imaginary part; within a row column 0 is the radial component and
        column 1 the angular component, in the orthonormal polar frame. The
        quantity represented is the matrix element of the nuclear gradient with
        the upper state on the left and the lower state on the right.

    Raises
    ------
    ValueError
        If any argument is not a finite scalar, or if Q, g or Delta is not
        strictly positive.
    '''
    return coupling

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


def _oracle_interstate_coupling(
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

    V = stacked[0] + 1j * stacked[1]
    lower, upper = V[:, 0], V[:, 1]

    hs = _oracle_bo_hamiltonian(
        Q,
        theta,
        g,
        K,
        Delta,
    )
    energies = np.linalg.eigvalsh(
        hs[0] + 1j * hs[1]
    )

    s1 = np.array([
        [0.0, 1.0],
        [1.0, 0.0],
    ])

    s3 = np.array([
        [1.0, 0.0],
        [0.0, -1.0],
    ])

    dH_dQ = (
        K * Q * np.eye(2)
        + g * (
            s3 * np.cos(theta)
            - s1 * np.sin(theta)
        )
    )

    dH_dtheta_over_Q = g * (
        -s3 * np.sin(theta)
        - s1 * np.cos(theta)
    )

    denom = energies[0] - energies[1]

    b = np.array([
        np.vdot(upper, dH_dQ @ lower) / denom,
        np.vdot(upper, dH_dtheta_over_Q @ lower) / denom,
    ])

    return np.stack([b.real, b.imag])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {"setup": "import numpy as np\n",
         "call": "np.round(interstate_coupling(0.20, 0.7, 0.60, 1.00, 0.12), 9)",
         "gold_call": "np.round(_oracle_interstate_coupling(0.20, 0.7, 0.60, 1.00, 0.12), 9)"},
        {"setup": "import numpy as np\n",
         "call": "np.round(interstate_coupling(0.20, 0.0, 0.60, 1.00, 0.12), 9)",
         "gold_call": "np.round(_oracle_interstate_coupling(0.20, 0.0, 0.60, 1.00, 0.12), 9)"},
        {"setup": "import numpy as np\n",
         "call": "np.round(interstate_coupling(0.20, np.pi, 0.60, 1.00, 0.12), 9)",
         "gold_call": "np.round(_oracle_interstate_coupling(0.20, np.pi, 0.60, 1.00, 0.12), 9)"},
        {"setup": "import numpy as np\n",
         "call": "np.round(interstate_coupling(0.20, 5.10, 0.60, 1.00, 0.12), 9)",
         "gold_call": "np.round(_oracle_interstate_coupling(0.20, 5.10, 0.60, 1.00, 0.12), 9)"},
        {"setup": "import numpy as np\n",
         "call": "np.round(interstate_coupling(2.40, 1.90, 0.80, 1.50, 0.03), 9)",
         "gold_call": "np.round(_oracle_interstate_coupling(2.40, 1.90, 0.80, 1.50, 0.03), 9)"},
        {"setup": "import numpy as np\n",
         "call": "np.round(interstate_coupling(0.02, 2.90, 0.20, 1.00, 0.70), 9)",
         "gold_call": "np.round(_oracle_interstate_coupling(0.02, 2.90, 0.20, 1.00, 0.70), 9)"},
    ]
