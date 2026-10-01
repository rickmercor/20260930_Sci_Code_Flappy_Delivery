"""
Diagonalise that Hamiltonian and return the two adiabatic states with their phases fixed by requiring the second component of each to be real and non-negative, lower state first.

Diagonalising the Hamiltonian of the previous step at each nuclear point gives a lower and an upper adiabatic state, but an eigensolver returns each of them only up to an arbitrary phase, and an arbitrary phase destroys every derivative taken later. The convention adopted here fixes that phase completely: the second component of each returned state is made real and non-negative. Because Delta is strictly positive that component never vanishes anywhere on the loop, so the convention is well defined at every theta and the resulting states are smooth and single valued as theta advances through 2 pi. The step returns the real and the imaginary part of the 2 x 2 matrix whose first column is the lower state and whose second column is the upper state.

Returns
-------
np.ndarray of shape (2, 2, 2): real and imaginary part of the matrix of phase-fixed adiabatic states, lower state in column 0
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def adiabatic_states(Q: float, theta: float, g: float, K: float,
                     Delta: float) -> "np.ndarray":
    '''Phase-fixed adiabatic electronic states at one nuclear configuration.

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
    stacked : np.ndarray
        Real array of shape (2, 2, 2). stacked[0] holds the real part and
        stacked[1] the imaginary part of a 2 x 2 matrix whose column 0 is the
        lower adiabatic state and whose column 1 is the upper adiabatic state.
        The phase of each column is fixed by requiring its second component to
        be real and non-negative.

    Raises
    ------
    ValueError
        If any argument is not a finite scalar, or if Q, g or Delta is not
        strictly positive.
    '''
    return stacked

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


def _oracle_adiabatic_states(
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

    stacked = _oracle_bo_hamiltonian(
        Q,
        theta,
        g,
        K,
        Delta,
    )

    H = stacked[0] + 1j * stacked[1]
    _, vecs = np.linalg.eigh(H)

    out = np.empty((2, 2), dtype=complex)

    for k in range(2):
        v = vecs[:, k]
        out[:, k] = v * np.exp(
            -1j * np.angle(v[1])
        )

    return np.stack([out.real, out.imag])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {"setup": "import numpy as np\n",
         "call": "adiabatic_states(0.20, 0.7, 0.60, 1.00, 0.12)",
         "gold_call": "_oracle_adiabatic_states(0.20, 0.7, 0.60, 1.00, 0.12)"},
        {"setup": "import numpy as np\n",
         "call": "adiabatic_states(0.20, 0.0, 0.60, 1.00, 0.12)",
         "gold_call": "_oracle_adiabatic_states(0.20, 0.0, 0.60, 1.00, 0.12)"},
        {"setup": "import numpy as np\n",
         "call": "adiabatic_states(0.20, np.pi, 0.60, 1.00, 0.12)",
         "gold_call": "_oracle_adiabatic_states(0.20, np.pi, 0.60, 1.00, 0.12)"},
        {"setup": "import numpy as np\n",
         "call": "adiabatic_states(0.20, 6.0, 0.60, 1.00, 0.12)",
         "gold_call": "_oracle_adiabatic_states(0.20, 6.0, 0.60, 1.00, 0.12)"},
        {"setup": "import numpy as np\n",
         "call": "adiabatic_states(3.20, 2.40, 0.90, 1.00, 0.01)",
         "gold_call": "_oracle_adiabatic_states(3.20, 2.40, 0.90, 1.00, 0.01)"},
        {"setup": "import numpy as np\n",
         "call": "adiabatic_states(0.05, 1.57, 0.25, 1.00, 0.80)",
         "gold_call": "_oracle_adiabatic_states(0.05, 1.57, 0.25, 1.00, 0.80)"},
    ]
