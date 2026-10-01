"""
Return the divergence, in the nuclear plane, of the derivative-coupling field of the previous step.

The operator that couples electrons to nuclei brings in the divergence of the derivative-coupling field of the previous step. The divergence wanted here is the one taken in the nuclear plane itself; the field arrives as components along the radial and angular unit vectors, which is an orthonormal frame but not a Cartesian one. The quantity is complex, and both parts are returned. It is finite everywhere on a loop of non-zero radius as long as the gap term is non-zero.

Returns
-------
np.ndarray of shape (2,): real part and imaginary part of the divergence of the derivative-coupling field
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def coupling_divergence(Q: float, theta: float, g: float, K: float,
                        Delta: float) -> "np.ndarray":
    '''Divergence of the derivative-coupling field at one nuclear point.

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
    divergence : np.ndarray
        Real array of shape (2,) holding the real part and then the imaginary
        part of the divergence of the derivative-coupling field, the field
        being the one returned by the previous step.

    Raises
    ------
    ValueError
        If any argument is not a finite scalar, or if Q, g or Delta is not
        strictly positive.
    '''
    return divergence

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


def _oracle_coupling_divergence(
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

    c = np.cos(theta)
    s = np.sin(theta)

    W2 = g * g * Q * Q + Delta * Delta
    W = np.sqrt(W2)

    S2 = g * g * Q * Q * s * s + Delta * Delta
    S = np.sqrt(S2)

    radial = (
        (Delta * Delta * g * c / 2.0)
        * (
            W2 * S2
            - g * g * Q * Q
            * (2.0 * S2 + W2 * s * s)
        )
        / (W2 * W2 * S2 * S)
        + 1j
        * (Delta * g * s / 2.0)
        * (
            W2 * S2
            - g * g * Q * Q
            * (S2 + W2 * s * s)
        )
        / (W2 * W * S2 * S)
    )

    angular = (
        -(g / 2.0)
        * (c * Delta * Delta / (S2 * S))
        - 1j
        * (Delta * g / 2.0)
        * (
            s
            * (
                S2
                + g * g * Q * Q * c * c
            )
            / (W * S2 * S)
        )
    )

    value = (radial + angular) / Q

    return np.array([
        value.real,
        value.imag,
    ])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {"setup": "import numpy as np\n",
         "call": "np.round(coupling_divergence(0.20, 0.7, 0.60, 1.00, 0.12), 5)",
         "gold_call": "np.round(_oracle_coupling_divergence(0.20, 0.7, 0.60, 1.00, 0.12), 5)"},
        {"setup": "import numpy as np\n",
         "call": "np.round(coupling_divergence(0.20, 0.0, 0.60, 1.00, 0.12), 5)",
         "gold_call": "np.round(_oracle_coupling_divergence(0.20, 0.0, 0.60, 1.00, 0.12), 5)"},
        {"setup": "import numpy as np\n",
         "call": "np.round(coupling_divergence(0.20, np.pi / 2, 0.60, 1.00, 0.12), 5)",
         "gold_call": "np.round(_oracle_coupling_divergence(0.20, np.pi / 2, 0.60, 1.00, 0.12), 5)"},
        {"setup": "import numpy as np\n",
         "call": "np.round(coupling_divergence(0.20, 4.40, 0.60, 1.00, 0.12), 5)",
         "gold_call": "np.round(_oracle_coupling_divergence(0.20, 4.40, 0.60, 1.00, 0.12), 5)"},
        {"setup": "import numpy as np\n",
         "call": "np.round(coupling_divergence(1.30, 2.60, 0.50, 1.00, 0.20), 5)",
         "gold_call": "np.round(_oracle_coupling_divergence(1.30, 2.60, 0.50, 1.00, 0.20), 5)"},
        {"setup": "import numpy as np\n",
         "call": "np.round(coupling_divergence(0.60, 1.20, 0.40, 1.00, 1.50), 5)",
         "gold_call": "np.round(_oracle_coupling_divergence(0.60, 1.20, 0.40, 1.00, 1.50), 5)"},
    ]
