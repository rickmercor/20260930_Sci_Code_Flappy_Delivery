"""
Return the matrix element of the electron-nuclear correlation operator between the upper adiabatic state on the left and the lower one on the right, without the mass-ratio prefactor.

The perturbation of the electronic equation of motion in the exact-factorization framework is the electron-nuclear correlation operator. It is built from the nuclear momentum function and from the nuclear gradient made covariant by the Berry connection, and it is scaled by the nuclear mass; the mass-ratio parameter that multiplies the whole perturbation is not applied at this step. Only its matrix element between the upper adiabatic state on the left and the lower one on the right is wanted, which is why the step returns a single complex number rather than an operator. The momentum function is passed in rather than recomputed, because it does not depend on the polar angle and solving for it again at every angle would be wasteful.

Returns
-------
np.ndarray of shape (2,): real part and imaginary part of the upper-lower matrix element of the electron-nuclear correlation operator
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def correlation_coupling(Q: float, theta: float, g: float, K: float,
                         Delta: float, M0: float,
                         momentum: "np.ndarray") -> "np.ndarray":
    '''Off-diagonal element of the electron-nuclear correlation operator.

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
    M0 : float
        Nuclear mass carried by the two normal modes, strictly positive. The
        mass-ratio prefactor of the perturbation is not applied here.
    momentum : np.ndarray
        The nuclear momentum function at this radius, in the shape returned by
        the previous step: real part in row 0, imaginary part in row 1, radial
        component in column 0 and angular component in column 1.

    Returns
    -------
    element : np.ndarray
        Real array of shape (2,) holding the real part and then the imaginary
        part of the matrix element, with the upper adiabatic state on the left
        and the lower one on the right.

    Raises
    ------
    ValueError
        If any of Q, theta, g, K, Delta is not a finite scalar, if Q, g or
        Delta is not strictly positive, if M0 is not a finite positive scalar,
        or if momentum is not a finite real array of shape (2, 2).
    '''
    return element

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


def _check_momentum(momentum):
    p = np.asarray(momentum, dtype=float)

    if p.shape != (2, 2) or not np.all(np.isfinite(p)):
        raise ValueError(
            "momentum must be a finite real array of shape (2, 2)"
        )

    return p[0] + 1j * p[1]


def _oracle_correlation_coupling(
    Q: float,
    theta: float,
    g: float,
    K: float,
    Delta: float,
    M0: float,
    momentum: "np.ndarray",
) -> "np.ndarray":
    Q, theta, g, K, Delta = _check_point(Q, theta, g, K, Delta)

    if (
        np.ndim(M0) != 0
        or not np.isfinite(float(M0))
        or float(M0) <= 0.0
    ):
        raise ValueError("M0 must be a finite positive scalar")

    M0 = float(M0)
    p = _check_momentum(momentum)

    bs = _oracle_interstate_coupling(Q, theta, g, K, Delta)
    b = bs[0] + 1j * bs[1]

    conn = _oracle_berry_connections(Q, theta, g, K, Delta)
    dA = conn[1] - conn[0]

    dv = _oracle_coupling_divergence(Q, theta, g, K, Delta)
    div = dv[0] + 1j * dv[1]

    value = (
        (-div - 1j * np.dot(b, dA)) / (2.0 * M0)
        - 1j * np.dot(p, b) / M0
    )

    return np.array([value.real, value.imag])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {"setup": """import numpy as np
P = np.array([[0.0, 9.29123009], [-19.45869523, 0.0]])
""",
         "call": "np.round(correlation_coupling(0.35, 0.7, 0.60, 1.00, 0.12, 100.0, P), 8)",
         "gold_call": "np.round(_oracle_correlation_coupling(0.35, 0.7, 0.60, 1.00, 0.12, 100.0, P), 8)"},
        {"setup": """import numpy as np
P = np.array([[0.0, 9.29123009], [-19.45869523, 0.0]])
""",
         "call": "np.round(correlation_coupling(0.35, 0.0, 0.60, 1.00, 0.12, 100.0, P), 8)",
         "gold_call": "np.round(_oracle_correlation_coupling(0.35, 0.0, 0.60, 1.00, 0.12, 100.0, P), 8)"},
        {"setup": """import numpy as np
P = np.array([[0.0, 9.29123009], [-19.45869523, 0.0]])
""",
         "call": "np.round(correlation_coupling(0.35, np.pi, 0.60, 1.00, 0.12, 100.0, P), 8)",
         "gold_call": "np.round(_oracle_correlation_coupling(0.35, np.pi, 0.60, 1.00, 0.12, 100.0, P), 8)"},
        {"setup": """import numpy as np
P = np.zeros((2, 2))
""",
         "call": "np.round(correlation_coupling(0.35, 2.20, 0.60, 1.00, 0.12, 100.0, P), 8)",
         "gold_call": "np.round(_oracle_correlation_coupling(0.35, 2.20, 0.60, 1.00, 0.12, 100.0, P), 8)"},
        {"setup": """import numpy as np
P = np.array([[1.5, -0.75], [0.25, 2.0]])
""",
         "call": "np.round(correlation_coupling(1.10, 4.60, 0.45, 0.80, 0.30, 40.0, P), 8)",
         "gold_call": "np.round(_oracle_correlation_coupling(1.10, 4.60, 0.45, 0.80, 0.30, 40.0, P), 8)"},
        {"setup": """import numpy as np
P = np.array([[0.0, 3.0], [-2.0, 0.0]])
""",
         "call": "np.round(correlation_coupling(0.04, 5.90, 0.60, 1.00, 0.12, 900.0, P), 8)",
         "gold_call": "np.round(_oracle_correlation_coupling(0.04, 5.90, 0.60, 1.00, 0.12, 900.0, P), 8)"},
    ]
