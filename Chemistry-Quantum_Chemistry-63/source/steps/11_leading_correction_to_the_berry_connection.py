"""
Return the first-order correction, in the mass ratio, to the Berry connection of the lower adiabatic state.

Treating the electron-nuclear correlation operator as a perturbation of the electronic equation of motion produces a correction to the conditional electronic factor at first order in the mass ratio, and through it a correction to the Berry connection of the lower state. That corrected connection is what this step returns, in the same orthonormal polar frame as the uncorrected one and, like it, real. The correction is first order in the mass-ratio parameter and vanishes with it. Both the matrix element of the previous step and the adiabatic energies enter.

Returns
-------
np.ndarray of shape (2,): radial and angular component of the leading correction to the Berry connection of the lower state
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def connection_correction(Q: float, theta: float, g: float, K: float,
                          Delta: float, M0: float, mu: float,
                          momentum: "np.ndarray") -> "np.ndarray":
    '''Leading correction to the Berry connection of the lower state.

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
        Nuclear mass carried by the two normal modes, strictly positive.
    mu : float
        Electron-to-nuclear mass ratio that scales the perturbation, strictly
        positive.
    momentum : np.ndarray
        The nuclear momentum function at this radius, in the shape returned by
        the nuclear momentum step.

    Returns
    -------
    correction : np.ndarray
        Real array of shape (2,) holding the radial and then the angular
        component of the correction to the Berry connection, in the orthonormal
        polar frame.

    Raises
    ------
    ValueError
        If any of Q, theta, g, K, Delta is not a finite scalar, if Q, g or
        Delta is not strictly positive, if M0 or mu is not a finite positive
        scalar, or if momentum is not a finite real array of shape (2, 2).
    '''
    return correction

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


def _oracle_connection_correction(
    Q: float,
    theta: float,
    g: float,
    K: float,
    Delta: float,
    M0: float,
    mu: float,
    momentum: "np.ndarray",
) -> "np.ndarray":
    Q, theta, g, K, Delta = _check_point(Q, theta, g, K, Delta)

    if (
        np.ndim(mu) != 0
        or not np.isfinite(float(mu))
        or float(mu) <= 0.0
    ):
        raise ValueError("mu must be a finite positive scalar")

    mu = float(mu)

    el = _oracle_correlation_coupling(
        Q,
        theta,
        g,
        K,
        Delta,
        M0,
        momentum,
    )
    element = el[0] + 1j * el[1]

    hs = _oracle_bo_hamiltonian(Q, theta, g, K, Delta)
    energies = np.linalg.eigvalsh(hs[0] + 1j * hs[1])

    kappa = -mu * element / (energies[1] - energies[0])

    bs = _oracle_interstate_coupling(Q, theta, g, K, Delta)
    b = bs[0] + 1j * bs[1]

    return 2.0 * np.real(np.conj(kappa) * (-1j * b))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {"setup": """import numpy as np
P = np.array([[0.0, 9.29123009], [-19.45869523, 0.0]])
""",
         "call": "np.round(connection_correction(0.35, 0.7, 0.60, 1.00, 0.12, 100.0, 0.01, P), 12)",
         "gold_call": "np.round(_oracle_connection_correction(0.35, 0.7, 0.60, 1.00, 0.12, 100.0, 0.01, P), 12)"},
        {"setup": """import numpy as np
P = np.array([[0.0, 9.29123009], [-19.45869523, 0.0]])
""",
         "call": "np.round(connection_correction(0.35, 0.0, 0.60, 1.00, 0.12, 100.0, 0.01, P), 12)",
         "gold_call": "np.round(_oracle_connection_correction(0.35, 0.0, 0.60, 1.00, 0.12, 100.0, 0.01, P), 12)"},
        {"setup": """import numpy as np
P = np.array([[0.0, 9.29123009], [-19.45869523, 0.0]])
""",
         "call": "np.round(connection_correction(0.35, np.pi, 0.60, 1.00, 0.12, 100.0, 0.01, P), 12)",
         "gold_call": "np.round(_oracle_connection_correction(0.35, np.pi, 0.60, 1.00, 0.12, 100.0, 0.01, P), 12)"},
        {"setup": """import numpy as np
P = np.zeros((2, 2))
""",
         "call": "np.round(connection_correction(0.35, 3.90, 0.60, 1.00, 0.12, 100.0, 0.01, P), 12)",
         "gold_call": "np.round(_oracle_connection_correction(0.35, 3.90, 0.60, 1.00, 0.12, 100.0, 0.01, P), 12)"},
        {"setup": """import numpy as np
P = np.array([[1.5, -0.75], [0.25, 2.0]])
""",
         "call": "np.round(connection_correction(0.95, 1.40, 0.55, 1.20, 0.35, 250.0, 0.004, P), 12)",
         "gold_call": "np.round(_oracle_connection_correction(0.95, 1.40, 0.55, 1.20, 0.35, 250.0, 0.004, P), 12)"},
        {"setup": """import numpy as np
P = np.array([[0.0, 3.0], [-2.0, 0.0]])
""",
         "call": "np.round(connection_correction(0.06, 2.75, 0.60, 1.00, 0.12, 100.0, 0.02, P), 12)",
         "gold_call": "np.round(_oracle_connection_correction(0.06, 2.75, 0.60, 1.00, 0.12, 100.0, 0.02, P), 12)"},
    ]
