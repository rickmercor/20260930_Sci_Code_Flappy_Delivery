"""
Return the term the exact factorization adds to the Born-Oppenheimer surface in the scalar potential of the nuclear equation, at one nuclear configuration.

Beyond the clamped-nucleus picture the scalar potential felt by the nuclei is not the Born-Oppenheimer surface alone. It carries a further term, weighted by the mass ratio, which is the expectation value in the lower adiabatic state of the same covariant nuclear kinetic operator that appears in the electron-nuclear correlation operator, divided by twice the nuclear mass. This step returns that term. It is real and non-negative, it is built from a quantity an earlier step already returns, and for this model it turns out not to depend on the polar angle at all, which is worth checking rather than assuming.

Returns
-------
float: the diagonal correction to the Born-Oppenheimer potential energy surface at the given nuclear configuration
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def diagonal_correction(Q: float, theta: float, g: float, K: float,
                        Delta: float, M0: float) -> float:
    '''Diagonal correction to the Born-Oppenheimer surface at one configuration.

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
        mass-ratio parameter that weights this term in the nuclear equation is
        not applied here.

    Returns
    -------
    correction : float
        The diagonal correction at that nuclear configuration.

    Raises
    ------
    ValueError
        If any of Q, theta, g, K, Delta is not a finite scalar, if Q, g or
        Delta is not strictly positive, or if M0 is not a finite positive
        scalar.
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


def _oracle_diagonal_correction(
    Q: float,
    theta: float,
    g: float,
    K: float,
    Delta: float,
    M0: float,
) -> float:
    Q, theta, g, K, Delta = _check_point(
        Q,
        theta,
        g,
        K,
        Delta,
    )

    if (
        np.ndim(M0) != 0
        or not np.isfinite(float(M0))
        or float(M0) <= 0.0
    ):
        raise ValueError(
            "M0 must be a finite positive scalar"
        )

    M0 = float(M0)

    bs = _oracle_interstate_coupling(
        Q,
        theta,
        g,
        K,
        Delta,
    )

    b = bs[0] + 1j * bs[1]

    return float(
        (abs(b[0]) ** 2 + abs(b[1]) ** 2)
        / (2.0 * M0)
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {"setup": "import numpy as np\n",
         "call": "round(diagonal_correction(0.35, 0.7, 0.60, 1.00, 0.12, 100.0), 12)",
         "gold_call": "round(_oracle_diagonal_correction(0.35, 0.7, 0.60, 1.00, 0.12, 100.0), 12)"},
        {"setup": "import numpy as np\n",
         "call": "round(diagonal_correction(0.35, 0.0, 0.60, 1.00, 0.12, 100.0), 12)",
         "gold_call": "round(_oracle_diagonal_correction(0.35, 0.0, 0.60, 1.00, 0.12, 100.0), 12)"},
        {"setup": "import numpy as np\n",
         "call": "round(diagonal_correction(0.35, np.pi, 0.60, 1.00, 0.12, 100.0), 12)",
         "gold_call": "round(_oracle_diagonal_correction(0.35, np.pi, 0.60, 1.00, 0.12, 100.0), 12)"},
        {"setup": "import numpy as np\n",
         "call": "round(diagonal_correction(0.02, 2.40, 0.60, 1.00, 0.12, 100.0), 12)",
         "gold_call": "round(_oracle_diagonal_correction(0.02, 2.40, 0.60, 1.00, 0.12, 100.0), 12)"},
        {"setup": "import numpy as np\n",
         "call": "round(diagonal_correction(1.90, 5.10, 0.45, 0.80, 0.30, 40.0), 12)",
         "gold_call": "round(_oracle_diagonal_correction(1.90, 5.10, 0.45, 0.80, 0.30, 40.0), 12)"},
        {"setup": "import numpy as np\n",
         "call": "round(diagonal_correction(0.35, 1.20, 0.60, 1.00, 0.90, 900.0), 12)",
         "gold_call": "round(_oracle_diagonal_correction(0.35, 1.20, 0.60, 1.00, 0.90, 900.0), 12)"},
    ]
