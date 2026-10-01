"""
Return the real and the imaginary part of the two-state Born-Oppenheimer Hamiltonian of the gapped two-mode vibronic model at one nuclear configuration.

Two degenerate vibrational normal modes span a plane, written in polar form as R = Q(cos theta, sin theta) with Q > 0 the loop radius. Two electronic states span the doubly degenerate orbital pair. The Hamiltonian carries an isotropic harmonic term proportional to the identity, a linear vibronic term that couples the two electronic states with a strength g Q and rotates with theta, and a constant term Delta that lifts the residual degeneracy at the origin. With the Pauli matrices built on the two diabatic states,

H(Q, theta) = (K/2) Q^2 * 1 + g Q (sigma_3 cos theta - sigma_1 sin theta) + Delta sigma_2 .

The only complex entries come from Delta sigma_2, so the step returns the real and the imaginary part of the matrix stacked along a leading axis.

Returns
-------
np.ndarray of shape (2, 2, 2): real and imaginary part of the 2 x 2 Born-Oppenheimer Hamiltonian
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def bo_hamiltonian(Q: float, theta: float, g: float, K: float,
                   Delta: float) -> "np.ndarray":
    '''Electronic Hamiltonian of the model at one nuclear configuration.

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
        stacked[1] the imaginary part of the 2 x 2 electronic Hamiltonian.

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


def _oracle_bo_hamiltonian(
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

    s1 = np.array([
        [0.0, 1.0],
        [1.0, 0.0],
    ])

    s3 = np.array([
        [1.0, 0.0],
        [0.0, -1.0],
    ])

    s2_imag = np.array([
        [0.0, -1.0],
        [1.0, 0.0],
    ])

    real = (
        0.5 * K * Q * Q * np.eye(2)
        + g * Q * (
            s3 * np.cos(theta)
            - s1 * np.sin(theta)
        )
    )

    imag = Delta * s2_imag

    return np.stack([real, imag])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {"setup": "import numpy as np\n",
         "call": "bo_hamiltonian(0.20, 0.7, 0.60, 1.00, 0.12)",
         "gold_call": "_oracle_bo_hamiltonian(0.20, 0.7, 0.60, 1.00, 0.12)"},
        {"setup": "import numpy as np\n",
         "call": "bo_hamiltonian(0.20, 0.0, 0.60, 1.00, 0.12)",
         "gold_call": "_oracle_bo_hamiltonian(0.20, 0.0, 0.60, 1.00, 0.12)"},
        {"setup": "import numpy as np\n",
         "call": "bo_hamiltonian(0.20, np.pi, 0.60, 1.00, 0.12)",
         "gold_call": "_oracle_bo_hamiltonian(0.20, np.pi, 0.60, 1.00, 0.12)"},
        {"setup": "import numpy as np\n",
         "call": "bo_hamiltonian(2.75, 4.10, 0.35, 0.40, 0.55)",
         "gold_call": "_oracle_bo_hamiltonian(2.75, 4.10, 0.35, 0.40, 0.55)"},
        {"setup": "import numpy as np\n",
         "call": "bo_hamiltonian(0.004, 5.90, 1.10, 2.00, 0.002)",
         "gold_call": "_oracle_bo_hamiltonian(0.004, 5.90, 1.10, 2.00, 0.002)"},
    ]
