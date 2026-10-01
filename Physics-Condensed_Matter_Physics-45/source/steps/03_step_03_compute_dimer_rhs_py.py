"""
Evaluate the four real semiclassical correlation derivatives for the isolated dissipative Heisenberg dimer.

The isolated dimer evolves through the real correlation variables $N^1$, $N^2$, $N^z$, and $M^z$. The finite-spin Casimir is

$$

Q=S(S+1)\hbar^2,

$$

and the bond correlation and semiclassical energy are

$$

C=2N^1+(M^z)^2-(N^z)^2,\qquad H_{\mathrm{sc}}=JC.

$$

The dissipative correlation dynamics is

$$

\dot N^1=-2JN^zN^2+2\eta N^1H_{\mathrm{sc}}-\eta JQ\left[Q-(M^z)^2-(N^z)^2\right],

$$



$$

\dot N^2=JN^z\left[2N^1+Q+(M^z)^2-(N^z)^2\right]+2\eta N^2H_{\mathrm{sc}},

$$

$$

\dot N^z=-2JN^2+\eta N^z\left(H_{\mathrm{sc}}+JQ\right),

$$

$$

\dot M^z=\eta M^z\left(H_{\mathrm{sc}}-JQ\right).

$$

Returns
-------
np.ndarray of shape (4,), containing $[\dot N^1,\dot N^2,\dot N^z,\dot M^z]$.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_dimer_rhs(
    t: float,
    state: "np.ndarray",
    J: float,
    eta: float,
    S: float,
    hbar: float,
) -> "np.ndarray":
    """Evaluate the four real dissipative dimer derivatives.

    Parameters
    ----------
    t : float
        Time. The equations are autonomous, but the argument is retained for
        use with numerical ODE solvers.
    state : np.ndarray
        Length-4 array ordered as $[N^1,N^2,N^z,M^z]$.
    J : float
        Heisenberg exchange constant.
    eta : float
        Correlation-level damping parameter.
    S : float
        Spin quantum number.
    hbar : float
        Reduced Planck constant.

    Returns
    -------
    derivative : np.ndarray
        Length-4 array containing
        $[\dot N^1,\dot N^2,\dot N^z,\dot M^z]$.
    """
    return derivative

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_compute_dimer_rhs(
    t: float,
    state: "np.ndarray",
    J: float,
    eta: float,
    S: float,
    hbar: float,
) -> "np.ndarray":
    """Reference implementation."""
    state = np.asarray(state, dtype=float)
    N1, N2, Nz, Mz = state

    Q = _oracle_compute_spin_casimir(S, hbar)
    C, H_sc = _oracle_compute_bond_quantities(state, J)

    dN1 = (
        -2.0 * J * Nz * N2
        + 2.0 * eta * N1 * H_sc
        - eta * J * Q * (Q - Mz * Mz - Nz * Nz)
    )

    dN2 = (
        J * Nz * (2.0 * N1 + Q + Mz * Mz - Nz * Nz)
        + 2.0 * eta * N2 * H_sc
    )

    dNz = -2.0 * J * N2 + eta * Nz * (H_sc + J * Q)

    dMz = eta * Mz * (H_sc - J * Q)

    return np.array([dN1, dN2, dNz, dMz], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
state = np.array([0.0, 0.0, 1.5, 0.0], dtype=float)
""",
            "call": "compute_dimer_rhs(0.0, state.copy(), 0.83, 0.06, 1.5, 1.0)",
            "gold_call": "_oracle_compute_dimer_rhs(0.0, state.copy(), 0.83, 0.06, 1.5, 1.0)",
        },
        {
            "setup": """import numpy as np
state = np.array([-0.2, 0.4, 0.9, 0.0], dtype=float)
""",
            "call": "compute_dimer_rhs(1.0, state.copy(), 1.2, 0.0, 1.5, 1.0)",
            "gold_call": "_oracle_compute_dimer_rhs(1.0, state.copy(), 1.2, 0.0, 1.5, 1.0)",
        },
        {
            "setup": """import numpy as np
state = np.array([0.25, -0.6, -0.8, 0.35], dtype=float)
""",
            "call": "compute_dimer_rhs(2.3, state.copy(), 1.8, 0.15, 2.0, 0.7)",
            "gold_call": "_oracle_compute_dimer_rhs(2.3, state.copy(), 1.8, 0.15, 2.0, 0.7)",
        },
    ]
