"""
Compute the equal-time Heisenberg bond correlation and the corresponding semiclassical exchange energy.

For the real dimer correlation variables $N^1$, $N^2$, $N^z$, and $M^z$, the equal-time spin correlation is

$$

C=2N^1+(M^z)^2-(N^z)^2.

$$

Using the exchange convention $H=J\,\mathbf S_1\cdot\mathbf S_2$, the corresponding semiclassical bond energy is

$$

H_{\mathrm{sc}}=JC.

$$

Returns
-------
np.ndarray of shape (2,), containing $[C,H_{\mathrm{sc}}]$.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_bond_quantities(state: "np.ndarray", J: float) -> "np.ndarray":
    """Compute the bond correlation and semiclassical exchange energy.

    Parameters
    ----------
    state : np.ndarray
        Length-4 array ordered as $[N^1,N^2,N^z,M^z]$.
    J : float
        Heisenberg exchange constant.

    Returns
    -------
    values : np.ndarray
        Length-2 array containing $[C,H_{\mathrm{sc}}]$.
    """
    return values

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_compute_bond_quantities(
    state: "np.ndarray",
    J: float,
) -> "np.ndarray":
    """Reference implementation."""
    state = np.asarray(state, dtype=float)

    N1 = state[0]
    Nz = state[2]
    Mz = state[3]

    C = 2.0 * N1 + Mz**2 - Nz**2
    H_sc = J * C

    return np.array([C, H_sc], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
state = np.array([0.0, 0.0, 1.5, 0.0], dtype=float)
J = 0.83
""",
            "call": "compute_bond_quantities(state.copy(), J)",
            "gold_call": "_oracle_compute_bond_quantities(state.copy(), J)",
        },
        {
            "setup": """import numpy as np
state = np.array([0.4, -0.2, 0.0, 0.0], dtype=float)
J = 1.0
""",
            "call": "compute_bond_quantities(state.copy(), J)",
            "gold_call": "_oracle_compute_bond_quantities(state.copy(), J)",
        },
        {
            "setup": """import numpy as np
state = np.array([-0.35, 0.7, -1.1, 0.45], dtype=float)
J = 1.8
""",
            "call": "compute_bond_quantities(state.copy(), J)",
            "gold_call": "_oracle_compute_bond_quantities(state.copy(), J)",
        },
    ]
