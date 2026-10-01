"""
Combine the endpoint-energy summary and midpoint boundary-port summary produced by the preceding subproblems into the one-step discrete power-balance result. Use the signed Hamiltonian increment stored in the endpoint-energy summary and the total midpoint boundary power stored in the boundary-port summary. Convert the midpoint power into one-step work using the supplied nonnegative time-step size, and form the signed energy-balance defect using the time-level ordering prescribed by the structure-preserving balance. Return a two-component NumPy array containing the one-step boundary work and the signed energy-balance defect, in that order.

The structure-preserving implicit-midpoint discretization relates the Hamiltonian change across two consecutive endpoint states to the mechanical work supplied through the midpoint system ports. Once the endpoint Hamiltonian increment and the midpoint boundary power have been evaluated separately, the discrete balance reduces to comparing the signed energy increment with the time-integrated midpoint power. This step isolates that temporal balance from the preceding Hamiltonian and boundary-port constructions.

Returns
-------
A 1D NumPy array of shape (2,) containing [boundary_work, signed_energy_balance_defect].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def compute_discrete_power_balance_defect(
    energy_summary: np.ndarray,
    port_summary: np.ndarray,
    h: float,
) -> np.ndarray:
    """Return the one-step boundary work and signed energy-balance defect."""
    return np.zeros(2, dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_discrete_power_balance_defect(
    energy_summary: np.ndarray,
    port_summary: np.ndarray,
    h: float,
) -> np.ndarray:
    import numpy as np

    energy_summary = np.asarray(energy_summary, dtype=float)
    port_summary = np.asarray(port_summary, dtype=float)

    if energy_summary.shape != (3,):
        raise ValueError("energy_summary must have shape (3,).")

    if port_summary.shape != (6,):
        raise ValueError("port_summary must have shape (6,).")

    if not np.all(np.isfinite(energy_summary)):
        raise ValueError("energy_summary must contain finite values.")

    if not np.all(np.isfinite(port_summary)):
        raise ValueError("port_summary must contain finite values.")

    h = float(h)

    if not np.isfinite(h):
        raise ValueError("h must be finite.")

    if h < 0.0:
        raise ValueError("h must be nonnegative.")

    delta_H = float(energy_summary[2])
    total_power = float(port_summary[5])

    boundary_work = h * total_power
    defect = delta_H - boundary_work

    return np.array(
        [boundary_work, defect],
        dtype=float,
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """
import numpy as np

energy_summary = np.array([
    0.095043250,
    0.116607585,
    0.021564335,
], dtype=float)

port_summary = np.array([
    0.12,
    -0.08,
    0.05,
    0.179550,
    -0.004500,
    0.175050,
], dtype=float)

h = 0.04
""",
            "call": """
compute_discrete_power_balance_defect(
    energy_summary,
    port_summary,
    h,
)
""",
            "gold_call": """
_oracle_compute_discrete_power_balance_defect(
    energy_summary,
    port_summary,
    h,
)
""",
        },

        {
            "setup": """
import numpy as np

energy_summary = np.array([
    1.2,
    0.8,
    -0.4,
], dtype=float)

port_summary = np.array([
    0.0,
    0.0,
    0.0,
    -0.9,
    -0.6,
    -1.5,
], dtype=float)

h = 0.2
""",
            "call": """
compute_discrete_power_balance_defect(
    energy_summary,
    port_summary,
    h,
)
""",
            "gold_call": """
_oracle_compute_discrete_power_balance_defect(
    energy_summary,
    port_summary,
    h,
)
""",
        },

        {
            "setup": """
import numpy as np

energy_summary = np.array([
    2.0,
    2.75,
    0.75,
], dtype=float)

port_summary = np.array([
    0.1,
    -0.2,
    0.3,
    4.0,
    -1.0,
    3.0,
], dtype=float)

h = 0.0
""",
            "call": """
compute_discrete_power_balance_defect(
    energy_summary,
    port_summary,
    h,
)
""",
            "gold_call": """
_oracle_compute_discrete_power_balance_defect(
    energy_summary,
    port_summary,
    h,
)
""",
        },
    ]
