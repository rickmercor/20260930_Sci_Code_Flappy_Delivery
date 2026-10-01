"""
Compute the independent Peierls--Boltzmann relaxation-time thermal conductivity from the modal quantum heat capacities, supplied mode velocities, and equilibrium phonon lifetimes.

The Peierls--Boltzmann relaxation-time comparison uses the equilibrium phonon lifetimes rather than the transport lifetime reconstructed from the heat-current correlations.

For the supplied modes,

$$

\kappa_{\mathrm{PB-RTA}}=\sum_n c_n v_n^2\tau_n^{ph},

$$

where $c_n$ and $\tau_n^{ph}$ are the modal properties determined from the frequencies, inverse temperature, and equilibrium phonon linewidths.

Returns
-------
float, the Peierls--Boltzmann relaxation-time thermal conductivity
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_pb_rta_conductivity(
    omega: "np.ndarray",
    beta: float,
    gamma_ph: "np.ndarray",
    velocity: "np.ndarray",
) -> float:
    """Compute Peierls--Boltzmann relaxation-time conductivity.

    Parameters
    ----------
    omega : np.ndarray
        One-dimensional array of positive mode frequencies.
    beta : float
        Positive inverse temperature.
    gamma_ph : np.ndarray
        Positive equilibrium phonon linewidths.
    velocity : np.ndarray
        Mode velocities with the same length as omega.

    Returns
    -------
    kappa_pb : float
        Peierls--Boltzmann relaxation-time conductivity.
    """
    return kappa_pb

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_compute_pb_rta_conductivity(
    omega: "np.ndarray",
    beta: float,
    gamma_ph: "np.ndarray",
    velocity: "np.ndarray",
) -> float:
    properties = _oracle_compute_modal_properties(
        omega,
        beta,
        gamma_ph,
    )

    heat_capacity = properties[0]
    lifetime = properties[1]

    velocity = np.asarray(
        velocity,
        dtype=float,
    )

    return float(
        np.sum(
            heat_capacity
            * velocity**2
            * lifetime
        )
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """
import numpy as np
omega = np.array([0.75, 1.20, 1.85, 2.55], dtype=float)
omega_gold = omega.copy()
beta = 1.5
gamma_ph = np.array([0.18, 0.22, 0.27, 0.34], dtype=float)
gamma_ph_gold = gamma_ph.copy()
velocity = np.array([1.20, 1.00, 0.78, 0.58], dtype=float)
velocity_gold = velocity.copy()
""",
            "call": "round(compute_pb_rta_conductivity(omega, beta, gamma_ph, velocity), 10)",
            "gold_call": "round(_oracle_compute_pb_rta_conductivity(omega_gold, beta, gamma_ph_gold, velocity_gold), 10)",
        },
        {
            "setup": """
import numpy as np
omega = np.array([1.0], dtype=float)
omega_gold = omega.copy()
beta = 1.0
gamma_ph = np.array([0.2], dtype=float)
gamma_ph_gold = gamma_ph.copy()
velocity = np.array([0.8], dtype=float)
velocity_gold = velocity.copy()
""",
            "call": "round(compute_pb_rta_conductivity(omega, beta, gamma_ph, velocity), 10)",
            "gold_call": "round(_oracle_compute_pb_rta_conductivity(omega_gold, beta, gamma_ph_gold, velocity_gold), 10)",
        },
        {
            "setup": """
import numpy as np
omega = np.array([1.0e-8, 0.5], dtype=float)
omega_gold = omega.copy()
beta = 0.75
gamma_ph = np.array([0.05, 0.40], dtype=float)
gamma_ph_gold = gamma_ph.copy()
velocity = np.array([1.0, 0.2], dtype=float)
velocity_gold = velocity.copy()
""",
            "call": "round(compute_pb_rta_conductivity(omega, beta, gamma_ph, velocity), 10)",
            "gold_call": "round(_oracle_compute_pb_rta_conductivity(omega_gold, beta, gamma_ph_gold, velocity_gold), 10)",
        },
    ]
