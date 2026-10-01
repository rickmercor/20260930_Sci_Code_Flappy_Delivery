"""
Compute the quantum harmonic heat capacities and equilibrium phonon lifetimes for the supplied vibrational modes. These modal properties provide the equilibrium-phonon quantities needed later for the Peierls--Boltzmann relaxation-time conductivity.

For a vibrational mode of frequency $\omega_n$ at inverse temperature $\beta$, the quantum harmonic heat capacity follows the Bose--Einstein occupation. In the dimensionless units used in this task, $k_B=\hbar=1$.

The modal heat capacity is

$$

c_n=(\beta\omega_n)^2 \frac{e^{\beta\omega_n}} {(e^{\beta\omega_n}-1)^2}.

$$

The equilibrium phonon lifetime is obtained independently from the equilibrium phonon linewidth,

$$

\tau_n^{ph}=\frac{1}{2\Gamma_n^{ph}}.

$$

These equilibrium lifetimes are distinct from the transport lifetime inferred later from the heat-current reconstruction.

Returns
-------
np.ndarray of shape (2, N): row 0 contains modal heat capacities and row 1 contains equilibrium phonon lifetimes
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_modal_properties(
    omega: "np.ndarray",
    beta: float,
    gamma_ph: "np.ndarray",
) -> "np.ndarray":
    """Compute modal heat capacities and equilibrium phonon lifetimes.

    Parameters
    ----------
    omega : np.ndarray
        One-dimensional array of positive vibrational frequencies.
    beta : float
        Positive inverse temperature in units with k_B = hbar = 1.
    gamma_ph : np.ndarray
        One-dimensional array of positive equilibrium phonon linewidths
        with the same length as omega.

    Returns
    -------
    properties : np.ndarray
        Array of shape (2, N). Row 0 contains the modal heat capacities
        and row 1 contains the equilibrium phonon lifetimes.
    """
    return properties

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_compute_modal_properties(
    omega: "np.ndarray",
    beta: float,
    gamma_ph: "np.ndarray",
) -> "np.ndarray":
    omega = np.asarray(omega, dtype=float)
    gamma_ph = np.asarray(gamma_ph, dtype=float)

    x = float(beta) * omega

    denominator = -np.expm1(-x)
    heat_capacity = x**2 * np.exp(-x) / denominator**2

    lifetime = 1.0 / (2.0 * gamma_ph)

    return np.vstack((heat_capacity, lifetime)).astype(float)

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
""",
            "call": "np.round(compute_modal_properties(omega, beta, gamma_ph), 10)",
            "gold_call": "np.round(_oracle_compute_modal_properties(omega_gold, beta, gamma_ph_gold), 10)",
        },
        {
            "setup": """
import numpy as np
omega = np.array([0.75], dtype=float)
omega_gold = omega.copy()
beta = 1.5
gamma_ph = np.array([0.18], dtype=float)
gamma_ph_gold = gamma_ph.copy()
""",
            "call": "np.round(compute_modal_properties(omega, beta, gamma_ph), 10)",
            "gold_call": "np.round(_oracle_compute_modal_properties(omega_gold, beta, gamma_ph_gold), 10)",
        },
        {
            "setup": """
import numpy as np
omega = np.array([1.0e-8, 0.50], dtype=float)
omega_gold = omega.copy()
beta = 0.75
gamma_ph = np.array([0.05, 0.40], dtype=float)
gamma_ph_gold = gamma_ph.copy()
""",
            "call": "np.round(compute_modal_properties(omega, beta, gamma_ph), 10)",
            "gold_call": "np.round(_oracle_compute_modal_properties(omega_gold, beta, gamma_ph_gold), 10)",
        },
    ]
