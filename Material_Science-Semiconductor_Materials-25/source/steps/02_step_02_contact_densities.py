"""
Compute the charge-neutral thermal-equilibrium potential and carrier densities of silicon for a given net doping, used at Ohmic contacts and as the Newton initial guess.

At an Ohmic contact the semiconductor is held in thermal equilibrium and is locally charge-neutral, so n - p = N and n p = n_ie^2. Solving these gives the familiar expression n = (N + sqrt(N^2 + 4 n_ie^2))/2, but on a p-type contact that formula subtracts two numbers of size |N| to produce a density of order n_ie^2/|N|. With N = -1e17 cm^-3 the result is about 1.18e3 cm^-3, and double-precision cancellation leaves it wrong in the third significant figure. The stable route is to compute the majority density as (|N| + sqrt(N^2 + 4 n_ie^2))/2, which never cancels, and the minority density as n_ie^2 divided by it. The electrostatic potential is then fixed by the Boltzmann relation, psi = V_applied + V_T ln(n/n_ie), so the contact potential already contains the built-in potential of the junction. Silicon at 300 K is used throughout: V_T = 0.025852 V and n_ie = 1.087386e10 cm^-3.

Returns
-------
np.ndarray of shape (3,) + N.shape, float: psi (V), n (cm^-3), p (cm^-3).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def contact_densities(N: "np.ndarray", V_applied: float = 0.0) -> "np.ndarray":
    """Charge-neutral equilibrium psi, n, p of silicon (300 K) for net doping N.

    Parameters
    ----------
    N : "np.ndarray"
        Net doping N_D - N_A in cm^-3, any shape (positive = n-type).
    V_applied : float
        Applied contact voltage in volts.

    Returns
    -------
    state : "np.ndarray"
        Shape (3,) + N.shape: row 0 is psi in volts, row 1 is n and row 2 is p
        in cm^-3. Use V_T = 0.025852 V and n_ie = 1.087386e10 cm^-3, compute the
        majority density as (|N| + sqrt(N^2 + 4 n_ie^2)) / 2 and the minority
        density as n_ie^2 / majority, and set psi = V_applied + V_T ln(n / n_ie).
        For N = 0 both densities equal n_ie.

    Notes
    -----
    Include every import your implementation needs inside the function body.
    """
    return state  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_contact_densities(N: "np.ndarray", V_applied: float = 0.0) -> "np.ndarray":
    VT = 0.025852
    nie = 1.087386e10
    N = np.asarray(N, dtype=float)
    maj = 0.5 * (np.abs(N) + np.sqrt(N * N + 4.0 * nie * nie))
    minor = nie * nie / maj
    n = np.where(N >= 0.0, maj, minor)
    p = np.where(N >= 0.0, minor, maj)
    psi = V_applied + VT * np.log(n / nie)
    return np.stack([psi, n, p])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    view = "(lambda s: np.vstack([s[0], np.log10(s[1]), np.log10(s[2])]))"
    return [
        # Normal: the three n-type doping levels of the study at a grounded contact.
        {"setup": "import numpy as np\nN = np.array([1e16, 3e16, 1e17])\n",
         "call": view + "(contact_densities(N, 0.0))",
         "gold_call": view + "(_oracle_contact_densities(N, 0.0))",
         "tol": 1e-9},
        # Edge: p-type contacts at 0.5 V, where the naive closed form cancels catastrophically.
        {"setup": "import numpy as np\nN = np.array([-1e16, -3e16, -1e17])\n",
         "call": view + "(contact_densities(N, 0.5))",
         "gold_call": view + "(_oracle_contact_densities(N, 0.5))",
         "tol": 1e-9},
        # Boundary: intrinsic material, N = 0, so n = p = n_ie and psi = V_applied.
        {"setup": "import numpy as np\nN = np.zeros(3)\n",
         "call": view + "(contact_densities(N, 1.0))",
         "gold_call": view + "(_oracle_contact_densities(N, 1.0))",
         "tol": 1e-9},
        # Normal: mixed 2D doping array spanning both types and several decades.
        {"setup": "import numpy as np\nN = np.array([[1e12, -1e12], [5e18, -5e18]])\n",
         "call": view + "(contact_densities(N, -0.2))",
         "gold_call": view + "(_oracle_contact_densities(N, -0.2))",
         "tol": 1e-9},
    ]
