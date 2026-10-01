"""
Step 03 - Carnahan-Starling hard-sphere reference terms.

The protein solution is treated as a one-component compressible fluid of colloidal particles: the aqueous medium and all low-molecular-weight cosolutes are absorbed into the background, so the protein osmotic pressure plays the role of a gas pressure and liquid-liquid phase separation is a gas-liquid transition. Working with the reduced Helmholtz free energy a = beta * (V_P / V) * A, where V_P is the particle volume and beta = 1 / T with k_B = 1 (so energies carry temperature units), the hard-sphere reference contribution is

a_HS(phi) = phi * mu0 + phi * (ln(phi) - 1) + (4 - 3*phi) / (1 - phi)**2 * phi**2

The first term is an inconsequential standard-state constant, the second is the ideal contribution from translational motion, and the third is the Carnahan-Starling excess free energy for excluded-volume repulsion. Take mu0 = 0 throughout; it cancels from every phase-equilibrium condition. Differentiating gives the reduced chemical potential and, through pi = phi * mu - a, the reduced osmotic pressure:

mu_HS(phi) = ln(phi) + (8 - 9*phi + 3*phi**2) * phi / (1 - phi)**3 pi_HS(phi) = phi * (1 + phi + phi**2 - phi**3) / (1 - phi)**3

The spinodal and critical-point conditions further downstream need the first and second derivatives of mu_HS with respect to phi, so return those as well. Both are elementary derivatives of the rational expression above; take them analytically.

Returns
-------
np.ndarray of shape (5,) and dtype float, holding [a_HS, mu_HS, pi_HS, d(mu_HS)/d(phi), d^2(mu_HS)/d(phi)^2]
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def hard_sphere_reference(phi: float) -> "np.ndarray":
    '''Carnahan-Starling reduced free energy, potentials and mu-derivatives.

    Parameters
    ----------
    phi : float
        Protein volume fraction of the fluid phase, 0 < phi < 1.

    Returns
    -------
    out : np.ndarray
        Array of shape (5,), [a_HS, mu_HS, pi_HS, d(mu_HS)/d(phi),
        d^2(mu_HS)/d(phi)^2], with the standard-state constant mu0 set to zero.

    Raises
    ------
    ValueError
        If phi is not strictly inside (0, 1) or is not finite.
    '''
    return out  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_hard_sphere_reference(phi: float) -> "np.ndarray":
    import numpy as np
    if not np.isfinite(phi):
        raise ValueError("phi must be finite")
    phi = float(phi)
    if not (0.0 < phi < 1.0):
        raise ValueError("phi must satisfy 0 < phi < 1")
    om = 1.0 - phi
    a = phi * (np.log(phi) - 1.0) + (4.0 - 3.0 * phi) / om ** 2 * phi ** 2
    # h = P / V with P = 8 phi - 9 phi^2 + 3 phi^3 and V = (1 - phi)^3, by Leibniz
    P0 = 8.0 * phi - 9.0 * phi ** 2 + 3.0 * phi ** 3
    P1 = 8.0 - 18.0 * phi + 9.0 * phi ** 2
    P2 = -18.0 + 18.0 * phi
    V0 = om ** 3
    V1 = -3.0 * om ** 2
    V2 = 6.0 * om
    h0 = P0 / V0
    h1 = (P1 - h0 * V1) / V0
    h2 = (P2 - 2.0 * h1 * V1 - h0 * V2) / V0
    mu = np.log(phi) + h0
    pi = phi * (1.0 + phi + phi ** 2 - phi ** 3) / om ** 3
    return np.array([a, mu, pi, 1.0 / phi + h1, -1.0 / phi ** 2 + h2], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: near the critical volume fraction ---
        {
            "setup": """import numpy as np
phi = 0.20
""",
            "call": "hard_sphere_reference(phi)",
            "gold_call": "_oracle_hard_sphere_reference(phi)",
        },
        # --- Normal: dense coexisting branch ---
        {
            "setup": """import numpy as np
phi = 0.3474
""",
            "call": "hard_sphere_reference(phi)",
            "gold_call": "_oracle_hard_sphere_reference(phi)",
        },
        # --- Boundary: dilute solution where the ideal term dominates ---
        {
            "setup": """import numpy as np
phi = 3.6e-4
""",
            "call": "hard_sphere_reference(phi)",
            "gold_call": "_oracle_hard_sphere_reference(phi)",
        },
        # --- Boundary: crystal-like packing where the CS terms are stiff ---
        {
            "setup": """import numpy as np
phi = 0.57
""",
            "call": "hard_sphere_reference(phi)",
            "gold_call": "_oracle_hard_sphere_reference(phi)",
        },
        # --- Edge: phi = 0 must raise ValueError (logarithm diverges) ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        hard_sphere_reference(0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_hard_sphere_reference(0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Edge: phi >= 1 must raise ValueError ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        hard_sphere_reference(1.2)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_hard_sphere_reference(1.2)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
