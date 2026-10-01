"""
Step 02 - Average number of well neighbours and its first four derivatives.

The first-order perturbation term of the square-well free energy is proportional to nu_HS, the average number of neighbours a particle has inside the attractive well,

nu_HS(phi, lambda) = 24 * phi * Integral_1^lambda g_HS(x, phi) x^2 dx = 8 * (lambda**3 - 1) * phi * g_HS(1, phi')

where 8*(lambda**3 - 1)*phi is the dilute limit of nu_HS and g_HS(1, phi') is the Carnahan-Starling contact value

g_HS(1, p) = (1 - p/2) / (1 - p)**3

evaluated at the effective volume fraction phi' = phi'(phi, lambda) of Step 01. Downstream steps differentiate this quantity repeatedly. The second-order perturbation term is built from d(nu_HS)/d(phi); the chemical potential differentiates that term once more; the spinodal condition differentiates it a second time; and the critical point, where the two spinodal roots merge, needs one derivative beyond that. Four derivatives of nu_HS are therefore required. Obtain all of them analytically. nu_HS is a rational function of phi, because phi' is rational in phi and the contact value is rational in phi', so exact expressions exist; the chain rule for a composition taken to fourth order is the only machinery needed. Numerical differencing is not accurate enough here: the fourth derivative enters the critical point, where a relative error of 1e-8 in the derivative chain already moves the critical temperature in the third decimal.

Returns
-------
np.ndarray of shape (5,) and dtype float, holding nu_HS and its first four derivatives with respect to phi
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def contact_number_derivatives(phi: float, lam: float) -> "np.ndarray":
    '''Average well-neighbour count nu_HS and its first four phi-derivatives.

    Parameters
    ----------
    phi : float
        Protein volume fraction of the fluid phase, 0 < phi < 1.
    lam : float
        Reduced square-well range lambda, lam > 1.

    Returns
    -------
    out : np.ndarray
        Array of shape (5,), [nu_HS, d1, d2, d3, d4], where dk is the k-th
        derivative of nu_HS with respect to phi.

    Raises
    ------
    ValueError
        If phi is not strictly inside (0, 1), or if lam <= 1, or if either
        argument is not finite, or if the effective volume fraction reaches 1
        so that the contact value diverges.
    '''
    return out  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _pade_derivatives(phi, lam):
    """phi' and its first four derivatives, from p * M = N by Leibniz."""
    import numpy as np
    c1, c2, c3 = _pade_coefficients(lam)
    phi = float(phi)
    N0 = c1 * phi + c2 * phi ** 2
    N1 = c1 + 2.0 * c2 * phi
    N2 = 2.0 * c2
    N3 = 0.0
    N4 = 0.0
    b = 1.0 + c3 * phi
    M0 = b ** 3
    M1 = 3.0 * c3 * b ** 2
    M2 = 6.0 * c3 ** 2 * b
    M3 = 6.0 * c3 ** 3
    M4 = 0.0
    if M0 == 0.0:
        raise ValueError("degenerate Pade denominator")
    p0 = _oracle_pade_effective_volume_fraction(phi, lam)
    p1 = (N1 - p0 * M1) / M0
    p2 = (N2 - 2.0 * p1 * M1 - p0 * M2) / M0
    p3 = (N3 - 3.0 * p2 * M1 - 3.0 * p1 * M2 - p0 * M3) / M0
    p4 = (N4 - 4.0 * p3 * M1 - 6.0 * p2 * M2 - 4.0 * p1 * M3 - p0 * M4) / M0
    return p0, p1, p2, p3, p4


def _oracle_contact_number_derivatives(phi: float, lam: float) -> "np.ndarray":
    import numpy as np
    _validate_phi_lam(phi, lam)
    phi = float(phi)
    p0, p1, p2, p3, p4 = _pade_derivatives(phi, lam)
    u = 1.0 - p0
    if u <= 0.0:
        raise ValueError("effective volume fraction reached 1; contact value diverges")
    # g and its p-derivatives, written in u = 1 - p where g = (u**-3 + u**-2) / 2
    g0 = 0.5 * (u ** -3 + u ** -2)
    g1 = 1.5 * u ** -4 + u ** -3
    g2 = 6.0 * u ** -5 + 3.0 * u ** -4
    g3 = 30.0 * u ** -6 + 12.0 * u ** -5
    g4 = 180.0 * u ** -7 + 60.0 * u ** -6
    # Faa di Bruno for G(phi) = g(p(phi))
    G0 = g0
    G1 = g1 * p1
    G2 = g2 * p1 ** 2 + g1 * p2
    G3 = g3 * p1 ** 3 + 3.0 * g2 * p1 * p2 + g1 * p3
    G4 = (g4 * p1 ** 4 + 6.0 * g3 * p1 ** 2 * p2 + 3.0 * g2 * p2 ** 2
          + 4.0 * g2 * p1 * p3 + g1 * p4)
    K = 8.0 * (float(lam) ** 3 - 1.0)
    return np.array([K * phi * G0,
                     K * (G0 + phi * G1),
                     K * (2.0 * G1 + phi * G2),
                     K * (3.0 * G2 + phi * G3),
                     K * (4.0 * G3 + phi * G4)], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: at the critical volume fraction of this range ---
        {
            "setup": """import numpy as np
phi = 0.20
lam = 1.3
""",
            "call": "contact_number_derivatives(phi, lam)",
            "gold_call": "_oracle_contact_number_derivatives(phi, lam)",
        },
        # --- Normal: dense coexisting branch ---
        {
            "setup": """import numpy as np
phi = 0.3474
lam = 1.3
""",
            "call": "contact_number_derivatives(phi, lam)",
            "gold_call": "_oracle_contact_number_derivatives(phi, lam)",
        },
        # --- Boundary: dilute limit, nu_HS -> 8*(lam**3-1)*phi ---
        {
            "setup": """import numpy as np
phi = 1e-7
lam = 1.3
""",
            "call": "contact_number_derivatives(phi, lam)",
            "gold_call": "_oracle_contact_number_derivatives(phi, lam)",
        },
        # --- Boundary: shortest range for which the theory is recommended ---
        {
            "setup": """import numpy as np
phi = 0.25
lam = 1.2
""",
            "call": "contact_number_derivatives(phi, lam)",
            "gold_call": "_oracle_contact_number_derivatives(phi, lam)",
        },
        # --- Boundary: crystal-like packing, where the high derivatives are stiffest ---
        {
            "setup": """import numpy as np
phi = 0.55
lam = 1.4
""",
            "call": "contact_number_derivatives(phi, lam)",
            "gold_call": "_oracle_contact_number_derivatives(phi, lam)",
        },
        # --- Edge: invalid volume fraction must raise ValueError ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        contact_number_derivatives(1.0, 1.3)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_contact_number_derivatives(1.0, 1.3)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Edge: non-finite input must raise ValueError ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        contact_number_derivatives(float('nan'), 1.3)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_contact_number_derivatives(float('nan'), 1.3)
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
