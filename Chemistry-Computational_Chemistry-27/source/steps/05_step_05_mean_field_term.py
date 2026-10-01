"""
Step 05 - First-order mean-field perturbation term.

Splitting the reduced Helmholtz free energy as a = a_HS + a_R, the residual part is itself expanded about the hard-sphere reference. Its leading contribution is the rigorous first-order (mean-field) correction, which counts every neighbour inside the well once and weights it by the well depth,

a_mf = -(1/2) * nu_HS(phi, lambda) * phi * beta * eps

The factor 1/2 avoids double counting pair interactions. Here beta*eps is the dimensionless well depth that actually acts between particles; for anisotropic protein-protein contacts that is the effective depth beta*eps_eff of Step 04, not the bare one. Because the chemical potential, the spinodal and the critical point each differentiate the free energy once more than the last, return a_mf together with its first three derivatives with respect to phi. Every one of them follows from the product rule applied to nu_HS * phi, using the derivatives of nu_HS supplied by Step 02; beta*eps does not depend on phi.

Returns
-------
np.ndarray of shape (4,) and dtype float, holding a_mf and its first three derivatives with respect to phi
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def mean_field_term(phi: float, lam: float, beta_eps_eff: float) -> "np.ndarray":
    '''First-order mean-field free-energy term and its first three phi-derivatives.

    Parameters
    ----------
    phi : float
        Protein volume fraction of the fluid phase, 0 < phi < 1.
    lam : float
        Reduced square-well range lambda, lam > 1.
    beta_eps_eff : float
        Dimensionless effective well depth beta * eps_eff, must be >= 0.

    Returns
    -------
    out : np.ndarray
        Array of shape (4,), [a_mf, d1, d2, d3], where dk is the k-th derivative
        of a_mf with respect to phi.

    Raises
    ------
    ValueError
        If phi is not strictly inside (0, 1), or lam <= 1, or beta_eps_eff is
        negative, or any argument is not finite.
    '''
    return out  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_mean_field_term(phi: float, lam: float, beta_eps_eff: float) -> "np.ndarray":
    import numpy as np
    _validate_phi_lam(phi, lam)
    if not np.isfinite(beta_eps_eff):
        raise ValueError("beta_eps_eff must be finite")
    B = float(beta_eps_eff)
    if B < 0.0:
        raise ValueError("beta_eps_eff must be >= 0")
    phi = float(phi)
    n0, n1, n2, n3, _ = _oracle_contact_number_derivatives(phi, lam)
    return np.array([-0.5 * B * n0 * phi,
                     -0.5 * B * (n0 + phi * n1),
                     -0.5 * B * (2.0 * n1 + phi * n2),
                     -0.5 * B * (3.0 * n2 + phi * n3)], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: dense branch of the additive system ---
        {
            "setup": """import numpy as np
phi = 0.3474
lam = 1.3
beta_eps_eff = 1.0552
""",
            "call": "mean_field_term(phi, lam, beta_eps_eff)",
            "gold_call": "_oracle_mean_field_term(phi, lam, beta_eps_eff)",
        },
        # --- Normal: dilute branch at the same state point ---
        {
            "setup": """import numpy as np
phi = 0.0411
lam = 1.3
beta_eps_eff = 1.0552
""",
            "call": "mean_field_term(phi, lam, beta_eps_eff)",
            "gold_call": "_oracle_mean_field_term(phi, lam, beta_eps_eff)",
        },
        # --- Boundary: zero attraction switches the whole term off ---
        {
            "setup": """import numpy as np
phi = 0.25
lam = 1.3
beta_eps_eff = 0.0
""",
            "call": "mean_field_term(phi, lam, beta_eps_eff)",
            "gold_call": "_oracle_mean_field_term(phi, lam, beta_eps_eff)",
        },
        # --- Boundary: longer range strengthens the mean-field term ---
        {
            "setup": """import numpy as np
phi = 0.20
lam = 1.5
beta_eps_eff = 0.9
""",
            "call": "mean_field_term(phi, lam, beta_eps_eff)",
            "gold_call": "_oracle_mean_field_term(phi, lam, beta_eps_eff)",
        },
        # --- Edge: negative dimensionless depth must raise ValueError ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        mean_field_term(0.2, 1.3, -0.5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_mean_field_term(0.2, 1.3, -0.5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Edge: invalid volume fraction must raise ValueError ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        mean_field_term(-0.1, 1.3, 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_mean_field_term(-0.1, 1.3, 1.0)
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
