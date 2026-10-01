"""
Step 07 - Assembled reduced potentials and their volume-fraction derivatives.

With the reference and residual pieces in hand the fluid-phase thermodynamics is

a(phi)  = a_HS(phi) + a_mf(phi) + a_R2(phi) mu(phi) = ( d a / d phi )_beta pi(phi) = phi * mu(phi) - a(phi)

where mu is the reduced protein chemical potential and pi the reduced osmotic pressure, both in the units fixed in Step 03 (mu_hat = beta*mu, pi_hat = beta*Pi*V_P). The chemical potential describes insertion of one protein accompanied by isochoric removal of solvent, which is why the osmotic pressure rather than the total pressure is the mechanical variable that has to be matched between coexisting phases. The spinodal is the locus where d(mu)/d(phi) vanishes, and the critical point is where d(mu)/d(phi) and d^2(mu)/d(phi)^2 vanish together, so this step returns those two derivatives alongside a, mu and pi. Every contribution is available analytically from Steps 03, 05 and 06, so all five outputs follow by summing derivative contributions term by term; nothing here should be obtained by differencing the assembled function. This step takes the bare dimensionless depth beta*eps_SW and the degeneracy factor alpha and converts them to the effective depth with Step 04 before combining the contributions.

Returns
-------
np.ndarray of shape (5,) and dtype float, holding [a, mu, pi, d(mu)/d(phi), d^2(mu)/d(phi)^2]
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def reduced_potentials(phi: float, lam: float, beta_eps_sw: float,
                       alpha: float) -> "np.ndarray":
    '''Reduced free energy, potentials and the two mu-derivatives of the fluid.

    Parameters
    ----------
    phi : float
        Protein volume fraction of the fluid phase, 0 < phi < 1.
    lam : float
        Reduced square-well range lambda, lam > 1.
    beta_eps_sw : float
        Dimensionless bare square-well depth beta * eps_SW, must be >= 0.
    alpha : float
        Degeneracy factor, 0 < alpha <= 1.

    Returns
    -------
    out : np.ndarray
        Array of shape (5,), [a, mu, pi, d(mu)/d(phi), d^2(mu)/d(phi)^2], with
        the standard-state constant mu0 set to zero.

    Raises
    ------
    ValueError
        If phi is not strictly inside (0, 1), or lam <= 1, or beta_eps_sw is
        negative, or alpha is outside (0, 1], or any argument is not finite.
    '''
    return out  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_reduced_potentials(phi: float, lam: float, beta_eps_sw: float,
                               alpha: float) -> "np.ndarray":
    import numpy as np
    B = _oracle_effective_well_depth(beta_eps_sw, alpha)
    hs = _oracle_hard_sphere_reference(phi)
    mf = _oracle_mean_field_term(phi, lam, B)
    r2 = _oracle_second_order_term(phi, lam, B)
    phi = float(phi)
    a = hs[0] + mf[0] + r2[0]
    mu = hs[1] + mf[1] + r2[1]
    dmu = hs[3] + mf[2] + r2[2]
    d2mu = hs[4] + mf[3] + r2[3]
    return np.array([a, mu, phi * mu - a, dmu, d2mu], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: dense branch of the additive system at the working temperature ---
        {
            "setup": """import numpy as np
phi = 0.3474
lam = 1.3
beta_eps_sw = 1732.0 / 250.0
alpha = 0.0293
""",
            "call": "reduced_potentials(phi, lam, beta_eps_sw, alpha)",
            "gold_call": "_oracle_reduced_potentials(phi, lam, beta_eps_sw, alpha)",
        },
        # --- Normal: reference system dilute branch ---
        {
            "setup": """import numpy as np
phi = 0.01734
lam = 1.3
beta_eps_sw = 1687.0 / 250.0
alpha = 0.0372
""",
            "call": "reduced_potentials(phi, lam, beta_eps_sw, alpha)",
            "gold_call": "_oracle_reduced_potentials(phi, lam, beta_eps_sw, alpha)",
        },
        # --- Boundary: at the critical volume fraction, where both mu-derivatives
        #     approach zero together ---
        {
            "setup": """import numpy as np
phi = 0.199253156667
lam = 1.3
beta_eps_sw = 1732.0 / 265.7541454524
alpha = 0.0293
""",
            "call": "reduced_potentials(phi, lam, beta_eps_sw, alpha)",
            "gold_call": "_oracle_reduced_potentials(phi, lam, beta_eps_sw, alpha)",
        },
        # --- Boundary: no attraction reduces the result to hard spheres ---
        {
            "setup": """import numpy as np
phi = 0.30
lam = 1.3
beta_eps_sw = 0.0
alpha = 0.05
""",
            "call": "reduced_potentials(phi, lam, beta_eps_sw, alpha)",
            "gold_call": "_oracle_reduced_potentials(phi, lam, beta_eps_sw, alpha)",
        },
        # --- Boundary: very dilute solution on the solubility branch ---
        {
            "setup": """import numpy as np
phi = 3.64e-4
lam = 1.3
beta_eps_sw = 1732.0 / 250.0
alpha = 0.0293
""",
            "call": "reduced_potentials(phi, lam, beta_eps_sw, alpha)",
            "gold_call": "_oracle_reduced_potentials(phi, lam, beta_eps_sw, alpha)",
        },
        # --- Edge: alpha outside (0, 1] must raise ValueError ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        reduced_potentials(0.2, 1.3, 6.9, 1.5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_reduced_potentials(0.2, 1.3, 6.9, 1.5)
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
        reduced_potentials(1.0, 1.3, 6.9, 0.03)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_reduced_potentials(1.0, 1.3, 6.9, 0.03)
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
