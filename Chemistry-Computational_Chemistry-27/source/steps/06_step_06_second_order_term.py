"""
Step 06 - Second-order residual term under the local-compressibility approximation.

Beyond the mean-field term, the residual free energy carries a second contribution a_R2 that measures how the density around a central particle fluctuates. In the local compressibility approximation those fluctuations are described by the isothermal compressibility of the hard-sphere reference, which turns the term into

a_R2 = -(1/2) * phi**2 * ( d(nu_HS) / d(pi_HS) ) * W(beta*eps) d(nu_HS) / d(pi_HS) = ( d(nu_HS)/d(phi) ) * ( d(phi)/d(pi_HS) )

with the Carnahan-Starling compressibility factor

d(phi)/d(pi_HS) = (1 - phi)**4 / (1 + 4*phi + 4*phi**2 - 4*phi**3 + phi**4)

Everything above is standard. What is not standard is the well factor W. The textbook closure expands the Mayer function of the well to quadratic order and takes W = (beta*eps)**2 / 2. That choice is convenient but it leaves the theory inconsistent with the second virial coefficient: it makes the model predict a coefficient in which the well factor is truncated after the quadratic term, whereas the exact square-well result is

B / 4 = 1 - (lambda**3 - 1) * ( exp(beta*eps) - 1 )

Since the second virial coefficient is the one interaction measure that experiment reports directly, the model must reproduce it exactly. Work out the well factor W that does so while still agreeing with the quadratic truncation through second order in beta*eps, and use that factor. Do not use the quadratic truncation itself: at the well depths reached here the two differ enough to move the dense coexisting branch by several percent. As before beta*eps is the effective depth of Step 04. Return a_R2 together with its first three derivatives with respect to phi, which the chemical potential, the spinodal and the critical point consume in turn. They follow from the product rule applied to phi**2, d(nu_HS)/d(phi) and the compressibility factor, and therefore need the higher derivatives of nu_HS from Step 02 as well as derivatives of the compressibility factor.

Returns
-------
np.ndarray of shape (4,) and dtype float, holding a_R2 and its first three derivatives with respect to phi
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def second_order_term(phi: float, lam: float, beta_eps_eff: float) -> "np.ndarray":
    '''Second-order residual free-energy term and its first three phi-derivatives.

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
        Array of shape (4,), [a_R2, d1, d2, d3], where dk is the k-th derivative
        of a_R2 with respect to phi.

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


def _compressibility_factor(phi):
    """d(phi)/d(pi_HS) and its first three derivatives, from D * Qd = (1-phi)^4."""
    import numpy as np
    phi = float(phi)
    om = 1.0 - phi
    U0 = om ** 4
    U1 = -4.0 * om ** 3
    U2 = 12.0 * om ** 2
    U3 = -24.0 * om
    Q0 = 1.0 + 4.0 * phi + 4.0 * phi ** 2 - 4.0 * phi ** 3 + phi ** 4
    Q1 = 4.0 + 8.0 * phi - 12.0 * phi ** 2 + 4.0 * phi ** 3
    Q2 = 8.0 - 24.0 * phi + 12.0 * phi ** 2
    Q3 = -24.0 + 24.0 * phi
    D0 = U0 / Q0
    D1 = (U1 - D0 * Q1) / Q0
    D2 = (U2 - 2.0 * D1 * Q1 - D0 * Q2) / Q0
    D3 = (U3 - 3.0 * D2 * Q1 - 3.0 * D1 * Q2 - D0 * Q3) / Q0
    return D0, D1, D2, D3


def _oracle_second_order_term(phi: float, lam: float, beta_eps_eff: float) -> "np.ndarray":
    import numpy as np
    _validate_phi_lam(phi, lam)
    if not np.isfinite(beta_eps_eff):
        raise ValueError("beta_eps_eff must be finite")
    B = float(beta_eps_eff)
    if B < 0.0:
        raise ValueError("beta_eps_eff must be >= 0")
    phi = float(phi)
    _, n1, n2, n3, n4 = _oracle_contact_number_derivatives(phi, lam)
    D0, D1, D2, D3 = _compressibility_factor(phi)
    # the well factor that restores the exact square-well second virial coefficient
    F = np.expm1(B) - B
    R0, R1, R2, R3 = phi ** 2, 2.0 * phi, 2.0, 0.0
    S0, S1, S2, S3 = n1, n2, n3, n4
    T0, T1, T2, T3 = D0, D1, D2, D3
    Q0 = R0 * S0 * T0
    Q1 = R1 * S0 * T0 + R0 * S1 * T0 + R0 * S0 * T1
    Q2 = (R2 * S0 * T0 + R0 * S2 * T0 + R0 * S0 * T2
          + 2.0 * (R1 * S1 * T0 + R1 * S0 * T1 + R0 * S1 * T1))
    Q3 = (R3 * S0 * T0 + R0 * S3 * T0 + R0 * S0 * T3
          + 3.0 * (R2 * S1 * T0 + R2 * S0 * T1 + R1 * S2 * T0
                   + R0 * S2 * T1 + R1 * S0 * T2 + R0 * S1 * T2)
          + 6.0 * R1 * S1 * T1)
    return np.array([-0.5 * F * Q0, -0.5 * F * Q1, -0.5 * F * Q2, -0.5 * F * Q3],
                    dtype=float)

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
            "call": "second_order_term(phi, lam, beta_eps_eff)",
            "gold_call": "_oracle_second_order_term(phi, lam, beta_eps_eff)",
        },
        # --- Normal: dilute branch, where the term is small but not negligible ---
        {
            "setup": """import numpy as np
phi = 0.0411
lam = 1.3
beta_eps_eff = 1.0552
""",
            "call": "second_order_term(phi, lam, beta_eps_eff)",
            "gold_call": "_oracle_second_order_term(phi, lam, beta_eps_eff)",
        },
        # --- Boundary: zero attraction must switch the term off exactly ---
        {
            "setup": """import numpy as np
phi = 0.30
lam = 1.3
beta_eps_eff = 0.0
""",
            "call": "second_order_term(phi, lam, beta_eps_eff)",
            "gold_call": "_oracle_second_order_term(phi, lam, beta_eps_eff)",
        },
        # --- Boundary: strong attraction, where the restored and truncated
        #     well factors differ the most ---
        {
            "setup": """import numpy as np
phi = 0.25
lam = 1.3
beta_eps_eff = 3.0
""",
            "call": "second_order_term(phi, lam, beta_eps_eff)",
            "gold_call": "_oracle_second_order_term(phi, lam, beta_eps_eff)",
        },
        # --- Boundary: shallow depth, where the restored and truncated well factors
        # differ by only a few parts in 1e8 of the term itself ---
        {
            "setup": """import numpy as np
phi = 0.25
lam = 1.3
beta_eps_eff = 1e-2
""",
            "call": "second_order_term(phi, lam, beta_eps_eff)",
            "gold_call": "_oracle_second_order_term(phi, lam, beta_eps_eff)",
        },
        # --- Edge: negative dimensionless depth must raise ValueError ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        second_order_term(0.25, 1.3, -2.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_second_order_term(0.25, 1.3, -2.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Edge: lam <= 1 must raise ValueError ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        second_order_term(0.25, 0.9, 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_second_order_term(0.25, 0.9, 1.0)
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
