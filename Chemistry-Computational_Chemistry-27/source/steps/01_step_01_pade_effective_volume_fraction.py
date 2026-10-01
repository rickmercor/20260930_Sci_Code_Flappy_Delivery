"""
Step 01 - Effective volume fraction of the Pade contact-value map.

The residual (attractive) part of the Barker-Henderson perturbation theory for a square-well fluid is built from the average number of neighbours a particle has inside the well. That quantity is obtained from the hard-sphere pair distribution function integrated over the well, and the standard way to evaluate the integral in closed form is to replace it by the Carnahan-Starling contact value evaluated not at the true packing fraction phi but at an effective packing fraction phi' that absorbs the range dependence. The effective volume fraction is the Pade form

phi'(phi, lambda) = phi * (c1 + c2 * phi) / (1 + c3 * phi)**3

with range-dependent coefficients

c_i = sum_{j=1..4} s_ij * lambda**(-j),      i = 1, 2, 3

and the fixed 3 x 4 coefficient matrix

s = [[ -3.1649,   13.3501,  -14.8057,    5.7029], [ 43.0042, -191.6623,  273.8968, -128.9334], [ 65.0419, -266.4627,  361.0431, -162.6996]]

lambda is the reduced square-well range (attraction acts for sigma <= r <= lambda*sigma), and phi is the protein volume fraction of the fluid phase.

Returns
-------
float, the effective volume fraction phi' as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def pade_effective_volume_fraction(phi: float, lam: float) -> float:
    '''Effective volume fraction entering the hard-sphere contact value.

    Parameters
    ----------
    phi : float
        Protein volume fraction of the fluid phase, 0 < phi < 1.
    lam : float
        Reduced square-well range lambda, lam > 1.

    Returns
    -------
    phi_eff : float
        Effective volume fraction phi' as a native Python float.

    Raises
    ------
    ValueError
        If phi is not strictly inside (0, 1), or if lam <= 1, or if either
        argument is not finite.
    '''
    return phi_eff  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _pade_coefficients(lam):
    """Range-dependent Pade coefficients c1, c2, c3."""
    import numpy as np
    s_ij = np.array([[-3.1649, 13.3501, -14.8057, 5.7029],
                     [43.0042, -191.6623, 273.8968, -128.9334],
                     [65.0419, -266.4627, 361.0431, -162.6996]], dtype=float)
    j = np.arange(1, 5, dtype=float)
    return s_ij @ (float(lam) ** (-j))


def _validate_phi_lam(phi, lam):
    """Shared argument validation for the volume fraction and the well range."""
    import numpy as np
    if not np.isfinite(phi) or not np.isfinite(lam):
        raise ValueError("phi and lam must be finite")
    if not (0.0 < float(phi) < 1.0):
        raise ValueError("phi must satisfy 0 < phi < 1")
    if float(lam) <= 1.0:
        raise ValueError("lam must be > 1")


def _oracle_pade_effective_volume_fraction(phi: float, lam: float) -> float:
    import numpy as np
    _validate_phi_lam(phi, lam)
    phi = float(phi)
    c1, c2, c3 = _pade_coefficients(lam)
    denom = (1.0 + c3 * phi) ** 3
    if denom == 0.0:
        raise ValueError("degenerate Pade denominator")
    return float(phi * (c1 + c2 * phi) / denom)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: moderate packing at the range used for protein solutions ---
        {
            "setup": """import numpy as np
phi = 0.20
lam = 1.3
""",
            "call": "pade_effective_volume_fraction(phi, lam)",
            "gold_call": "_oracle_pade_effective_volume_fraction(phi, lam)",
        },
        # --- Normal: dense liquid branch ---
        {
            "setup": """import numpy as np
phi = 0.38
lam = 1.3
""",
            "call": "pade_effective_volume_fraction(phi, lam)",
            "gold_call": "_oracle_pade_effective_volume_fraction(phi, lam)",
        },
        # --- Boundary: very dilute, phi' must go to zero linearly ---
        {
            "setup": """import numpy as np
phi = 1e-8
lam = 1.3
""",
            "call": "pade_effective_volume_fraction(phi, lam)",
            "gold_call": "_oracle_pade_effective_volume_fraction(phi, lam)",
        },
        # --- Boundary: longer range, where the coefficients change sign structure ---
        {
            "setup": """import numpy as np
phi = 0.30
lam = 1.6
""",
            "call": "pade_effective_volume_fraction(phi, lam)",
            "gold_call": "_oracle_pade_effective_volume_fraction(phi, lam)",
        },
        # --- Edge: phi out of range must raise ValueError ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        pade_effective_volume_fraction(0.0, 1.3)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_pade_effective_volume_fraction(0.0, 1.3)
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
        pade_effective_volume_fraction(0.2, 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_pade_effective_volume_fraction(0.2, 1.0)
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
