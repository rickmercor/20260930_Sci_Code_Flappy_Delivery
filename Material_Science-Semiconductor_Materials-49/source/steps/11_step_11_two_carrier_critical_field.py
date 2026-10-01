"""
Critical field at avalanche breakdown with the electron and hole ionisation coefficients kept separate.

Take a one sided abrupt junction on an n-type drift layer and measure x from the blocking junction, where the field is largest. The field falls linearly, E(x) = E_cr - q N_D x/eps, across a triangular profile of width W = eps E_cr/(q N_D). Electrons drift away from the junction and holes drift back towards it, so the multiplication factor diverges when int_0^W alpha_n exp(-int_0^x (alpha_n - alpha_p) dx') dx = 1, with alpha_n = a_n exp(-b_n/E) and alpha_p = a_p exp(-b_p/E). No effective coefficient is formed: both species enter separately. Solve for the peak field E_cr at which that condition holds, converged to a relative precision of at least 1e-11, using the elementary charge 1.602e-19 C and the vacuum permittivity 8.854e-14 F/cm, with doping in cm^-3.

Returns
-------
float, the two-carrier critical field in V/cm
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def two_carrier_critical_field(a_n: float, b_n: float, a_p: float, b_p: float, eps_r: float, doping: float) -> float:
    '''Peak field at which the electron-initiated two-carrier ionisation integral reaches one.

    Parameters
    ----------
    a_n, a_p : float
        Electron and hole ionisation prefactors in cm^-1.
    b_n, b_p : float
        Electron and hole ionisation field scales in V/cm.
    eps_r : float
        Relative permittivity.
    doping : float
        Drift layer doping in cm^-3.

    Returns
    -------
    float
        Critical field in V/cm, converged to a relative precision of at least 1e-11.

    Raises
    ------
    ValueError
        If any of a_n, b_n, a_p, b_p, eps_r, doping is not finite or is not strictly positive.
    '''
    return 0.0  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.special import exp1
from scipy.integrate import quad
from scipy.optimize import brentq

def _ionisation_antiderivative(a: float, b: float, e: float) -> float:
    import numpy as np
    from scipy.special import exp1
    return a*(e*np.exp(-b/e) - b*exp1(b/e))

def _oracle_two_carrier_critical_field(a_n: float, b_n: float, a_p: float, b_p: float, eps_r: float, doping: float) -> float:
    import numpy as np
    from scipy.integrate import quad
    from scipy.optimize import brentq
    for _v in (a_n, b_n, a_p, b_p, eps_r, doping):
        if not np.isfinite(_v) or _v <= 0.0:
            raise ValueError('a_n, b_n, a_p, b_p, eps_r, doping must be finite and strictly positive')
    q = 1.602e-19
    k = eps_r*8.854e-14/(q*doping)

    def _condition(e_cr: float) -> float:
        fn_top = _ionisation_antiderivative(a_n, b_n, e_cr)
        fp_top = _ionisation_antiderivative(a_p, b_p, e_cr)

        def _integrand(e: float) -> float:
            inner = k*((fn_top - _ionisation_antiderivative(a_n, b_n, e)) - (fp_top - _ionisation_antiderivative(a_p, b_p, e)))
            return k*a_n*np.exp(-b_n/e)*np.exp(-inner)

        val = quad(_integrand, 1.0e-6*e_cr, e_cr, limit=400, epsabs=0.0, epsrel=1.0e-13)[0]
        return val - 1.0

    lo, hi = 1.0e4, 1.0e5
    while _condition(hi) < 0.0:
        lo, hi = hi, 2.0*hi
        if hi > 1.0e9:
            raise ValueError('no breakdown field below 1e9 V/cm')
    return float(brentq(_condition, lo, hi, xtol=1.0e-13*hi, rtol=1.0e-15, maxiter=300))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nfrom scipy.special import exp1\nfrom scipy.integrate import quad\nfrom scipy.optimize import brentq",
            "call": "round(two_carrier_critical_field(8.2e9, 3.94e7, 4.5e6, 1.28e7, 9.7, 3.0e16)*1e-6, 6)",
            "gold_call": "round(_oracle_two_carrier_critical_field(8.2e9, 3.94e7, 4.5e6, 1.28e7, 9.7, 3.0e16)*1e-6, 6)",
        },
        {
            "setup": "import numpy as np\nfrom scipy.special import exp1\nfrom scipy.integrate import quad\nfrom scipy.optimize import brentq",
            "call": "round(two_carrier_critical_field(8.2e9, 3.94e7, 4.5e6, 1.28e7, 9.7, 2.0e15)*1e-6, 6)",
            "gold_call": "round(_oracle_two_carrier_critical_field(8.2e9, 3.94e7, 4.5e6, 1.28e7, 9.7, 2.0e15)*1e-6, 6)",
        },
        {
            "setup": "import numpy as np\nfrom scipy.special import exp1\nfrom scipy.integrate import quad\nfrom scipy.optimize import brentq",
            "call": "round(two_carrier_critical_field(8.2e9, 3.94e7, 4.5e6, 1.28e7, 9.7, 2.0e17)*1e-6, 6)",
            "gold_call": "round(_oracle_two_carrier_critical_field(8.2e9, 3.94e7, 4.5e6, 1.28e7, 9.7, 2.0e17)*1e-6, 6)",
        },
        {
            "setup": "import numpy as np\nfrom scipy.special import exp1\nfrom scipy.integrate import quad\nfrom scipy.optimize import brentq",
            "call": "round(two_carrier_critical_field(1.92094e8, 2.61e7, 1.92094e8, 2.61e7, 9.7, 3.0e16)*1e-6, 6)",
            "gold_call": "round(_oracle_two_carrier_critical_field(1.92094e8, 2.61e7, 1.92094e8, 2.61e7, 9.7, 3.0e16)*1e-6, 6)",
        },
        {
            "setup": "import numpy as np\nfrom scipy.special import exp1\nfrom scipy.integrate import quad\nfrom scipy.optimize import brentq\ndef run_model():\n    try:\n        two_carrier_critical_field(8.2e9, 3.94e7, 4.5e6, 1.28e7, 9.7, 0.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_two_carrier_critical_field(8.2e9, 3.94e7, 4.5e6, 1.28e7, 9.7, 0.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
