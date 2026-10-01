"""
Form the dimensionless group that controls the breakdown condition.

Once a single effective coefficient is in hand, the breakdown integral over a triangular field profile depends on the material and the drift doping only through one dimensionless combination built from the effective prefactor, the effective field scale, the permittivity and the doping. Grouping the problem this way is what makes one solution serve every semiconductor. Use the elementary charge 1.602e-19 C and the vacuum permittivity 8.854e-14 F/cm, with doping in cm^-3.

Returns
-------
float, the dimensionless group (no units)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def ionization_dimensionless_group(a_eff: float, b_eff: float, eps_r: float, doping: float) -> float:
    '''Dimensionless group controlling the avalanche condition.

    Parameters
    ----------
    a_eff : float
        Effective prefactor in cm^-1.
    b_eff : float
        Effective field scale in V/cm.
    eps_r : float
        Relative permittivity of the semiconductor.
    doping : float
        Drift region doping in cm^-3.

    Returns
    -------
    float
        The dimensionless group.

    Raises
    ------
    ValueError
        If any of a_eff, b_eff, eps_r, doping is not finite or is not strictly positive.
    '''
    return 0.0  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_ionization_dimensionless_group(a_eff: float, b_eff: float, eps_r: float, doping: float) -> float:
    for _v in (a_eff, b_eff, eps_r, doping):
        if not np.isfinite(_v) or _v <= 0.0:
            raise ValueError('a_eff, b_eff, eps_r, doping must be finite and strictly positive')
    q = 1.602e-19; eps0 = 8.854e-14
    return float(a_eff*b_eff*eps_r*eps0/(q*doping))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": "import numpy as np",
            "call": "ionization_dimensionless_group(1.920937271e8, 2.61e7, 9.7, 4.4e15)",
            "gold_call": "_oracle_ionization_dimensionless_group(1.920937271e8, 2.61e7, 9.7, 4.4e15)",
        },
        {
            "setup": "import numpy as np",
            "call": "ionization_dimensionless_group(1.058300525e6, 1.615e6, 11.7, 1.0e15)",
            "gold_call": "_oracle_ionization_dimensionless_group(1.058300525e6, 1.615e6, 11.7, 1.0e15)",
        },
        {
            "setup": "import numpy as np",
            "call": "ionization_dimensionless_group(1.920937271e8, 2.61e7, 9.7, 1.0e17)",
            "gold_call": "_oracle_ionization_dimensionless_group(1.920937271e8, 2.61e7, 9.7, 1.0e17)",
        },
        {
            "setup": "import numpy as np",
            "call": "ionization_dimensionless_group(3.887158e7, 2.695e7, 8.9, 1.0e16)",
            "gold_call": "_oracle_ionization_dimensionless_group(3.887158e7, 2.695e7, 8.9, 1.0e16)",
        },
        {
            "setup": "import numpy as np",
            "call": "ionization_dimensionless_group(9.486833e4, 1.895e7, 5.7, 1.0e14)",
            "gold_call": "_oracle_ionization_dimensionless_group(9.486833e4, 1.895e7, 5.7, 1.0e14)",
        },
        {
            "setup": "import numpy as np\ndef run_model():\n    try:\n        ionization_dimensionless_group(1.92e8, 2.61e7, 9.7, 0.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_ionization_dimensionless_group(1.92e8, 2.61e7, 9.7, 0.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
