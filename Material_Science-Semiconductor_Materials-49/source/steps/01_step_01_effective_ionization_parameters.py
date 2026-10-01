"""
Collapse the separate electron and hole ionization coefficients into one effective pair.

Avalanche multiplication in a one sided abrupt junction is driven by two carrier species whose ionization rates follow Chynoweth exponentials with their own prefactor and field scale. Carrying both species through the breakdown integral leaves an expression that cannot be reduced in closed form. A single effective coefficient of the same Chynoweth shape removes that obstruction, and the way the two species are combined fixes every number computed afterwards. Prefactors are in reciprocal centimetres and field scales in volts per centimetre.

Returns
-------
numpy.ndarray of length two, the effective prefactor in cm^-1 and field scale in V/cm
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def effective_ionization_parameters(a_n: float, b_n: float, a_p: float, b_p: float) -> np.ndarray:
    '''Effective Chynoweth prefactor and field scale for the two carrier species.

    Parameters
    ----------
    a_n, a_p : float
        Electron and hole ionization prefactors in cm^-1.
    b_n, b_p : float
        Electron and hole ionization field scales in V/cm.

    Returns
    -------
    numpy.ndarray
        Two element array [a_eff, b_eff]; a_eff in cm^-1, b_eff in V/cm.

    Raises
    ------
    ValueError
        If any of a_n, b_n, a_p, b_p is not finite or is not strictly positive.
    '''
    return [0.0, 0.0]  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_effective_ionization_parameters(a_n: float, b_n: float, a_p: float, b_p: float) -> np.ndarray:
    for _v in (a_n, b_n, a_p, b_p):
        if not np.isfinite(_v) or _v <= 0.0:
            raise ValueError('a_n, b_n, a_p, b_p must be finite and strictly positive')
    return np.array([np.sqrt(a_n*a_p), 0.5*(b_n + b_p)], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": "import numpy as np",
            "call": "effective_ionization_parameters(8.2e9, 3.94e7, 4.5e6, 1.28e7)",
            "gold_call": "_oracle_effective_ionization_parameters(8.2e9, 3.94e7, 4.5e6, 1.28e7)",
        },
        {
            "setup": "import numpy as np",
            "call": "effective_ionization_parameters(7.0e5, 1.23e6, 1.6e6, 2.00e6)",
            "gold_call": "_oracle_effective_ionization_parameters(7.0e5, 1.23e6, 1.6e6, 2.00e6)",
        },
        {
            "setup": "import numpy as np",
            "call": "effective_ionization_parameters(2.8e8, 3.43e7, 5.4e6, 1.96e7)",
            "gold_call": "_oracle_effective_ionization_parameters(2.8e8, 3.43e7, 5.4e6, 1.96e7)",
        },
        {
            "setup": "import numpy as np",
            "call": "effective_ionization_parameters(7.1e5, 2.10e7, 7.1e5, 2.10e7)",
            "gold_call": "_oracle_effective_ionization_parameters(7.1e5, 2.10e7, 7.1e5, 2.10e7)",
        },
        {
            "setup": "import numpy as np",
            "call": "effective_ionization_parameters(1.5e5, 2.40e7, 6.0e4, 1.39e7)",
            "gold_call": "_oracle_effective_ionization_parameters(1.5e5, 2.40e7, 6.0e4, 1.39e7)",
        },
        {
            "setup": "import numpy as np\ndef run_model():\n    try:\n        effective_ionization_parameters(-8.2e9, 3.94e7, 4.5e6, 1.28e7)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_effective_ionization_parameters(-8.2e9, 3.94e7, 4.5e6, 1.28e7)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
