"""
Return the pressure of one phase from its density and specific internal energy under the stiffened-gas equation of state the source uses, given that phase's two constants.

A stiffened-gas closure extends the ideal-gas law with a constant that represents the molecular attraction of a liquid, which is what lets a single algebraic form cover both a gas and a nearly incompressible liquid.

Returns
-------
A float64 array shaped like rho holding the phasic pressure.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def stiffened_gas_pressure(rho: "np.ndarray", e: "np.ndarray", gamma: float, pi_: float) -> "np.ndarray":
    """Return the pressure of one phase from its density and specific internal energy under
    the stiffened-gas equation of state the source uses, given that phase's two constants.

    Args:
        rho: Phasic density, any array shape.
        e: Phasic specific internal energy, shaped like rho.
        gamma: Ratio of specific heats of the phase, greater than one.
        pi_: Stiffening constant of the phase, non-negative.

    Returns:
        A float64 array shaped like rho holding the phasic pressure.

    Raises:
        ValueError: If gamma is not greater than one, if pi_ is negative, or if any density
            is not positive.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_stiffened_gas_pressure(rho: "np.ndarray", e: "np.ndarray", gamma: float, pi_: float) -> "np.ndarray":
    rho = np.asarray(rho, dtype=np.float64); e = np.asarray(e, dtype=np.float64)
    if gamma <= 1.0 or pi_ < 0.0:
        raise ValueError("need gamma > 1 and pi >= 0")
    if np.any(rho <= 0.0):
        raise ValueError("density must be positive")
    return (gamma - 1.0) * rho * e - gamma * pi_

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\nrho = np.array([1.0e3, 8.0e2])\ne = np.array([1.0e6, 2.0e6])\ngamma = 4.4\npi_ = 6.0e8\n',
         'call': 'stiffened_gas_pressure(rho, e, gamma, pi_)',
         'gold_call': '_oracle_stiffened_gas_pressure(rho, e, gamma, pi_)', 'tol': 1e-6},
        {'setup': 'import numpy as np\nrho = np.array([1.2, 0.8, 2.5])\ne = np.array([2.0e5, 3.0e5, 1.0e5])\ngamma = 1.4\npi_ = 0.0\n',
         'call': 'stiffened_gas_pressure(rho, e, gamma, pi_)',
         'gold_call': '_oracle_stiffened_gas_pressure(rho, e, gamma, pi_)', 'tol': 1e-8},
        {'setup': 'import numpy as np\nrho = np.array([[1.0e3]])\ne = np.array([[0.0]])\ngamma = 4.4\npi_ = 6.0e8\n',
         'call': 'stiffened_gas_pressure(rho, e, gamma, pi_)',
         'gold_call': '_oracle_stiffened_gas_pressure(rho, e, gamma, pi_)', 'tol': 1e-6},
        {'setup': 'import numpy as np\n# invalid input: a ratio of specific heats of one is not greater than one and must raise ValueError\nrho = np.array([1.2])\ne = np.array([2.0e5])\ngamma = 1.0\npi_ = 0.0\ndef _catches_value_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return np.array([1.0])\n    return np.array([0.0])\n',
         'call': '_catches_value_error(lambda: stiffened_gas_pressure(rho, e, gamma, pi_))',
         'gold_call': '_catches_value_error(lambda: _oracle_stiffened_gas_pressure(rho, e, gamma, pi_))'},
    ]
