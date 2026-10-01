"""
Evaluate the source's degradation function, Eq. (6), together with its derivative with respect to the phase-field parameter and the positive source coefficient that the phase-field balance, Eq. (22), carries. Note carefully which quantity this function degrades in this formulation, because it is not the quantity a standard phase-field model degrades, and the small floor parameter it carries is named accordingly.

In this formulation the phase-field parameter records the loss of cohesive capacity rather than the loss of stiffness, so the degradation function multiplies a different term of the energy than in a Griffith-type phase-field model. The floor parameter keeps the degraded quantity's derivative well defined once the point is fully broken. The phase-field balance of Eq. (22) inherits the same floor through the derivative.

Returns
-------
A (3,) float64 array [d, dd/dphi, source coefficient], where the source coefficient is the phase-field-independent positive factor multiplying the crack driving force in the source term of Eq. (22).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def degradation_state(phi: float, kappa: float) -> "np.ndarray":
    """Evaluate the source's degradation function, Eq. (6), together with its derivative with
    respect to the phase-field parameter and the positive source coefficient that the phase-
    field balance, Eq. (22), carries.

    Args:
        phi: Phase-field parameter, in [0, 1].
        kappa: The small floor parameter carried by the degradation function, in (0, 1).

    Returns:
        A (3,) float64 array [d, dd/dphi, source coefficient], where the source coefficient
        is the phase-field-independent positive factor multiplying the crack driving force in
        the source term of Eq. (22).

    Raises:
        ValueError: If phi is not a finite number in [0, 1] or kappa is not a finite number
            in (0, 1).
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_degradation_state(phi: float, kappa: float) -> "np.ndarray":
    if not np.isfinite(phi) or phi < 0.0 or phi > 1.0:
        raise ValueError("phi must lie in [0, 1]")
    if not np.isfinite(kappa) or kappa <= 0.0 or kappa >= 1.0:
        raise ValueError("kappa must lie in (0, 1)")
    d = (1.0 - kappa) * (1.0 - phi) ** 2 + kappa
    dd = -2.0 * (1.0 - kappa) * (1.0 - phi)
    return np.array([float(d), float(dd), float(2.0 * (1.0 - kappa))])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\nphi = 0.0\nkappa = 1.0e-3\n',
         'call': 'degradation_state(phi, kappa)',
         'gold_call': '_oracle_degradation_state(phi, kappa)'},
        {'setup': 'import numpy as np\nphi = 0.37\nkappa = 1.0e-3\n',
         'call': 'degradation_state(phi, kappa)',
         'gold_call': '_oracle_degradation_state(phi, kappa)'},
        {'setup': 'import numpy as np\nphi = 0.91\nkappa = 5.0e-2\n',
         'call': 'degradation_state(phi, kappa)',
         'gold_call': '_oracle_degradation_state(phi, kappa)'},
        {'setup': 'import numpy as np\n# invalid input: a phase-field parameter above one lies outside [0, 1] and must raise ValueError\nphi = 1.2\nkappa = 1.0e-3\ndef _catches_value_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return np.array([1.0])\n    return np.array([0.0])\n',
         'call': '_catches_value_error(lambda: degradation_state(phi, kappa))',
         'gold_call': '_catches_value_error(lambda: _oracle_degradation_state(phi, kappa))'},
    ]
