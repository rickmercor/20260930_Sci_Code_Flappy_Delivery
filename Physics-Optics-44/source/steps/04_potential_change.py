"""
Return the change in the effective potential of the reduced description between two widths, that is its value at the second width minus its value at the first. The potential is the function whose negative derivative with respect to the width is the predicted width acceleration, so only differences of it are defined. Raise ValueError if either width, the power or the response length is not strictly positive.

The reduced description behaves like a particle moving in a one-dimensional potential. At fixed medium parameters, the work done by the effective force as the width changes from the first value to the second is the negative of this potential change.

Returns
-------
float: the potential at the second width minus the potential at the first.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def potential_change(a1: float, a2: float, power: float, gamma: float, alpha: float, sigma: float) -> float:
    """Return the change in the effective potential of the reduced description between two widths, that is its value at the second width minus its value at the first. The potential is the function whose negative derivative with respect to the width is the predicted width acceleration, so only differences of it are defined. Raise ValueError if either width, the power or the response length is not strictly positive.

    Returns
    -------
    float: the potential at the second width minus the potential at the first.

    Raises
    ------
    ValueError
        If a1, a2, power or sigma is not strictly positive.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_potential_change(a1: float, a2: float, power: float, gamma: float, alpha: float,
                             sigma: float) -> float:
    """Change in the effective potential between two widths: V(a2) - V(a1), with a'' = -dV/da."""
    a1 = float(a1); a2 = float(a2); power = float(power); sigma = float(sigma)
    if a1 <= 0.0 or a2 <= 0.0 or power <= 0.0 or sigma <= 0.0:
        raise ValueError("a1, a2, power and sigma must be positive")

    def _v(a: float) -> float:
        return (1.0 / (2.0 * a ** 2)
                + alpha ** 2 * a ** 2
                - power * gamma / (np.sqrt(2.0 * np.pi) * a)
                - power / (np.sqrt(np.pi) * np.sqrt(2.0 * a ** 2 + sigma ** 2)))

    return float(_v(a2) - _v(a1))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the step test specifications."""
    return [{'setup': 'import numpy as np',
      'call': 'potential_change(0.4610400, 0.8928089, 12.0, 0.2, 0.15, 4.0)',
      'gold_call': '_oracle_potential_change(0.4610400, 0.8928089, 12.0, 0.2, 0.15, 4.0)',
      'tol': 0.001},
     {'setup': 'import numpy as np',
      'call': 'potential_change(0.5, 1.5, 12.0, 0.2, 0.15, 0.8)',
      'gold_call': '_oracle_potential_change(0.5, 1.5, 12.0, 0.2, 0.15, 0.8)',
      'tol': 0.001},
     {'setup': 'import numpy as np',
      'call': 'potential_change(0.3, 2.0, 12.0, 0.2, 0.15, 2.4)',
      'gold_call': '_oracle_potential_change(0.3, 2.0, 12.0, 0.2, 0.15, 2.4)',
      'tol': 0.001},
     {'setup': 'import numpy as np',
      'call': 'potential_change(0.6, 0.6, 12.0, 0.2, 0.15, 2.0)',
      'gold_call': '_oracle_potential_change(0.6, 0.6, 12.0, 0.2, 0.15, 2.0)'},
     {'setup': 'import numpy as np',
      'call': 'potential_change(0.9, 0.90001, 12.0, 0.2, 0.15, 3.0)',
      'gold_call': '_oracle_potential_change(0.9, 0.90001, 12.0, 0.2, 0.15, 3.0)',
      'tol': 1e-10},
     {'setup': 'import numpy as np',
      'call': 'potential_change(1.6, 1.59999, 12.0, 0.2, 0.15, 0.7)',
      'gold_call': '_oracle_potential_change(1.6, 1.59999, 12.0, 0.2, 0.15, 0.7)',
      'tol': 1e-10},
     {'setup': 'import numpy as np\n'
               'def probe_public():\n'
               '    try:\n'
               '        potential_change(0.0, 1.0, 12.0, 0.2, 0.15, 4.0)\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '    return 0\n'
               'def probe_gold():\n'
               '    try:\n'
               '        _oracle_potential_change(0.0, 1.0, 12.0, 0.2, 0.15, 4.0)\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '    return 0\n',
      'call': 'probe_public()',
      'gold_call': 'probe_gold()'}]
