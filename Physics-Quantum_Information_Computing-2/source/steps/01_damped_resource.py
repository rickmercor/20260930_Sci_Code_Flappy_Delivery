"""
Construct the amplitude-damped Bell resource.

Apply the two local channels defined in the global background to Phi+. The tensor order is Alice, Bob. This state-preparation step uses standard Kraus evolution.

Returns
-------
return result
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def damped_resource(a: float, b: float) -> np.ndarray:
    """Construct the amplitude-damped Bell resource.

    a, b : float
        Alice and Bob damping probabilities in [0,1].
    Returns
    -------
    ndarray, shape (4,4), float
        Resource density matrix in the ordered basis 00,01,10,11.

    All quantities are dimensionless. Inputs are preserved.
    Raises
    ------
    ValueError
        For invalid input shapes or data outside the stated domain.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.integrate import quad
from scipy.special import betaln
from scipy.optimize import least_squares, linprog

def _oracle_damped_resource(a: float, b: float) -> np.ndarray:
    if not np.isfinite([a, b]).all() or not (0 <= a <= 1 and 0 <= b <= 1):
        raise ValueError('Damping parameters must lie in [0,1].')
    c = np.sqrt((1 - a) * (1 - b))
    return np.array([[1 + a * b, 0, 0, c], [0, a * (1 - b), 0, 0], [0, 0, (1 - a) * b, 0], [c, 0, 0, (1 - a) * (1 - b)]]) / 2

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np',
  'call': 'damped_resource(0, 0)',
  'gold_call': '_oracle_damped_resource(0, 0)',
  'tol': 1e-08},
 {'setup': 'import numpy as np',
  'call': 'damped_resource(0, 1)',
  'gold_call': '_oracle_damped_resource(0, 1)',
  'tol': 1e-08},
 {'setup': 'import numpy as np',
  'call': 'damped_resource(1, 0)',
  'gold_call': '_oracle_damped_resource(1, 0)',
  'tol': 1e-08},
 {'setup': 'import numpy as np',
  'call': 'damped_resource(1, 1)',
  'gold_call': '_oracle_damped_resource(1, 1)',
  'tol': 1e-08},
 {'setup': 'import numpy as np',
  'call': 'damped_resource(0.23, 0.74)',
  'gold_call': '_oracle_damped_resource(0.23, 0.74)',
  'tol': 1e-08},
 {'setup': 'import numpy as np',
  'call': 'damped_resource(0.74, 0.23)',
  'gold_call': '_oracle_damped_resource(0.74, 0.23)',
  'tol': 1e-08},
 {'setup': 'import numpy as np',
  'call': 'damped_resource(0.5, 0.5)',
  'gold_call': '_oracle_damped_resource(0.5, 0.5)',
  'tol': 1e-08},
 {'setup': 'import numpy as np',
  'call': 'damped_resource(1e-08, 0.99999999)',
  'gold_call': '_oracle_damped_resource(1e-08, 0.99999999)',
  'tol': 1e-08},
 {'setup': 'import numpy as np\n'
           'def _exception_code(fn):\n'
           '    try:\n'
           '        fn()\n'
           '    except ValueError:\n'
           '        return 1.0\n'
           '    except Exception:\n'
           '        return 2.0\n'
           '    return 0.0',
  'call': '_exception_code(lambda: damped_resource(-.1, .2))',
  'gold_call': '_exception_code(lambda: _oracle_damped_resource(-.1, .2))',
  'tol': 0.0}]
