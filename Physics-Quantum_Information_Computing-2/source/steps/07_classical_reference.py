"""
Compute the classical reference for either record assessment.

Use computational-basis measurement followed by preparation of the observed basis state. Mode 0 records the classical outcome; mode 1 averages the prepared output states for each input. Return the corresponding importance expectations and CDF.

Returns
-------
return result
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def classical_reference(priors: np.ndarray, thresholds: np.ndarray, mode: int = 0) -> np.ndarray:
    """Compute the classical reference for either record assessment.

    priors : ndarray, shape (K,2)
        Finite beta shapes [alpha,beta], each >=1; K may be zero.
    thresholds : ndarray, shape (T,)
        Finite CDF thresholds, including values outside [0,1]; T may be zero.
    mode : int
        0 for the recorded classical outcome R; 1 for erased outcome E.
    Returns
    -------
    ndarray, shape (K+T,), float
        First K entries are classical E[W_k(F)]; last T entries are
        classical Pr(F<=thresholds[i]), each in supplied order.

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

def _oracle_classical_reference(priors: np.ndarray, thresholds: np.ndarray, mode: int=0) -> np.ndarray:
    if mode not in (0, 1):
        raise ValueError('Mode must be 0 (recorded) or 1 (erased).')
    priors = np.asarray(priors, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)
    if priors.ndim != 2 or priors.shape[1] != 2 or (not np.isfinite(priors).all()) or np.any(priors < 1) or (thresholds.ndim != 1) or (not np.isfinite(thresholds).all()):
        raise ValueError('Expected beta shapes >= 1 and finite thresholds.')
    if mode == 0:
        return np.r_[2 * priors[:, 0] / priors.sum(axis=1), np.clip(thresholds, 0, 1) ** 2]
    expectations = []
    for (a, b) in priors:

        def _integrand(z):
            f = (1 + z * z) / 2
            return np.exp(-betaln(a, b)) * f ** (a - 1) * (1 - f) ** (b - 1)
        expectations.append(quad(_integrand, 0, 1, epsabs=2e-11, epsrel=2e-11)[0])
    return np.r_[expectations, np.sqrt(np.clip(2 * thresholds - 1, 0, 1))]

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\n'
               'priors=np.array([[1, 1]], dtype=float).reshape((-1,2))\n'
               'thresholds=np.array([0.0, 0.5, 1.0], dtype=float)',
      'call': 'classical_reference(priors, thresholds)',
      'gold_call': '_oracle_classical_reference(priors, thresholds)',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               'priors=np.array([[2, 1], [5, 1]], dtype=float).reshape((-1,2))\n'
               'thresholds=np.array([-1, 0, 1, 2], dtype=float)',
      'call': 'classical_reference(priors, thresholds)',
      'gold_call': '_oracle_classical_reference(priors, thresholds)',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               'priors=np.array([[1, 2], [1, 5]], dtype=float).reshape((-1,2))\n'
               'thresholds=np.array([0.1, 0.9], dtype=float)',
      'call': 'classical_reference(priors, thresholds)',
      'gold_call': '_oracle_classical_reference(priors, thresholds)',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               'priors=np.array([[2, 2], [7, 7]], dtype=float).reshape((-1,2))\n'
               'thresholds=np.array([0.7, 0.2, 0.7], dtype=float)',
      'call': 'classical_reference(priors, thresholds)',
      'gold_call': '_oracle_classical_reference(priors, thresholds)',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               'priors=np.array([[1.2, 3.4]], dtype=float).reshape((-1,2))\n'
               'thresholds=np.array([0.45], dtype=float)',
      'call': 'classical_reference(priors, thresholds)',
      'gold_call': '_oracle_classical_reference(priors, thresholds)',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               'priors=np.array([[12.0, 2.5]], dtype=float).reshape((-1,2))\n'
               'thresholds=np.array([], dtype=float)',
      'call': 'classical_reference(priors, thresholds)',
      'gold_call': '_oracle_classical_reference(priors, thresholds)',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               'priors=np.array([], dtype=float).reshape((-1,2))\n'
               'thresholds=np.array([-0.1, 0.0, 0.5, 1.0, 1.1], dtype=float)',
      'call': 'classical_reference(priors, thresholds)',
      'gold_call': '_oracle_classical_reference(priors, thresholds)',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               'priors=np.array([[30, 1], [1, 30]], dtype=float).reshape((-1,2))\n'
               'thresholds=np.array([0.001, 0.999], dtype=float)',
      'call': 'classical_reference(priors, thresholds)',
      'gold_call': '_oracle_classical_reference(priors, thresholds)',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               'priors=np.array([[2, 8], [8, 2]], dtype=float).reshape((-1,2))\n'
               'thresholds=np.array([1.0, 0.8, 0.6, 0.0], dtype=float)',
      'call': 'classical_reference(priors, thresholds)',
      'gold_call': '_oracle_classical_reference(priors, thresholds)',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               'priors=np.array([[1.0001, 1.0], [1.0, 1.0001]], dtype=float).reshape((-1,2))\n'
               'thresholds=np.array([0.333333333333], dtype=float)',
      'call': 'classical_reference(priors, thresholds)',
      'gold_call': '_oracle_classical_reference(priors, thresholds)',
      'tol': 1e-08},
     {'setup': 'import numpy as np',
      'call': 'classical_reference(np.array([[2, 1]], dtype=float), np.array([-.1,.49,.5,.75,1.,1.1]), 1)',
      'gold_call': '_oracle_classical_reference(np.array([[2, 1]], dtype=float), '
                   'np.array([-.1,.49,.5,.75,1.,1.1]), 1)',
      'tol': 2e-07},
     {'setup': 'import numpy as np',
      'call': 'classical_reference(np.array([[3, 1], [8, 1]], dtype=float), '
              'np.array([-.1,.49,.5,.75,1.,1.1]), 1)',
      'gold_call': '_oracle_classical_reference(np.array([[3, 1], [8, 1]], dtype=float), '
                   'np.array([-.1,.49,.5,.75,1.,1.1]), 1)',
      'tol': 2e-07},
     {'setup': 'import numpy as np',
      'call': 'classical_reference(np.array([[1, 2], [2, 1]], dtype=float), '
              'np.array([-.1,.49,.5,.75,1.,1.1]), 1)',
      'gold_call': '_oracle_classical_reference(np.array([[1, 2], [2, 1]], dtype=float), '
                   'np.array([-.1,.49,.5,.75,1.,1.1]), 1)',
      'tol': 2e-07},
     {'setup': 'import numpy as np',
      'call': 'classical_reference(np.array([[2, 2], [4, 4]], dtype=float), '
              'np.array([-.1,.49,.5,.75,1.,1.1]), 1)',
      'gold_call': '_oracle_classical_reference(np.array([[2, 2], [4, 4]], dtype=float), '
                   'np.array([-.1,.49,.5,.75,1.,1.1]), 1)',
      'tol': 2e-07},
     {'setup': 'import numpy as np',
      'call': 'classical_reference(np.array([[12.0, 2.5]], dtype=float), np.array([-.1,.49,.5,.75,1.,1.1]), '
              '1)',
      'gold_call': '_oracle_classical_reference(np.array([[12.0, 2.5]], dtype=float), '
                   'np.array([-.1,.49,.5,.75,1.,1.1]), 1)',
      'tol': 2e-07},
     {'setup': 'import numpy as np',
      'call': 'classical_reference(np.array([[1, 12], [12, 1]], dtype=float), '
              'np.array([-.1,.49,.5,.75,1.,1.1]), 1)',
      'gold_call': '_oracle_classical_reference(np.array([[1, 12], [12, 1]], dtype=float), '
                   'np.array([-.1,.49,.5,.75,1.,1.1]), 1)',
      'tol': 2e-07},
     {'setup': 'import numpy as np\n'
               'def _exception_code(fn):\n'
               '    try:\n'
               '        fn()\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    except Exception:\n'
               '        return 2.0\n'
               '    return 0.0',
      'call': '_exception_code(lambda: classical_reference(np.array([[0.,2.]]), np.array([.5])))',
      'gold_call': '_exception_code(lambda: _oracle_classical_reference(np.array([[0.,2.]]), '
                   'np.array([.5])))',
      'tol': 0.0}]
