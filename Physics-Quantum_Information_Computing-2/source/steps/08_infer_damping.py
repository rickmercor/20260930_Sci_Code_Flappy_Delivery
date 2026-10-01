"""
Infer the ordered reference damping pair from distributional calibration.

This constructed inverse problem uses the paper's first two moments and fidelity CDF. Inputs are consistent with a unique ordered pair in [0.01,0.96]^2, at absolute observation residual <=1e-9. Compose damped_resource, conditional_fidelity_law, fidelity_moments and fidelity_cdf.

Returns
-------
return result
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def infer_damping(observations: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Infer the ordered reference damping pair from distributional calibration.

    observations : ndarray, shape (2+T,)
        [E[F], E[F**2], CDF(thresholds[0]), ..., CDF(thresholds[T-1])].
    thresholds : ndarray, shape (T,)
        Finite thresholds in supplied order. The data satisfy the stated
        unique-pair identifiability condition.
    Returns
    -------
    ndarray, shape (2,), float
        Inferred [p_A,p_B] in physical-link order.
        Inconsistent observations with best residual norm >1e-8 raise ValueError.

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

def _oracle_infer_damping(observations: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    observations = np.asarray(observations, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)
    if thresholds.ndim != 1 or observations.shape != (2 + len(thresholds),) or (not np.isfinite(observations).all()) or (not np.isfinite(thresholds).all()):
        raise ValueError('Expected [E F,E F^2,CDF values] and their thresholds.')

    def _residual(x):
        L = _oracle_conditional_fidelity_law(_oracle_damped_resource(*x))
        return np.r_[_oracle_fidelity_moments(L, [1, 2]), _oracle_fidelity_cdf(L, thresholds)] - observations
    solutions = []
    for a in [0.1, 0.35, 0.65, 0.88]:
        for b in [0.1, 0.35, 0.65, 0.88]:
            fit = least_squares(_residual, [a, b], bounds=([0.01, 0.01], [0.96, 0.96]), xtol=2e-12, ftol=2e-12, gtol=2e-12, max_nfev=250)
            solutions.append((float(np.linalg.norm(fit.fun)), fit.x))
    (error, answer) = min(solutions, key=lambda item: item[0])
    if error > 1e-08:
        raise ValueError('Calibration is inconsistent with the physical model.')
    return answer.copy()

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\n'
               'observations=np.array([0.7231882775473688, 0.5572218954564145, 0.20852676659117841, '
               '0.38807975204329076, 0.6633375335863507], dtype=float)\n'
               'thresholds=np.array([0.55, 0.7, 0.85], dtype=float)',
      'call': 'infer_damping(observations, thresholds)',
      'gold_call': '_oracle_infer_damping(observations, thresholds)',
      'tol': 2e-06},
     {'setup': 'import numpy as np\n'
               'observations=np.array([0.7231882775473688, 0.5246163746252991, 0.0, 0.35444029382514236, '
               '0.9999999999999998], dtype=float)\n'
               'thresholds=np.array([0.55, 0.7, 0.85], dtype=float)',
      'call': 'infer_damping(observations, thresholds)',
      'gold_call': '_oracle_infer_damping(observations, thresholds)',
      'tol': 2e-06},
     {'setup': 'import numpy as np\n'
               'observations=np.array([0.6252795558988649, 0.45802813889173716, 0.152640058775516, '
               '0.43001741309454333, 0.8201101174886166], dtype=float)\n'
               'thresholds=np.array([0.3, 0.6, 0.9], dtype=float)',
      'call': 'infer_damping(observations, thresholds)',
      'gold_call': '_oracle_infer_damping(observations, thresholds)',
      'tol': 2e-06},
     {'setup': 'import numpy as np\n'
               'observations=np.array([0.6252795558988649, 0.39504247463471376, 0.0, 0.7038892266354584, '
               '0.9835463780003222], dtype=float)\n'
               'thresholds=np.array([0.52, 0.65, 0.8], dtype=float)',
      'call': 'infer_damping(observations, thresholds)',
      'gold_call': '_oracle_infer_damping(observations, thresholds)',
      'tol': 2e-06},
     {'setup': 'import numpy as np\n'
               'observations=np.array([0.749933329999333, 0.5692774995555574, 0.07849250407879499, '
               '0.3776887155912776, 0.9999999999999999], dtype=float)\n'
               'thresholds=np.array([0.6, 0.75, 0.85], dtype=float)',
      'call': 'infer_damping(observations, thresholds)',
      'gold_call': '_oracle_infer_damping(observations, thresholds)',
      'tol': 2e-06},
     {'setup': 'import numpy as np\n'
               'observations=np.array([0.8924572710215865, 0.7980367565068479, 0.17170758629017224, '
               '0.9999999999999997, 0.9999999999999997], dtype=float)\n'
               'thresholds=np.array([0.85, 0.93, 0.98], dtype=float)',
      'call': 'infer_damping(observations, thresholds)',
      'gold_call': '_oracle_infer_damping(observations, thresholds)',
      'tol': 2e-06},
     {'setup': 'import numpy as np\n'
               'observations=np.array([0.6728564615223281, 0.49100192876364895, 0.049107657954781773, '
               '0.40478708742856784, 0.8898200250677972], dtype=float)\n'
               'thresholds=np.array([0.3, 0.65, 0.9], dtype=float)',
      'call': 'infer_damping(observations, thresholds)',
      'gold_call': '_oracle_infer_damping(observations, thresholds)',
      'tol': 2e-06},
     {'setup': 'import numpy as np\n'
               'observations=np.array([0.7316264038725353, 0.5601691381686221, 0.7014132758927534, '
               '0.16305444991419757, 0.7014132758927534, 0.35408732210301164], dtype=float)\n'
               'thresholds=np.array([0.85, 0.55, 0.85, 0.7], dtype=float)',
      'call': 'infer_damping(observations, thresholds)',
      'gold_call': '_oracle_infer_damping(observations, thresholds)',
      'tol': 2e-06},
     {'setup': 'import numpy as np\n'
               'def _exception_code(fn):\n'
               '    try:\n'
               '        fn()\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    except Exception:\n'
               '        return 2.0\n'
               '    return 0.0',
      'call': '_exception_code(lambda: infer_damping(np.array([0.]), np.array([.5])))',
      'gold_call': '_exception_code(lambda: _oracle_infer_damping(np.array([0.]), np.array([.5])))',
      'tol': 0.0}]
