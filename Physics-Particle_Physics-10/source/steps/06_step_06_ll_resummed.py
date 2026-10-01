"""
Evaluate the paper's all-order leading-logarithmic Compton total cross section and its remainder beyond a supplied fixed order.

Use the resummation selected in the problem statement. The integer n_max specifies the number of low ladder orders already included in the fixed-order prediction, with n_max = 3 for NNLO. The second returned value is the part of the resummation beyond those orders. Both returned quantities are in GeV^-2.

Returns
-------
tuple (float, float) — the resummed leading-logarithmic cross section and its part beyond the first n_max ladder orders, in GeV^-2.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def ll_resummed(tau: float, s: float, alpha: float, n_max: int) -> tuple:
    '''Resummed leading logarithms and their remainder beyond n_max ladder orders.

    Parameters
    ----------
    tau : float
        m^2/s, with 0 < tau < 1.
    s : float
        Mandelstam s in GeV^2, > 0.
    alpha : float
        Fine-structure constant, > 0.
    n_max : int
        Number of leading ladder orders already contained in the fixed-order
        result (n_max = 3 for an NNLO calculation), >= 0.

    Returns
    -------
    result : tuple of (float, float)
        (sigma_LL, remainder): the paper's closed-form resummed
        leading-logarithmic cross section and sigma_LL minus the sum of the
        ladder terms n = 1..n_max (uses ll_ladder_term), both in GeV^-2 as
        native Python floats.

    Raises
    ------
    ValueError
        If tau is outside (0, 1), s or alpha is not strictly positive, or
        n_max < 0.
    '''
    return sigma_LL, remainder  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
import math
from scipy.special import iv

def _oracle_ll_resummed(tau: float, s: float, alpha: float, n_max: int) -> tuple:
    tau = float(tau); s = float(s); alpha = float(alpha); n_max = int(n_max)
    if not (0.0 < tau < 1.0) or not (s > 0.0) or not (alpha > 0.0) or n_max < 0:
        raise ValueError("require 0 < tau < 1, s > 0, alpha > 0, n_max >= 0")
    lt = math.log(tau)
    # eq. (13): sigma_LL = -(8 pi^2 alpha / (s ln tau)) I_2( sqrt(2 alpha/pi) ln tau )
    sigma_ll = -8.0 * math.pi ** 2 * alpha / (s * lt) * float(iv(2, math.sqrt(2.0 * alpha / math.pi) * lt))
    fixed = sum(_oracle_ll_ladder_term(n, tau, s, alpha) for n in range(1, n_max + 1))
    return float(sigma_ll), float(sigma_ll - fixed)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return test specifications with independent model/oracle dependency chains."""
    return [{'setup': 'def _candidate_case():\n'
               '    import numpy as np\n'
               '    import copy\n'
               '    m = 0.00051099895\n'
               '    a = 1.0 / 137.035999084\n'
               '    (s, tau, _) = compton_kinematics_lo(*copy.deepcopy((8.0, m, a)))\n'
               '    return np.array(ll_resummed(*copy.deepcopy((tau, s, a, 3))))\n'
               '\n'
               'def _reference_case():\n'
               '    import numpy as np\n'
               '    import copy\n'
               '    m = 0.00051099895\n'
               '    a = 1.0 / 137.035999084\n'
               '    (s, tau, _) = _oracle_compton_kinematics_lo(*copy.deepcopy((8.0, m, a)))\n'
               '    return np.array(_oracle_ll_resummed(*copy.deepcopy((tau, s, a, 3))))\n',
      'call': '_candidate_case()',
      'gold_call': '_reference_case()',
      'tol': 1e-09},
     {'setup': 'import numpy as np\nimport copy\n',
      'call': 'np.array(ll_resummed(*copy.deepcopy((1.0e-5, 3.0, 1.0 / 137.035999084, 0))))',
      'gold_call': 'np.array(_oracle_ll_resummed(*copy.deepcopy((1.0e-5, 3.0, 1.0 / 137.035999084, 0))))',
      'tol': 1e-09},
     {'setup': 'def _candidate_case():\n'
               '    import numpy as np\n'
               '    import copy\n'
               '    m = 0.00051099895\n'
               '    a = 1.0 / 137.035999084\n'
               '    (s, tau, _) = compton_kinematics_lo(*copy.deepcopy(((1.0 - m * m) / (2.0 * m), m, a)))\n'
               '    series = sum((ll_ladder_term(n, tau, s, a) for n in range(1, 13)))\n'
               '\n'
               '    def run_model():\n'
               '        return float(abs(ll_resummed(*copy.deepcopy((tau, s, a, 0)))[0] / series - 1.0) < '
               '1e-12)\n'
               '    return run_model()\n'
               '\n'
               'def _reference_case():\n'
               '    import numpy as np\n'
               '    import copy\n'
               '    m = 0.00051099895\n'
               '    a = 1.0 / 137.035999084\n'
               '    (s, tau, _) = _oracle_compton_kinematics_lo(*copy.deepcopy(((1.0 - m * m) / (2.0 * m), m, '
               'a)))\n'
               '    series = sum((_oracle_ll_ladder_term(n, tau, s, a) for n in range(1, 13)))\n'
               '\n'
               '    def run_gold():\n'
               '        return float(abs(_oracle_ll_resummed(*copy.deepcopy((tau, s, a, 0)))[0] / series - 1.0) '
               '< 1e-12)\n'
               '    return run_gold()\n',
      'call': '_candidate_case()',
      'gold_call': '_reference_case()'},
     {'setup': 'import numpy as np\nimport copy\n',
      'call': 'np.array(ll_resummed(*copy.deepcopy((0.05, 1.0, 0.3, 2))))',
      'gold_call': 'np.array(_oracle_ll_resummed(*copy.deepcopy((0.05, 1.0, 0.3, 2))))',
      'tol': 1e-09},
     {'setup': 'def _candidate_case():\n'
               '    import numpy as np\n'
               '    import copy\n'
               '\n'
               '    def run_model():\n'
               '        try:\n'
               '            ll_resummed(*copy.deepcopy((1.0, 1.0, 1.0 / 137.035999084, 3)))\n'
               '            return 0\n'
               '        except ValueError:\n'
               '            return 1\n'
               '        except Exception:\n'
               '            return 2\n'
               '    return run_model()\n'
               '\n'
               'def _reference_case():\n'
               '    import numpy as np\n'
               '    import copy\n'
               '\n'
               '    def run_gold():\n'
               '        try:\n'
               '            _oracle_ll_resummed(*copy.deepcopy((1.0, 1.0, 1.0 / 137.035999084, 3)))\n'
               '            return 0\n'
               '        except ValueError:\n'
               '            return 1\n'
               '        except Exception:\n'
               '            return 2\n'
               '    return run_gold()\n',
      'call': '_candidate_case()',
      'gold_call': '_reference_case()'}]
