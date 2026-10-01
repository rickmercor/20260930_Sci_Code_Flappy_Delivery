"""
Evaluate the paper's all-order leading-logarithmic contribution of the n-loop ladder to the Compton total cross section at order alpha^{n+1}, i.e. the N^{n-1}LO leading logarithm ln^{2n-1} tau with its closed-form coefficient.

The paper traces the high-energy double logarithms of Compton scattering to the t-channel ladder diagrams in which the exchanged electron momenta are strongly ordered in their light-cone components, with one Glauber-mode electron exchange separating collinear from anti-collinear exchanges; the (2n-1)-fold nested phase-space integration over this region produces exactly the (2n-1)-th power of the logarithm at order alpha^{n+1}, while non-ladder diagrams carry no leading logarithms. The resulting closed-form coefficient (eq. (12) of the paper; not restated here) involves factorials of n that suppress the series strongly, and it reproduces the L, L^3 and L^5 terms of the fixed-order expansion at LO, NLO and NNLO; the paper also verified it against an explicit N^3LO ladder calculation. Consult the paper for the coefficient.

Returns
-------
float — the leading-logarithmic order-alpha^{n+1} ladder contribution to the Compton total cross section, in GeV^-2.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def ll_ladder_term(n: int, tau: float, s: float, alpha: float) -> float:
    '''Leading-logarithmic N^{n-1}LO ladder contribution to the total cross section.

    Parameters
    ----------
    n : int
        Loop order of the ladder, >= 1 (n = 1 is the leading-order
        logarithm, n = 2 the NLO double logarithm, ...).
    tau : float
        m^2/s, with 0 < tau < 1.
    s : float
        Mandelstam s in GeV^2, > 0.
    alpha : float
        Fine-structure constant, > 0.

    Returns
    -------
    sigma_LL_n : float
        The order-alpha^{n+1} leading-logarithmic term as defined by the
        paper, in GeV^-2, as a native Python float.

    Raises
    ------
    ValueError
        If n < 1, tau is outside (0, 1), or s or alpha is not strictly positive.
    '''
    return sigma_LL_n  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
import math

def _oracle_ll_ladder_term(n: int, tau: float, s: float, alpha: float) -> float:
    n = int(n); tau = float(tau); s = float(s); alpha = float(alpha)
    if n < 1 or not (0.0 < tau < 1.0) or not (s > 0.0) or not (alpha > 0.0):
        raise ValueError("require n >= 1, 0 < tau < 1, s > 0, alpha > 0")
    # eq. (12): sigma_LL^{N^{n-1}LO} = -2 alpha^{n+1} ln^{2n-1} tau / ((2 pi)^{n-2} s (n-1)! (n+1)!)
    lt = math.log(tau)
    return float(-2.0 * alpha ** (n + 1) * lt ** (2 * n - 1)
                 / ((2.0 * math.pi) ** (n - 2) * s * math.factorial(n - 1) * math.factorial(n + 1)))

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
               '    return ll_ladder_term(*copy.deepcopy((1, tau, s, a)))\n'
               '\n'
               'def _reference_case():\n'
               '    import numpy as np\n'
               '    import copy\n'
               '    m = 0.00051099895\n'
               '    a = 1.0 / 137.035999084\n'
               '    (s, tau, _) = _oracle_compton_kinematics_lo(*copy.deepcopy((8.0, m, a)))\n'
               '    return _oracle_ll_ladder_term(*copy.deepcopy((1, tau, s, a)))\n',
      'call': '_candidate_case()',
      'gold_call': '_reference_case()',
      'tol': 1e-09},
     {'setup': 'def _candidate_case():\n'
               '    import numpy as np\n'
               '    import copy\n'
               '    m = 0.00051099895\n'
               '    a = 1.0 / 137.035999084\n'
               '    (s, tau, _) = compton_kinematics_lo(*copy.deepcopy((8.0, m, a)))\n'
               '    return ll_ladder_term(*copy.deepcopy((2, tau, s, a)))\n'
               '\n'
               'def _reference_case():\n'
               '    import numpy as np\n'
               '    import copy\n'
               '    m = 0.00051099895\n'
               '    a = 1.0 / 137.035999084\n'
               '    (s, tau, _) = _oracle_compton_kinematics_lo(*copy.deepcopy((8.0, m, a)))\n'
               '    return _oracle_ll_ladder_term(*copy.deepcopy((2, tau, s, a)))\n',
      'call': '_candidate_case()',
      'gold_call': '_reference_case()',
      'tol': 1e-09},
     {'setup': 'def _candidate_case():\n'
               '    import numpy as np\n'
               '    import copy\n'
               '    m = 0.00051099895\n'
               '    a = 1.0 / 137.035999084\n'
               '    (s, tau, _) = compton_kinematics_lo(*copy.deepcopy((8.0, m, a)))\n'
               '    return ll_ladder_term(*copy.deepcopy((3, tau, s, a)))\n'
               '\n'
               'def _reference_case():\n'
               '    import numpy as np\n'
               '    import copy\n'
               '    m = 0.00051099895\n'
               '    a = 1.0 / 137.035999084\n'
               '    (s, tau, _) = _oracle_compton_kinematics_lo(*copy.deepcopy((8.0, m, a)))\n'
               '    return _oracle_ll_ladder_term(*copy.deepcopy((3, tau, s, a)))\n',
      'call': '_candidate_case()',
      'gold_call': '_reference_case()',
      'tol': 1e-09},
     {'setup': 'def _candidate_case():\n'
               '    import numpy as np\n'
               '    import copy\n'
               '    m = 0.00051099895\n'
               '    a = 1.0 / 137.035999084\n'
               '    (s, tau, _) = compton_kinematics_lo(*copy.deepcopy((8.0, m, a)))\n'
               '    return ll_ladder_term(*copy.deepcopy((4, tau, s, a)))\n'
               '\n'
               'def _reference_case():\n'
               '    import numpy as np\n'
               '    import copy\n'
               '    m = 0.00051099895\n'
               '    a = 1.0 / 137.035999084\n'
               '    (s, tau, _) = _oracle_compton_kinematics_lo(*copy.deepcopy((8.0, m, a)))\n'
               '    return _oracle_ll_ladder_term(*copy.deepcopy((4, tau, s, a)))\n',
      'call': '_candidate_case()',
      'gold_call': '_reference_case()',
      'tol': 1e-09},
     {'setup': 'def _candidate_case():\n'
               '    import numpy as np\n'
               '    import copy\n'
               '    m = 0.00051099895\n'
               '    a = 1.0 / 137.035999084\n'
               '    (s, tau, _) = compton_kinematics_lo(*copy.deepcopy((8.0, m, a)))\n'
               '    return ll_ladder_term(*copy.deepcopy((5, tau, s, a)))\n'
               '\n'
               'def _reference_case():\n'
               '    import numpy as np\n'
               '    import copy\n'
               '    m = 0.00051099895\n'
               '    a = 1.0 / 137.035999084\n'
               '    (s, tau, _) = _oracle_compton_kinematics_lo(*copy.deepcopy((8.0, m, a)))\n'
               '    return _oracle_ll_ladder_term(*copy.deepcopy((5, tau, s, a)))\n',
      'call': '_candidate_case()',
      'gold_call': '_reference_case()',
      'tol': 1e-09},
     {'setup': 'def _candidate_case():\n'
               '    import numpy as np\n'
               '    import copy\n'
               '    tau = 1e-05\n'
               '    s = 3.0\n'
               '    a = 1.0 / 137.035999084\n'
               '\n'
               '    def run_model():\n'
               '        return float(abs(ll_ladder_term(*copy.deepcopy((1, tau, s, a))) / (2.0 * np.pi * a ** 2 '
               '/ s * np.log(1.0 / tau)) - 1.0) < 1e-12)\n'
               '    return run_model()\n'
               '\n'
               'def _reference_case():\n'
               '    import numpy as np\n'
               '    import copy\n'
               '    tau = 1e-05\n'
               '    s = 3.0\n'
               '    a = 1.0 / 137.035999084\n'
               '\n'
               '    def run_gold():\n'
               '        return float(abs(_oracle_ll_ladder_term(*copy.deepcopy((1, tau, s, a))) / (2.0 * np.pi * '
               'a ** 2 / s * np.log(1.0 / tau)) - 1.0) < 1e-12)\n'
               '    return run_gold()\n',
      'call': '_candidate_case()',
      'gold_call': '_reference_case()'},
     {'setup': 'def _candidate_case():\n'
               '    import numpy as np\n'
               '    import copy\n'
               '    m = 0.00051099895\n'
               '    a = 1.0 / 137.035999084\n'
               '    (s, tau, _) = compton_kinematics_lo(*copy.deepcopy(((1.0 - m * m) / (2.0 * m), m, a)))\n'
               '    return ll_ladder_term(*copy.deepcopy((2, tau, s, a)))\n'
               '\n'
               'def _reference_case():\n'
               '    import numpy as np\n'
               '    import copy\n'
               '    m = 0.00051099895\n'
               '    a = 1.0 / 137.035999084\n'
               '    (s, tau, _) = _oracle_compton_kinematics_lo(*copy.deepcopy(((1.0 - m * m) / (2.0 * m), m, '
               'a)))\n'
               '    return _oracle_ll_ladder_term(*copy.deepcopy((2, tau, s, a)))\n',
      'call': '_candidate_case()',
      'gold_call': '_reference_case()',
      'tol': 1e-09},
     {'setup': 'def _candidate_case():\n'
               '    import numpy as np\n'
               '    import copy\n'
               '    m = 0.00051099895\n'
               '    a = 1.0 / 137.035999084\n'
               '    (s, tau, _) = compton_kinematics_lo(*copy.deepcopy(((1.0 - m * m) / (2.0 * m), m, a)))\n'
               '    return ll_ladder_term(*copy.deepcopy((3, tau, s, a)))\n'
               '\n'
               'def _reference_case():\n'
               '    import numpy as np\n'
               '    import copy\n'
               '    m = 0.00051099895\n'
               '    a = 1.0 / 137.035999084\n'
               '    (s, tau, _) = _oracle_compton_kinematics_lo(*copy.deepcopy(((1.0 - m * m) / (2.0 * m), m, '
               'a)))\n'
               '    return _oracle_ll_ladder_term(*copy.deepcopy((3, tau, s, a)))\n',
      'call': '_candidate_case()',
      'gold_call': '_reference_case()',
      'tol': 1e-09},
     {'setup': 'import numpy as np\nimport copy\n',
      'call': 'll_ladder_term(*copy.deepcopy((8, 0.2, 1.5, 0.1)))',
      'gold_call': '_oracle_ll_ladder_term(*copy.deepcopy((8, 0.2, 1.5, 0.1)))',
      'tol': 1e-09},
     {'setup': 'def _candidate_case():\n'
               '    import numpy as np\n'
               '    import copy\n'
               '\n'
               '    def run_model():\n'
               '        try:\n'
               '            ll_ladder_term(*copy.deepcopy((0, 0.0001, 1.0, 1.0 / 137.035999084)))\n'
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
               '            _oracle_ll_ladder_term(*copy.deepcopy((0, 0.0001, 1.0, 1.0 / 137.035999084)))\n'
               '            return 0\n'
               '        except ValueError:\n'
               '            return 1\n'
               '        except Exception:\n'
               '            return 2\n'
               '    return run_gold()\n',
      'call': '_candidate_case()',
      'gold_call': '_reference_case()'}]
