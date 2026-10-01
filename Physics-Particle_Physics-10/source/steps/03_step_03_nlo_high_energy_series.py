"""
Evaluate the paper's high-energy next-to-leading-order (order alpha^3) correction to the Compton total cross section, through the tau^3 term.

Use the next-to-leading-order expansion selected in the problem statement for 0 < tau <= 0.01, where tau = m^2/s. Apply the stated on-shell and final-state conventions. This step supplies the next-to-leading-order correction for the matched prediction.

Returns
-------
float — the next-to-leading-order (alpha^3) correction to the Compton total cross section, in GeV^-2.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def nlo_high_energy_series(tau: float, s: float, alpha: float) -> float:
    '''High-energy expansion of the order-alpha^3 correction.

    Parameters
    ----------
    tau : float
        m^2/s, with 0 < tau <= 0.01.
    s : float
        Mandelstam s in GeV^2, > 0.
    alpha : float
        Fine-structure constant, > 0.

    Returns
    -------
    sigma_NLO : float
        The order-alpha^3 correction to the total cross section from the
        paper's expansion (through tau^3), in GeV^-2, as a native Python
        float.

    Raises
    ------
    ValueError
        If tau is outside (0, 0.01], or if s or alpha is not strictly positive.
    '''
    return sigma_NLO  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
import math

def _oracle_nlo_high_energy_series(tau: float, s: float, alpha: float) -> float:
    tau = float(tau); s = float(s); alpha = float(alpha)
    if not (0.0 < tau <= 0.01) or not (s > 0.0) or not (alpha > 0.0):
        raise ValueError("require 0 < tau <= 0.01, s > 0, alpha > 0")
    L = math.log(1.0 / tau)
    # eq. (7), next-to-leading-order line
    poly = (L ** 3 / 3.0 - L ** 2 / 2.0 + 17.0 * L / 4.0 - 9.5016) \
        + tau * (2.0 * L ** 3 + 13.0 * L ** 2 - 36.5340 * L + 0.55139) \
        + tau ** 2 * (38.0 * L ** 3 / 3.0 + 151.0 * L ** 2 / 2.0 - 89.091 * L + 47.062) \
        + tau ** 3 * (157.0 * L ** 3 / 3.0 + 344.0 * L ** 2 / 3.0 - 220.07 * L + 149.73)
    return float(alpha ** 3 / s * poly)

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
               '    return nlo_high_energy_series(*copy.deepcopy((tau, s, a)))\n'
               '\n'
               'def _reference_case():\n'
               '    import numpy as np\n'
               '    import copy\n'
               '    m = 0.00051099895\n'
               '    a = 1.0 / 137.035999084\n'
               '    (s, tau, _) = _oracle_compton_kinematics_lo(*copy.deepcopy((8.0, m, a)))\n'
               '    return _oracle_nlo_high_energy_series(*copy.deepcopy((tau, s, a)))\n',
      'call': '_candidate_case()',
      'gold_call': '_reference_case()',
      'tol': 1e-09},
     {'setup': 'def _candidate_case():\n'
               '    import numpy as np\n'
               '    import copy\n'
               '    m = 0.00051099895\n'
               '    a = 1.0 / 137.035999084\n'
               '    (s, tau, _) = compton_kinematics_lo(*copy.deepcopy(((1.0 - m * m) / (2.0 * m), m, a)))\n'
               '    return nlo_high_energy_series(*copy.deepcopy((tau, s, a)))\n'
               '\n'
               'def _reference_case():\n'
               '    import numpy as np\n'
               '    import copy\n'
               '    m = 0.00051099895\n'
               '    a = 1.0 / 137.035999084\n'
               '    (s, tau, _) = _oracle_compton_kinematics_lo(*copy.deepcopy(((1.0 - m * m) / (2.0 * m), m, '
               'a)))\n'
               '    return _oracle_nlo_high_energy_series(*copy.deepcopy((tau, s, a)))\n',
      'call': '_candidate_case()',
      'gold_call': '_reference_case()',
      'tol': 1e-09},
     {'setup': 'import numpy as np\nimport copy\n',
      'call': 'nlo_high_energy_series(*copy.deepcopy((0.01, 2.0, 1.0 / 137.035999084)))',
      'gold_call': '_oracle_nlo_high_energy_series(*copy.deepcopy((0.01, 2.0, 1.0 / 137.035999084)))',
      'tol': 1e-09},
     {'setup': 'import numpy as np\nimport copy\n',
      'call': 'nlo_high_energy_series(*copy.deepcopy((1.0e-8, 5.0, 0.05)))',
      'gold_call': '_oracle_nlo_high_energy_series(*copy.deepcopy((1.0e-8, 5.0, 0.05)))',
      'tol': 1e-09},
     {'setup': 'def _candidate_case():\n'
               '    import numpy as np\n'
               '    import copy\n'
               '\n'
               '    def run_model():\n'
               '        try:\n'
               '            nlo_high_energy_series(*copy.deepcopy((0.2, 1.0, 1.0 / 137.035999084)))\n'
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
               '            _oracle_nlo_high_energy_series(*copy.deepcopy((0.2, 1.0, 1.0 / 137.035999084)))\n'
               '            return 0\n'
               '        except ValueError:\n'
               '            return 1\n'
               '        except Exception:\n'
               '            return 2\n'
               '    return run_gold()\n',
      'call': '_candidate_case()',
      'gold_call': '_reference_case()'}]
