"""
Evaluate the paper's high-energy next-to-next-to-leading-order (order alpha^4) correction to the Compton total cross section, through the tau^1 term.

Use the next-to-next-to-leading-order expansion selected in the problem statement for 0 < tau <= 0.01, where tau = m^2/s. Apply the stated on-shell and final-state conventions. This step supplies the next-to-next-to-leading-order correction for the matched prediction.

Returns
-------
float — the next-to-next-to-leading-order (alpha^4) correction to the Compton total cross section, in GeV^-2.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def nnlo_high_energy_series(tau: float, s: float, alpha: float) -> float:
    '''High-energy expansion of the order-alpha^4 correction.

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
    sigma_NNLO : float
        The order-alpha^4 correction to the total cross section from the
        paper's expansion (through tau^1), in GeV^-2, as a native Python
        float.

    Raises
    ------
    ValueError
        If tau is outside (0, 0.01], or if s or alpha is not strictly positive.
    '''
    return sigma_NNLO  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
import math

def _oracle_nnlo_high_energy_series(tau: float, s: float, alpha: float) -> float:
    tau = float(tau); s = float(s); alpha = float(alpha)
    if not (0.0 < tau <= 0.01) or not (s > 0.0) or not (alpha > 0.0):
        raise ValueError("require 0 < tau <= 0.01, s > 0, alpha > 0")
    L = math.log(1.0 / tau)
    # eq. (7), next-to-next-to-leading-order line
    poly = (L ** 5 / 48.0 - 0.149 * L ** 4 + 1.34 * L ** 3 - 1.17 * L ** 2 - 16.6 * L + 23.8) \
        + tau * (1.13 * L ** 5 - 6.54 * L ** 4 - 44.6 * L ** 3 + 442.0 * L ** 2 - 36.0 * L - 2.64e3)
    return float(alpha ** 4 / (math.pi * s) * poly)

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
               '    return nnlo_high_energy_series(*copy.deepcopy((tau, s, a)))\n'
               '\n'
               'def _reference_case():\n'
               '    import numpy as np\n'
               '    import copy\n'
               '    m = 0.00051099895\n'
               '    a = 1.0 / 137.035999084\n'
               '    (s, tau, _) = _oracle_compton_kinematics_lo(*copy.deepcopy((8.0, m, a)))\n'
               '    return _oracle_nnlo_high_energy_series(*copy.deepcopy((tau, s, a)))\n',
      'call': '_candidate_case()',
      'gold_call': '_reference_case()',
      'tol': 1e-09},
     {'setup': 'def _candidate_case():\n'
               '    import numpy as np\n'
               '    import copy\n'
               '    m = 0.00051099895\n'
               '    a = 1.0 / 137.035999084\n'
               '    (s, tau, _) = compton_kinematics_lo(*copy.deepcopy(((1.0 - m * m) / (2.0 * m), m, a)))\n'
               '    return nnlo_high_energy_series(*copy.deepcopy((tau, s, a)))\n'
               '\n'
               'def _reference_case():\n'
               '    import numpy as np\n'
               '    import copy\n'
               '    m = 0.00051099895\n'
               '    a = 1.0 / 137.035999084\n'
               '    (s, tau, _) = _oracle_compton_kinematics_lo(*copy.deepcopy(((1.0 - m * m) / (2.0 * m), m, '
               'a)))\n'
               '    return _oracle_nnlo_high_energy_series(*copy.deepcopy((tau, s, a)))\n',
      'call': '_candidate_case()',
      'gold_call': '_reference_case()',
      'tol': 1e-09},
     {'setup': 'import numpy as np\nimport copy\n',
      'call': 'nnlo_high_energy_series(*copy.deepcopy((0.01, 2.0, 1.0 / 137.035999084)))',
      'gold_call': '_oracle_nnlo_high_energy_series(*copy.deepcopy((0.01, 2.0, 1.0 / 137.035999084)))',
      'tol': 1e-09},
     {'setup': 'def _candidate_case():\n'
               '    import numpy as np\n'
               '    import copy\n'
               '    m = 0.00051099895\n'
               '    a = 1.0 / 137.035999084\n'
               '    (s, tau, _) = compton_kinematics_lo(*copy.deepcopy(((1000000.0 - m * m) / (2.0 * m), m, '
               'a)))\n'
               '    return nnlo_high_energy_series(*copy.deepcopy((tau, s, a)))\n'
               '\n'
               'def _reference_case():\n'
               '    import numpy as np\n'
               '    import copy\n'
               '    m = 0.00051099895\n'
               '    a = 1.0 / 137.035999084\n'
               '    (s, tau, _) = _oracle_compton_kinematics_lo(*copy.deepcopy(((1000000.0 - m * m) / (2.0 * m), '
               'm, a)))\n'
               '    return _oracle_nnlo_high_energy_series(*copy.deepcopy((tau, s, a)))\n',
      'call': '_candidate_case()',
      'gold_call': '_reference_case()',
      'tol': 1e-09},
     {'setup': 'def _candidate_case():\n'
               '    import numpy as np\n'
               '    import copy\n'
               '\n'
               '    def run_model():\n'
               '        try:\n'
               '            nnlo_high_energy_series(*copy.deepcopy((0.0001, 0.0, 1.0 / 137.035999084)))\n'
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
               '            _oracle_nnlo_high_energy_series(*copy.deepcopy((0.0001, 0.0, 1.0 / 137.035999084)))\n'
               '            return 0\n'
               '        except ValueError:\n'
               '            return 1\n'
               '        except Exception:\n'
               '            return 2\n'
               '    return run_gold()\n',
      'call': '_candidate_case()',
      'gold_call': '_reference_case()'}]
