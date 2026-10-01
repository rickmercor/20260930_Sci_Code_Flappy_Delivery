"""
Evaluate the paper's high-energy leading-order Compton total cross section through the tau^4 term.

Use the leading-order expansion selected in the problem statement for 0 < tau <= 0.01, where tau = m^2/s. This step supplies the leading-order contribution for the matched prediction.

Returns
-------
float — the leading-order Compton total cross section from the high-energy expansion, in GeV^-2.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def lo_high_energy_series(tau: float, s: float, alpha: float) -> float:
    '''High-energy expansion of the leading-order total cross section.

    Parameters
    ----------
    tau : float
        m^2/s, with 0 < tau <= 0.01 (the domain of the expansion).
    s : float
        Mandelstam s in GeV^2, > 0.
    alpha : float
        Fine-structure constant, > 0.

    Returns
    -------
    sigma_LO : float
        The truncated high-energy expansion of the leading-order cross
        section in GeV^-2, as a native Python float.

    Raises
    ------
    ValueError
        If tau is outside (0, 0.01], or if s or alpha is not strictly positive.
    '''
    return sigma_LO  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
import math

def _oracle_lo_high_energy_series(tau: float, s: float, alpha: float) -> float:
    tau = float(tau); s = float(s); alpha = float(alpha)
    if not (0.0 < tau <= 0.01) or not (s > 0.0) or not (alpha > 0.0):
        raise ValueError("require 0 < tau <= 0.01, s > 0, alpha > 0")
    L = math.log(1.0 / tau)
    # eq. (7), leading-order line
    poly = (2.0 * L + 1.0) + tau * (-6.0 * L + 17.0) + tau ** 2 * (-30.0 * L + 32.0) \
        + tau ** 3 * (-70.0 * L + 48.0) + tau ** 4 * (-126.0 * L + 64.0)
    return float(math.pi * alpha ** 2 / s * poly)

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
               '    return lo_high_energy_series(*copy.deepcopy((tau, s, a)))\n'
               '\n'
               'def _reference_case():\n'
               '    import numpy as np\n'
               '    import copy\n'
               '    m = 0.00051099895\n'
               '    a = 1.0 / 137.035999084\n'
               '    (s, tau, _) = _oracle_compton_kinematics_lo(*copy.deepcopy((8.0, m, a)))\n'
               '    return _oracle_lo_high_energy_series(*copy.deepcopy((tau, s, a)))\n',
      'call': '_candidate_case()',
      'gold_call': '_reference_case()',
      'tol': 1e-09},
     {'setup': 'import numpy as np\nimport copy\n',
      'call': 'lo_high_energy_series(*copy.deepcopy((0.01, 2.0, 1.0 / 137.035999084)))',
      'gold_call': '_oracle_lo_high_energy_series(*copy.deepcopy((0.01, 2.0, 1.0 / 137.035999084)))',
      'tol': 1e-09},
     {'setup': 'def _candidate_case():\n'
               '    import numpy as np\n'
               '    import copy\n'
               '    m = 0.00051099895\n'
               '    a = 1.0 / 137.035999084\n'
               '    (s, tau, kn) = compton_kinematics_lo(*copy.deepcopy(((1.0 - m * m) / (2.0 * m), m, a)))\n'
               '\n'
               '    def run_model():\n'
               '        return float(abs(lo_high_energy_series(*copy.deepcopy((tau, s, a))) / kn - 1.0) < '
               '1e-12)\n'
               '    return run_model()\n'
               '\n'
               'def _reference_case():\n'
               '    import numpy as np\n'
               '    import copy\n'
               '    m = 0.00051099895\n'
               '    a = 1.0 / 137.035999084\n'
               '    (s, tau, kn) = _oracle_compton_kinematics_lo(*copy.deepcopy(((1.0 - m * m) / (2.0 * m), m, '
               'a)))\n'
               '\n'
               '    def run_gold():\n'
               '        return float(abs(_oracle_lo_high_energy_series(*copy.deepcopy((tau, s, a))) / kn - 1.0) '
               '< 1e-12)\n'
               '    return run_gold()\n',
      'call': '_candidate_case()',
      'gold_call': '_reference_case()'},
     {'setup': 'def _candidate_case():\n'
               '    import numpy as np\n'
               '    import copy\n'
               '    m = 0.00051099895\n'
               '    a = 1.0 / 137.035999084\n'
               '    (s, tau, _) = compton_kinematics_lo(*copy.deepcopy(((1000000.0 - m * m) / (2.0 * m), m, '
               'a)))\n'
               '    return lo_high_energy_series(*copy.deepcopy((tau, s, a)))\n'
               '\n'
               'def _reference_case():\n'
               '    import numpy as np\n'
               '    import copy\n'
               '    m = 0.00051099895\n'
               '    a = 1.0 / 137.035999084\n'
               '    (s, tau, _) = _oracle_compton_kinematics_lo(*copy.deepcopy(((1000000.0 - m * m) / (2.0 * m), '
               'm, a)))\n'
               '    return _oracle_lo_high_energy_series(*copy.deepcopy((tau, s, a)))\n',
      'call': '_candidate_case()',
      'gold_call': '_reference_case()',
      'tol': 1e-09},
     {'setup': 'def _candidate_case():\n'
               '    import numpy as np\n'
               '    import copy\n'
               '\n'
               '    def run_model():\n'
               '        try:\n'
               '            lo_high_energy_series(*copy.deepcopy((0.05, 1.0, 1.0 / 137.035999084)))\n'
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
               '            _oracle_lo_high_energy_series(*copy.deepcopy((0.05, 1.0, 1.0 / 137.035999084)))\n'
               '            return 0\n'
               '        except ValueError:\n'
               '            return 1\n'
               '        except Exception:\n'
               '            return 2\n'
               '    return run_gold()\n',
      'call': '_candidate_case()',
      'gold_call': '_reference_case()'}]
