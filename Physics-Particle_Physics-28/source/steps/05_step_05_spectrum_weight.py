"""
Evaluate the Coulomb-corrected positron momentum spectrum of a superallowed decay, i.e. the phase-space weight p_e^2 Ebar^2 Fbar(beta, mubar) used to average the outer radiative corrections over the spectrum.

With the Fermi function factored out as in eq. (2.25), the differential rate in the positron energy is proportional to p_e E_e Ebar^2 Fbar(beta, mubar) times the outer-correction factor. The paper defines phase-space averages of the outer corrections with exactly this weight (eq. (4.4)): deltabar'_R = integral dE_e p_e E_e Ebar^2 Fbar delta'_R(E_e) / integral dE_e p_e E_e Ebar^2 Fbar, with the phase-space integral of eq. (4.3) taken for a point-like nucleus. Since E_e dE_e = p_e dp_e, the same weight expressed per unit momentum is w(p_e) = p_e^2 Ebar^2 Fbar(beta, mubar), with E_e = sqrt(p_e^2 + m_e^2) and beta, Ebar from the kinematics of Step 1 (uses lepton_kinematics); this is the form convenient for quadrature because it vanishes smoothly at both ends of the spectrum (as p_e^2 at threshold, where the Coulomb repulsion of the positron adds an exponential suppression, and as Ebar^2 at the endpoint). Energies and momenta in MeV.

Returns
-------
float — the phase-space weight p_e^2 Ebar^2 Fbar(beta, mubar) in MeV^4.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def spectrum_weight(p_e: float, Z: int, E_0: float, mubar: float, m_e: float, alpha: float) -> float:
    '''Phase-space weight per unit momentum, p_e^2 Ebar^2 Fbar(beta, mubar).

    Parameters
    ----------
    p_e : float
        Positron momentum in MeV, with 0 < p_e < sqrt(E_0^2 - m_e^2).
    Z : int
        Charge of the daughter nucleus, >= 1.
    E_0 : float
        Endpoint energy in MeV, > m_e.
    mubar : float
        MS-bar renormalization scale in MeV, > 0.
    m_e : float
        Positron mass in MeV, > 0.
    alpha : float
        Fine-structure constant, with 0 < alpha * Z < 1.

    Returns
    -------
    w : float
        p_e^2 (E_0 - E_e)^2 Fbar(beta, mubar) in MeV^4 (uses
        lepton_kinematics and fermi_function_msbar), as a native Python
        float.

    Raises
    ------
    ValueError
        If p_e is not strictly inside (0, sqrt(E_0^2 - m_e^2)), or propagated
        from lepton_kinematics or fermi_function_msbar for invalid inputs.
    '''
    return w  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math
import numpy as np

def _oracle_spectrum_weight(p_e: float, Z: int, E_0: float, mubar: float, m_e: float, alpha: float) -> float:
    p_e = float(p_e); E_0 = float(E_0); m_e = float(m_e)
    if m_e <= 0.0 or E_0 <= m_e:
        raise ValueError("require m_e > 0 and E_0 > m_e")
    p_max = math.sqrt((E_0 - m_e) * (E_0 + m_e))
    if not (0.0 < p_e < p_max):
        raise ValueError("require 0 < p_e < sqrt(E_0^2 - m_e^2)")
    E_e = math.sqrt(p_e * p_e + m_e * m_e)
    beta, p_chk, Ebar = _oracle_lepton_kinematics(E_e, m_e, E_0)      # Step 1
    Fbar = _oracle_fermi_function_msbar(beta, Z, mubar, m_e, alpha)  # Step 2
    return float(p_chk * p_chk * Ebar * Ebar * Fbar)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Independent configurations; each reference call invokes this step once."""
    return [
        {
            # Case 1
            "setup": """import numpy as np
import copy
m = 0.51099895; a = 1.0 / 137.035999084
E0 = 4.2327 - m; mu = 2.0 * E0 * np.exp(-1.0)
""",
            'call': 'spectrum_weight(*copy.deepcopy((0.3, 12, E0, mu, m, a)))',
            'gold_call': '_oracle_spectrum_weight(*copy.deepcopy((0.3, 12, E0, mu, m, a)))',
            'tol': 1e-09,
        },
        {
            # Case 2
            "setup": """import numpy as np
import copy
m = 0.51099895; a = 1.0 / 137.035999084
E0 = 4.2327 - m; mu = 2.0 * E0 * np.exp(-1.0)
""",
            'call': 'spectrum_weight(*copy.deepcopy((1.0, 12, E0, mu, m, a)))',
            'gold_call': '_oracle_spectrum_weight(*copy.deepcopy((1.0, 12, E0, mu, m, a)))',
            'tol': 1e-09,
        },
        {
            # Case 3
            "setup": """import numpy as np
import copy
m = 0.51099895; a = 1.0 / 137.035999084
E0 = 4.2327 - m; mu = 2.0 * E0 * np.exp(-1.0)
""",
            'call': 'spectrum_weight(*copy.deepcopy((2.0, 12, E0, mu, m, a)))',
            'gold_call': '_oracle_spectrum_weight(*copy.deepcopy((2.0, 12, E0, mu, m, a)))',
            'tol': 1e-09,
        },
        {
            # Case 4
            "setup": """import numpy as np
import copy
m = 0.51099895; a = 1.0 / 137.035999084
E0 = 4.2327 - m; mu = 2.0 * E0 * np.exp(-1.0)
""",
            'call': 'spectrum_weight(*copy.deepcopy((3.0, 12, E0, mu, m, a)))',
            'gold_call': '_oracle_spectrum_weight(*copy.deepcopy((3.0, 12, E0, mu, m, a)))',
            'tol': 1e-09,
        },
        {
            # Case 5
            "setup": """import numpy as np
import copy
m = 0.51099895; a = 1.0 / 137.035999084
E0 = 4.2327 - m; mu = 2.0 * E0 * np.exp(-1.0)
""",
            'call': 'spectrum_weight(*copy.deepcopy((3.6, 12, E0, mu, m, a)))',
            'gold_call': '_oracle_spectrum_weight(*copy.deepcopy((3.6, 12, E0, mu, m, a)))',
            'tol': 1e-09,
        },
        {
            # Case 6
            "setup": """import numpy as np
import copy
m = 0.51099895; a = 1.0 / 137.035999084
E0 = 8.2444 - m; mu = 2.0 * E0 * np.exp(-1.0)
""",
            'call': 'spectrum_weight(*copy.deepcopy((4.0, 26, E0, mu, m, a)))',
            'gold_call': '_oracle_spectrum_weight(*copy.deepcopy((4.0, 26, E0, mu, m, a)))',
            'tol': 1e-09,
        },
        {
            # Case 7
            "setup": """import numpy as np
import copy
m = 0.51099895; a = 1.0 / 137.035999084
E0 = 4.2327 - m; mu = 2.0 * E0 * np.exp(-1.0)
pmax = np.sqrt(E0 * E0 - m * m)
""",
            'call': 'spectrum_weight(*copy.deepcopy((1.0e-3, 12, E0, mu, m, a)))',
            'gold_call': '_oracle_spectrum_weight(*copy.deepcopy((1.0e-3, 12, E0, mu, m, a)))',
            'tol': 1e-08,
        },
        {
            # Case 8
            "setup": """import numpy as np
import copy
m = 0.51099895; a = 1.0 / 137.035999084
E0 = 4.2327 - m; mu = 2.0 * E0 * np.exp(-1.0)
pmax = np.sqrt(E0 * E0 - m * m)
""",
            'call': 'spectrum_weight(*copy.deepcopy((pmax - 1.0e-4, 12, E0, mu, m, a)))',
            'gold_call': '_oracle_spectrum_weight(*copy.deepcopy((pmax - 1.0e-4, 12, E0, mu, m, a)))',
            'tol': 1e-08,
        },
        {
            # Case 9
            "setup": """import numpy as np
import copy
m = 0.51099895
""",
            'call': 'spectrum_weight(*copy.deepcopy((1.5, 1, 3.0, 1.0, m, 1.0e-6)))',
            'gold_call': '_oracle_spectrum_weight(*copy.deepcopy((1.5, 1, 3.0, 1.0, m, 1.0e-6)))',
            'tol': 1e-09,
        },
        {
            # Case 10
            "setup": """import numpy as np
import copy
def run_model():
    try:
        spectrum_weight(*copy.deepcopy((5.0, 12, 3.7217, 2.7, 0.51099895, 1.0 / 137.035999084)))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_spectrum_weight(*copy.deepcopy((5.0, 12, 3.7217, 2.7, 0.51099895, 1.0 / 137.035999084)))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            'call': 'run_model()',
            'gold_call': 'run_gold()',
        },
    ]
