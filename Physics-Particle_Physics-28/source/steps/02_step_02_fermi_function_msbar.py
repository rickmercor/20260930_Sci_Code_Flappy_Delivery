"""
Evaluate the all-orders-in-alpha*Z Fermi function of a positron emitter in the MS-bar scheme of the heavy-particle effective theory, at a given lepton velocity and renormalization scale.

The Coulomb interaction between the outgoing positron and the daughter nucleus of charge Z enhances or suppresses the decay rate by the Fermi function, which sums the potential-photon exchanges to all orders in alpha*Z. In the effective-field-theory treatment used by the paper the Fermi function is renormalized in the MS-bar scheme and reads (eq. (2.29)) Fbar(beta, mubar) = [4 eta / (1 + eta)^2] * [2 (1 + eta) / Gamma(2 eta + 1)^2] * |Gamma(eta + i y)|^2 * exp(pi y) * (2 p_e e^{-gamma_E} / mubar)^{2 (eta - 1)}, with eta = sqrt(1 - alpha^2 Z^2), y = -Z alpha / beta for positron emission (+Z alpha / beta for electron emission), p_e = beta E_e = beta m_e / sqrt(1 - beta^2) the lepton momentum, gamma_E the Euler-Mascheroni constant and mubar the MS-bar renormalization scale; Gamma is the complex gamma function. Setting mubar = R^{-1} e^{-gamma_E} with R the nuclear radius recovers the traditional Fermi function up to the factor 4 eta / (1 + eta)^2. Its expansion in alpha*Z starts as 1 - pi alpha Z / beta + ... for positrons. Energies in MeV; alpha is the fine-structure constant.

Returns
-------
float — the MS-bar Fermi function Fbar(beta, mubar) of the positron emitter.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def fermi_function_msbar(beta: float, Z: int, mubar: float, m_e: float, alpha: float) -> float:
    '''MS-bar Fermi function of a positron emitter.

    Parameters
    ----------
    beta : float
        Positron velocity, 0 < beta < 1.
    Z : int
        Charge of the daughter (final-state) nucleus, >= 1.
    mubar : float
        MS-bar renormalization scale in MeV, > 0.
    m_e : float
        Positron mass in MeV, > 0.
    alpha : float
        Fine-structure constant, with 0 < alpha * Z < 1.

    Returns
    -------
    Fbar : float
        The MS-bar Fermi function for positron emission, as a native Python
        float.

    Raises
    ------
    ValueError
        If beta is outside (0, 1), Z < 1, mubar <= 0, m_e <= 0, or
        alpha * Z is not in (0, 1).
    '''
    return Fbar  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math
import numpy as np
from scipy.special import loggamma

def _oracle_fermi_function_msbar(beta: float, Z: int, mubar: float, m_e: float, alpha: float) -> float:
    beta = float(beta); Z = int(Z); mubar = float(mubar); m_e = float(m_e); alpha = float(alpha)
    if not (0.0 < beta < 1.0) or Z < 1 or mubar <= 0.0 or m_e <= 0.0 or not (0.0 < alpha * Z < 1.0):
        raise ValueError("require 0 < beta < 1, Z >= 1, mubar > 0, m_e > 0, 0 < alpha Z < 1")
    eta = math.sqrt(1.0 - (alpha * Z) ** 2)
    y = -Z * alpha / beta                       # positron emission
    p_e = beta * m_e / math.sqrt(1.0 - beta * beta)
    # |Gamma(eta + i y)|^2 e^{pi y} evaluated through log-gamma for stability
    log_abs_gamma_sq = 2.0 * loggamma(complex(eta, y)).real
    pref = 4.0 * eta / (1.0 + eta) ** 2 * 2.0 * (1.0 + eta) / math.gamma(2.0 * eta + 1.0) ** 2
    scale = (2.0 * p_e * math.exp(-np.euler_gamma) / mubar) ** (2.0 * (eta - 1.0))
    return float(pref * math.exp(log_abs_gamma_sq + math.pi * y) * scale)

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
beta = float(np.sqrt((2.0 - m) * (2.0 + m))) / 2.0
""",
            'call': 'fermi_function_msbar(*copy.deepcopy((beta, 12, mu, m, a)))',
            'gold_call': '_oracle_fermi_function_msbar(*copy.deepcopy((beta, 12, mu, m, a)))',
            'tol': 1e-09,
        },
        {
            # Case 2
            "setup": """import numpy as np
import copy
m = 0.51099895; a = 1.0 / 137.035999084
mu = 5.0
""",
            'call': 'fermi_function_msbar(*copy.deepcopy((0.3, 26, mu, m, a)))',
            'gold_call': '_oracle_fermi_function_msbar(*copy.deepcopy((0.3, 26, mu, m, a)))',
            'tol': 1e-09,
        },
        {
            # Case 3
            "setup": """import numpy as np
import copy
m = 0.51099895; a = 1.0 / 137.035999084
mu = 5.0
""",
            'call': 'fermi_function_msbar(*copy.deepcopy((0.6, 26, mu, m, a)))',
            'gold_call': '_oracle_fermi_function_msbar(*copy.deepcopy((0.6, 26, mu, m, a)))',
            'tol': 1e-09,
        },
        {
            # Case 4
            "setup": """import numpy as np
import copy
m = 0.51099895; a = 1.0 / 137.035999084
mu = 5.0
""",
            'call': 'fermi_function_msbar(*copy.deepcopy((0.9, 26, mu, m, a)))',
            'gold_call': '_oracle_fermi_function_msbar(*copy.deepcopy((0.9, 26, mu, m, a)))',
            'tol': 1e-09,
        },
        {
            # Case 5
            "setup": """import numpy as np
import copy
m = 0.51099895; a = 1.0 / 137.035999084
mu = 5.0
""",
            'call': 'fermi_function_msbar(*copy.deepcopy((0.99, 26, mu, m, a)))',
            'gold_call': '_oracle_fermi_function_msbar(*copy.deepcopy((0.99, 26, mu, m, a)))',
            'tol': 1e-09,
        },
        {
            # Case 6
            "setup": """import numpy as np
import copy
m = 0.51099895; a = 1.0 / 137.035999084
""",
            'call': 'fermi_function_msbar(*copy.deepcopy((0.02, 12, 2.7, m, a)))',
            'gold_call': '_oracle_fermi_function_msbar(*copy.deepcopy((0.02, 12, 2.7, m, a)))',
            'tol': 1e-09,
        },
        {
            # Case 7
            "setup": """import numpy as np
import copy
m = 0.51099895
""",
            'call': 'fermi_function_msbar(*copy.deepcopy((0.5, 1, 1.0, m, 1.0e-4)))',
            'gold_call': '_oracle_fermi_function_msbar(*copy.deepcopy((0.5, 1, 1.0, m, 1.0e-4)))',
            'tol': 1e-09,
        },
        {
            # Case 8
            "setup": """import numpy as np
import copy
m = 0.51099895
""",
            'call': 'fermi_function_msbar(*copy.deepcopy((0.5, 3, 1.0, m, 0.2)))',
            'gold_call': '_oracle_fermi_function_msbar(*copy.deepcopy((0.5, 3, 1.0, m, 0.2)))',
            'tol': 1e-09,
        },
        {
            # Case 9
            "setup": """import numpy as np
import copy
def run_model():
    try:
        fermi_function_msbar(*copy.deepcopy((0.5, 140, 1.0, 0.51099895, 1.0 / 137.035999084)))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_fermi_function_msbar(*copy.deepcopy((0.5, 140, 1.0, 0.51099895, 1.0 / 137.035999084)))
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
