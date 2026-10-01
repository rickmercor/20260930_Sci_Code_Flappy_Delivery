"""
Evaluate the one-loop order-alpha outer radiative correction g^(1) to the positron spectrum of a superallowed decay (the Sirlin function in the heavy-particle effective theory, MS-bar scheme) at a given velocity, neutrino energy and renormalization scale.

Beyond the Coulomb (Fermi-function) effects, the exchange and emission of ultrasoft photons by the charged lepton corrects the differential rate by a factor 1 + alpha g^(1)(beta, Ebar, mubar) at first order. In the heavy-particle effective theory the paper writes this function as (eq. (2.30)) g^(1) = (1 / (2 pi)) { (3/2) L_mu - (4 / beta) [ Li_2(2 beta / (1 + beta)) + (1/4) log^2((1 + beta)/(1 - beta)) ] + 2 log(m_e^2 / (4 Ebar^2)) + 8 - (4/3) Ebar / E_e + (1 / beta) log((1 + beta)/(1 - beta)) [ -log(m_e^2 / (4 Ebar^2)) - 2 + beta^2 + Ebar^2 / (12 E_e^2) + (2/3) Ebar / E_e ] }, where L_mu = log(mubar^2 / m_e^2), E_e = m_e / sqrt(1 - beta^2) is the lepton energy, Ebar = E_0 - E_e the neutrino energy and Li_2 the dilogarithm. This is Sirlin's classic function up to the scheme-dependent constant (it differs from Sirlin's original by the constant 11/4, absorbed in the matching coefficient) and up to the explicit MS-bar logarithm L_mu that compensates the running of the vector coupling; it is the same for electron and positron emission. Energies in MeV.

Returns
-------
float — the one-loop outer correction function g^(1)(beta, Ebar, mubar).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def sirlin_g1(beta: float, Ebar: float, mubar: float, m_e: float) -> float:
    '''Order-alpha outer correction g^(1)(beta, Ebar, mubar).

    Parameters
    ----------
    beta : float
        Lepton velocity, 0 < beta < 1.
    Ebar : float
        Neutrino energy E_0 - E_e in MeV, > 0.
    mubar : float
        MS-bar renormalization scale in MeV, > 0.
    m_e : float
        Lepton mass in MeV, > 0.

    Returns
    -------
    g1 : float
        The dimensionless function g^(1) (the correction factor is
        1 + alpha g^(1)), as a native Python float.

    Raises
    ------
    ValueError
        If beta is outside (0, 1), or Ebar, mubar or m_e is not strictly
        positive.
    '''
    return g1  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math
import numpy as np
from scipy.special import spence

def _oracle_sirlin_g1(beta: float, Ebar: float, mubar: float, m_e: float) -> float:
    beta = float(beta); Ebar = float(Ebar); mubar = float(mubar); m_e = float(m_e)
    if not (0.0 < beta < 1.0) or Ebar <= 0.0 or mubar <= 0.0 or m_e <= 0.0:
        raise ValueError("require 0 < beta < 1, Ebar > 0, mubar > 0, m_e > 0")
    Li2 = lambda x: float(spence(1.0 - x))          # scipy's spence(1-x) is Li_2(x)
    E_e = m_e / math.sqrt((1.0 - beta) * (1.0 + beta))
    L_mu = math.log(mubar * mubar / (m_e * m_e))
    L = math.log((1.0 + beta) / (1.0 - beta))
    lm = math.log(m_e * m_e / (4.0 * Ebar * Ebar))
    r = Ebar / E_e
    val = (1.5 * L_mu
           - 4.0 / beta * (Li2(2.0 * beta / (1.0 + beta)) + 0.25 * L * L)
           + 2.0 * lm + 8.0 - 4.0 / 3.0 * r
           + L / beta * (-lm - 2.0 + beta * beta + r * r / 12.0 + 2.0 / 3.0 * r))
    return float(val / (2.0 * math.pi))

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
m = 0.51099895
E0 = 4.2327 - m; mu = 2.0 * E0 * np.exp(-1.0)
p = float(np.sqrt((2.0 - m) * (2.0 + m)))
beta = p / 2.0
Ebar = E0 - 2.0
""",
            'call': 'sirlin_g1(*copy.deepcopy((beta, Ebar, mu, m)))',
            'gold_call': '_oracle_sirlin_g1(*copy.deepcopy((beta, Ebar, mu, m)))',
            'tol': 1e-09,
        },
        {
            # Case 2
            "setup": """import numpy as np
import copy
m = 0.51099895
E0 = 4.2327 - m; mu = 2.0 * E0 * np.exp(-1.0)
E = 0.6
p = float(np.sqrt((E - m) * (E + m)))
beta = p / E
Ebar = E0 - E
""",
            'call': 'sirlin_g1(*copy.deepcopy((beta, Ebar, mu, m)))',
            'gold_call': '_oracle_sirlin_g1(*copy.deepcopy((beta, Ebar, mu, m)))',
            'tol': 1e-09,
        },
        {
            # Case 3
            "setup": """import numpy as np
import copy
m = 0.51099895
E0 = 4.2327 - m; mu = 2.0 * E0 * np.exp(-1.0)
E = 1.0
p = float(np.sqrt((E - m) * (E + m)))
beta = p / E
Ebar = E0 - E
""",
            'call': 'sirlin_g1(*copy.deepcopy((beta, Ebar, mu, m)))',
            'gold_call': '_oracle_sirlin_g1(*copy.deepcopy((beta, Ebar, mu, m)))',
            'tol': 1e-09,
        },
        {
            # Case 4
            "setup": """import numpy as np
import copy
m = 0.51099895
E0 = 4.2327 - m; mu = 2.0 * E0 * np.exp(-1.0)
E = 2.5
p = float(np.sqrt((E - m) * (E + m)))
beta = p / E
Ebar = E0 - E
""",
            'call': 'sirlin_g1(*copy.deepcopy((beta, Ebar, mu, m)))',
            'gold_call': '_oracle_sirlin_g1(*copy.deepcopy((beta, Ebar, mu, m)))',
            'tol': 1e-09,
        },
        {
            # Case 5
            "setup": """import numpy as np
import copy
m = 0.51099895
E0 = 4.2327 - m; mu = 2.0 * E0 * np.exp(-1.0)
E = 3.5
p = float(np.sqrt((E - m) * (E + m)))
beta = p / E
Ebar = E0 - E
""",
            'call': 'sirlin_g1(*copy.deepcopy((beta, Ebar, mu, m)))',
            'gold_call': '_oracle_sirlin_g1(*copy.deepcopy((beta, Ebar, mu, m)))',
            'tol': 1e-09,
        },
        {
            # Case 6
            "setup": """import numpy as np
import copy
m = 0.51099895
""",
            'call': 'sirlin_g1(*copy.deepcopy((0.7, 1.2, 4.0, m)))',
            'gold_call': '_oracle_sirlin_g1(*copy.deepcopy((0.7, 1.2, 4.0, m)))',
            'tol': 1e-09,
        },
        {
            # Case 7
            "setup": """import numpy as np
import copy
m = 0.51099895
""",
            'call': 'sirlin_g1(*copy.deepcopy((0.7, 1.2, 1.0, m)))',
            'gold_call': '_oracle_sirlin_g1(*copy.deepcopy((0.7, 1.2, 1.0, m)))',
            'tol': 1e-09,
        },
        {
            # Case 8
            "setup": """import numpy as np
import copy
m = 0.51099895
""",
            'call': 'sirlin_g1(*copy.deepcopy((0.95, 1.0e-4, 2.7, m)))',
            'gold_call': '_oracle_sirlin_g1(*copy.deepcopy((0.95, 1.0e-4, 2.7, m)))',
            'tol': 1e-09,
        },
        {
            # Case 9
            "setup": """import numpy as np
import copy
m = 0.51099895
""",
            'call': 'sirlin_g1(*copy.deepcopy((0.01, 3.0, 2.7, m)))',
            'gold_call': '_oracle_sirlin_g1(*copy.deepcopy((0.01, 3.0, 2.7, m)))',
            'tol': 1e-09,
        },
        {
            # Case 10
            "setup": """import numpy as np
import copy
def run_model():
    try:
        sirlin_g1(*copy.deepcopy((1.0, 1.0, 2.7, 0.51099895)))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_sirlin_g1(*copy.deepcopy((1.0, 1.0, 2.7, 0.51099895)))
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
