"""
Compute the phase-space normalization and the spectrum-averaged one-loop and two-loop outer correction functions of a superallowed positron emitter by Gauss-Legendre quadrature over the positron momentum.

The paper compares the size of the radiative corrections through their phase-space averages (eq. (4.4)): the average of a correction function h(E_e) is integral dE_e p_e E_e Ebar^2 Fbar h / integral dE_e p_e E_e Ebar^2 Fbar. Written per unit momentum with the weight w(p_e) = p_e^2 Ebar^2 Fbar of the previous step, both integrals run over 0 < p_e < p_max = sqrt(E_0^2 - m_e^2). Because the weight vanishes at both ends and the correction functions are smooth on the open interval (g^(1) has an integrable log(Ebar) at the endpoint and ghat^(2) a 1/beta behaviour at threshold that is killed by the p_e^2 factor and the Coulomb suppression), an n-node Gauss-Legendre rule mapped onto [0, p_max] converges rapidly: the nodes x_i and weights w_i on [-1, 1] give p_i = p_max (x_i + 1)/2 and the integrals sum_i (p_max/2) w_i w(p_i) h(p_i). Convergence to better than 1e-9 relative is reached well below 200 nodes for the superallowed emitters. At every node the velocity and neutrino energy come from the kinematics of Step 1 (uses lepton_kinematics). This step returns the normalization integral N = integral w dp_e, the averaged one-loop function gbar^(1) (uses sirlin_g1) and the averaged two-loop function gbar^(2) (uses g2_alpha2z), all at the same MS-bar scale.

Returns
-------
np.ndarray of shape (3,) — the normalization integral (MeV^5), the averaged g^(1) and the averaged ghat^(2).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def phase_space_average(Z: int, E_0: float, mubar: float, m_e: float, alpha: float, n_nodes: int) -> "np.ndarray":
    '''Normalization and spectrum-averaged outer corrections by Gauss-Legendre quadrature.

    Parameters
    ----------
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
    n_nodes : int
        Number of Gauss-Legendre nodes on [0, p_max], >= 1.

    Returns
    -------
    result : np.ndarray of shape (3,)
        [N, gbar1, gbar2]: the normalization integral of the weight in MeV^5,
        the phase-space average of g^(1)(beta, Ebar, mubar) and the
        phase-space average of ghat^(2)(beta, mubar).

    Raises
    ------
    ValueError
        If n_nodes < 1 or E_0 <= m_e, or propagated from the earlier steps for
        invalid Z, mubar, m_e or alpha.
    '''
    return result  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math
import numpy as np

def _oracle_phase_space_average(Z: int, E_0: float, mubar: float, m_e: float, alpha: float, n_nodes: int) -> "np.ndarray":
    E_0 = float(E_0); m_e = float(m_e); n_nodes = int(n_nodes)
    if n_nodes < 1 or m_e <= 0.0 or E_0 <= m_e:
        raise ValueError("require n_nodes >= 1 and E_0 > m_e > 0")
    p_max = math.sqrt((E_0 - m_e) * (E_0 + m_e))
    x, w = np.polynomial.legendre.leggauss(n_nodes)
    norm = 0.0; s1 = 0.0; s2 = 0.0
    for x_i, w_i in zip(x, w):
        p = 0.5 * p_max * (float(x_i) + 1.0)
        wt = 0.5 * p_max * float(w_i) * _oracle_spectrum_weight(p, Z, E_0, mubar, m_e, alpha)
        E_e = math.sqrt(p * p + m_e * m_e)
        beta, _p, Ebar = _oracle_lepton_kinematics(E_e, m_e, E_0)   # Step 1
        norm += wt
        s1 += wt * _oracle_sirlin_g1(beta, Ebar, mubar, m_e)
        s2 += wt * _oracle_g2_alpha2z(beta, mubar, m_e)
    return np.array([norm, s1 / norm, s2 / norm], dtype=float)

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
            'call': 'phase_space_average(*copy.deepcopy((12, E0, mu, m, a, 200)))',
            'gold_call': '_oracle_phase_space_average(*copy.deepcopy((12, E0, mu, m, a, 200)))',
            'tol': 1e-09,
        },
        {
            # Case 2
            "setup": """import numpy as np
import copy
m = 0.51099895; a = 1.0 / 137.035999084
""",
            'call': 'phase_space_average(*copy.deepcopy((5, 1.90799 - m, 2.0 * (1.90799 - m) * np.exp(-1.0), m, a, 200)))',
            'gold_call': '_oracle_phase_space_average(*copy.deepcopy((5, 1.90799 - m, 2.0 * (1.90799 - m) * np.exp(-1.0), m, a, 200)))',
            'tol': 1e-09,
        },
        {
            # Case 3
            "setup": """import numpy as np
import copy
m = 0.51099895; a = 1.0 / 137.035999084
""",
            'call': 'phase_space_average(*copy.deepcopy((26, 8.2444 - m, 2.0 * (8.2444 - m) * np.exp(-1.0), m, a, 200)))',
            'gold_call': '_oracle_phase_space_average(*copy.deepcopy((26, 8.2444 - m, 2.0 * (8.2444 - m) * np.exp(-1.0), m, a, 200)))',
            'tol': 1e-09,
        },
        {
            # Case 4
            "setup": """import numpy as np
import copy
m = 0.51099895; a = 1.0 / 137.035999084
E0 = 4.2327 - m; mu = 2.0 * E0 * np.exp(-1.0)
""",
            'call': 'phase_space_average(*copy.deepcopy((12, E0, mu, m, a, 1)))',
            'gold_call': '_oracle_phase_space_average(*copy.deepcopy((12, E0, mu, m, a, 1)))',
            'tol': 1e-09,
        },
        {
            # Case 5
            "setup": """import numpy as np
import copy
m = 0.51099895; a = 1.0 / 137.035999084
""",
            'call': 'phase_space_average(*copy.deepcopy((20, 0.9, 1.5, m, a, 64)))',
            'gold_call': '_oracle_phase_space_average(*copy.deepcopy((20, 0.9, 1.5, m, a, 64)))',
            'tol': 1e-09,
        },
        {
            # Case 6
            "setup": """import numpy as np
import copy
def run_model():
    try:
        phase_space_average(*copy.deepcopy((12, 3.7217, 2.7, 0.51099895, 1.0 / 137.035999084, 0)))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_phase_space_average(*copy.deepcopy((12, 3.7217, 2.7, 0.51099895, 1.0 / 137.035999084, 0)))
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
