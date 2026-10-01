"""
[ORCHESTRATOR] Compute the phase-space-averaged outer radiative correction deltabar'_R of a superallowed positron emitter through order alpha^2 Z, at the paper's central renormalization scale, from the transition's electron-capture Q value and daughter charge.

With the Fermi function factored out of the rate, the energy-dependent ("outer") radiative correction of a superallowed transition is delta'_R(E_e, mubar) = alpha g^(1)(beta, Ebar, mubar) + alpha^2 Z ghat^(2)(beta, mubar) (eq. (4.1)), and the quantity that enters the corrected half-life is its phase-space average deltabar'_R(mubar) with the weight p_e E_e Ebar^2 Fbar (eq. (4.4)). The paper evaluates it at the low-energy scale mu_ext = 2 E_0 of the MS-chi scheme, which corresponds to the MS-bar scale mubar = 2 E_0 e^{-1} (eq. (2.6)), with E_0 = Q_EC - m_e the endpoint and Z the charge of the daughter nucleus; alpha is held fixed at its low-energy value, the scale dependence being carried by the explicit logarithms of g^(1) and ghat^(2). This step composes the pipeline end to end: kinematics (Step 1), the MS-bar Fermi function (Step 2), the one-loop function (Step 3), the paper's two-loop function (Step 4), the spectral weight (Step 5) and the Gauss-Legendre averages (Step 6), returning alpha gbar^(1) + alpha^2 Z gbar^(2). For the emitters from 10C to 54Co the alpha^2 Z term is between 0.4e-3 and 2.3e-3 and partially cancels the negative one-loop average.

Returns
-------
float — the phase-space-averaged outer radiative correction deltabar'_R through order alpha^2 Z.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def averaged_outer_correction(Q_EC: float, Z: int, m_e: float, alpha: float, n_nodes: int) -> float:
    '''Phase-space-averaged outer correction deltabar'_R at mubar = 2 E_0 / e.

    Parameters
    ----------
    Q_EC : float
        Electron-capture Q value of the transition in MeV, > 2 m_e.
    Z : int
        Charge of the daughter nucleus, >= 1.
    m_e : float
        Positron mass in MeV, > 0.
    alpha : float
        Fine-structure constant, with 0 < alpha * Z < 1.
    n_nodes : int
        Number of Gauss-Legendre nodes for the spectral averages, >= 1.

    Returns
    -------
    delta_R : float
        deltabar'_R = alpha gbar^(1) + alpha^2 Z gbar^(2), dimensionless, as a
        native Python float.

    Raises
    ------
    ValueError
        If Q_EC <= 2 m_e, or propagated from the earlier steps for invalid Z,
        m_e, alpha or n_nodes.
    '''
    return delta_R  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math
import numpy as np

def _oracle_averaged_outer_correction(Q_EC: float, Z: int, m_e: float, alpha: float, n_nodes: int) -> float:
    Q_EC = float(Q_EC); m_e = float(m_e); alpha = float(alpha); Z = int(Z)
    if m_e <= 0.0 or Q_EC <= 2.0 * m_e:
        raise ValueError("require Q_EC > 2 m_e > 0")
    E_0 = Q_EC - m_e
    mubar = 2.0 * E_0 * math.exp(-1.0)
    _norm, gbar1, gbar2 = _oracle_phase_space_average(Z, E_0, mubar, m_e, alpha, n_nodes)
    return float(alpha * gbar1 + alpha * alpha * Z * gbar2)

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
""",
            'call': 'averaged_outer_correction(*copy.deepcopy((4.2327, 12, m, a, 200)))',
            'gold_call': '_oracle_averaged_outer_correction(*copy.deepcopy((4.2327, 12, m, a, 200)))',
            'tol': 1e-09,
        },
        {
            # Case 2
            "setup": """import numpy as np
import copy
m = 0.51099895; a = 1.0 / 137.035999084
""",
            'call': 'averaged_outer_correction(*copy.deepcopy((1.90799, 5, m, a, 200)))',
            'gold_call': '_oracle_averaged_outer_correction(*copy.deepcopy((1.90799, 5, m, a, 200)))',
            'tol': 1e-09,
        },
        {
            # Case 3
            "setup": """import numpy as np
import copy
m = 0.51099895; a = 1.0 / 137.035999084
""",
            'call': 'averaged_outer_correction(*copy.deepcopy((8.2444, 26, m, a, 200)))',
            'gold_call': '_oracle_averaged_outer_correction(*copy.deepcopy((8.2444, 26, m, a, 200)))',
            'tol': 1e-09,
        },
        {
            # Case 4
            "setup": """import numpy as np
import copy
m = 0.51099895; a = 1.0 / 137.035999084
""",
            'call': 'averaged_outer_correction(*copy.deepcopy((4.2327, 12, m, a, 1)))',
            'gold_call': '_oracle_averaged_outer_correction(*copy.deepcopy((4.2327, 12, m, a, 1)))',
            'tol': 1e-09,
        },
        {
            # Case 5
            "setup": """import numpy as np
import copy
m = 0.51099895
""",
            'call': 'averaged_outer_correction(*copy.deepcopy((1.5, 30, m, 1.0 / 137.035999084, 128)))',
            'gold_call': '_oracle_averaged_outer_correction(*copy.deepcopy((1.5, 30, m, 1.0 / 137.035999084, 128)))',
            'tol': 1e-09,
        },
        {
            # Case 6
            "setup": """import numpy as np
import copy
m = 0.51099895
""",
            'call': 'averaged_outer_correction(*copy.deepcopy((4.2327, 12, m, 1.0e-4, 64)))',
            'gold_call': '_oracle_averaged_outer_correction(*copy.deepcopy((4.2327, 12, m, 1.0e-4, 64)))',
            'tol': 1e-09,
        },
        {
            # Case 7
            "setup": """import numpy as np
import copy
def run_model():
    try:
        averaged_outer_correction(*copy.deepcopy((0.8, 12, 0.51099895, 1.0 / 137.035999084, 64)))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_averaged_outer_correction(*copy.deepcopy((0.8, 12, 0.51099895, 1.0 / 137.035999084, 64)))
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
