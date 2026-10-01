"""
Step 05 - Exciton density of the self-consistent condensate.

Exciton density of the self-consistent condensate.

The number of condensed excitons per unit area is n_ex = int d^2k/(2 pi)^2 v_k^2 for the converged state of the
previous step. It vanishes for E_G >= E_b and, close to the transition, approaches (E_b - E_G)/(d mu_ex/d n_ex)
from below, the deviation growing linearly with n_ex. Only the reduced mass enters the self-consistency
condition, so the density does not depend on the mass ratio; the mass ratio only enters the quasiparticle bands.

Inputs: E_G (Ry*, finite), d >= 0 (a_B*). Output: n_ex in units of (a_B*)^-2 as a float, converged to 1e-9
relative accuracy. Raises ValueError if E_G is not finite, d is negative or not finite, or the self-consistent
iteration does not converge within 400 iterations.

Returns
-------
float, the exciton density n_ex in (a_B*)^-2 (0.0 for E_G >= E_b)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def exciton_density(EG: float, d: float) -> float:
    '''Exciton density n_ex of the self-consistent condensate.

    Parameters
    ----------
    EG : float
        Bare band gap E_G in Ry*, finite.
    d : float
        Interlayer distance in units of a_B*, >= 0.

    Returns
    -------
    result : float
        n_ex = int d^2k/(2 pi)^2 v_k^2 in units of (a_B*)^-2 (0.0 for E_G >= E_b), converged to a relative
        accuracy of 1e-9.

    Raises
    ------
    ValueError
        If E_G is not finite, d is negative or not finite, or the self-consistent iteration does not converge
        within 400 iterations.
    '''
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_exciton_density(EG: float, d: float) -> float:
    """Reference implementation."""
    EG = _check_scalar(EG, "EG")
    d = _check_scalar(d, "d", nonneg=True)
    return float(_hf(EG, d, 1.0)["n"])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: monolayer at the source's figure regime; deviates from first order by about 0.8 percent ---
        {
            "setup": "import numpy as np\n"
                     "from scipy.special import ellipk, ellipe\n"
                     "from scipy.linalg import eigh\n",
            "call": "exciton_density(3.848585, 0.0) * 100",
            "gold_call": "_oracle_exciton_density(3.848585, 0.0) * 100",
            "tol": 1e-6,
        },
        # --- Normal: the heterobilayer of the task ---
        {
            "setup": "import numpy as np\n"
                     "from scipy.special import ellipk, ellipe\n"
                     "from scipy.linalg import eigh\n",
            "call": "exciton_density(1.7815, 0.25) * 100",
            "gold_call": "_oracle_exciton_density(1.7815, 0.25) * 100",
            "tol": 1e-6,
        },
        # --- Boundary: gap close to E_b, very dilute condensate at a full Bohr radius of separation ---
        {
            "setup": "import numpy as np\n"
                     "from scipy.special import ellipk, ellipe\n"
                     "from scipy.linalg import eigh\n",
            "call": "exciton_density(0.85, 1.0) * 1000",
            "gold_call": "_oracle_exciton_density(0.85, 1.0) * 1000",
            "tol": 1e-6,
        },
        # --- Edge: gap above E_b, no condensate ---
        {
            "setup": "import numpy as np\n"
                     "from scipy.special import ellipk, ellipe\n"
                     "from scipy.linalg import eigh\n",
            "call": "exciton_density(4.5, 0.0)",
            "gold_call": "_oracle_exciton_density(4.5, 0.0)",
            "tol": 1e-12,
        },
    ]
