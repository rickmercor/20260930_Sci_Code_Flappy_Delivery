"""
Return the mean extension in nm of a single strand of n_nt nucleotides under force f (pN) at thermal energy k_B T (pN nm), modelled as a worm-like chain of contour length l_ss n_nt with persistence length lambda_ss through the Marko-Siggia interpolation formula f lambda_ss / k_B T = 1 / (4 (1 - u)^2) - 1/4 + u, u = <z>/L, solved for u on (0, 1); zero nucleotides give zero extension.

Single-stranded DNA is flexible on the scale of a nucleotide, and its force-extension response is well described by a worm-like chain; the released strands of a partially ruptured duplex lengthen under the shear force, which is what makes each base-pair transition force dependent.

Returns
-------
float, the strand extension in nm.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def ssdna_extension(n_nt: int, force: float, k_bt: float, persistence_length: float,
                            interphosphate_distance: float) -> float:
    """Return the mean extension in nm of a single strand of n_nt nucleotides under force f (pN) at thermal energy k_B T (pN nm), modelled as a worm-like chain of contour length l_ss n_nt with persistence length lambda_ss through the Marko-Siggia interpolation formula f lambda_ss / k_B T = 1 / (4 (1 - u)^2) - 1/4 + u, u = <z>/L, solved for u on (0, 1); zero nucleotides give zero extension.

    Parameters
    ----------
    n_nt : int
        Number of nucleotides in the released strand (>= 0).
    force : float
        Force in pN (> 0).
    k_bt : float
        Thermal energy in pN nm (> 0).
    persistence_length : float
        lambda_ss in nm (> 0).
    interphosphate_distance : float
        l_ss in nm (> 0).

    Returns
    -------
    extension : float
        Extension in nm.

    Raises
    ------
    ValueError
        If n_nt is not a nonnegative integer or any other argument is not finite and positive.
    """
    return extension

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, curve_fit


def _check_pos(x, name):
    if isinstance(x, bool) or not np.isfinite(x) or float(x) <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return float(x)


def _oracle_ssdna_extension(n_nt: int, force: float, k_bt: float, persistence_length: float,
                            interphosphate_distance: float) -> float:
    """Marko-Siggia worm-like-chain extension (nm) of an n_nt-nucleotide strand of contour length l_ss n_nt.

    Solves f lambda / kT = 1/(4 (1 - u)^2) - 1/4 + u for u = <z>/L on (0, 1) and returns L u; zero nucleotides
    give zero extension.
    """
    if isinstance(n_nt, bool) or int(n_nt) != n_nt or int(n_nt) < 0:
        raise ValueError("n_nt must be a nonnegative integer")
    m = int(n_nt)
    f = _check_pos(force, "force")
    kt = _check_pos(k_bt, "k_bt")
    lam = _check_pos(persistence_length, "persistence_length")
    l_ss = _check_pos(interphosphate_distance, "interphosphate_distance")
    if m == 0:
        return 0.0
    L = l_ss * m
    g = f * lam / kt
    u = brentq(lambda v: 1.0 / (4.0 * (1.0 - v) ** 2) - 0.25 + v - g, 0.0, 1.0 - 1e-12, xtol=1e-14, rtol=1e-14)
    return float(L * u)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nn_nt, force, k_bt, persistence_length, interphosphate_distance = 5, 6.0, 4.186, 0.77, 0.7\n",
            "call": "ssdna_extension(n_nt, force, k_bt, persistence_length, interphosphate_distance)",
            "gold_call": "_oracle_ssdna_extension(n_nt, force, k_bt, persistence_length, interphosphate_distance)",
        },
        {
            "setup": "import numpy as np\nn_nt, force, k_bt, persistence_length, interphosphate_distance = 0, 6.0, 4.186, 0.77, 0.7\n",
            "call": "ssdna_extension(n_nt, force, k_bt, persistence_length, interphosphate_distance)",
            "gold_call": "_oracle_ssdna_extension(n_nt, force, k_bt, persistence_length, interphosphate_distance)",
        },
        {
            "setup": "import numpy as np\nn_nt, force, k_bt, persistence_length, interphosphate_distance = 40, 0.2, 4.1, 1.2, 0.65\n",
            "call": "ssdna_extension(n_nt, force, k_bt, persistence_length, interphosphate_distance)",
            "gold_call": "_oracle_ssdna_extension(n_nt, force, k_bt, persistence_length, interphosphate_distance)",
        },
        {
            "setup": "import numpy as np\ndef run_model():\n    try:\n        ssdna_extension(3, 6.0, 4.186, -0.77, 0.7)\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_ssdna_extension(3, 6.0, 4.186, -0.77, 0.7)\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
