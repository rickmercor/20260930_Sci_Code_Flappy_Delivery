"""
Step 06 - Bare gap that sustains a prescribed exciton density.

Bare gap that sustains a prescribed exciton density.

In a gated bilayer the bare gap E_G is the experimental knob, while the quantity a local probe measures is the
exciton density. This step inverts the previous one: given a target density n_target, find the bare gap E_G at
which the self-consistent condensate has exactly that density. The first-order estimate E_G ~ E_b - (d mu_ex/d n_ex)
n_target is the natural starting point and the self-consistent density at that gap falls short of n_target by a
fraction that grows linearly with n_target (of order one percent per hundredth of (a_B*)^-2), so the root must be refined iteratively with the full self-consistent
solution; the density is a smooth, monotonically decreasing function of E_G on (-inf, E_b), so the root is unique.
For n_target = 0 the answer is E_b itself.

Inputs: n_target >= 0 in (a_B*)^-2; d >= 0 (a_B*). Output: E_G in Ry* as a float, converged so that the
self-consistent density reproduces n_target to 1e-10 relative. Raises ValueError if n_target is negative or not
finite, or if d is negative or not finite.

Returns
-------
float, the bare gap E_G in Ry* at which the condensate density equals n_target
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def gap_for_density(n_target: float, d: float) -> float:
    '''Bare gap E_G at which the self-consistent condensate has exciton density n_target.

    Parameters
    ----------
    n_target : float
        Target exciton density in units of (a_B*)^-2, >= 0.
    d : float
        Interlayer distance in units of a_B*, >= 0.

    Returns
    -------
    result : float
        E_G in Ry* (E_b when n_target is 0), such that the self-consistent density equals n_target to a relative
        accuracy of 1e-10.

    Raises
    ------
    ValueError
        If n_target is negative or not finite, or if d is negative or not finite.
    '''
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _gap_for_density(n_target, d):
    """Secant iteration on E_G with warm-started self-consistent solutions."""
    Eb, phi = _exciton(d)
    dmu = _oracle_inverse_compressibility(d)
    if n_target == 0.0:
        return Eb
    E0 = Eb - dmu * n_target
    s0 = _hf(E0, d, 1.0)
    f0 = s0["n"] - n_target
    E1 = Eb - dmu * n_target * (n_target / s0["n"]) if s0["n"] > 0 else E0 - 0.5 * dmu * n_target
    s1 = _hf(E1, d, 1.0, start=s0)
    f1 = s1["n"] - n_target
    for _ in range(40):
        if f1 == f0:
            break
        E2 = min(E1 - f1 * (E1 - E0) / (f1 - f0), Eb - 1e-12)
        s2 = _hf(E2, d, 1.0, start=s1)
        f2 = s2["n"] - n_target
        E0, f0, E1, f1, s1 = E1, f1, E2, f2, s2
        if abs(f2) < 1e-13 * n_target:
            break
    return float(E1)


def _oracle_gap_for_density(n_target: float, d: float) -> float:
    """Reference implementation."""
    n_target = _check_scalar(n_target, "n_target", nonneg=True)
    d = _check_scalar(d, "d", nonneg=True)
    return _gap_for_density(n_target, d)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: the heterobilayer of the task at one hundredth of an inverse Bohr radius squared ---
        {
            "setup": "import numpy as np\n"
                     "from scipy.special import ellipk, ellipe\n"
                     "from scipy.linalg import eigh\n",
            "call": "gap_for_density(0.01, 0.25)",
            "gold_call": "_oracle_gap_for_density(0.01, 0.25)",
            "tol": 1e-6,
        },
        # --- Normal: monolayer at the source's figure density ---
        {
            "setup": "import numpy as np\n"
                     "from scipy.special import ellipk, ellipe\n"
                     "from scipy.linalg import eigh\n",
            "call": "gap_for_density(0.025, 0.0)",
            "gold_call": "_oracle_gap_for_density(0.025, 0.0)",
            "tol": 1e-6,
        },
        # --- Boundary: half a Bohr radius, very dilute ---
        {
            "setup": "import numpy as np\n"
                     "from scipy.special import ellipk, ellipe\n"
                     "from scipy.linalg import eigh\n",
            "call": "gap_for_density(0.003, 0.5)",
            "gold_call": "_oracle_gap_for_density(0.003, 0.5)",
            "tol": 1e-6,
        },
        # --- Edge: zero density returns the binding energy of the interlayer exciton ---
        {
            "setup": "import numpy as np\n"
                     "from scipy.special import ellipk, ellipe\n"
                     "from scipy.linalg import eigh\n",
            "call": "gap_for_density(0.0, 1.0)",
            "gold_call": "_oracle_gap_for_density(0.0, 1.0)",
            "tol": 1e-6,
        },
    ]
