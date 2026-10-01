"""
Companion angles from the conserved linear combinations

Because $\Sigma_1$ and $\Sigma_2$ are constants of the orbit-averaged motion, the remaining angle cosines follow algebraically from $\cos\kappa_1(t)$:

$$\cos\kappa_2(t) = \Sigma_2

   - \frac{m_2}{m_1}\,\frac{s_1}{s_2}\,\cos\kappa_1(t), \qquad

\cos\gamma(t) = \Sigma_1

   - \frac{m_1 - m_2}{m_1}\,\frac{l}{s_2}\,\cos\kappa_1(t).$$

Returns
-------
np.ndarray of shape (2, N): [cos kappa2; cos gamma]
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def companion_angles(cos_kappa1: np.ndarray, m1: float, m2: float, l: float,
                     s1: float, s2: float, big_sigma1: float,
                     big_sigma2: float) -> np.ndarray:
    '''cos kappa2 and cos gamma from cos kappa1 and the invariants.

    Parameters
    ----------
    cos_kappa1 : np.ndarray
        1-D array of cos kappa1 values, entries finite in [-1, 1].
    m1 : float
        Primary mass, must satisfy m1 > m2 > 0.
    m2 : float
        Secondary mass.
    l : float
        Reduced orbital angular momentum magnitude, must be > 0.
    s1 : float
        Reduced spin magnitude of the primary, must be > 0.
    s2 : float
        Reduced spin magnitude of the secondary, must be > 0.
    big_sigma1 : float
        Conserved quantity Sigma1.
    big_sigma2 : float
        Conserved quantity Sigma2.

    Returns
    -------
    angles : np.ndarray
        Array of shape (2, N): row 0 holds cos kappa2, row 1 holds
        cos gamma.

    Raises
    ------
    ValueError
        If cos_kappa1 is not a nonempty 1-D array with finite entries in
        [-1, 1]; if the masses do not satisfy m1 > m2 > 0; or if l, s1
        or s2 <= 0.
    '''
    return None  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_companion_angles(cos_kappa1: np.ndarray, m1: float, m2: float,
                             l: float, s1: float, s2: float,
                             big_sigma1: float, big_sigma2: float) -> np.ndarray:
    x = np.asarray(cos_kappa1, dtype=float)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("cos_kappa1 must be a nonempty 1-D array")
    if not np.all(np.isfinite(x)) or np.any(np.abs(x) > 1.0):
        raise ValueError("cos_kappa1 entries must be finite with |x| <= 1")
    vals = [m1, m2, l, s1, s2, big_sigma1, big_sigma2]
    if not all(np.isscalar(v) and np.isfinite(float(v)) for v in vals):
        raise ValueError("all parameters must be finite scalars")
    m1, m2, l = float(m1), float(m2), float(l)
    s1, s2 = float(s1), float(s2)
    sg1, sg2 = float(big_sigma1), float(big_sigma2)
    if not (m1 > m2 > 0.0):
        raise ValueError("masses must satisfy m1 > m2 > 0")
    if not (l > 0.0 and s1 > 0.0 and s2 > 0.0):
        raise ValueError("l, s1 and s2 must be > 0")
    ck2 = sg2 - (m2 / m1) * (s1 / s2) * x
    cg = sg1 - (m1 - m2) / m1 * (l / s2) * x
    return np.vstack([ck2, cg])

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np 

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: nutation band of the task configuration ---
        {
            "setup": ("import numpy as np\n"
                      "x = np.linspace(0.84123286, 0.93672625, 101)\n"
                      "args = (2.0/3.0, 1.0/3.0, 8.965, 1.8, 0.4,\n"
                      "        10.09122436, 2.04728098)\n"),
            "call": "companion_angles(x, *args)",
            "gold_call": "_oracle_companion_angles(x, *args)",
        },
        # --- Normal: alternative configuration ---
        {
            "setup": ("import numpy as np\n"
                      "x = np.linspace(-0.2, 0.6, 50)\n"
                      "args = (0.8, 0.2, 12.0, 2.0, 0.075, 60.34202014, 4.04044011)\n"),
            "call": "companion_angles(x, *args)",
            "gold_call": "_oracle_companion_angles(x, *args)",
        },
        # --- Boundary: single value at a band edge ---
        {
            "setup": ("import numpy as np\n"
                      "x = np.array([0.84123286])\n"
                      "args = (2.0/3.0, 1.0/3.0, 8.965, 1.8, 0.4,\n"
                      "        10.09122436, 2.04728098)\n"),
            "call": "companion_angles(x, *args)",
            "gold_call": "_oracle_companion_angles(x, *args)",
        },
        # --- Edge: out-of-range cosine must raise ValueError ---
        {
            "setup": """import numpy as np
x = np.array([0.5, 1.2])
def run_model():
    try:
        companion_angles(x, 2.0/3.0, 1.0/3.0, 8.965, 1.8, 0.4, 10.0, 2.0)
        return 0
    except ValueError:
        return 1
def run_gold():
    try:
        _oracle_companion_angles(x, 2.0/3.0, 1.0/3.0, 8.965, 1.8, 0.4, 10.0, 2.0)
        return 0
    except ValueError:
        return 1
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Edge: equal masses must raise ValueError ---
        {
            "setup": """import numpy as np
x = np.array([0.5])
def run_model():
    try:
        companion_angles(x, 0.5, 0.5, 8.965, 1.8, 0.4, 10.0, 2.0)
        return 0
    except ValueError:
        return 1
def run_gold():
    try:
        _oracle_companion_angles(x, 0.5, 0.5, 8.965, 1.8, 0.4, 10.0, 2.0)
        return 0
    except ValueError:
        return 1
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
