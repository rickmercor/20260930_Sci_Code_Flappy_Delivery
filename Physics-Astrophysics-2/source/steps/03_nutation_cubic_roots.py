"""
Nutation cubic and its ordered roots.

The single dynamical degree of freedom of the orbit-averaged spin system is $x = \cos\kappa_1$. The squared triple product obeys $[\frac{\mathbf{l}\cdot(\mathbf{s}_1\times\mathbf{s}_2)}{l s_1}]^2 = A_3 x^3 + A_2 x^2 + A_1 x + A_0$, with coefficients built from the invariants (all in reduced variables, $m = m_1 + m_2$):

$A_3 = \frac{2 (m_1 - m_2)\, m_2}{m_1^2}\, l s_1$,

$A_2 = -\frac{1}{m_1^2}[(m_1 - m_2)^2 l^2 + m_2^2 s_1^2 + m_1^2 s_2^2 + 2 m_1 m_2 s_1 s_2 \Sigma_1 + 2 m_1 (m_1 - m_2)\, l s_2 \Sigma_2]$,

$A_1 = \frac{2 s_2}{m_1}[(m_1 - m_2)\, l\, \Sigma_1 + (m_2 s_1 + m_1 s_2 \Sigma_1)\, \Sigma_2]$,

$A_0 = (1 - \Sigma_1^2 - \Sigma_2^2)\, s_2^2$.

The cubic factorizes as $A (x - x_-)(x - x_+)(x - x_3)$ with real roots ordered $x_- < x_+ < x_3$, the physical nutation being confined to $x_- \le x \le x_+$, and $A = \frac{9}{2}\,\frac{m_2 (m_1 - m_2)}{m^2}\,(1 - \lambda)^2\, l s_1 > 0$.

Returns
-------
np.ndarray of shape (4,): [x_minus, x_plus, x_3, A]
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def nutation_cubic_roots(m1: float, m2: float, l: float, s1: float,
                         s2: float, lam: float, big_sigma1: float,
                         big_sigma2: float) -> np.ndarray:
    '''Ordered roots of the nutation cubic and its overall factor A.

    Parameters
    ----------
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
    lam : float
        Conserved quantity lambda = (l . s0) / l^2, must satisfy lam != 1.
    big_sigma1 : float
        Conserved quantity Sigma1.
    big_sigma2 : float
        Conserved quantity Sigma2.

    Returns
    -------
    out : np.ndarray
        Array of shape (4,): [x_minus, x_plus, x_3, A] with
        x_minus < x_plus < x_3 and A > 0.

    Raises
    ------
    ValueError
        If any input is not a finite scalar; if the masses do not satisfy
        m1 > m2 > 0; if l, s1 or s2 <= 0; if lam == 1; or if the cubic
        does not have three real roots.
    '''
    return None  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_nutation_cubic_roots(m1: float, m2: float, l: float, s1: float,
                                 s2: float, lam: float, big_sigma1: float,
                                 big_sigma2: float) -> np.ndarray:
    vals = [m1, m2, l, s1, s2, lam, big_sigma1, big_sigma2]
    if not all(np.isscalar(v) and np.isfinite(float(v)) for v in vals):
        raise ValueError("all inputs must be finite scalars")
    m1, m2, l, s1, s2 = float(m1), float(m2), float(l), float(s1), float(s2)
    lam, sg1, sg2 = float(lam), float(big_sigma1), float(big_sigma2)
    if not (m1 > m2 > 0.0):
        raise ValueError("masses must satisfy m1 > m2 > 0")
    if not (l > 0.0 and s1 > 0.0 and s2 > 0.0):
        raise ValueError("l, s1 and s2 must be > 0")
    if lam == 1.0:
        raise ValueError("lambda must differ from 1")
    m = m1 + m2
    a3 = 2.0 * (m1 - m2) * m2 / m1 ** 2 * l * s1
    a2 = -(1.0 / m1 ** 2) * ((m1 - m2) ** 2 * l ** 2 + m2 ** 2 * s1 ** 2
                             + m1 ** 2 * s2 ** 2 + 2.0 * m1 * m2 * s1 * s2 * sg1
                             + 2.0 * m1 * (m1 - m2) * l * s2 * sg2)
    a1 = (2.0 * s2 / m1) * ((m1 - m2) * l * sg1 + (m2 * s1 + m1 * s2 * sg1) * sg2)
    a0 = (1.0 - sg1 ** 2 - sg2 ** 2) * s2 ** 2
    roots = np.roots([a3, a2, a1, a0])
    scale = max(1.0, np.max(np.abs(roots.real)))
    if np.max(np.abs(roots.imag)) > 1e-9 * scale:
        raise ValueError("the nutation cubic must have three real roots")
    roots = np.sort(roots.real)
    big_a = 4.5 * m2 * (m1 - m2) / m ** 2 * (1.0 - lam) ** 2 * l * s1
    return np.array([roots[0], roots[1], roots[2], big_a], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: invariants of the task configuration ---
        {
            "setup": ("import numpy as np\n"
                      "args = (2.0/3.0, 1.0/3.0, 8.965, 1.8, 0.4,\n"
                      "        0.06089683, 10.09122436, 2.04728098)\n"),
            "call": "nutation_cubic_roots(*args)",
            "gold_call": "_oracle_nutation_cubic_roots(*args)",
        },
        # --- Normal: alternative binary configuration (m=1, q=0.25 invariants) ---
        {
            "setup": ("import numpy as np\n"
                      "args = (0.8, 0.2, 12.0, 2.0, 0.075,\n"
                      "        0.02020220, 60.34202014, 4.04044011)\n"),
            "call": "nutation_cubic_roots(*args)",
            "gold_call": "_oracle_nutation_cubic_roots(*args)",
        },
        # --- Boundary: mild mass asymmetry (small A3, well-separated roots) ---
        {
            "setup": ("import numpy as np\n"
                      "args = (0.52, 0.48, 9.0, 0.55, 0.45,\n"
                      "        0.05, 1.4, 1.1)\n"),
            "call": "nutation_cubic_roots(*args)",
            "gold_call": "_oracle_nutation_cubic_roots(*args)",
        },
        # --- Edge: lambda equal to 1 must raise ValueError ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        nutation_cubic_roots(2.0/3.0, 1.0/3.0, 8.965, 1.8, 0.4, 1.0, 10.0, 2.0)
        return 0
    except ValueError:
        return 1
def run_gold():
    try:
        _oracle_nutation_cubic_roots(2.0/3.0, 1.0/3.0, 8.965, 1.8, 0.4, 1.0, 10.0, 2.0)
        return 0
    except ValueError:
        return 1
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Edge: nonpositive spin magnitude must raise ValueError ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        nutation_cubic_roots(2.0/3.0, 1.0/3.0, 8.965, 0.0, 0.4, 0.06, 10.0, 2.0)
        return 0
    except ValueError:
        return 1
def run_gold():
    try:
        _oracle_nutation_cubic_roots(2.0/3.0, 1.0/3.0, 8.965, 0.0, 0.4, 0.06, 10.0, 2.0)
        return 0
    except ValueError:
        return 1
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
