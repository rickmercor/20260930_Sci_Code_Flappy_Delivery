"""
Mass parameters, reduced spins, and conserved invariants.

The orbit-averaged 2PN spin dynamics of a black-hole binary is integrable because, on top of the conserved magnitudes $(l, s_1, s_2)$ and the total angular momentum $\mathbf{j} = \mathbf{l} + \mathbf{s}_1 + \mathbf{s}_2$, orbit averaging produces additional constants of motion. Working in reduced variables ($G = 1$, masses $m_1 > m_2$, total mass $m = m_1 + m_2$, reduced mass $\mu = m_1 m_2/m$, symmetric mass ratio $\nu = \mu/m$), the spins are $s_a = \chi_a m_a^2/(\mu m c)$, and the mass-weighted couplings are $\delta_a = 2\nu(1 + \frac{3 m_b}{4 m_a})$, $\sigma_a = \nu(1 + \frac{m_b}{m_a})$, with $(a, b) \in \{(1, 2), (2, 1)\}$.  With $\mathbf{s}_0 = \sigma_1 \mathbf{s}_1 + \sigma_2 \mathbf{s}_2$, the orbit-averaged system conserves $\lambda = \frac{\mathbf{l}\cdot\mathbf{s}_0}{l^2} = \frac{\sigma_1 s_1 \cos\kappa_1 + \sigma_2 s_2 \cos\kappa_2}{l}$, $\Sigma_1 = \cos\gamma + \frac{m_1 - m_2}{m_1}\,\frac{l}{s_2}\cos\kappa_1$, $\Sigma_2 = \cos\kappa_2 + \frac{m_2}{m_1}\,\frac{s_1}{s_2}\cos\kappa_1$, where $\kappa_a$ is the angle between $\mathbf{l}$ and $\mathbf{s}_a$ and $\gamma$ the angle between the two spins. The magnitude of the total reduced angular momentum follows from the initial angles: $j^2 = l^2 + s_1^2 + s_2^2 + 2 l s_1 \cos\kappa_1 + 2 l s_2 \cos\kappa_2 + 2 s_1 s_2 \cos\gamma$.

Returns
-------
np.ndarray of shape (12,): [nu, mu, delta1, delta2, sigma1, sigma2, s1, s2, j, lambda, Sigma1, Sigma2]
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def binary_invariants(m1: float, m2: float, chi1: float, chi2: float,
                      l: float, kappa1_0: float, kappa2_0: float,
                      gamma_0: float, c: float = 1.0) -> np.ndarray:
    '''Mass parameters, reduced spin magnitudes, and conserved invariants.

    Parameters
    ----------
    m1 : float
        Primary mass, must satisfy m1 > m2 > 0.
    m2 : float
        Secondary mass.
    chi1 : float
        Dimensionless spin magnitude of the primary, in (0, 1].
    chi2 : float
        Dimensionless spin magnitude of the secondary, in (0, 1].
    l : float
        Reduced orbital angular momentum magnitude, must be > 0.
    kappa1_0 : float
        Initial angle between l and s1 in radians, in (0, pi).
    kappa2_0 : float
        Initial angle between l and s2 in radians, in (0, pi).
    gamma_0 : float
        Initial angle between s1 and s2 in radians, in (0, pi).
    c : float
        Speed of light in reduced units, must be > 0.

    Returns
    -------
    inv : np.ndarray
        Array of shape (12,):
        [nu, mu, delta1, delta2, sigma1, sigma2, s1, s2, j, lam,
         Sigma1, Sigma2].

    Raises
    ------
    ValueError
        If any input is not a finite scalar; if the masses do not satisfy
        m1 > m2 > 0; if chi1 or chi2 lies outside (0, 1] (both spins must be
        nonvanishing for the conserved-quantity construction); if l <= 0 or
        c <= 0; or if any initial angle lies outside (0, pi).
    '''
    return None  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_binary_invariants(m1: float, m2: float, chi1: float, chi2: float,
                              l: float, kappa1_0: float, kappa2_0: float,
                              gamma_0: float, c: float = 1.0) -> np.ndarray:
    vals = [m1, m2, chi1, chi2, l, kappa1_0, kappa2_0, gamma_0, c]
    if not all(np.isscalar(v) and np.isfinite(float(v)) for v in vals):
        raise ValueError("all inputs must be finite scalars")
    m1, m2, chi1, chi2 = float(m1), float(m2), float(chi1), float(chi2)
    l, c = float(l), float(c)
    k1, k2, g = float(kappa1_0), float(kappa2_0), float(gamma_0)
    if not (m1 > m2 > 0.0):
        raise ValueError("masses must satisfy m1 > m2 > 0")
    if not (0.0 < chi1 <= 1.0 and 0.0 < chi2 <= 1.0):
        raise ValueError("chi1 and chi2 must lie in (0, 1]")
    if not (l > 0.0 and c > 0.0):
        raise ValueError("l and c must be > 0")
    for ang in (k1, k2, g):
        if not (0.0 < ang < np.pi):
            raise ValueError("initial angles must lie in (0, pi)")
    m = m1 + m2
    mu = m1 * m2 / m
    nu = mu / m
    delta1 = 2.0 * nu * (1.0 + 3.0 * m2 / (4.0 * m1))
    delta2 = 2.0 * nu * (1.0 + 3.0 * m1 / (4.0 * m2))
    sigma1 = nu * (1.0 + m2 / m1)
    sigma2 = nu * (1.0 + m1 / m2)
    s1 = chi1 * m1 ** 2 / (mu * m * c)
    s2 = chi2 * m2 ** 2 / (mu * m * c)
    ck1, ck2, cg = np.cos(k1), np.cos(k2), np.cos(g)
    lam = (sigma1 * s1 * ck1 + sigma2 * s2 * ck2) / l
    big1 = cg + (m1 - m2) / m1 * (l / s2) * ck1
    big2 = ck2 + (m2 / m1) * (s1 / s2) * ck1
    j2 = (l ** 2 + s1 ** 2 + s2 ** 2 + 2.0 * l * s1 * ck1
          + 2.0 * l * s2 * ck2 + 2.0 * s1 * s2 * cg)
    return np.array([nu, mu, delta1, delta2, sigma1, sigma2, s1, s2,
                     np.sqrt(j2), lam, big1, big2], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Edge: vanishing secondary spin must raise ValueError ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        binary_invariants(2.0/3.0, 1.0/3.0, 0.9, 0.0, 8.965,
                          np.deg2rad(32.0), np.deg2rad(82.0), np.deg2rad(54.0))
        return 0
    except ValueError:
        return 1
def run_gold():
    try:
        _oracle_binary_invariants(2.0/3.0, 1.0/3.0, 0.9, 0.0, 8.965,
                                  np.deg2rad(32.0), np.deg2rad(82.0), np.deg2rad(54.0))
        return 0
    except ValueError:
        return 1
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Normal: the task configuration ---
        {
            "setup": ("import numpy as np\n"
                      "args = (2.0/3.0, 1.0/3.0, 0.9, 0.8, 8.965,\n"
                      "        np.deg2rad(32.0), np.deg2rad(82.0), np.deg2rad(54.0), 1.0)\n"),
            "call": "binary_invariants(*args)",
            "gold_call": "_oracle_binary_invariants(*args)",
        },
        # --- Normal: different mass ratio and spins ---
        {
            "setup": ("import numpy as np\n"
                      "args = (0.8, 0.2, 0.5, 0.3, 12.0,\n"
                      "        np.deg2rad(60.0), np.deg2rad(45.0), np.deg2rad(70.0), 1.0)\n"),
            "call": "binary_invariants(*args)",
            "gold_call": "_oracle_binary_invariants(*args)",
        },
        # --- Boundary: near-maximal spin, near-planar angle ---
        {
            "setup": ("import numpy as np\n"
                      "args = (0.6, 0.4, 1.0, 0.05, 9.5,\n"
                      "        np.deg2rad(89.0), np.deg2rad(150.0), np.deg2rad(120.0), 1.0)\n"),
            "call": "binary_invariants(*args)",
            "gold_call": "_oracle_binary_invariants(*args)",
        },
        # --- Edge: equal masses must raise ValueError ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        binary_invariants(0.5, 0.5, 0.9, 0.8, 8.965, 0.5, 1.0, 1.0)
        return 0
    except ValueError:
        return 1
def run_gold():
    try:
        _oracle_binary_invariants(0.5, 0.5, 0.9, 0.8, 8.965, 0.5, 1.0, 1.0)
        return 0
    except ValueError:
        return 1
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Edge: spin magnitude above the Kerr bound must raise ValueError ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        binary_invariants(2.0/3.0, 1.0/3.0, 1.2, 0.8, 8.965, 0.5, 1.0, 1.0)
        return 0
    except ValueError:
        return 1
def run_gold():
    try:
        _oracle_binary_invariants(2.0/3.0, 1.0/3.0, 1.2, 0.8, 8.965, 0.5, 1.0, 1.0)
        return 0
    except ValueError:
        return 1
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
