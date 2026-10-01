"""
Coefficients of the azimuthal precession equations

In the inertial frame anchored to the conserved total angular momentum $\mathbf{j}$, the azimuthal angles of $\mathbf{l}$ and $\mathbf{s}_1$ obey

$$\frac{d\phi}{dt} = \frac{1}{c^2 d^3}

\left[\frac{\beta_1}{\alpha_1 + \cos\kappa_1}

    - \frac{\beta_2}{\alpha_2 + \cos\kappa_1} + \beta_3\right],$$

with one coefficient set for $\mathbf{l}$ and one for $\mathbf{s}_1$ (all in reduced variables; $m = m_1 + m_2$, $\mu = m_1 m_2/m$, $\nu = \mu/m$, $\delta_a = 2\nu(1 + 3 m_b/(4 m_a))$). L-sector:




$$\alpha_{1L} = \frac{m_1}{m_1 - m_2}\,\frac{j + l + s_2 \Sigma_2}{s_1},

\qquad \beta_{3L} = \frac{j \nu}{2},$$




$$\beta_{1L} = \frac{3 m_1}{4 (m_1^2 - m_2^2)\, s_1}\,(1 - \lambda)

\left[m_1 j^2 + m_2 l^2 + (m_1 + m_2)\, j l

      - (m_1 - m_2)\, s_1 (s_1 + s_2 \Sigma_1)

      + (m_2 l + m_1 j)\, s_2 \Sigma_2\right].$$




$S_1$-sector:




$$\alpha_{1S_1} = \frac{m_1}{m_2}\,\frac{1}{l}\,(-j + s_1 + s_2 \Sigma_1),

\qquad \beta_{3S_1} = \left(\delta_2 - \frac{3 \mu \lambda}{2 m_2}\right) j,$$




$$\beta_{1S_1} = \frac{m_1}{4 m_2 l}\Bigg[

  \left(2\delta_2 - \frac{3\mu\lambda}{m_2}\right) j^2

- \left(2\delta_1 - \frac{3\mu\lambda}{m_1}\right) l^2

- \nu\left(s_1^2 + s_2^2 + 2 s_1 s_2 \Sigma_1 + 2 l s_2 \Sigma_2\right)

- \frac{3 (2 m_1 - m_2)}{m_1 + m_2}\,(1 - \lambda)\, j s_1

+ \frac{3 (m_1^2 - m_2^2)}{(m_1 + m_2)^2}\,(1 - \lambda)

  \left(s_1^2 + s_1 s_2 \Sigma_1\right)

- \frac{3 (1 - \lambda)}{m_1 + m_2}\,

  \left(m_1 j \Sigma_1 + m_2 l \Sigma_2\right) s_2\Bigg].$$




The second members of each set $(\alpha_2, \beta_2)$ follow from the first by the substitution $j \to -j$.

Returns
-------
np.ndarray of shape (10,): [alpha1L, beta1L, alpha2L, beta2L, beta3L, alpha1S1, beta1S1, alpha2S1, beta2S1, beta3S1]
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def azimuthal_coefficients(m1: float, m2: float, l: float, s1: float,
                           s2: float, j: float, lam: float,
                           big_sigma1: float, big_sigma2: float) -> np.ndarray:
    '''Coefficient sets of the azimuthal equations for l and s1.

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
    j : float
        Magnitude of the total reduced angular momentum, must be > 0.
    lam : float
        Conserved quantity lambda.
    big_sigma1 : float
        Conserved quantity Sigma1.
    big_sigma2 : float
        Conserved quantity Sigma2.

    Returns
    -------
    coeffs : np.ndarray
        Array of shape (10,):
        [alpha1L, beta1L, alpha2L, beta2L, beta3L,
         alpha1S1, beta1S1, alpha2S1, beta2S1, beta3S1].

    Raises
    ------
    ValueError
        If any input is not a finite scalar; if the masses do not satisfy
        m1 > m2 > 0; or if l, s1, s2 or j <= 0.
    '''
    return None  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_azimuthal_coefficients(m1: float, m2: float, l: float, s1: float,
                                   s2: float, j: float, lam: float,
                                   big_sigma1: float,
                                   big_sigma2: float) -> np.ndarray:
    vals = [m1, m2, l, s1, s2, j, lam, big_sigma1, big_sigma2]
    if not all(np.isscalar(v) and np.isfinite(float(v)) for v in vals):
        raise ValueError("all inputs must be finite scalars")
    m1, m2, l = float(m1), float(m2), float(l)
    s1, s2, j = float(s1), float(s2), float(j)
    lam, sg1, sg2 = float(lam), float(big_sigma1), float(big_sigma2)
    if not (m1 > m2 > 0.0):
        raise ValueError("masses must satisfy m1 > m2 > 0")
    if not (l > 0.0 and s1 > 0.0 and s2 > 0.0 and j > 0.0):
        raise ValueError("l, s1, s2 and j must be > 0")
    m = m1 + m2
    mu = m1 * m2 / m
    nu = mu / m
    delta1 = 2.0 * nu * (1.0 + 3.0 * m2 / (4.0 * m1))
    delta2 = 2.0 * nu * (1.0 + 3.0 * m1 / (4.0 * m2))

    def l_sector(jj):
        a1 = m1 / (m1 - m2) * (jj + l + s2 * sg2) / s1
        b1 = (3.0 * m1 / (4.0 * (m1 ** 2 - m2 ** 2)) / s1 * (1.0 - lam)
              * (m1 * jj ** 2 + m2 * l ** 2 + (m1 + m2) * jj * l
                 - (m1 - m2) * s1 * (s1 + s2 * sg1)
                 + (m2 * l + m1 * jj) * s2 * sg2))
        return a1, b1

    def s1_sector(jj):
        a1 = m1 / m2 / l * (-jj + s1 + s2 * sg1)
        b1 = (m1 / (4.0 * m2) / l
              * ((2.0 * delta2 - 3.0 * mu / m2 * lam) * jj ** 2
                 - (2.0 * delta1 - 3.0 * mu / m1 * lam) * l ** 2
                 - nu * (s1 ** 2 + s2 ** 2 + 2.0 * s1 * s2 * sg1
                         + 2.0 * l * s2 * sg2)
                 - 3.0 * ((2.0 * m1 - m2) / (m1 + m2)) * (1.0 - lam) * jj * s1
                 + 3.0 * (m1 ** 2 - m2 ** 2) / (m1 + m2) ** 2 * (1.0 - lam)
                 * (s1 ** 2 + s1 * s2 * sg1)
                 - 3.0 * ((1.0 - lam) / (m1 + m2))
                 * (m1 * jj * sg1 + m2 * l * sg2) * s2))
        return a1, b1

    a1l, b1l = l_sector(j)
    a2l, b2l = l_sector(-j)
    b3l = j * nu / 2.0
    a1s, b1s = s1_sector(j)
    a2s, b2s = s1_sector(-j)
    b3s = (delta2 - 3.0 * mu / (2.0 * m2) * lam) * j
    return np.array([a1l, b1l, a2l, b2l, b3l, a1s, b1s, a2s, b2s, b3s],
                    dtype=float)

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
                      "args = (2.0/3.0, 1.0/3.0, 8.965, 1.8, 0.4, 10.62947280,\n"
                      "        0.06089683, 10.09122436, 2.04728098)\n"),
            "call": "azimuthal_coefficients(*args)",
            "gold_call": "_oracle_azimuthal_coefficients(*args)",
        },
        # --- Normal: alternative binary configuration ---
        {
            "setup": ("import numpy as np\n"
                      "args = (0.8, 0.2, 12.0, 2.0, 0.075, 13.16742280,\n"
                      "        0.02020220, 60.34202014, 4.04044011)\n"),
            "call": "azimuthal_coefficients(*args)",
            "gold_call": "_oracle_azimuthal_coefficients(*args)",
        },
        # --- Boundary: mild mass asymmetry (large 1/(m1-m2) prefactors) ---
        {
            "setup": ("import numpy as np\n"
                      "args = (0.52, 0.48, 9.0, 0.55, 0.45, 9.8,\n"
                      "        0.05, 1.4, 1.1)\n"),
            "call": "azimuthal_coefficients(*args)",
            "gold_call": "_oracle_azimuthal_coefficients(*args)",
        },
        # --- Edge: equal masses must raise ValueError ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        azimuthal_coefficients(0.5, 0.5, 8.965, 1.8, 0.4, 10.6, 0.06, 10.0, 2.0)
        return 0
    except ValueError:
        return 1
def run_gold():
    try:
        _oracle_azimuthal_coefficients(0.5, 0.5, 8.965, 1.8, 0.4, 10.6, 0.06, 10.0, 2.0)
        return 0
    except ValueError:
        return 1
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Edge: nonpositive total angular momentum must raise ValueError ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        azimuthal_coefficients(2.0/3.0, 1.0/3.0, 8.965, 1.8, 0.4, 0.0, 0.06, 10.0, 2.0)
        return 0
    except ValueError:
        return 1
def run_gold():
    try:
        _oracle_azimuthal_coefficients(2.0/3.0, 1.0/3.0, 8.965, 1.8, 0.4, 0.0, 0.06, 10.0, 2.0)
        return 0
    except ValueError:
        return 1
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
