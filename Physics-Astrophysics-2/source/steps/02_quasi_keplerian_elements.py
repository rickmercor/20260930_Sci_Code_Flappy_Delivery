"""
1PN quasi-Keplerian orbital elements and the averaging radius

The slow precession frequencies of the orbit-averaged spin dynamics are set by the orbit average of $1/r^3$.  Averaging the leading spin-orbit terms over the 1PN quasi-Keplerian orbit (rather than the Newtonian one) improves the secular accuracy. From the reduced orbital energy $h < 0$ and angular momentum $l$, the 1PN quasi-Keplerian elements are $a_r = -\frac{1}{2h}[1 - (\nu - 7)\frac{h}{2c^2}]$, $e_r^2 = 1 + 2hl^2 - 2(6 - \nu)\frac{h}{c^2} - 5(3 - \nu)\frac{h^2 l^2}{c^2}$, $n = (-2h)^{3/2}[1 + (15 - \nu)\frac{h}{4c^2}]$, $e_t^2 = 1 + 2hl^2 + 4(1 - \nu)\frac{h}{c^2} + (17 - 7\nu)\frac{h^2 l^2}{c^2}$, and the orbit average $\langle 1/r^3 \rangle = 1/d^3$ is expressed through $e_\theta = \frac{3 e_r - e_t}{2}$, $d = a_r\sqrt{1 - e_\theta^2}$.

Returns
-------
np.ndarray of shape (6,): [a_r, e_r, n, e_t, e_theta, d]
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def quasi_keplerian_elements(h: float, l: float, nu: float,
                             c: float = 1.0) -> np.ndarray:
    '''1PN quasi-Keplerian elements from reduced energy and angular momentum.

    Parameters
    ----------
    h : float
        Reduced orbital energy, must be < 0 (bound orbit).
    l : float
        Reduced orbital angular momentum magnitude, must be > 0.
    nu : float
        Symmetric mass ratio, in (0, 0.25].
    c : float
        Speed of light in reduced units, must be > 0.

    Returns
    -------
    elements : np.ndarray
        Array of shape (6,): [a_r, e_r, n, e_t, e_theta, d].

    Raises
    ------
    ValueError
        If any input is not a finite scalar; if h >= 0; if l <= 0 or
        c <= 0; if nu lies outside (0, 0.25]; or if (h, l) do not
        describe an eccentric bound orbit (e_r^2 <= 0, e_t^2 <= 0, or
        |e_theta| >= 1).
    '''
    return None  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_quasi_keplerian_elements(h: float, l: float, nu: float,
                                     c: float = 1.0) -> np.ndarray:
    vals = [h, l, nu, c]
    if not all(np.isscalar(v) and np.isfinite(float(v)) for v in vals):
        raise ValueError("all inputs must be finite scalars")
    h, l, nu, c = float(h), float(l), float(nu), float(c)
    if not (h < 0.0):
        raise ValueError("h must be < 0 for a bound orbit")
    if not (l > 0.0 and c > 0.0):
        raise ValueError("l and c must be > 0")
    if not (0.0 < nu <= 0.25):
        raise ValueError("nu must lie in (0, 0.25]")
    a_r = -1.0 / (2.0 * h) * (1.0 - (nu - 7.0) * h / (2.0 * c ** 2))
    er2 = (1.0 + 2.0 * h * l ** 2 - 2.0 * (6.0 - nu) * h / c ** 2
           - 5.0 * (3.0 - nu) * h ** 2 * l ** 2 / c ** 2)
    n = (-2.0 * h) ** 1.5 * (1.0 + (15.0 - nu) * h / (4.0 * c ** 2))
    et2 = (1.0 + 2.0 * h * l ** 2 + 4.0 * (1.0 - nu) * h / c ** 2
           + (17.0 - 7.0 * nu) * h ** 2 * l ** 2 / c ** 2)
    if er2 <= 0.0 or et2 <= 0.0:
        raise ValueError("h and l must describe an eccentric bound orbit "
                         "with e_r^2 > 0 and e_t^2 > 0")
    e_r, e_t = np.sqrt(er2), np.sqrt(et2)
    e_theta = (3.0 * e_r - e_t) / 2.0
    if not (abs(e_theta) < 1.0):
        raise ValueError("e_theta must satisfy |e_theta| < 1")
    d = a_r * np.sqrt(1.0 - e_theta ** 2)
    return np.array([a_r, e_r, n, e_t, e_theta, d], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: the task configuration ---
        {
            "setup": "import numpy as np\nh, l, nu = -1.0/256.0, 8.965, 2.0/9.0\n",
            "call": "quasi_keplerian_elements(h, l, nu)",
            "gold_call": "_oracle_quasi_keplerian_elements(h, l, nu)",
        },
        # --- Normal: tighter, more relativistic orbit ---
        {
            "setup": "import numpy as np\nh, l, nu = -1.0/64.0, 5.0, 0.16\n",
            "call": "quasi_keplerian_elements(h, l, nu)",
            "gold_call": "_oracle_quasi_keplerian_elements(h, l, nu)",
        },
        # --- Boundary: near-circular orbit (small eccentricity) ---
        {
            "setup": "import numpy as np\nh, l = -1.0/256.0, 11.28\nnu = 0.25\n",
            "call": "quasi_keplerian_elements(h, l, nu)",
            "gold_call": "_oracle_quasi_keplerian_elements(h, l, nu)",
        },
        # --- Edge: unbound orbit must raise ValueError ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        quasi_keplerian_elements(0.01, 8.965, 2.0/9.0)
        return 0
    except ValueError:
        return 1
def run_gold():
    try:
        _oracle_quasi_keplerian_elements(0.01, 8.965, 2.0/9.0)
        return 0
    except ValueError:
        return 1
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Edge: invalid symmetric mass ratio must raise ValueError ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        quasi_keplerian_elements(-1.0/256.0, 8.965, 0.3)
        return 0
    except ValueError:
        return 1
def run_gold():
    try:
        _oracle_quasi_keplerian_elements(-1.0/256.0, 8.965, 0.3)
        return 0
    except ValueError:
        return 1
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
