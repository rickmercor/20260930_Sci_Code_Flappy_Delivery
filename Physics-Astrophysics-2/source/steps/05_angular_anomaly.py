"""
Continuous angular anomaly of the 1PN quasi-Keplerian orbit

The hybrid phase accumulation is driven by the angular anomaly built from the angular eccentricity $e_\theta$:
$v_\theta = u + 2\arctan(\frac{\beta_\theta \sin u}{1 - \beta_\theta \cos u}\right)$, $\beta_\theta = \frac{e_\theta}{1 + \sqrt{1 - e_\theta^2}}$,

with $u$ the continuous eccentric anomaly. Like $u$, the returned anomaly is a continuous accumulation over all orbits, not a per-orbit reduction.

Returns
-------
np.ndarray, same shape as u, the continuous angular anomaly v_theta(u) in radians
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def angular_anomaly(u: np.ndarray, e_theta: float) -> np.ndarray:
    '''Continuous angular anomaly v_theta from the eccentric anomaly.

    Parameters
    ----------
    u : np.ndarray
        1-D array of continuous eccentric-anomaly values in radians,
        entries finite.
    e_theta : float
        Angular eccentricity, must satisfy 0 <= e_theta < 1.

    Returns
    -------
    v_theta : np.ndarray
        1-D array, same shape as u, the continuous angular anomaly in
        radians.

    Raises
    ------
    ValueError
        If u is not a nonempty 1-D array with finite entries, or if
        e_theta is not a finite scalar in [0, 1).
    '''
    return None  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_angular_anomaly(u: np.ndarray, e_theta: float) -> np.ndarray:
    uu = np.asarray(u, dtype=float)
    if uu.ndim != 1 or uu.size == 0 or not np.all(np.isfinite(uu)):
        raise ValueError("u must be a nonempty 1-D array of finite values")
    if not (np.isscalar(e_theta) and np.isfinite(float(e_theta))):
        raise ValueError("e_theta must be a finite scalar")
    e_theta = float(e_theta)
    if not (0.0 <= e_theta < 1.0):
        raise ValueError("e_theta must lie in [0, 1)")
    b = e_theta / (1.0 + np.sqrt(1.0 - e_theta ** 2))
    return uu + 2.0 * np.arctan(b * np.sin(uu) / (1.0 - b * np.cos(uu)))

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np 

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: hundreds of orbits at the pipeline eccentricity ---
        {
            "setup": ("import numpy as np\n"
                      "u = np.linspace(0.0, 3500.0, 512)\n"),
            "call": "angular_anomaly(u, 0.64115810)",
            "gold_call": "_oracle_angular_anomaly(u, 0.64115810)",
        },
        # --- Normal: high eccentricity, negative anomalies included ---
        {
            "setup": ("import numpy as np\n"
                      "u = np.linspace(-20.0, 20.0, 257)\n"),
            "call": "angular_anomaly(u, 0.95)",
            "gold_call": "_oracle_angular_anomaly(u, 0.95)",
        },
        # --- Boundary: circular orbit gives v_theta = u exactly ---
        {
            "setup": ("import numpy as np\n"
                      "u = np.linspace(0.0, 50.0, 64)\n"),
            "call": "angular_anomaly(u, 0.0)",
            "gold_call": "_oracle_angular_anomaly(u, 0.0)",
        },
        # --- Boundary: exact periapsis passages leave the anomaly unchanged ---
        {
            "setup": ("import numpy as np\n"
                      "u = 2.0 * np.pi * np.arange(6, dtype=float)\n"),
            "call": "angular_anomaly(u, 0.61553846)",
            "gold_call": "_oracle_angular_anomaly(u, 0.61553846)",
        },
        # --- Edge: parabolic eccentricity must raise ValueError ---
        {
            "setup": """import numpy as np
u = np.linspace(0.0, 1.0, 5)
def run_model():
    try:
        angular_anomaly(u, 1.0)
        return 0
    except ValueError:
        return 1
def run_gold():
    try:
        _oracle_angular_anomaly(u, 1.0)
        return 0
    except ValueError:
        return 1
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Edge: non-finite anomaly entries must raise ValueError ---
        {
            "setup": """import numpy as np
u = np.array([0.0, np.inf, 1.0])
def run_model():
    try:
        angular_anomaly(u, 0.5)
        return 0
    except ValueError:
        return 1
def run_gold():
    try:
        _oracle_angular_anomaly(u, 0.5)
        return 0
    except ValueError:
        return 1
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
