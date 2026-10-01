"""
Jacobi-elliptic nutation angle under the hybrid evolution

The hybrid model evolves the spins with the instantaneous $1/r^3$ of the 1PN quasi-Keplerian orbit rather than its orbit average, which replaces the orbit-averaged linear phase with an orbit-modulated accumulation. The spin-orbit angle follows the closed-form nutation law 

$\cos\kappa_1(t) = x_- + (x_+ - x_-)\,\mathrm{sn}^2(Υ(t), \beta)$, $\beta = \sqrt{\frac{x_+ - x_-}{x_3 - x_-}}$, 

with the hybrid phase argument $Υ(t) = \frac{\sqrt{A (x_3 - x_-)}}{2}[\alpha + \frac{1}{c^2 d^3}\,\frac{v_\theta + e_\theta \sin v_\theta}{n}]$, where $v_\theta$ is the continuous angular anomaly of the orbit and the phase constant carries the initial condition, $\alpha = \mathrm{sign}_0\,\frac{2}{\sqrt{A (x_3 - x_-)}}F(\arcsin\sqrt{\frac{x_0 - x_-}{x_+ - x_-}},\ \beta)$, with $F(\phi, k)$ the incomplete elliptic integral of the first kind of modulus $k = \beta$ (parameter $m = k^2$), $x_0 = \cos\kappa_1$ at $t = 0$, and $\mathrm{sign}_0 = +1$ if $d\cos\kappa_1/dt > 0$ at $t = 0$ and $-1$ otherwise.

Returns
-------
np.ndarray, same shape as t_grid, the nutation cosine cos kappa1(t) under the hybrid evolution
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def nutation_angle(t_grid: np.ndarray, x_minus: float, x_plus: float,
                   x_3: float, big_a: float, d: float, n: float,
                   e_t: float, e_theta: float, x0: float,
                   sign0: float, c: float = 1.0) -> np.ndarray:
    '''Nutation cosine cos kappa1(t) under the hybrid evolution.

    Parameters
    ----------
    t_grid : np.ndarray
        1-D array of reduced times, entries finite.
    x_minus, x_plus, x_3 : float
        Ordered nutation-cubic roots, x_minus < x_plus < x_3.
    big_a : float
        Overall cubic factor A, must be > 0.
    d : float
        1PN averaging radius, must be > 0.
    n : float
        1PN mean motion, must be > 0.
    e_t : float
        Time eccentricity of the Kepler equation, in [0, 1).
    e_theta : float
        Angular eccentricity of the anomaly and its modulation, in [0, 1).
    x0 : float
        Initial value cos kappa1(0), in [x_minus, x_plus].
    sign0 : float
        Sign of d cos kappa1 / dt at t = 0; must be +1.0 or -1.0.
    c : float
        Speed of light in reduced units, must be > 0.

    Returns
    -------
    cos_kappa1 : np.ndarray
        1-D array, same shape as t_grid, the values of cos kappa1(t).

    Raises
    ------
    ValueError
        If t_grid is not a nonempty 1-D array of finite times; if the
        roots do not satisfy x_minus < x_plus < x_3; if big_a, d, n or
        c <= 0; if e_t or e_theta lies outside [0, 1); if x0 lies
        outside [x_minus, x_plus]; or if sign0 is not +1.0 or -1.0.
    '''
    return None  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.special import ellipj, ellipkinc

def _validate_hybrid_args(t, x_minus, x_plus, x_3, big_a, d, n, e_t,
                          e_theta, x0, sign0, c):
    if t.ndim != 1 or t.size == 0 or not np.all(np.isfinite(t)):
        raise ValueError("t_grid must be a nonempty 1-D array of finite times")
    vals = [x_minus, x_plus, x_3, big_a, d, n, e_t, e_theta, x0, sign0, c]
    if not all(np.isscalar(v) and np.isfinite(float(v)) for v in vals):
        raise ValueError("all parameters must be finite scalars")
    if not (float(x_minus) < float(x_plus) < float(x_3)):
        raise ValueError("roots must satisfy x_minus < x_plus < x_3")
    if not (float(big_a) > 0.0 and float(d) > 0.0 and float(n) > 0.0
            and float(c) > 0.0):
        raise ValueError("big_a, d, n and c must be > 0")
    if not (0.0 <= float(e_t) < 1.0 and 0.0 <= float(e_theta) < 1.0):
        raise ValueError("e_t and e_theta must lie in [0, 1)")
    if not (float(x_minus) <= float(x0) <= float(x_plus)):
        raise ValueError("x0 must lie in [x_minus, x_plus]")
    if float(sign0) not in (-1.0, 1.0):
        raise ValueError("sign0 must be +1.0 or -1.0")


def _hybrid_upsilon(t, x_minus, x_plus, x_3, big_a, d, n, e_t, e_theta,
                    x0, sign0, c):
    xm, xp, x3 = float(x_minus), float(x_plus), float(x_3)
    big_a, d, n, c = float(big_a), float(d), float(n), float(c)
    e_theta = float(e_theta)
    mpar = (xp - xm) / (x3 - xm)
    sqax = np.sqrt(big_a * (x3 - xm))
    phi0 = np.arcsin(np.sqrt((float(x0) - xm) / (xp - xm)))
    alpha0 = float(sign0) * 2.0 / sqax * ellipkinc(phi0, mpar)
    u = _oracle_kepler_eccentric_anomaly(t, n, float(e_t))
    v_th = _oracle_angular_anomaly(u, e_theta)
    return sqax / 2.0 * (alpha0 + (v_th + e_theta * np.sin(v_th))
                         / (n * c ** 2 * d ** 3))


def _oracle_nutation_angle(t_grid: np.ndarray, x_minus: float, x_plus: float,
                           x_3: float, big_a: float, d: float, n: float,
                           e_t: float, e_theta: float, x0: float,
                           sign0: float, c: float = 1.0) -> np.ndarray:
    t = np.asarray(t_grid, dtype=float)
    _validate_hybrid_args(t, x_minus, x_plus, x_3, big_a, d, n, e_t,
                          e_theta, x0, sign0, c)
    ups = _hybrid_upsilon(t, x_minus, x_plus, x_3, big_a, d, n, e_t,
                          e_theta, x0, sign0, c)
    mpar = (float(x_plus) - float(x_minus)) / (float(x_3) - float(x_minus))
    sn, _, _, _ = ellipj(ups, mpar)
    return float(x_minus) + (float(x_plus) - float(x_minus)) * sn ** 2

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: pipeline configuration over several nutation cycles ---
        {
            "setup": ("import numpy as np\n"
                      "t = np.linspace(0.0, 5.0e6, 256)\n"
                      "args = (0.84123303, 0.93672577, 2.64294205, 7.11572656,\n"
                      "        96.92799691, 6.80568578e-4, 0.61553846, 0.64115810,\n"
                      "        0.84804810, 1.0)\n"),
            "call": "nutation_angle(t, *args)",
            "gold_call": "_oracle_nutation_angle(t, *args)",
        },
        # --- Normal: negative initial slope, near-1 modulus root triple ---
        {
            "setup": ("import numpy as np\n"
                      "t = np.linspace(0.0, 8.0e5, 128)\n"
                      "args = (0.20, 0.80, 0.85, 7.11572656,\n"
                      "        96.92799691, 6.80568578e-4, 0.61553846, 0.64115810,\n"
                      "        0.55, -1.0)\n"),
            "call": "nutation_angle(t, *args)",
            "gold_call": "_oracle_nutation_angle(t, *args)",
        },
        # --- Boundary: t = 0 returns exactly x0 ---
        {
            "setup": ("import numpy as np\n"
                      "t = np.array([0.0])\n"
                      "args = (0.84123303, 0.93672577, 2.64294205, 7.11572656,\n"
                      "        96.92799691, 6.80568578e-4, 0.61553846, 0.64115810,\n"
                      "        0.90, 1.0)\n"),
            "call": "nutation_angle(t, *args)",
            "gold_call": "_oracle_nutation_angle(t, *args)",
        },
        # --- Boundary: circular orbit collapses the modulation ---
        {
            "setup": ("import numpy as np\n"
                      "t = np.linspace(0.0, 2.0e6, 64)\n"
                      "args = (0.84123303, 0.93672577, 2.64294205, 7.11572656,\n"
                      "        96.92799691, 6.80568578e-4, 0.0, 0.0,\n"
                      "        0.84123303, 1.0)\n"),
            "call": "nutation_angle(t, *args)",
            "gold_call": "_oracle_nutation_angle(t, *args)",
        },
        # --- Edge: x0 outside the nutation band must raise ValueError ---
        {
            "setup": """import numpy as np
t = np.linspace(0.0, 10.0, 5)
def run_model():
    try:
        nutation_angle(t, 0.84, 0.94, 2.64, 7.1, 96.9, 6.8e-4,
                       0.6, 0.64, 0.99, 1.0)
        return 0
    except ValueError:
        return 1
def run_gold():
    try:
        _oracle_nutation_angle(t, 0.84, 0.94, 2.64, 7.1, 96.9, 6.8e-4,
                               0.6, 0.64, 0.99, 1.0)
        return 0
    except ValueError:
        return 1
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Edge: invalid sign0 must raise ValueError ---
        {
            "setup": """import numpy as np
t = np.linspace(0.0, 10.0, 5)
def run_model():
    try:
        nutation_angle(t, 0.84, 0.94, 2.64, 7.1, 96.9, 6.8e-4,
                       0.6, 0.64, 0.85, 0.5)
        return 0
    except ValueError:
        return 1
def run_gold():
    try:
        _oracle_nutation_angle(t, 0.84, 0.94, 2.64, 7.1, 96.9, 6.8e-4,
                               0.6, 0.64, 0.85, 0.5)
        return 0
    except ValueError:
        return 1
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
