"""
Accumulated azimuthal phase under the hybrid evolution

Substituting the Jacobi-elliptic nutation solution into the azimuthal precession equation and integrating gives the accumulated phase as the difference of a primitive evaluated at the hybrid phase argument, 


$$\Delta\phi(t) = F(Υ(t)) - F(Υ(0)), \qquad

F(u) = \frac{2}{\sqrt{A (x_3 - x_-)}}

\left[\frac{\beta_1\,\Pi(n_1, \mathrm{am}(u, \beta), \beta)}{\alpha_1 + x_-}

    - \frac{\beta_2\,\Pi(n_2, \mathrm{am}(u, \beta), \beta)}{\alpha_2 + x_-}

    + \beta_3\,u\right],$$

with characteristics $n_i = (x_- - x_+)/(\alpha_i + x_-)$, $\Pi$ the Legendre incomplete elliptic integral of the third kind, and $\mathrm{am}$ the Jacobi amplitude.  The hybrid phase argument $Υ(t)$ is built exactly as in the nutation solution: the orbit-modulated accumulation $(v_\theta + e_\theta \sin v_\theta)/(c^2 d^3 n)$ added to the phase constant $\alpha$ fixed by $x_0$ and $\mathrm{sign}_0$. Because the phase argument grows without bound while the elliptic integrals are defined on a bounded angular range, the evaluation must return the continuous accumulation between the two arguments.

Returns
-------
np.ndarray, same shape as t_grid, the accumulated azimuthal phase in radians
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def azimuthal_phase(t_grid: np.ndarray, x_minus: float, x_plus: float,
                    x_3: float, big_a: float, d: float, n: float,
                    e_t: float, e_theta: float, x0: float, sign0: float,
                    alpha1: float, beta1: float, alpha2: float,
                    beta2: float, beta3: float,
                    c: float = 1.0) -> np.ndarray:
    '''Azimuthal phase accumulated since t = 0 under the hybrid evolution.

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
    alpha1, beta1, alpha2, beta2, beta3 : float
        Finite azimuthal coefficients of the vector being tracked. For
        each active term (beta_i != 0, i = 1, 2), alpha_i + x must
        be nonzero for every x in [x_minus, x_plus]. Equivalently,
        alpha_i + x_minus and alpha_i + x_plus must have the same
        strict sign. Terms with beta_i == 0 are omitted entirely
        and impose no restriction on alpha_i beyond finiteness.
    c : float
        Speed of light in reduced units, must be > 0.

    Returns
    -------
    dphi : np.ndarray
        1-D array, same shape as t_grid, the phase accumulated since
        t = 0, in radians.

    Raises
    ------
    ValueError
        If t_grid is not a nonempty 1-D array of finite times; if the
        roots do not satisfy x_minus < x_plus < x_3; if big_a, d, n or
        c <= 0; if e_t or e_theta lies outside [0, 1); if x0 lies
        outside [x_minus, x_plus]; if sign0 is not +1.0 or -1.0; or if
        any coefficient is nonfinite; or if an active denominator
        vanishes anywhere in [x_minus, x_plus].
    FloatingPointError
        If numerical evaluation cannot produce finite phase values.
    '''
    return None  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.special import ellipj, ellipk, elliprf, elliprj


def _am_continuous(u, mpar):
    """Continuous Jacobi amplitude for arbitrary real u."""
    big_k = ellipk(mpar)
    ncyc = np.floor((u + big_k) / (2.0 * big_k))
    _, _, _, ph = ellipj(u - 2.0 * big_k * ncyc, mpar)
    return ph + np.pi * ncyc


def _ellip_pi(n, phi, mpar):
    """Legendre incomplete Pi(n; phi | m) extended to arbitrary real phi."""
    ncyc = np.floor((phi + np.pi / 2.0) / np.pi)
    phir = phi - np.pi * ncyc
    sp, cp = np.sin(phir), np.cos(phir)
    y = 1.0 - mpar * sp ** 2
    inc = (sp * elliprf(cp ** 2, y, 1.0)
           + (n / 3.0) * sp ** 3 * elliprj(cp ** 2, y, 1.0, 1.0 - n * sp ** 2))
    comp = (elliprf(0.0, 1.0 - mpar, 1.0)
            + (n / 3.0) * elliprj(0.0, 1.0 - mpar, 1.0, 1.0 - n))
    return inc + 2.0 * ncyc * comp


def _oracle_azimuthal_phase(t_grid: np.ndarray, x_minus: float, x_plus: float,
                            x_3: float, big_a: float, d: float, n: float,
                            e_t: float, e_theta: float, x0: float,
                            sign0: float, alpha1: float, beta1: float,
                            alpha2: float, beta2: float, beta3: float,
                            c: float = 1.0) -> np.ndarray:
    t = np.asarray(t_grid, dtype=float)
    _validate_hybrid_args(t, x_minus, x_plus, x_3, big_a, d, n, e_t,
                          e_theta, x0, sign0, c)
    vals = [alpha1, beta1, alpha2, beta2, beta3]
    if not all(np.isscalar(v) and np.isfinite(float(v)) for v in vals):
        raise ValueError("all coefficients must be finite scalars")
    xm, xp, x3 = float(x_minus), float(x_plus), float(x_3)
    a1, b1, a2, b2, b3 = (float(alpha1), float(beta1), float(alpha2),
                          float(beta2), float(beta3))
    for alpha_i, beta_i in ((a1, b1), (a2, b2)):
        if beta_i != 0.0:
            left, right = alpha_i + xm, alpha_i + xp
            if not ((left > 0.0 and right > 0.0) or
                    (left < 0.0 and right < 0.0)):
                raise ValueError("active alpha_i + x must be nonzero "
                                 "throughout [x_minus, x_plus]")
    mpar = (xp - xm) / (x3 - xm)
    sqax = np.sqrt(float(big_a) * (x3 - xm))
    n1 = (xm - xp) / (a1 + xm) if b1 != 0.0 else 0.0
    n2 = (xm - xp) / (a2 + xm) if b2 != 0.0 else 0.0

    def primitive(u):
        ph = _am_continuous(u, mpar)
        value = b3 * u
        if b1 != 0.0:
            value = value + b1 / (a1 + xm) * _ellip_pi(n1, ph, mpar)
        if b2 != 0.0:
            value = value - b2 / (a2 + xm) * _ellip_pi(n2, ph, mpar)
        return (2.0 / sqax) * value

    ups_t = _hybrid_upsilon(t, x_minus, x_plus, x_3, big_a, d, n, e_t,
                            e_theta, x0, sign0, c)
    ups_0 = _hybrid_upsilon(np.array([0.0]), x_minus, x_plus, x_3, big_a,
                            d, n, e_t, e_theta, x0, sign0, c)
    result = primitive(ups_t) - primitive(ups_0)[0]
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("azimuthal phase evaluation is nonfinite")
    return result

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np 

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: S1-sector coefficients, pipeline configuration ---
        {
            "setup": ("import numpy as np\n"
                      "t = np.linspace(0.0, 5.0e6, 256)\n"
                      "geom = (0.84123303, 0.93672577, 2.64294205, 7.11572656,\n"
                      "        96.92799691, 6.80568578e-4, 0.61553846, 0.64115810,\n"
                      "        0.84804810, 1.0)\n"
                      "coeffs = (-16.44121026, -37.62335661, 26.08337534,\n"
                      "          71.12783064, 11.63102968)\n"),
            "call": "azimuthal_phase(t, *geom, *coeffs)",
            "gold_call": "_oracle_azimuthal_phase(t, *geom, *coeffs)",
        },
        # --- Normal: L-sector coefficients, long interval ---
        {
            "setup": ("import numpy as np\n"
                      "t = np.linspace(0.0, 6.5e6, 256)\n"
                      "geom = (0.84123303, 0.93672577, 2.64294205, 7.11572656,\n"
                      "        96.92799691, 6.80568578e-4, 0.61553846, 0.64115810,\n"
                      "        0.90, -1.0)\n"
                      "coeffs = (22.61061195, 132.36348010, -20.79993039,\n"
                      "          -106.72178398, 1.18105253)\n"),
            "call": "azimuthal_phase(t, *geom, *coeffs)",
            "gold_call": "_oracle_azimuthal_phase(t, *geom, *coeffs)",
        },
        # --- Normal: near-unit modulus with a near-pole characteristic ---
        {
            "setup": ("import numpy as np\n"
                      "t = np.linspace(0.0, 3.0e6, 200)\n"
                      "geom = (0.20, 0.80, 0.85, 7.11572656,\n"
                      "        96.92799691, 6.80568578e-4, 0.61553846, 0.64115810,\n"
                      "        0.55, 1.0)\n"
                      "coeffs = (-0.15, 2.0, 3.5, 1.2, 0.8)\n"),
            "call": "azimuthal_phase(t, *geom, *coeffs)",
            "gold_call": "_oracle_azimuthal_phase(t, *geom, *coeffs)",
        },
        # --- Normal: characteristic close to one on the second pole ---
        {
            "setup": ("import numpy as np\n"
                      "t = np.linspace(0.0, 2.0e6, 128)\n"
                      "geom = (0.84123303, 0.93672577, 2.64294205, 7.11572656,\n"
                      "        96.92799691, 6.80568578e-4, 0.61553846, 0.64115810,\n"
                      "        0.86, 1.0)\n"
                      "coeffs = (3.0, 1.0, -0.94059, 0.5, 0.2)\n"),
            "call": "azimuthal_phase(t, *geom, *coeffs)",
            "gold_call": "_oracle_azimuthal_phase(t, *geom, *coeffs)",
        },
        # --- Boundary: t = 0 gives exactly zero accumulated phase ---
        {
            "setup": ("import numpy as np\n"
                      "t = np.array([0.0])\n"
                      "geom = (0.84123303, 0.93672577, 2.64294205, 7.11572656,\n"
                      "        96.92799691, 6.80568578e-4, 0.61553846, 0.64115810,\n"
                      "        0.84804810, 1.0)\n"
                      "coeffs = (-16.44121026, -37.62335661, 26.08337534,\n"
                      "          71.12783064, 11.63102968)\n"),
            "call": "azimuthal_phase(t, *geom, *coeffs)",
            "gold_call": "_oracle_azimuthal_phase(t, *geom, *coeffs)",
        },
        # --- Edge: vanishing alpha_i + x_minus must raise ValueError ---
        {
            "setup": """import numpy as np
t = np.linspace(0.0, 10.0, 5)
def run_model():
    try:
        azimuthal_phase(t, 0.84123303, 0.93672577, 2.64294205, 7.11572656,
                        96.92799691, 6.8e-4, 0.6, 0.64, 0.85, 1.0,
                        -0.84123303, 1.0, 3.0, 1.0, 1.0)
        return 0
    except ValueError:
        return 1
def run_gold():
    try:
        _oracle_azimuthal_phase(t, 0.84123303, 0.93672577, 2.64294205,
                                7.11572656, 96.92799691, 6.8e-4, 0.6, 0.64,
                                0.85, 1.0, -0.84123303, 1.0, 3.0, 1.0, 1.0)
        return 0
    except ValueError:
        return 1
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Edge: nonpositive cubic factor must raise ValueError ---
        {
            "setup": """import numpy as np
t = np.linspace(0.0, 10.0, 5)
def run_model():
    try:
        azimuthal_phase(t, 0.84123303, 0.93672577, 2.64294205, 0.0,
                        96.92799691, 6.8e-4, 0.6, 0.64, 0.85, 1.0,
                        3.0, 1.0, -1.1, 1.0, 1.0)
        return 0
    except ValueError:
        return 1
def run_gold():
    try:
        _oracle_azimuthal_phase(t, 0.84123303, 0.93672577, 2.64294205, 0.0,
                                96.92799691, 6.8e-4, 0.6, 0.64, 0.85, 1.0,
                                3.0, 1.0, -1.1, 1.0, 1.0)
        return 0
    except ValueError:
        return 1
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Regression: reject right-endpoint poles in either active term ---
        {
            "setup": """import numpy as np
t = np.linspace(0.0, 3.0e6, 16)
geom = (0.2, 0.8, 0.85, 7.11572656, 96.92799691,
        6.80568578e-4, 0.61553846, 0.64115810, 0.55, 1.0)
coefficients = [(-0.8, 1.0, 3.5, 1.2, 0.8),
                (3.5, 1.2, -0.8, 1.0, 0.8)]
def check_rejection(solver):
    rejected = 0
    for coeffs in coefficients:
        try:
            solver(t, *geom, *coeffs)
        except ValueError:
            rejected += 1
    return int(rejected == len(coefficients))
""",
            "call": "check_rejection(azimuthal_phase)",
            "gold_call": "check_rejection(_oracle_azimuthal_phase)",
        },
        # --- Regression: reject interior poles in either active term ---
        {
            "setup": """import numpy as np
t = np.linspace(0.0, 3.0e6, 16)
geom = (0.2, 0.8, 0.85, 7.11572656, 96.92799691,
        6.80568578e-4, 0.61553846, 0.64115810, 0.55, 1.0)
coefficients = [(-0.5, 1.0, 3.5, 1.2, 0.8),
                (3.5, 1.2, -0.5, 1.0, 0.8)]
def check_rejection(solver):
    rejected = 0
    for coeffs in coefficients:
        try:
            solver(t, *geom, *coeffs)
        except ValueError:
            rejected += 1
    return int(rejected == len(coefficients))
""",
            "call": "check_rejection(azimuthal_phase)",
            "gold_call": "check_rejection(_oracle_azimuthal_phase)",
        },
        # --- Boundary: omit both zero-weight terms even with endpoint poles ---
        {
            "setup": """import numpy as np
t = np.linspace(0.0, 3.0e6, 16)
geom = (0.2, 0.8, 0.85, 7.11572656, 96.92799691,
        6.80568578e-4, 0.61553846, 0.64115810, 0.55, 1.0)
coeffs = (-0.2, 0.0, -0.8, 0.0, 0.8)
""",
            "call": "azimuthal_phase(t, *geom, *coeffs)",
            "gold_call": "_oracle_azimuthal_phase(t, *geom, *coeffs)",
        },
    ]
