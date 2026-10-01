"""
Determine the smallest Bernstein-ellipse radius whose ellipse encloses a supplied

set of complex points, such as a spectrum.



Inputs

------

points: array-like of complex numbers, non-empty and finite



Returns

-------

rho: float, the smallest Bernstein radius rho >= 1 enclosing every supplied point



Raises

------

ValueError: if points is empty or contains a non-finite value

Chebyshev polynomials of the first kind are bounded between -1 and 1 on the real

interval [-1, 1], but grow rapidly off it. The natural coordinates for that growth are

the Bernstein ellipses, the image of the circle of radius rho under the Joukowski map,



    E_rho = { z = (1/2) (rho e^{i theta} + rho^{-1} e^{-i theta}), theta in [0, 2 pi) },



a confocal family with foci at z = +-1 that degenerates onto the segment [-1, 1] at

rho = 1. On E_rho the classical bound is



    (1/2) (rho^m - rho^{-m}) <= |T_m(z)| <= (1/2) (rho^m + rho^{-m}),



so the polynomials grow geometrically at rate rho with the expansion order m. This

identifies rho, rather than the spectral radius or the spectral half-width, as the

quantity that controls the numerical behaviour of a Chebyshev expansion: two spectra

with the same largest eigenvalue modulus can sit on very different ellipses.



Recovering rho from a point z means inverting the Joukowski map, that is solving

w^2 - 2 z w + 1 = 0. Its two roots w = z +- sqrt(z^2 - 1) multiply to 1, so they are

reciprocals and exactly one lies outside the unit circle. Taking the larger modulus,



    rho(z) = max( |z + sqrt(z^2 - 1)|, |z - sqrt(z^2 - 1)| ) >= 1,



and the radius enclosing a whole set is the maximum of rho(z) over the set. Points

lying inside [-1, 1] return exactly 1, as they must.

Returns
-------
float, the smallest Bernstein-ellipse radius rho >= 1 whose ellipse contains every supplied point, as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def bernstein_radius(points: np.ndarray) -> float:
    '''Compute the smallest Bernstein-ellipse radius enclosing a set of points.

    Parameters
    ----------
    points : np.ndarray
        Array-like of complex (or real) points to be enclosed. Must be non-empty
        and contain only finite values.

    Returns
    -------
    rho : float
        The smallest radius rho >= 1 such that every supplied point lies inside or
        on the Bernstein ellipse E_rho.

    Raises
    ------
    ValueError
        Raised if points is empty or contains a non-finite value.
    '''
    return rho

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_bernstein_radius(points: np.ndarray) -> float:
    z = np.asarray(points, dtype=complex).ravel()
    if z.size == 0:
        raise ValueError("points must contain at least one value")
    if not np.all(np.isfinite(z)):
        raise ValueError("points must contain only finite values")

    root = np.sqrt(z * z - 1.0 + 0.0j)
    rho = np.maximum(np.abs(z + root), np.abs(z - root))

    return float(np.max(rho))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: the elliptical spectrum of the task configuration ---
        {
            "setup": """import numpy as np
a = np.arange(1, 101)
theta = 2.0 * np.pi * a / 100
points = 0.85 * 1.35 * np.exp(-1j * theta) + 0.85 * 0.65 * np.exp(1j * theta)
""",
            "call": "bernstein_radius(points)",
            "gold_call": "_oracle_bernstein_radius(points)",
        },
        # --- Normal: purely imaginary points ---
        {
            "setup": """import numpy as np
points = np.array([0.5j, -1.5j, 2.25j, -0.75j])
""",
            "call": "bernstein_radius(points)",
            "gold_call": "_oracle_bernstein_radius(points)",
        },
        # --- Boundary: points inside [-1, 1] must give exactly rho = 1 ---
        {
            "setup": """import numpy as np
points = np.array([-1.0, -0.5, 0.0, 0.25, 1.0], dtype=complex)
""",
            "call": "bernstein_radius(points)",
            "gold_call": "_oracle_bernstein_radius(points)",
        },
        # --- Edge: a single real point far outside the focal interval ---
        {
            "setup": """import numpy as np
points = np.array([7.5 + 0.0j])
""",
            "call": "bernstein_radius(points)",
            "gold_call": "_oracle_bernstein_radius(points)",
        },
        # --- Edge: real input array rather than complex ---
        {
            "setup": """import numpy as np
points = np.linspace(-2.5, 2.5, 11)
""",
            "call": "bernstein_radius(points)",
            "gold_call": "_oracle_bernstein_radius(points)",
        },
        # --- Invalid: empty input ---
        {
            "setup": """import numpy as np
points = np.array([], dtype=complex)
def run_model():
    _fn = bernstein_radius
    try:
        _fn(points)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    _fn = _oracle_bernstein_radius
    try:
        _fn(points)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
        # --- Invalid: non-finite entry ---
        {
            "setup": """import numpy as np
points = np.array([1.0 + 0.0j, np.inf + 0.0j])
def run_model():
    _fn = bernstein_radius
    try:
        _fn(points)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    _fn = _oracle_bernstein_radius
    try:
        _fn(points)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
