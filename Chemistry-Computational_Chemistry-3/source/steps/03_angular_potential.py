"""
Evaluate the angular part of the model potential at a Cartesian body-frame position, for the symmetry-breaking strength b.

The departing atom feels a hindrance that depends on how far it has bent away from the fragment axis, modulated in azimuth by the symmetry of the fragment left behind. Trajectories cross the symmetry axis, so the returned value must be finite and smooth there.

Returns
-------
ndarray with the broadcast shape of x, y and z: the angular potential in kcal/mol.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def angular_potential(x: "np.ndarray", y: "np.ndarray", z: "np.ndarray", ve: float, alpha: float, re: float, b: float) -> "np.ndarray":
    """Evaluate the angular part of the model potential at a Cartesian body-frame position, for the symmetry-breaking strength b.

    Parameters
    ----------
    x : np.ndarray
        Cartesian body-frame x component(s), in angstroms.
    y : np.ndarray
        Cartesian body-frame y component(s), in angstroms.
    z : np.ndarray
        Cartesian body-frame z component(s), in angstroms.
    ve : float
        Finite amplitude of the radial envelope, in kcal/mol.
    alpha : float
        Finite Gaussian width parameter of the envelope, in inverse square angstroms.
    re : float
        Reference separation of the envelope, in angstroms.
    b : float
        Non-negative dimensionless strength of the symmetry-breaking term.

    Returns
    -------
    v_angular : np.ndarray
        ndarray with the broadcast shape of x, y and z: the angular potential in kcal/mol.

    Raises
    ------
    ValueError
        if ve, alpha or b is not finite, if b is negative, or if the origin is in the domain.
    """
    return v_angular

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.integrate import solve_ivp


def _oracle_angular_potential(x: "np.ndarray", y: "np.ndarray", z: "np.ndarray", ve: float, alpha: float, re: float, b: float) -> "np.ndarray":
    x = np.asarray(x); y = np.asarray(y); z = np.asarray(z)
    cplx = np.iscomplexobj(x) or np.iscomplexobj(y) or np.iscomplexobj(z)
    if not cplx:
        x = x.astype(float); y = y.astype(float); z = z.astype(float)
    if not np.isfinite(ve) or not np.isfinite(alpha) or not np.isfinite(b):
        raise ValueError("ve, alpha and b must be finite")
    if b < 0.0:
        raise ValueError("b must be non-negative")
    r2 = x * x + y * y + z * z
    if not cplx and np.any(r2 <= 0.0):
        raise ValueError("the origin is not in the domain")
    r = np.sqrt(r2)
    v0 = ve * np.exp(-alpha * (r - re) ** 2)
    return v0 * ((x * x + y * y) / r2 + b * (x ** 3 - 3.0 * x * y * y) / r ** 3)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test-case specifications for this step."""
    return [
        {"setup": "import numpy as np",
         "call": "angular_potential(0.9, 0.3, 3.2, 55.0, 1.0, 1.1, 0.0)",
         "gold_call": "_oracle_angular_potential(0.9, 0.3, 3.2, 55.0, 1.0, 1.1, 0.0)"},   # normal
        {"setup": "import numpy as np",
         "call": "angular_potential(0.9, 0.3, 3.2, 55.0, 1.0, 1.1, 0.45)",
         "gold_call": "_oracle_angular_potential(0.9, 0.3, 3.2, 55.0, 1.0, 1.1, 0.45)"},   # normal
        {"setup": "import numpy as np",
         "call": "angular_potential(0.0, 0.0, 3.5, 55.0, 1.0, 1.1, 0.45)",
         "gold_call": "_oracle_angular_potential(0.0, 0.0, 3.5, 55.0, 1.0, 1.1, 0.45)"},   # boundary
        {"setup": "import numpy as np",
         "call": "angular_potential(np.array([1.0,-1.0]), np.array([0.2,0.2]), np.array([3.0,3.0]), 55.0, 1.0, 1.1, 0.3)",
         "gold_call": "_oracle_angular_potential(np.array([1.0,-1.0]), np.array([0.2,0.2]), np.array([3.0,3.0]), 55.0, 1.0, 1.1, 0.3)"},   # edge
        {"setup": "import numpy as np\ndef _c(fn):\n    try:\n        fn(0.9, 0.3, 3.2, 55.0, 1.0, 1.1, -0.1)\n    except ValueError:\n        return 1\n    except Exception:\n        return 0\n    return 0",
         "call": "_c(angular_potential)", "gold_call": "_c(_oracle_angular_potential)"},   # exception contract
    ]
