"""
Return the source's two Local Discontinuous Galerkin numerical fluxes on one face, the velocity flux first and the traction flux second. On an interior face each is built from the average across the face plus a term in the jump, and the traction flux carries an additional penalty on the velocity jump. The sign that the flux parameter carries in each of the two fluxes is the source's convention and the two are NOT the same; recover it from the paper. Take the jump of a quantity as its value in the element owning the given outward normal minus its value in the neighbour. On a boundary face the no-slip condition fixes the velocity flux and the traction flux is the element traction corrected by a term in the velocity whose coefficient the source states. Raise ValueError unless the flux parameters satisfy the source's admissible ranges.

A discontinuous method communicates between elements only through these fluxes, so their form decides both the consistency of the scheme and whether it is stable.

Returns
-------
return (2, 2) float64: the velocity flux in row 0 and the traction flux in row 1
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def ldg_numerical_fluxes(uL, uR, sL, sR, n, a, b, on_boundary):
    """uL, uR: (2,) velocities either side; sL, sR: (2, 2) stresses either side;
    n: (2,) outward normal of the left element; a, b: flux parameters;
    on_boundary: True on a boundary face, where uR and sR are ignored.
    Returns (2, 2) float64."""
    return np.zeros((2, 2))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Gold oracle: ldg_numerical_fluxes."""

import numpy as np
import warnings
warnings.filterwarnings("ignore")




def _oracle_ldg_numerical_fluxes(uL, uR, sL, sR, n, a, b, on_boundary):
    """Eqs 42-45. Returns (2, 2): row 0 the velocity flux, row 1 the traction flux."""
    if not (0.0 <= a < 0.5) or b <= 0:
        raise ValueError("need a in [0, 0.5) and b > 0")
    uL = np.asarray(uL, dtype=np.float64); n = np.asarray(n, dtype=np.float64)
    snL = np.asarray(sL, dtype=np.float64) @ n
    if on_boundary:
        return np.stack([np.zeros(2), snL - (b / (0.5 - a)) * uL])
    uR = np.asarray(uR, dtype=np.float64); snR = np.asarray(sR, dtype=np.float64) @ n
    return np.stack([0.5 * (uL + uR) + a * (uL - uR),
                     0.5 * (snL + snR) - a * (snL - snR) - b * (uL - uR)])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": 'import numpy as np\nrng = np.random.default_rng(3)\nuL = rng.standard_normal(2)\nuR = rng.standard_normal(2)\n_m = rng.standard_normal((2, 2)); sL = 0.5 * (_m + _m.T)\n_m = rng.standard_normal((2, 2)); sR = 0.5 * (_m + _m.T)\nn = np.array([1.0, 0.0])\na = 0.25\nb = 3.0', "call": 'ldg_numerical_fluxes(uL, uR, sL, sR, n, a, b, False)', "gold_call": '_oracle_ldg_numerical_fluxes(uL, uR, sL, sR, n, a, b, False)', "tol": 1e-12},
        {"setup": 'import numpy as np\nrng = np.random.default_rng(7)\nuL = rng.standard_normal(2)\nuR = rng.standard_normal(2)\n_m = rng.standard_normal((2, 2)); sL = 0.5 * (_m + _m.T)\n_m = rng.standard_normal((2, 2)); sR = 0.5 * (_m + _m.T)\nn = np.array([0.0, -1.0])\na = 0.4\nb = 12.0', "call": 'ldg_numerical_fluxes(uL, uR, sL, sR, n, a, b, False)', "gold_call": '_oracle_ldg_numerical_fluxes(uL, uR, sL, sR, n, a, b, False)', "tol": 1e-12},
        {"setup": 'import numpy as np\nuL = np.array([0.3, -0.2])\nsL = np.array([[1.0e3, -2.0e2], [-2.0e2, 5.0e2]])\nn = np.array([-1.0, 0.0])\na = 0.25\nb = 3.0', "call": 'ldg_numerical_fluxes(uL, None, sL, None, n, a, b, True)', "gold_call": '_oracle_ldg_numerical_fluxes(uL, None, sL, None, n, a, b, True)', "tol": 1e-10},
    ]
