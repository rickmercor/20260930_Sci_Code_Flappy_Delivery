"""
Return the axial separation eigenvalues of the via, together with the norm of each axial eigenfunction and its integral over the via height, for a via that is insulated on its bottom face and held at the ambient temperature on its top face.

Separating the axisymmetric heat equation in cylindrical coordinates splits the temperature difference field into a product of a radial factor, an axial factor and a decaying exponential in time. The axial factor obeys a one-dimensional Helmholtz equation whose separation constant is fixed entirely by the two end conditions on the via, and, because those two conditions are shared by every material layer, the axial eigenvalue is a property of the via as a whole rather than of any one layer. That is what allows the layered problem to be reduced to a family of purely radial problems, one per axial eigenvalue.




For a face on which the axial heat flux vanishes and an opposite face pinned to the ambient temperature, the admissible axial eigenfunctions are the cosines that are stationary at the insulated face and change sign at the pinned face, so the eigenvalues are the odd multiples of a quarter wavelength across the height. This half-integer spacing, rather than the integer spacing of a pair of like end conditions, is what makes the fundamental axial mode decay four times more slowly than it otherwise would and therefore dominates the transient at the times of engineering interest.




Two integrals of each eigenfunction are needed downstream and are returned here. The first is the square norm over the height, which appears in the denominator of every expansion coefficient because the axial eigenfunctions are orthogonal but not normalised. The second is the plain integral of the eigenfunction over the height, which is the axial moment of a spatially uniform initial temperature and therefore the only place the initial condition enters the axial direction; it alternates in sign with the mode index and decays as the reciprocal of the eigenvalue, which is the origin of the slow, alternating convergence of the series near the pinned face.

Returns
-------
np.ndarray of shape (n_modes, 3), float: axial eigenvalue (1/m), axial eigenfunction square norm (m) and axial eigenfunction integral (m), one row per mode, in SI units.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def compute_axial_modes(height: float, n_modes: int) -> np.ndarray:
    """Return the axial separation eigenvalues and their two integrals.

    Parameters
    ----------
    height : float
        Height of the via in m, measured from the insulated face at z = 0 to
        the face held at the ambient temperature (height > 0).
    n_modes : int
        Number of axial modes to return, ordered by increasing eigenvalue
        (n_modes >= 1).

    Returns
    -------
    axial : np.ndarray
        Array of shape (n_modes, 3) whose columns are the axial eigenvalue in
        1/m, the square norm of the corresponding axial eigenfunction over the
        height in m, and the integral of that eigenfunction over the height
        in m.

    Raises
    ------
    ValueError
        If any argument violates the constraints stated above; invalid input
        must raise ValueError rather than return a sentinel value.
    """
    return axial  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_axial_modes(height: float, n_modes: int) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes
    # it in isolation.
    import numpy as np

    if not (isinstance(n_modes, (int, np.integer)) and not isinstance(n_modes, bool)
            and int(n_modes) >= 1):
        raise ValueError("n_modes must be an integer >= 1")
    if not (isinstance(height, (int, float, np.floating, np.integer))
            and not isinstance(height, bool)
            and np.isfinite(height) and float(height) > 0.0):
        raise ValueError("height must be a finite number > 0")

    height = float(height)
    n_modes = int(n_modes)

    # Insulated at z = 0 and pinned at z = h selects the cosines whose
    # quarter wavelengths fit an odd number of times into the height.
    index = np.arange(1, n_modes + 1, dtype=float)
    eta = (2.0 * index - 1.0) * np.pi / (2.0 * height)

    # The cosines are orthogonal on the height with a norm of half the height,
    # and their plain integral alternates in sign with the mode index.
    norm = np.full(n_modes, 0.5 * height)
    moment = np.sin(eta * height) / eta

    return np.column_stack([eta, norm, moment])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: benchmark via height (normal scenario) ---
        {
            "setup": """import numpy as np
height = 200.0e-6
n_modes = 12
""",
            "call": "compute_axial_modes(height, n_modes)",
            "gold_call": "_oracle_compute_axial_modes(height, n_modes)",
        },
        # --- Valid: short via with many modes ---
        {
            "setup": """import numpy as np
height = 50.0e-6
n_modes = 40
""",
            "call": "compute_axial_modes(height, n_modes)",
            "gold_call": "_oracle_compute_axial_modes(height, n_modes)",
        },
        # --- Boundary: a single mode ---
        {
            "setup": """import numpy as np
height = 1.0e-3
n_modes = 1
""",
            "call": "compute_axial_modes(height, n_modes)",
            "gold_call": "_oracle_compute_axial_modes(height, n_modes)",
        },
        # --- Edge: unit height makes the eigenvalues plain odd multiples of pi/2 ---
        {
            "setup": """import numpy as np
height = 1.0
n_modes = 5
""",
            "call": "compute_axial_modes(height, n_modes)",
            "gold_call": "_oracle_compute_axial_modes(height, n_modes)",
        },
        # --- Invalid: non-positive height ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        compute_axial_modes(0.0, 4)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_axial_modes(0.0, 4)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: zero modes requested ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        compute_axial_modes(200.0e-6, 0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_axial_modes(200.0e-6, 0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
