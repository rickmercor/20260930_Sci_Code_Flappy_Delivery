"""
Rayleigh–Sommerfeld cell operator

Use the complex cell-center Rayleigh–Sommerfeld kernel H_mn=A_s z(1-ikR_mn) exp(ikR_mn)/(2 pi R_mn^3), k=2 pi/lambda. Coordinates are transverse micrometres; z>0. Source cells have equal area A_s. This fixes the outgoing-wave sign and retains the near-field term.

Returns
-------
complex ndarray (M,N). Element [m,n] maps unit source-cell field to observation field; dimensionless.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def rs_operator(
    source_xy: "np.ndarray",
    target_xy: "np.ndarray",
    wavelength: float,
    z: float,
    source_area: float,
) -> "np.ndarray":
    """Rayleigh–Sommerfeld cell operator.

    Use the complex cell-center Rayleigh–Sommerfeld kernel H_mn=A_s
    z(1-ikR_mn) exp(ikR_mn)/(2 pi R_mn^3), k=2 pi/lambda. Coordinates are
    transverse micrometres; z>0. Source cells have equal area A_s. This
    fixes the outgoing-wave sign and retains the near-field term.

    Parameters
    source_xy : float ndarray (N,2): source-cell centers [x,y],
    micrometres.
    target_xy : float ndarray (M,2): observation centers [x,y],
    micrometres, in target order.
    wavelength : positive float: wavelength in micrometres.
    z : positive float: propagation distance in micrometres.
    source_area : positive float: one source-cell area in square
    micrometres.

    Returns
    complex ndarray (M,N). Element [m,n] maps unit source-cell field to
    observation field; dimensionless.

    Raises
    ValueError for invalid dimensions, nonfinite inputs or parameters
    outside the stated domain.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_rs_operator(
    source_xy: "np.ndarray",
    target_xy: "np.ndarray",
    wavelength: float,
    z: float,
    source_area: float,
) -> "np.ndarray":
    source_xy = np.asarray(source_xy, dtype=float)
    target_xy = np.asarray(target_xy, dtype=float)
    if (
        source_xy.ndim != 2
        or target_xy.ndim != 2
        or source_xy.shape[1:] != (2,)
        or target_xy.shape[1:] != (2,)
        or min(len(source_xy), len(target_xy)) == 0
    ):
        raise ValueError(
            "Coordinates must be nonempty (N,2) and (M,2) arrays."
        )
    if (
        not all(
            np.isfinite(x).all()
            for x in [source_xy, target_xy, wavelength, z, source_area]
        )
        or min(wavelength, z, source_area) <= 0
    ):
        raise ValueError(
            "Finite coordinates and positive optical scales are required."
        )
    r = np.sqrt(
        np.sum((target_xy[:, None, :] - source_xy[None, :, :]) ** 2, axis=-1)
        + z * z
    )
    k = 2 * np.pi / wavelength
    return (
        source_area
        * z
        * (1 - 1j * k * r)
        * np.exp(1j * k * r)
        / (2 * np.pi * r**3)
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return independently initialized differential cases."""
    return [
        {
            "setup": (
                """from copy import deepcopy
import numpy as np

s = np.array([[0.0, 0.0]])
t = s.copy()
lam = 0.633
z = 4.0
a = 0.0625
candidate_args = deepcopy((s, t, lam, z, a))
oracle_args = deepcopy((s, t, lam, z, a))
"""
            ),
            "call": "rs_operator(*candidate_args)",
            "gold_call": "_oracle_rs_operator(*oracle_args)",
            "tol": 1e-06,
        },
        {
            "setup": (
                """from copy import deepcopy
import numpy as np

s = np.array([[0.2, -0.4]])
t = np.array([[-0.3, 0.5]])
lam = 0.405
z = 0.6
a = 0.02
candidate_args = deepcopy((s, t, lam, z, a))
oracle_args = deepcopy((s, t, lam, z, a))
"""
            ),
            "call": "rs_operator(*candidate_args)",
            "gold_call": "_oracle_rs_operator(*oracle_args)",
            "tol": 1e-06,
        },
        {
            "setup": (
                """from copy import deepcopy
import numpy as np

s = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]])
t = np.array([[0.4, 0.2], [-0.5, 0.7]])
lam = 0.633
z = 2.0
a = 0.08
candidate_args = deepcopy((s, t, lam, z, a))
oracle_args = deepcopy((s, t, lam, z, a))
"""
            ),
            "call": "rs_operator(*candidate_args)",
            "gold_call": "_oracle_rs_operator(*oracle_args)",
            "tol": 1e-06,
        },
        {
            "setup": (
                """from copy import deepcopy
import numpy as np

s = np.array([[0.0, 0.0]])
t = np.array([[0.002, -0.001]])
lam = 1.0
z = 0.003
a = 1e-06
candidate_args = deepcopy((s, t, lam, z, a))
oracle_args = deepcopy((s, t, lam, z, a))
"""
            ),
            "call": "rs_operator(*candidate_args)",
            "gold_call": "_oracle_rs_operator(*oracle_args)",
            "tol": 1e-06,
        },
        {
            "setup": (
                """from copy import deepcopy
import numpy as np

s = np.array([[0.0, 0.0], [0.2, 0.1]])
t = np.array([[1.0, 2.0]])
lam = 0.633
z = 200.0
a = 0.0625
candidate_args = deepcopy((s, t, lam, z, a))
oracle_args = deepcopy((s, t, lam, z, a))
"""
            ),
            "call": "rs_operator(*candidate_args)",
            "gold_call": "_oracle_rs_operator(*oracle_args)",
            "tol": 1e-06,
        },
        {
            "setup": (
                """from copy import deepcopy
import numpy as np

s = np.array([[10.0, -9.0], [10.5, -9.3]])
t = np.array([[10.1, -8.8], [9.7, -9.2]])
lam = 0.532
z = 1.7
a = 0.03
candidate_args = deepcopy((s, t, lam, z, a))
oracle_args = deepcopy((s, t, lam, z, a))
"""
            ),
            "call": "rs_operator(*candidate_args)",
            "gold_call": "_oracle_rs_operator(*oracle_args)",
            "tol": 1e-06,
        },
        {
            "setup": (
                """from copy import deepcopy
import numpy as np

s = 3 * np.array([[0.0, 0.0], [0.2, 0.1]])
t = 3 * np.array([[0.4, -0.3]])
lam = 3 * 0.633
z = 3 * 2.0
a = 9 * 0.0625
candidate_args = deepcopy((s, t, lam, z, a))
oracle_args = deepcopy((s, t, lam, z, a))
"""
            ),
            "call": "rs_operator(*candidate_args)",
            "gold_call": "_oracle_rs_operator(*oracle_args)",
            "tol": 1e-06,
        },
        {
            "setup": (
                """from copy import deepcopy
import numpy as np

s = np.array([[0.4, -0.2], [0.0, 0.0], [-0.1, 0.5]])
t = np.array([[0.2, 0.3], [-0.4, -0.5]])
lam = 0.633
z = 1.1
a = 0.04
candidate_args = deepcopy((s, t, lam, z, a))
oracle_args = deepcopy((s, t, lam, z, a))
"""
            ),
            "call": "rs_operator(*candidate_args)",
            "gold_call": "_oracle_rs_operator(*oracle_args)",
            "tol": 1e-06,
        },
    ]
