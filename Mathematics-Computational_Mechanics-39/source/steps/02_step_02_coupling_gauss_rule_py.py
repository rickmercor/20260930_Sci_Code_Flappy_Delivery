"""
Construct the one-dimensional Gauss-Legendre rule used for integration of the MPM-FEM coupling interface.

Determine the integration order from the supplied finite-element interface side lengths and mortar-side grid spacings using the empirical coupling-integration prescription defined in the cited paper.

Apply the floor operation to the mathematical length ratio. If floating-point evaluation places a value within 1e-12 of an integer, treat it as that integer before applying the floor.

Return the selected order together with the corresponding one-dimensional Gauss points and weights on [-1,1].

Nonmatching interface discretizations can introduce spatial variation at scales smaller than the finite-element interface element itself.

Accurate mortar integration therefore requires a quadrature resolution that reflects both the finite-element surface size and the grid scale of the opposing discretization.

Returns
-------
A tuple (n_gauss, points, weights).  n_gauss is a native integer.  points and weights are one-dimensional NumPy arrays of length n_gauss containing the Gauss-Legendre rule on [-1,1].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def coupling_gauss_rule(
    fe_lengths: np.ndarray,
    mpm_spacing: np.ndarray,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Return source coupling Gauss order, points, and weights."""

    return (
        1,
        np.zeros(1, dtype=float),
        np.zeros(1, dtype=float),
    )

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math
import numpy as np


def _oracle_coupling_gauss_rule(
    fe_lengths: np.ndarray,
    mpm_spacing: np.ndarray,
):

    lfe = np.asarray(
        fe_lengths,
        dtype=float,
    )

    hmpm = np.asarray(
        mpm_spacing,
        dtype=float,
    )

    ratio = (
        np.max(lfe)
        / np.min(hmpm)
    )

    nearest_integer = np.rint(
        ratio
    )

    if abs(
        ratio
        - nearest_integer
    ) <= 1.0e-12:
        ratio = float(
            nearest_integer
        )

    n_gauss = int(
        math.floor(
            ratio
        )
        + 2
    )

    points, weights = (
        np.polynomial.legendre.leggauss(
            n_gauss
        )
    )

    return (
        n_gauss,
        points,
        weights,
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """
import numpy as np
fe_lengths=np.array([2.4,1.6])
mpm_spacing=np.array([.8,.8])
""",
            "call": """
(lambda z: np.concatenate((
[float(z[0])],z[1],z[2]
)))(coupling_gauss_rule(fe_lengths,mpm_spacing))
""",
            "gold_call": """
(lambda z: np.concatenate((
[float(z[0])],z[1],z[2]
)))(_oracle_coupling_gauss_rule(fe_lengths,mpm_spacing))
""",
            "tol": 1e-14,
        },
        {
            "setup": """
import numpy as np
fe_lengths=np.array([2.0,1.5])
mpm_spacing=np.array([1.0,.5])
""",
            "call": """
(lambda z: np.concatenate(([float(z[0])],z[1],z[2])))(
coupling_gauss_rule(fe_lengths,mpm_spacing))
""",
            "gold_call": """
(lambda z: np.concatenate(([float(z[0])],z[1],z[2])))(
_oracle_coupling_gauss_rule(fe_lengths,mpm_spacing))
""",
            "tol": 1e-14,
        },
        {
            "setup": """
import numpy as np
fe_lengths=np.array([3.0,1.8])
mpm_spacing=np.array([.75,.6])
""",
            "call": """
(lambda z: np.concatenate(([float(z[0])],z[1],z[2])))(
coupling_gauss_rule(fe_lengths,mpm_spacing))
""",
            "gold_call": """
(lambda z: np.concatenate(([float(z[0])],z[1],z[2])))(
_oracle_coupling_gauss_rule(fe_lengths,mpm_spacing))
""",
            "tol": 1e-14,
        },
        {
            "setup": """
import numpy as np
fe_lengths=np.array([1.8,1.2])
mpm_spacing=np.array([.6,.4])
""",
            "call": """
(lambda z: np.concatenate(([float(z[0])],z[1],z[2])))(
coupling_gauss_rule(fe_lengths,mpm_spacing))
""",
            "gold_call": """
(lambda z: np.concatenate(([float(z[0])],z[1],z[2])))(
_oracle_coupling_gauss_rule(fe_lengths,mpm_spacing))
""",
            "tol": 1e-14,
        },
    ]
