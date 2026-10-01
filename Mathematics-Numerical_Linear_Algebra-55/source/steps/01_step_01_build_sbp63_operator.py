"""
Recover a sixth/third-order SBP closure and an arbitrary calibration jet.

For $N$ uniform nodes on $[0,1]$ with $h=1/(N-1)$, write $D=H^{-1}Q$. The matrix $H$ is positive diagonal, equals $h$ except in its first and last six entries, and is centrosymmetric. The boundary form satisfies $Q+Q^T=\\operatorname{diag}(-1,0,\\ldots,0,1)$ and $Q=-JQJ$ for the exchange matrix $J$.



Every row outside the first and last six uses the seven-point numerator $(-1/60,3/20,-3/4,0,3/4,-3/20,1/60)$, divided by $h$. The first six rows of $Q$ vanish beyond zero-based column eight and differentiate $1,\\xi,\\xi^2,\\xi^3$ exactly. Together with the SBP and reflection identities, these equations determine $H$ but leave a one-dimensional affine family $B(\\theta)=B_0+\\theta S$ for the left $6\\times9$ closure block.



Let a calibration variable $s$ move both the selector and its target. The two inputs are equally long raw-derivative jets: entry $k$ is $q^{(k)}(0)$ or $F^{(k)}(0)$, and their common length $r$ may be any integer from one through nine. At every $s$, select $B(s)=B_0+\\theta(s)S$ by $\\langle F(s),B(s)\\rangle=q(s)$. Recover the closure family from the SBP constraints and differentiate this product identity with the full Leibniz rule through order $r-1$. Reject mismatched jet lengths and a value functional $F(0)$ that annihilates $S$.



A missing target jet means $(342523/518400,1,0)$, while a missing functional jet uses the length-three entry-functional jet at $(4,5)$ with zero derivatives. If only one jet is supplied, its length must therefore be three. The selected value at $s=0$ is the standard member for which zero-based closure row five also differentiates $\\xi^4$ exactly.



Return $H$ followed by the interleaved pairs $Q^{(k)}(0),D^{(k)}(0)$ for $k=0,\\ldots,r-1$. Complete every derivative by reflection. Every positive derivative level must preserve skew symmetry, reflection, closure support, and zero degree-zero through degree-three boundary moments. At least eighteen nodes are required so the two closure supports do not overlap.

Returns
-------
shape (2*r+1, n, n): H followed by Q^(k), D^(k), k=0,...,r-1
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def build_sbp63_operator(
    n: int,
    target_jet: np.ndarray = None,
    functional_jet: np.ndarray = None,
) -> np.ndarray:
    """Return a functionally selected operator and its runtime-order jet.

    The grid consists of ``n`` uniformly spaced nodes on ``[0, 1]``. The
    returned matrices must satisfy the normalization, reflection, support,
    interior-order, and degree-three boundary-moment conditions stated in the
    scientific background. The two equally long raw-derivative jets prescribe
    a scalar target and a Frobenius functional at zero. They select a smooth
    path through the one-parameter closure family. Jet length ``r`` requests
    derivatives zero through ``r - 1`` and may range from one to nine.

    Parameters
    ----------
    n : int
        Number of nodes; must be an integer at least eighteen.
    target_jet : np.ndarray
        ``None`` or a finite real raw-derivative jet of shape ``(r,)``, where
        ``1 <= r <= 9``. ``None`` selects ``[342523/518400, 1, 0]``.
    functional_jet : np.ndarray
        ``None`` or a finite real array of shape ``(r, 6, 9)`` containing the
        raw derivatives of ``F``. Its jet length must equal that of
        ``target_jet``. The value functional must not annihilate the
        affine-family direction. ``None`` uses a length-three jet whose value
        is the entry functional at ``[4, 5]`` and whose derivatives vanish.

    Returns
    -------
    np.ndarray
        Array of shape ``(2*r + 1, n, n)`` containing ``H`` followed by the
        interleaved raw-derivative pairs ``Q^(k)(0), D^(k)(0)`` for
        ``k = 0, ..., r - 1``.

    Raises
    ------
    ValueError
        If ``n`` is not an integer or is below eighteen; if either jet has an
        invalid shape or contains complex or non-finite data; if the jet
        lengths disagree; or if ``F(0)`` does not select a unique family member.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

from math import comb

import numpy as np


def _validated_node_count(n: int) -> int:
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)):
        raise ValueError("n must be an integer")
    if int(n) < 18:
        raise ValueError("n must be at least 18")
    return int(n)


def _validated_closure_jets(target_jet, functional_jet):
    if target_jet is None:
        targets = np.array([342523.0 / 518400.0, 1.0, 0.0])
    else:
        raw_targets = np.asarray(target_jet)
        if (
            raw_targets.ndim != 1
            or not 1 <= raw_targets.size <= 9
            or np.iscomplexobj(raw_targets)
            or not np.all(np.isfinite(raw_targets))
        ):
            raise ValueError(
                "target_jet must have between one and nine finite real entries"
            )
        targets = np.asarray(raw_targets, dtype=float)
    if functional_jet is None:
        functionals = np.zeros((3, 6, 9), dtype=float)
        functionals[0, 4, 5] = 1.0
    else:
        raw_functionals = np.asarray(functional_jet)
        if (
            raw_functionals.ndim != 3
            or raw_functionals.shape[1:] != (6, 9)
            or not 1 <= raw_functionals.shape[0] <= 9
            or np.iscomplexobj(raw_functionals)
            or not np.all(np.isfinite(raw_functionals))
        ):
            raise ValueError(
                "functional_jet must have finite real shape (r, 6, 9), 1 <= r <= 9"
            )
        functionals = np.asarray(raw_functionals, dtype=float)
    if targets.size != functionals.shape[0]:
        raise ValueError("target_jet and functional_jet must have the same jet length")
    return targets, functionals


_BOUNDARY_NORM = (
    13649.0 / 43200.0,
    12013.0 / 8640.0,
    2711.0 / 4320.0,
    5359.0 / 4320.0,
    7877.0 / 8640.0,
    43801.0 / 43200.0,
)
_BOUNDARY_BLOCK = np.array(
    [
        [
            -1.0 / 2.0,
            104009.0 / 172800.0,
            30443.0 / 259200.0,
            -33311.0 / 86400.0,
            5621.0 / 28800.0,
            -601.0 / 20736.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            -104009.0 / 172800.0,
            0.0,
            -311.0 / 51840.0,
            6743.0 / 5760.0,
            -24337.0 / 34560.0,
            36661.0 / 259200.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            -30443.0 / 259200.0,
            311.0 / 51840.0,
            0.0,
            -2231.0 / 5184.0,
            41287.0 / 51840.0,
            -7333.0 / 28800.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            33311.0 / 86400.0,
            -6743.0 / 5760.0,
            2231.0 / 5184.0,
            0.0,
            4147.0 / 17280.0,
            25427.0 / 259200.0,
            1.0 / 60.0,
            0.0,
            0.0,
        ],
        [
            -5621.0 / 28800.0,
            24337.0 / 34560.0,
            -41287.0 / 51840.0,
            -4147.0 / 17280.0,
            0.0,
            342523.0 / 518400.0,
            -3.0 / 20.0,
            1.0 / 60.0,
            0.0,
        ],
        [
            601.0 / 20736.0,
            -36661.0 / 259200.0,
            7333.0 / 28800.0,
            -25427.0 / 259200.0,
            -342523.0 / 518400.0,
            0.0,
            3.0 / 4.0,
            -3.0 / 20.0,
            1.0 / 60.0,
        ],
    ],
    dtype=float,
)
_BOUNDARY_BLOCK_SLOPE = np.array(
    [
        [0.0, 1.0, -4.0, 6.0, -4.0, 1.0, 0.0, 0.0, 0.0],
        [-1.0, 0.0, 10.0, -20.0, 15.0, -4.0, 0.0, 0.0, 0.0],
        [4.0, -10.0, 0.0, 20.0, -20.0, 6.0, 0.0, 0.0, 0.0],
        [-6.0, 20.0, -20.0, 0.0, 10.0, -4.0, 0.0, 0.0, 0.0],
        [4.0, -15.0, 20.0, -10.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [-1.0, 4.0, -6.0, 4.0, -1.0, 0.0, 0.0, 0.0, 0.0],
    ],
    dtype=float,
)
_INTERIOR_STENCIL = np.array(
    [-1.0 / 60.0, 3.0 / 20.0, -3.0 / 4.0, 0.0, 3.0 / 4.0, -3.0 / 20.0, 1.0 / 60.0]
)


def _oracle_build_sbp63_operator(
    n: int,
    target_jet: np.ndarray = None,
    functional_jet: np.ndarray = None,
) -> np.ndarray:
    """Reference functional selection and exact runtime-order path jet."""
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)):
        raise ValueError("n must be an integer")
    if int(n) < 18:
        raise ValueError("n must be at least 18")
    n = int(n)
    targets, functionals = _validated_closure_jets(target_jet, functional_jet)
    directional_values = np.einsum("kij,ij->k", functionals, _BOUNDARY_BLOCK_SLOPE)
    reference_values = np.einsum("kij,ij->k", functionals, _BOUNDARY_BLOCK)
    functional_direction = directional_values[0]
    functional_tolerance = (
        64.0
        * np.finfo(float).eps
        * max(
            1.0,
            float(np.linalg.norm(functionals[0]))
            * float(np.linalg.norm(_BOUNDARY_BLOCK_SLOPE)),
        )
    )
    if abs(functional_direction) <= functional_tolerance:
        raise ValueError("F(0) does not select a unique family member")
    spacing = 1.0 / (n - 1)

    weights = np.ones(n, dtype=float)
    weights[:6] = _BOUNDARY_NORM
    weights[-6:] = weights[5::-1]
    h_matrix = spacing * np.diag(weights)

    jet_length = targets.size
    family_jet = np.zeros(jet_length, dtype=float)
    for order in range(jet_length):
        known = reference_values[order]
        for functional_order in range(1, order + 1):
            known += (
                comb(order, functional_order)
                * directional_values[functional_order]
                * family_jet[order - functional_order]
            )
        family_jet[order] = (targets[order] - known) / functional_direction

    output = [h_matrix]
    for order, family_derivative in enumerate(family_jet):
        boundary = family_derivative * _BOUNDARY_BLOCK_SLOPE
        if order == 0:
            boundary = _BOUNDARY_BLOCK + boundary
        q_derivative = np.zeros((n, n), dtype=float)
        q_derivative[:6, :9] = boundary
        q_derivative[-6:, -9:] = -boundary[::-1, ::-1]
        if order == 0:
            for row in range(6, n - 6):
                q_derivative[row, row - 3 : row + 4] = _INTERIOR_STENCIL
        d_derivative = q_derivative / (spacing * weights)[:, None]
        output.extend((q_derivative, d_derivative))
    return np.stack(output)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return three runtime jet orders and combined invalid cases."""
    return [
        {
            "setup": """import numpy as np
n = 23
target_jet = np.array([0.60, -0.08, 0.14, -0.05, 0.03, 0.02, -0.01])
functional_jet = np.zeros((7, 6, 9))
functional_jet[0, 4, 5] = 1.0
functional_jet[0, 0, 3] = 0.01
functional_jet[1, 1, 4] = -0.03
functional_jet[1, 5, 2] = 0.02
functional_jet[2, 0, 1] = 0.04
functional_jet[2, 4, 3] = -0.015
functional_jet[3, 2, 5] = 0.018
functional_jet[4, 3, 1] = -0.011
functional_jet[5, 5, 4] = 0.007
functional_jet[6, 0, 2] = -0.009
""",
            "call": "build_sbp63_operator(n, target_jet, functional_jet)",
            "gold_call": "_oracle_build_sbp63_operator(n, target_jet, functional_jet)",
        },
        {
            "setup": """n = 18
""",
            "call": "build_sbp63_operator(n)",
            "gold_call": "_oracle_build_sbp63_operator(n)",
        },
        {
            "setup": """import numpy as np
n = 41
target_jet = np.array([0.70, 0.11, -0.09, 0.06, -0.04, 0.03, 0.02, -0.01, 0.005])
functional_jet = np.zeros((9, 6, 9))
functional_jet[0, 4, 5] = 1.0
functional_jet[0, 0, 1] = 0.03
functional_jet[0, 2, 3] = -0.02
functional_jet[1, 3, 5] = 0.025
functional_jet[2, 1, 2] = -0.035
functional_jet[3, 4, 1] = 0.021
functional_jet[4, 0, 5] = -0.013
functional_jet[5, 2, 4] = 0.008
functional_jet[6, 5, 1] = -0.006
functional_jet[7, 1, 3] = 0.004
functional_jet[8, 3, 2] = -0.003
""",
            "call": "build_sbp63_operator(n, target_jet, functional_jet)",
            "gold_call": "_oracle_build_sbp63_operator(n, target_jet, functional_jet)",
        },
        {
            "setup": """import numpy as np
n = 17
def run_model():
    try:
        build_sbp63_operator(n)
        return np.array([0, 0])
    except ValueError:
        first = 1
    except Exception:
        first = 2
    try:
        build_sbp63_operator(18, np.array([0.5, 0.0]), np.zeros((3, 6, 9)))
        second = 0
    except ValueError:
        second = 1
    except Exception:
        second = 2
    return np.array([first, second])
def run_gold():
    try:
        _oracle_build_sbp63_operator(n)
        return np.array([0, 0])
    except ValueError:
        first = 1
    except Exception:
        first = 2
    try:
        _oracle_build_sbp63_operator(18, np.array([0.5, 0.0]), np.zeros((3, 6, 9)))
        second = 0
    except ValueError:
        second = 1
    except Exception:
        second = 2
    return np.array([first, second])
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
