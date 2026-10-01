"""
Run the source depth-increasing synthesis search using the fixed-depth

optimizer.



Starting at depth 1, globally solve each fixed-depth source synthesis

instance. If the required fidelity is not reached, record the globally

optimized fixed-depth fidelity and continue to the next depth. Stop at the

first depth for which the fixed-depth optimizer returns a feasibility

certificate.



The public output contains only consecutive depth numbers, Boolean

threshold-feasibility flags, and representation-independent fidelity

certificates.

Search synthesis depths consecutively starting from depth 1.



The five rotation-angle inputs must all be finite.



epsilon must be a finite real number satisfying



0 <= epsilon < 1,



and the required fidelity threshold is 1-epsilon.



max_depth must be an integer in



{1,2,3,4}.



For every searched depth, the fixed-depth optimizer returns a Boolean

threshold flag and a fidelity certificate. Rejected depths contain their

globally optimized fixed-depth fidelity. A feasible depth contains the

representation-independent feasibility certificate defined by the fixed-depth

public contract.



The search terminates at the first feasible depth. If no depth from 1 through

max_depth is feasible, the function raises ValueError.

Returns
-------
np.ndarray of shape (m,3), with consecutive rows [depth, threshold_met, certificate], where threshold_met is exactly 0 or 1 and the last row is the first feasible depth
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def search_source_depths_d4max(
    theta_rx0: float,
    theta_rx1: float,
    theta_rx2: float,
    theta_rz0: float,
    theta_rz2: float,
    epsilon: float,
    max_depth: int,
) -> np.ndarray:
    """
    Search source synthesis depths in increasing order.

    Parameters
    ----------
    theta_rx0 : float
        Finite RX angle for qubit 0.
    theta_rx1 : float
        Finite RX angle for qubit 1.
    theta_rx2 : float
        Finite RX angle for qubit 2.
    theta_rz0 : float
        Finite RZ angle for qubit 0.
    theta_rz2 : float
        Finite RZ angle for qubit 2.
    epsilon : float
        Approximation tolerance. Must be finite and satisfy
        0 <= epsilon < 1.
    max_depth : int
        Largest synthesis depth to search. Must be exactly one of
        {1, 2, 3, 4}.

    Returns
    -------
    depth_results : np.ndarray
        Array of shape (m,3), with one row per searched depth:
        [depth, threshold_met, certificate].
        depth values are consecutive integers starting at 1.
        threshold_met is exactly 0 or 1.
        certificate lies in [0,1].
        The final row is the first feasible depth.

    Raises
    ------
    ValueError
        If any rotation angle is non-finite; if epsilon is non-finite
        or does not satisfy 0 <= epsilon < 1; if max_depth is not an
        integer in {1,2,3,4}; or if no searched depth reaches the
        requested fidelity threshold.
    """
    return np.empty((0, 3), dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_search_source_depths_d4max(
    theta_rx0: float,
    theta_rx1: float,
    theta_rx2: float,
    theta_rz0: float,
    theta_rz2: float,
    epsilon: float,
    max_depth: int,
) -> np.ndarray:
    angles = np.asarray(
        [
            theta_rx0,
            theta_rx1,
            theta_rx2,
            theta_rz0,
            theta_rz2,
        ],
        dtype=float,
    )

    if not np.all(np.isfinite(angles)):
        raise ValueError(
            "rotation angles must be finite"
        )

    try:
        epsilon = float(epsilon)
    except Exception as exc:
        raise ValueError(
            "epsilon must be finite and satisfy 0 <= epsilon < 1"
        ) from exc

    if (
        not np.isfinite(epsilon)
        or epsilon < 0.0
        or epsilon >= 1.0
    ):
        raise ValueError(
            "epsilon must be finite and satisfy 0 <= epsilon < 1"
        )

    if (
        isinstance(max_depth, (bool, np.bool_))
        or not isinstance(max_depth, (int, np.integer))
        or int(max_depth) not in {1, 2, 3, 4}
    ):
        raise ValueError(
            "max_depth must be an integer in {1,2,3,4}"
        )

    threshold = 1.0 - epsilon

    rows = []

    for depth in range(
        1,
        int(max_depth) + 1,
    ):
        result = (
            _oracle_optimize_fixed_depth_d4max(
                theta_rx0,
                theta_rx1,
                theta_rx2,
                theta_rz0,
                theta_rz2,
                depth,
                threshold,
            )
        )

        result = np.asarray(
            result,
            dtype=float,
        )

        if (
            result.shape != (2,)
            or not np.all(np.isfinite(result))
        ):
            raise ValueError(
                "fixed-depth optimizer returned a malformed result"
            )

        flag = float(result[0])
        certificate = float(result[1])

        if flag not in (0.0, 1.0):
            raise ValueError(
                "fixed-depth threshold flag must be exactly 0 or 1"
            )

        if (
            certificate < 0.0
            or certificate > 1.0
        ):
            raise ValueError(
                "fixed-depth certificate must lie in [0,1]"
            )

        rows.append(
            [
                float(depth),
                flag,
                certificate,
            ]
        )

        if flag == 1.0:
            return np.asarray(
                rows,
                dtype=float,
            )

    raise ValueError(
        "no searched depth reaches the requested threshold"
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np

a,b,c,d,e = 0.8453,0.1729,0.6331,0.4217,0.2876
epsilon = 0.30
max_depth = 4
""",
            "call": "search_source_depths_d4max(a,b,c,d,e,epsilon,max_depth)",
            "gold_call": "_oracle_search_source_depths_d4max(a,b,c,d,e,epsilon,max_depth)",
        },
        {
            "setup": """import numpy as np

a,b,c,d,e = 0.8453,0.1729,0.6331,0.4217,0.2876
epsilon = 0.81
max_depth = 4
""",
            "call": "search_source_depths_d4max(a,b,c,d,e,epsilon,max_depth)",
            "gold_call": "_oracle_search_source_depths_d4max(a,b,c,d,e,epsilon,max_depth)",
        },
        {
            "setup": """import numpy as np

a,b,c,d,e = 0.8453,0.1729,0.6331,0.4217,0.2876
epsilon = -0.1
max_depth = 4

def run_model():
    try:
        search_source_depths_d4max(
            a,b,c,d,e,epsilon,max_depth
        )
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_search_source_depths_d4max(
            a,b,c,d,e,epsilon,max_depth
        )
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
