"""
Differentiate the weighted overlap area for a multi-coordinate state.



The endpoint-owner columns select either a fixed fiber endpoint or one of the

stacked bodies. Each selected root carries a gradient and Hessian. Accumulate

entry and exit contributions with stable line norms and cancellation-resistant

summation in every derivative channel.

On a fixed active-set branch, differentiating the weighted overlap estimate with respect to state channel $q_j$ gives



$$

\frac{\partial V_c}{\partial q_j}

=\sum_i w_i\lVert q_i\rVert

\left(

\frac{\partial h_{\mathrm{out},i}}{\partial q_j}

-

\frac{\partial h_{\mathrm{in},i}}{\partial q_j}

\right).

$$



The entry and exit terms have opposite signs. An owner code selects the matching body, root side, and complete $p$-component Jacobian; a clipping endpoint has a zero vector. Empty intervals contribute zero. Since large signed endpoint terms can nearly cancel, each state channel requires cancellation-resistant summation, and segment lengths require scaled hypotenuse evaluation.



On the same fixed active-set branch, the Hessian follows by replacing each

selected endpoint gradient with its root Hessian:



$$

\frac{\partial^2 V_c}{\partial q_j\partial q_k}

=\sum_i w_i\lVert q_i\rVert

\left(h_{\mathrm{out},i,jk}-h_{\mathrm{in},i,jk}\right).

$$



Owner selection must therefore move the gradient and Hessian together. Active

root Hessians must be finite and symmetric. Cancellation-resistant summation

is applied independently to every entry of the returned derivative table.

Returns
-------
finite float64 np.ndarray of shape (p + 1, p): row 0 is the generalized overlap gradient and rows 1..p are its symmetric Hessian
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import math
import numpy as np


def differentiate_overlap_area(
    interval_state: np.ndarray,
    root_sensitivities: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Return the generalized gradient and Hessian of the contact area.

    Parameters
    ----------
    interval_state : np.ndarray
        Shape ``(n_fibers, 4)`` with bounds and owner codes.
    root_sensitivities : np.ndarray
        Shape ``(n_bodies, n_fibers, 2, p + 1, p)``. Derivative row zero is
        the root gradient and rows 1 through ``p`` are its Hessian. Owner code
        ``j + 1`` selects body index ``j``.
    starts, ends : np.ndarray
        Matching fiber endpoint arrays of shape ``(n_fibers, 2)``.
    weights : np.ndarray
        Nonnegative line weights of shape ``(n_fibers,)`` in metres.

    Returns
    -------
    np.ndarray
        Finite float64 array with shape ``(p + 1, p)``. Row zero is the overlap
        gradient and rows 1 through ``p`` are its symmetric Hessian. Every
        entry is summed accurately when large contributions nearly cancel.

    Raises
    ------
    ValueError
        If an active owner code is outside ``{0, ..., n_bodies}`` or an active
        endpoint has no finite derivative, or an active root Hessian is not
        symmetric to numerical precision.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math
import numpy as np


# =============================================================================
# ORACLE SOLUTION
# =============================================================================


def _oracle_differentiate_overlap_area(
    interval_state: np.ndarray,
    root_sensitivities: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Reference endpoint-owner accumulation."""
    state = np.asarray(interval_state, dtype=np.float64)
    derivatives = np.asarray(root_sensitivities, dtype=np.float64)
    starts = np.asarray(starts, dtype=np.float64)
    ends = np.asarray(ends, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if state.ndim != 2 or state.shape[0] < 1 or state.shape[1] != 4:
        raise ValueError("interval_state must have shape (n_fibers, 4)")
    if (
        derivatives.ndim != 5
        or derivatives.shape[0] < 2
        or derivatives.shape[1:3] != (state.shape[0], 2)
        or derivatives.shape[4] < 1
        or derivatives.shape[3] != derivatives.shape[4] + 1
    ):
        raise ValueError(
            "root_sensitivities must have shape "
            "(n_bodies, n_fibers, 2, p + 1, p)"
        )
    expected_roots = (state.shape[0], 2)
    if starts.shape != expected_roots or ends.shape != expected_roots:
        raise ValueError("fiber endpoints must have shape (n_fibers, 2)")
    if weights.shape != (state.shape[0],):
        raise ValueError("weights must have shape (n_fibers,)")
    if not all(np.all(np.isfinite(value)) for value in (starts, ends, weights)):
        raise ValueError("fiber data and weights must be finite")
    if np.any(weights < 0.0) or not np.any(weights > 0.0):
        raise ValueError("weights must be nonnegative with at least one positive value")
    segments = ends - starts
    lengths = np.hypot(segments[:, 0], segments[:, 1])
    if np.any(lengths <= 0.0):
        raise ValueError("fibers must have positive length")

    active = np.isfinite(state[:, 0])
    if np.any(active & ~np.all(np.isfinite(state), axis=1)):
        raise ValueError("active interval rows must be finite")
    if np.any((~active) & ~np.all(np.isnan(state), axis=1)):
        raise ValueError("inactive interval rows must contain four NaNs")
    owners = state[:, 2:4]
    valid_owner_values = np.arange(derivatives.shape[0] + 1, dtype=np.float64)
    if np.any(active & ~np.all(np.isin(owners, valid_owner_values), axis=1)):
        raise ValueError("endpoint owner is outside the available body range")

    derivative_shape = (state.shape[0], derivatives.shape[3], derivatives.shape[4])
    lower_derivative = np.zeros(derivative_shape, dtype=np.float64)
    upper_derivative = np.zeros_like(lower_derivative)
    for fiber in np.flatnonzero(active):
        lower_owner = int(state[fiber, 2])
        upper_owner = int(state[fiber, 3])
        if lower_owner > 0:
            lower_derivative[fiber] = derivatives[lower_owner - 1, fiber, 0]
        if upper_owner > 0:
            upper_derivative[fiber] = derivatives[upper_owner - 1, fiber, 1]
        if not np.all(np.isfinite(lower_derivative[fiber])) or not np.all(
            np.isfinite(upper_derivative[fiber])
        ):
            raise ValueError("an active endpoint has no finite derivative")
        for endpoint in (lower_derivative[fiber], upper_derivative[fiber]):
            if not np.allclose(
                endpoint[1:], endpoint[1:].T, rtol=1e-10, atol=1e-12
            ):
                raise ValueError("an active root Hessian is not symmetric")
    contributions = (
        weights[:, None, None] * lengths[:, None, None]
        * (upper_derivative - lower_derivative)
    )
    derivative = np.empty(contributions.shape[1:], dtype=np.float64)
    for row in range(derivative.shape[0]):
        for column in range(derivative.shape[1]):
            derivative[row, column] = math.fsum(
                contributions[:, row, column].tolist()
            )
    if not np.all(np.isfinite(derivative)):
        raise ValueError("overlap derivative must be finite")
    return derivative

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return multi-body Hessian, cancellation, empty, and invalid cases."""
    return [
        {
            "setup": """
import numpy as np
state = np.array([
    [0.0, 0.5, 0.0, 1.0],
    [0.0, 0.5, 0.0, 1.0],
    [0.0, 0.5, 0.0, 1.0],
    [0.2, 0.7, 2.0, 3.0],
    [0.0, 0.8, 0.0, 0.0],
    [0.1, 0.9, 3.0, 2.0],
])
derivatives = np.zeros((3, 6, 2, 4, 3))
derivatives[0, 0, 1, 0] = [1.0e16, -1.0e16, 2.0]
derivatives[0, 1, 1, 0] = [1.0, 2.0, 3.0]
derivatives[0, 2, 1, 0] = [-1.0e16, 1.0e16, -2.0]
derivatives[1, 3, 0, 0] = [0.2, -0.1, 0.4]
derivatives[2, 3, 1, 0] = [0.7, 0.3, -0.2]
derivatives[2, 5, 0, 0] = [-0.4, 0.2, 0.1]
derivatives[1, 5, 1, 0] = [0.1, -0.3, 0.6]
derivatives[0, 0, 1, 1:] = np.diag([1.0e16, -1.0e16, 2.0])
derivatives[0, 1, 1, 1:] = np.array([
    [1.0, 0.2, -0.1], [0.2, 2.0, 0.3], [-0.1, 0.3, 3.0]
])
derivatives[0, 2, 1, 1:] = np.diag([-1.0e16, 1.0e16, -2.0])
derivatives[1, 3, 0, 1:] = np.array([
    [0.4, -0.2, 0.1], [-0.2, 0.3, 0.05], [0.1, 0.05, -0.1]
])
derivatives[2, 3, 1, 1:] = np.array([
    [0.8, 0.1, -0.3], [0.1, -0.2, 0.4], [-0.3, 0.4, 0.6]
])
derivatives[2, 5, 0, 1:] = np.diag([-0.4, 0.2, 0.1])
derivatives[1, 5, 1, 1:] = np.array([
    [0.1, -0.05, 0.02], [-0.05, -0.3, 0.04], [0.02, 0.04, 0.6]
])
starts = np.zeros((6, 2))
ends = np.array([
    [1.0, 0.0], [1.0, 0.0], [1.0, 0.0],
    [3.0, 4.0], [0.0, 2.0], [1.0, 1.0],
])
weights = np.array([1.0, 1.0, 1.0, 0.1, 0.0, 0.2])
""",
            "call": "differentiate_overlap_area(state, derivatives, starts, ends, weights)",
            "gold_call": "_oracle_differentiate_overlap_area(state, derivatives, starts, ends, weights)",
        },
        {
            "setup": """
import numpy as np
state = np.array([
    [0.0, 0.75, 0.0, 2.0],
    [np.nan, np.nan, np.nan, np.nan],
])
derivatives = np.full((2, 2, 2, 3, 2), np.nan)
derivatives[:, 0] = 0.0
derivatives[1, 0, 1, 0] = [-0.3, 0.4]
derivatives[1, 0, 1, 1:] = [[0.2, -0.1], [-0.1, 0.5]]
starts = np.array([[0.0, 0.0], [0.0, 1.0]])
ends = np.array([[1.0, 0.0], [1.0, 1.0]])
weights = np.array([0.4, 0.4])
""",
            "call": "differentiate_overlap_area(state, derivatives, starts, ends, weights)",
            "gold_call": "_oracle_differentiate_overlap_area(state, derivatives, starts, ends, weights)",
        },
        {
            "setup": """
import numpy as np
state = np.full((2, 4), np.nan)
derivatives = np.full((3, 2, 2, 5, 4), np.nan)
starts = np.array([[0.0, 0.0], [0.0, 1.0]])
ends = np.array([[1.0, 0.0], [1.0, 1.0]])
weights = np.array([0.4, 0.4])
""",
            "call": "differentiate_overlap_area(state, derivatives, starts, ends, weights)",
            "gold_call": "_oracle_differentiate_overlap_area(state, derivatives, starts, ends, weights)",
        },
        {
            "setup": """
import numpy as np
state = np.array([[0.2, 0.7, 4.0, 2.0]])
derivatives = np.zeros((3, 1, 2, 3, 2))
starts = np.array([[0.0, 0.0]])
ends = np.array([[1.0, 0.0]])
weights = np.array([0.4])
def run_model():
    try:
        differentiate_overlap_area(state, derivatives, starts, ends, weights)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_differentiate_overlap_area(state, derivatives, starts, ends, weights)
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
