"""
Build coupled first- and second-derivative blocks from a weighted grid state.



Implement the source's augmented generalized-Laguerre Tables 2 and 3 as one

operation.  A noncontiguous principal block retains the original polynomial

degree and parameter; it is not the differentiation matrix of the selected

nodes viewed as a smaller grid.

Returns
-------
finite float np.ndarray of shape (2, m, m), ordered as first and second derivative blocks
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def build_truncated_collocation_state(
    n: int,
    alpha: float,
    nodes: np.ndarray,
    weighted_derivatives: np.ndarray,
    selected_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Return coupled derivative blocks for selected full-grid indices.

    ``nodes`` and ``weighted_derivatives`` must be the ordered finite state for
    degree ``n`` and parameter ``alpha > -1``.  If supplied,
    ``selected_indices`` must be a nonempty, strictly increasing integer
    vector of unique full-grid indices.  ``None`` selects every node.  Invalid
    shapes, signs, indices, or nonfinite values raise ``ValueError``.

    Parameters
    ----------
    n : int
        Number of positive nodes; the full augmented size is ``n + 1``.
    alpha : float
        Generalized-Laguerre parameter, strictly greater than ``-1``.
    nodes : np.ndarray
        Full augmented node vector of shape ``(n + 1,)``.
    weighted_derivatives : np.ndarray
        Weighted degree-``n`` derivatives at positive nodes, shape ``(n,)``.
    selected_indices : np.ndarray or None, optional
        Increasing full-grid indices for the requested principal block.

    Returns
    -------
    np.ndarray
        Float array of shape ``(2, m, m)``.  Plane zero is the first-order
        block and plane one is the second-order block.
    """
    return derivative_blocks  # noqa: F821 - required model stub

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy import special

def _oracle_build_truncated_collocation_state(
    n: int,
    alpha: float,
    nodes: np.ndarray,
    weighted_derivatives: np.ndarray,
    selected_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Reference augmented Table 2/Table 3 construction."""
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)):
        raise ValueError("n must be an integer")
    if not np.isscalar(alpha) or not np.isfinite(alpha) or alpha <= -1.0:
        raise ValueError("alpha must be finite and greater than -1")
    n = int(n)
    alpha = float(alpha)
    x = np.asarray(nodes, dtype=float)
    derivatives = np.asarray(weighted_derivatives, dtype=float)
    if n < 1 or x.shape != (n + 1,) or derivatives.shape != (n,):
        raise ValueError("grid state has incompatible shapes")
    if (
        not np.all(np.isfinite(x))
        or x[0] != 0.0
        or np.any(x[1:] <= 0.0)
        or np.any(np.diff(x) <= 0.0)
    ):
        raise ValueError("nodes must start at zero and increase")
    expected_signs = np.where(np.arange(n) % 2 == 0, -1.0, 1.0)
    if (
        not np.all(np.isfinite(derivatives))
        or np.any(derivatives == 0.0)
        or not np.array_equal(np.sign(derivatives), expected_signs)
    ):
        raise ValueError("weighted derivatives have invalid signs or values")

    endpoint_log = (
        special.gammaln(n + alpha + 1.0)
        - special.gammaln(alpha + 1.0)
        - special.gammaln(n + 1.0)
    )
    endpoint_coefficient = float(np.exp(endpoint_log))
    coefficients = np.concatenate(([endpoint_coefficient], x[1:] * derivatives))
    if not np.all(np.isfinite(coefficients)) or np.any(coefficients == 0.0):
        raise ValueError("scaled coefficients must be finite and nonzero")

    total = n + 1
    if selected_indices is None:
        indices = np.arange(total, dtype=int)
    else:
        raw_indices = np.asarray(selected_indices)
        if (
            raw_indices.ndim != 1
            or raw_indices.size < 1
            or raw_indices.dtype.kind not in "iu"
        ):
            raise ValueError("selected_indices must be a nonempty integer vector")
        indices = raw_indices.astype(int, copy=False)
        if (
            np.any(indices < 0)
            or np.any(indices >= total)
            or np.any(np.diff(indices) <= 0)
        ):
            raise ValueError("selected_indices must increase within the full grid")

    selected_x = x[indices]
    selected_coefficients = coefficients[indices]
    count = indices.size
    differences = selected_x[:, None] - selected_x[None, :]
    ratios = selected_coefficients[:, None] / selected_coefficients[None, :]
    off_diagonal = ~np.eye(count, dtype=bool)
    safe_differences = np.where(off_diagonal, differences, 1.0)

    first = ratios / safe_differences
    first_diagonal = np.empty(count, dtype=float)
    at_endpoint = indices == 0
    first_diagonal[at_endpoint] = -0.5 - n / (alpha + 1.0)
    first_diagonal[~at_endpoint] = (1.0 - alpha) / (2.0 * selected_x[~at_endpoint])
    first[np.arange(count), np.arange(count)] = first_diagonal

    second = 2.0 / safe_differences * (ratios * first_diagonal[:, None] - first)
    second_diagonal = np.empty(count, dtype=float)
    second_diagonal[at_endpoint] = 0.25 + n * (n + alpha + 1.0) / (
        (alpha + 1.0) * (alpha + 2.0)
    )
    positive = selected_x[~at_endpoint]
    correction = 4.0 * (alpha + 1.0) * (alpha - 1.0)
    second_diagonal[~at_endpoint] = 1.0 / 12.0 - (
        2.0 * (2.0 * n + alpha + 1.0) * positive - correction
    ) / (12.0 * positive**2)
    second[np.arange(count), np.arange(count)] = second_diagonal

    blocks = np.stack((first, second))
    if not np.all(np.isfinite(blocks)):
        raise ValueError("derivative blocks are not finite")
    return blocks

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return classical, generalized, truncated, and invalid cases."""
    return [
        {
            "setup": """import numpy as np
n = 4
alpha = 0.0
nodes = np.array([0.0, 0.3225476896193923, 1.7457611011583467, 4.536620296921128, 9.395070912301133])
weighted_derivatives = np.array([-1.9295167263574815, 0.5288490840397949, -0.2463836645483675, 0.12809234113352305])
selected_indices = None
""",
            "call": "(lambda value: float(np.dot(value[0].ravel(), np.arange(1, value[0].size + 1, dtype=float))))(build_truncated_collocation_state(n, alpha, nodes, weighted_derivatives, selected_indices))",
            "gold_call": "(lambda value: float(np.dot(value[0].ravel(), np.arange(1, value[0].size + 1, dtype=float))))(_oracle_build_truncated_collocation_state(n, alpha, nodes, weighted_derivatives, selected_indices))",
        },
        {
            "setup": """import numpy as np
n = 1
alpha = 2.5
nodes = np.array([0.0, 3.5])
weighted_derivatives = np.array([-0.17377394345044514])
selected_indices = None
""",
            "call": "(lambda value: float(np.dot(value[1].ravel(), np.arange(1, value[1].size + 1, dtype=float))))(build_truncated_collocation_state(n, alpha, nodes, weighted_derivatives, selected_indices))",
            "gold_call": "(lambda value: float(np.dot(value[1].ravel(), np.arange(1, value[1].size + 1, dtype=float))))(_oracle_build_truncated_collocation_state(n, alpha, nodes, weighted_derivatives, selected_indices))",
        },
        {
            "setup": """import numpy as np
n = 6
alpha = -0.6
nodes = np.array([0.0, 0.07674267368646596, 0.8413066603703071, 2.4646529229830443, 5.079644083448372, 8.980233158929984, 14.957420500581827])
weighted_derivatives = np.array([-1.6174834229538, 0.5511666024408923, -0.33418432022223143, 0.23405282970252178, -0.17153085041572877, 0.12157870027129943])
selected_indices = np.array([0, 2, 6])
""",
            "call": "(lambda value: float(np.dot(value[0].ravel(), np.arange(1, value[0].size + 1, dtype=float))))(build_truncated_collocation_state(n, alpha, nodes, weighted_derivatives, selected_indices))",
            "gold_call": "(lambda value: float(np.dot(value[0].ravel(), np.arange(1, value[0].size + 1, dtype=float))))(_oracle_build_truncated_collocation_state(n, alpha, nodes, weighted_derivatives, selected_indices))",
        },
        {
            "setup": """import numpy as np
n = 7
alpha = 1.75
nodes = np.array([0.0, 0.6982270967000053, 1.9874269753445861, 3.9482194676942703, 6.675227468040911, 10.33707861234613, 15.268203467160129, 22.335616912713967])
weighted_derivatives = np.array([-10.64942754544752, 1.9640104665918596, -0.6368164570840024, 0.26536759846289293, -0.1258326521529564, 0.06295053576398975, -0.03012272036405051])
selected_indices = np.array([1, 3, 5, 7])
""",
            "call": "(lambda value: float(np.dot(value[1].ravel(), np.arange(1, value[1].size + 1, dtype=float))))(build_truncated_collocation_state(n, alpha, nodes, weighted_derivatives, selected_indices))",
            "gold_call": "(lambda value: float(np.dot(value[1].ravel(), np.arange(1, value[1].size + 1, dtype=float))))(_oracle_build_truncated_collocation_state(n, alpha, nodes, weighted_derivatives, selected_indices))",
        },
        {
            "setup": """import numpy as np
n = 3
alpha = 0.0
nodes = np.array([0.0, 0.4, 2.0, 5.0])
weighted_derivatives = np.array([-1.0, 0.5, -0.1])
selected_indices = np.array([0, 3, 2])
def run_model():
    try:
        build_truncated_collocation_state(n, alpha, nodes, weighted_derivatives, selected_indices)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_build_truncated_collocation_state(n, alpha, nodes, weighted_derivatives, selected_indices)
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
