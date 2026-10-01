"""
Map a full augmented second-derivative operator to the physical half-line.



The coordinate scale acts quadratically on the second derivative.  The

homogeneous endpoint condition is then imposed by removing the endpoint row

and column, while the positive physical nodes retain their original order.

Returns
-------
tuple of finite float arrays with shapes (N - 1,) and (N - 1, N - 1)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def scale_and_reduce_second_order(
    nodes: np.ndarray, second_order: np.ndarray, beta: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return positive physical nodes and the reduced physical operator.

    ``nodes`` must be a finite increasing vector beginning at zero,
    ``second_order`` must be a matching finite square matrix, and ``beta``
    must be a finite positive scalar.  Invalid inputs raise ``ValueError``.

    Parameters
    ----------
    nodes : np.ndarray
        Unscaled augmented nodes, shape ``(N,)``.
    second_order : np.ndarray
        Unscaled second-derivative matrix, shape ``(N, N)``.
    beta : float
        Positive coordinate scale.

    Returns
    -------
    physical_nodes : np.ndarray
        Positive physical nodes, shape ``(N - 1,)``.
    reduced_second_order : np.ndarray
        Physical second-derivative operator after endpoint deletion, shape
        ``(N - 1, N - 1)``.
    """
    return physical_nodes, reduced_second_order  # noqa: F821 - model stub

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_scale_and_reduce_second_order(
    nodes: np.ndarray, second_order: np.ndarray, beta: float
) -> tuple[np.ndarray, np.ndarray]:
    """Reference coordinate scaling and endpoint deletion."""
    x = np.asarray(nodes, dtype=float)
    matrix = np.asarray(second_order, dtype=float)
    if x.ndim != 1 or x.size < 2 or matrix.shape != (x.size, x.size):
        raise ValueError("nodes and second_order have incompatible shapes")
    if (
        not np.all(np.isfinite(x))
        or x[0] != 0.0
        or np.any(x[1:] <= 0.0)
        or np.any(np.diff(x) <= 0.0)
        or not np.all(np.isfinite(matrix))
    ):
        raise ValueError("nodes and second_order must be finite and ordered")
    if not np.isscalar(beta) or not np.isfinite(beta) or beta <= 0.0:
        raise ValueError("beta must be positive and finite")
    beta = float(beta)
    physical_nodes = x[1:] / beta
    reduced_second_order = beta**2 * matrix[1:, 1:]
    if not np.all(np.isfinite(reduced_second_order)):
        raise ValueError("scaled operator is not finite")
    return physical_nodes, reduced_second_order

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return ordinary, unit-scale, small-scale, and invalid cases."""
    return [
        {
            "setup": """import numpy as np
nodes = np.array([0.0, 0.25, 2.0, 8.0])
second_order = np.arange(16, dtype=float).reshape(4, 4) - 5.0
beta = 10.0
""",
            "call": "(lambda value: float(np.dot(value[0], np.arange(1, value[0].size + 1, dtype=float)) + np.dot(value[1].ravel(), np.arange(value[0].size + 1, value[0].size + value[1].size + 1, dtype=float))))(scale_and_reduce_second_order(nodes, second_order, beta))",
            "gold_call": "(lambda value: float(np.dot(value[0], np.arange(1, value[0].size + 1, dtype=float)) + np.dot(value[1].ravel(), np.arange(value[0].size + 1, value[0].size + value[1].size + 1, dtype=float))))(_oracle_scale_and_reduce_second_order(nodes, second_order, beta))",
        },
        {
            "setup": """import numpy as np
nodes = np.array([0.0, 1.0])
second_order = np.array([[1.25, -2.0], [0.5, -0.75]])
beta = 1.0
""",
            "call": "(lambda value: float(np.dot(value[0], np.arange(1, value[0].size + 1, dtype=float)) + np.dot(value[1].ravel(), np.arange(value[0].size + 1, value[0].size + value[1].size + 1, dtype=float))))(scale_and_reduce_second_order(nodes, second_order, beta))",
            "gold_call": "(lambda value: float(np.dot(value[0], np.arange(1, value[0].size + 1, dtype=float)) + np.dot(value[1].ravel(), np.arange(value[0].size + 1, value[0].size + value[1].size + 1, dtype=float))))(_oracle_scale_and_reduce_second_order(nodes, second_order, beta))",
        },
        {
            "setup": """import numpy as np
nodes = np.array([0.0, 1e-6, 3.0])
second_order = np.eye(3)
beta = 0.125
""",
            "call": "(lambda value: float(np.dot(value[0], np.arange(1, value[0].size + 1, dtype=float)) + np.dot(value[1].ravel(), np.arange(value[0].size + 1, value[0].size + value[1].size + 1, dtype=float))))(scale_and_reduce_second_order(nodes, second_order, beta))",
            "gold_call": "(lambda value: float(np.dot(value[0], np.arange(1, value[0].size + 1, dtype=float)) + np.dot(value[1].ravel(), np.arange(value[0].size + 1, value[0].size + value[1].size + 1, dtype=float))))(_oracle_scale_and_reduce_second_order(nodes, second_order, beta))",
        },
        {
            "setup": """import numpy as np
nodes = np.array([0.0, 1.0])
second_order = np.eye(2)
beta = 0.0
def run_model():
    try:
        scale_and_reduce_second_order(nodes, second_order, beta)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_scale_and_reduce_second_order(nodes, second_order, beta)
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
