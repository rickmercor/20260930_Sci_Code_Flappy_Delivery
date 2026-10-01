"""
Compute the scalar-normalized correction from the current projected direction.

The denominator is evaluated stably as the squared norm of the image of the projected direction under A, which is algebraically equivalent to the cancellation-prone weighted expression in exact arithmetic.

Returns
-------
tuple[np.ndarray, float], containing p_k and gamma
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_projected_update(
    A: np.ndarray,
    b: np.ndarray,
    x: np.ndarray,
    s: np.ndarray,
    projected_direction: np.ndarray,
    delta: float,
) -> tuple[np.ndarray, float]:
    """Compute the correction and scalar multiplier.

    ``delta`` is retained as the historical projection checkpoint. For stable
    numerical evaluation, use ``D = ||A @ projected_direction||_2^2`` directly
    and do not subtract nearly equal weighted quantities.

    Returns
    -------
    tuple[np.ndarray, float]
        The correction vector p and scalar multiplier gamma.

    Raises
    ------
    ValueError
        If inputs are invalid, the projected direction has zero image under
        A, or the multiplier/correction becomes non-finite.

    Notes
    -----
    The result is deterministic for identical inputs.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_compute_projected_update(
    A: np.ndarray,
    b: np.ndarray,
    x: np.ndarray,
    s: np.ndarray,
    projected_direction: np.ndarray,
    delta: float,
) -> tuple[np.ndarray, float]:
    """Deterministic stable reference implementation."""
    import numpy as np

    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    projected_direction = np.asarray(projected_direction, dtype=np.float64)

    if A.ndim != 2:
        raise ValueError("A must be a matrix")
    m, n = A.shape
    if b.ndim != 1 or x.ndim != 1 or s.ndim != 1:
        raise ValueError("b, x, and s must be one-dimensional")
    if b.shape != (m,):
        raise ValueError("b has incompatible shape")
    if x.shape != (n,):
        raise ValueError("x has incompatible shape")
    if s.shape != (m,):
        raise ValueError("s has incompatible shape")
    if projected_direction.shape != (n,):
        raise ValueError("projected direction has incompatible shape")
    if not np.isfinite(delta):
        raise ValueError("delta must be finite")
    if not (np.all(np.isfinite(A)) and np.all(np.isfinite(b)) and np.all(np.isfinite(x)) and np.all(np.isfinite(s)) and np.all(np.isfinite(projected_direction))):
        raise ValueError("inputs must be finite")

    residual = b - A @ x
    if not np.all(np.isfinite(residual)):
        raise ValueError("residual is non-finite")

    numerator = float(s @ residual)
    if not np.isfinite(numerator):
        raise ValueError("update numerator is non-finite")

    projected_image = A @ projected_direction
    if not np.all(np.isfinite(projected_image)):
        raise ValueError("projected direction image is non-finite")

    denominator = float(projected_image @ projected_image)
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("update denominator is zero or numerically invalid")

    gamma = numerator / denominator
    if not np.isfinite(gamma):
        raise ValueError("update multiplier is non-finite")

    p = gamma * projected_direction
    if not np.all(np.isfinite(p)):
        raise ValueError("correction vector is non-finite")

    return np.asarray(p, dtype=np.float64), float(gamma)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""

    return [
        {
            "setup": """import numpy as np

A = np.array([[1.0, 2.0], [2.0, 1.0], [1.0, 0.0]], dtype=np.float64)
b = np.array([1.0, 0.0, 2.0], dtype=np.float64)
x = np.zeros(2, dtype=np.float64)
s = np.array([1.0, 2.0, -1.0], dtype=np.float64)
projected_direction = np.linalg.lstsq(A, s, rcond=None)[0]
delta = 0.0
""",
            "call": "compute_projected_update(A, b, x, s, projected_direction, delta)",
            "gold_call": "_oracle_compute_projected_update(A, b, x, s, projected_direction, delta)",
        },
        {
            "setup": """import numpy as np

A = np.array([[2.0]], dtype=np.float64)
b = np.array([4.0], dtype=np.float64)
x = np.array([0.0], dtype=np.float64)
s = np.array([1.0], dtype=np.float64)
projected_direction = np.array([0.5], dtype=np.float64)
delta = 0.0
""",
            "call": "compute_projected_update(A, b, x, s, projected_direction, delta)",
            "gold_call": "_oracle_compute_projected_update(A, b, x, s, projected_direction, delta)",
        },
        {
            "setup": """import numpy as np

A = np.array([[1.0, 2.0], [2.0, 1.0], [1.0, 0.0]], dtype=np.float64)
b = np.array([1.0, 0.0, 2.0], dtype=np.float64)
x = np.zeros(2, dtype=np.float64)
s = np.array([0.5, -1.0, 0.7], dtype=np.float64)
P = np.array([[1.0], [0.2]], dtype=np.float64)
c = np.linalg.lstsq(A @ P, s, rcond=None)[0]
projected_direction = np.linalg.lstsq(A, s, rcond=None)[0] - P @ c
historical_component = A @ (P @ c)
delta = float(historical_component @ historical_component)
""",
            "call": "compute_projected_update(A, b, x, s, projected_direction, delta)",
            "gold_call": "_oracle_compute_projected_update(A, b, x, s, projected_direction, delta)",
        },
        {
            "setup": """import numpy as np

A = np.eye(2, dtype=np.float64)
b = np.array([1.0, 2.0], dtype=np.float64)
x = np.zeros(2, dtype=np.float64)
W = np.eye(2, dtype=np.float64)
s = np.array([1.0, 0.0], dtype=np.float64)
projected_direction = np.zeros(2, dtype=np.float64)
delta = 0.0

def catches_value_error(fn):
    try:
        fn(A, b, x, s, projected_direction, delta)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "catches_value_error(compute_projected_update)",
            "gold_call": "catches_value_error(_oracle_compute_projected_update)",
        },
    ]
