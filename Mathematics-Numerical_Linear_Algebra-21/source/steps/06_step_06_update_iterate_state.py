"""
Update the current iterate and append the newly computed correction direction.

After a correction p_k is obtained, the new iterate is formed by addition and the correction is appended to the historical matrix for later iterations.

Returns
-------
tuple[np.ndarray, np.ndarray], containing the updated iterate x_new and the updated historical matrix P_new
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def update_iterate_state(
    x: np.ndarray,
    P: np.ndarray,
    p: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Update the iterate and append the new correction direction.

    Parameters
    ----------
    x : np.ndarray
        Current iterate of shape (n,).
    P : np.ndarray
        Historical correction matrix of shape (n, r).
    p : np.ndarray
        New correction direction of shape (n,).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Updated iterate x_new and updated historical matrix P_new.

    Raises
    ------
    ValueError
        If x or p is not one-dimensional, p and x have different shapes,
        P is not two-dimensional with a compatible row count, an input
        contains non-finite values, the correction is zero, the historical
        correction space has exhausted the available dimension, or the
        correction is numerically dependent on the historical correction
        space.

    Notes
    -----
    The result is deterministic for identical inputs.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_update_iterate_state(
    x: np.ndarray,
    P: np.ndarray,
    p: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic reference implementation."""
    import numpy as np

    x = np.asarray(x, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)

    # -------------------------
    # Input validation
    # -------------------------
    if x.ndim != 1 or p.ndim != 1:
        raise ValueError("x and p must be one-dimensional")

    if P.ndim != 2:
        raise ValueError("P must be two-dimensional")

    if p.shape != x.shape:
        raise ValueError("p must have the same shape as x")

    if P.shape[0] != x.size:
        raise ValueError("P has incompatible row count")

    if not (
        np.all(np.isfinite(x))
        and np.all(np.isfinite(P))
        and np.all(np.isfinite(p))
    ):
        raise ValueError("inputs must be finite")

    n = x.size

    # -------------------------
    # Validate new correction
    # -------------------------
    p_norm = float(np.linalg.norm(p))

    if not np.isfinite(p_norm) or p_norm == 0.0:
        raise ValueError(
            "correction direction is zero or non-finite"
        )

    # -------------------------
    # Prevent exhaustion
    # -------------------------
    if P.shape[1] >= n:
        raise ValueError(
            "historical correction space has exhausted the "
            "available dimension"
        )

    # -------------------------
    # Check numerical independence
    # -------------------------
    if P.shape[1] > 0:
        try:
            coefficients, _, _, _ = np.linalg.lstsq(
                P,
                p,
                rcond=None,
            )
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "historical independence check failed"
            ) from exc

        residual = p - P @ coefficients
        residual_norm = float(np.linalg.norm(residual))

        if not np.isfinite(residual_norm):
            raise ValueError(
                "historical independence check is non-finite"
            )

        # Reject corrections whose component outside the historical
        # subspace is only at floating-point noise level.
        tolerance = (
            1e-10 * max(1.0, p_norm)
        )

        if residual_norm <= tolerance:
            raise ValueError(
                "correction direction is numerically contained "
                "in the historical subspace"
            )

    x_new = x + p

    if not np.all(np.isfinite(x_new)):
        raise ValueError("updated iterate is non-finite")

    P_new = np.column_stack((P, p))

    if not np.all(np.isfinite(P_new)):
        raise ValueError("updated historical matrix is non-finite")

    return (
        np.asarray(x_new, dtype=np.float64),
        np.asarray(P_new, dtype=np.float64),
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""

    return [
        {
            "setup": (
                "import numpy as np\n"
                "x = np.array([1.0, 2.0], dtype=np.float64)\n"
                "P = np.array([[0.5], [1.0]], dtype=np.float64)\n"
                "p = np.array([0.2, -0.3], dtype=np.float64)\n"

            ),
            "call": "update_iterate_state(x, P, p)",
            "gold_call": "_oracle_update_iterate_state(x, P, p)",
        },
        {
            "setup": (
                "import numpy as np\n"
                "x = np.array([2.0], dtype=np.float64)\n"
                "P = np.empty((1, 0), dtype=np.float64)\n"
                "p = np.array([-0.5], dtype=np.float64)\n"

            ),
            "call": "update_iterate_state(x, P, p)",
            "gold_call": "_oracle_update_iterate_state(x, P, p)",
        },
        {
            "setup": (
                "import numpy as np\n"
                "x = np.array([1.0, 2.0], dtype=np.float64)\n"
                "P = np.eye(2, dtype=np.float64)\n"
                "p = np.array([0.5, -0.5], dtype=np.float64)\n"
                "def catches_value_error(fn):\n"
                "    try:\n"
                "        fn(x, P, p)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2"
            ),
            "call": "catches_value_error(update_iterate_state)",
            "gold_call": "catches_value_error(_oracle_update_iterate_state)",
        },
        {
            "setup": (
                "import numpy as np\n"
                "x = np.array([1.0, 2.0], dtype=np.float64)\n"
                "P = np.array([[1.0], [2.0]], dtype=np.float64)\n"
                "p = np.array([2.0, 4.0], dtype=np.float64)\n"
                "def catches_value_error(fn):\n"
                "    try:\n"
                "        fn(x, P, p)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2"
            ),
            "call": "catches_value_error(update_iterate_state)",
            "gold_call": "catches_value_error(_oracle_update_iterate_state)",
        },
    ]
