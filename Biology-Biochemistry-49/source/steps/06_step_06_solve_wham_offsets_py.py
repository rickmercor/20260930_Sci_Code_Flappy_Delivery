"""
Iterate simultaneous relative-offset updates from zero to the documented inclusive convergence threshold.

The WHAM equations determine relative rather than absolute offsets. A fixed reference, zero initialization, simultaneous updates, and an inclusive maximum-change test make the deterministic fixed-point execution reproducible.

Returns
-------
tuple[np.ndarray, int]: converged gauge-fixed float64 offsets of shape (w,) and the positive integer number of simultaneous updates performed.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def solve_wham_offsets(
    bias_values: "np.ndarray",
    sample_counts: "np.ndarray",
    beta: float,
    tolerance: float,
    max_iterations: int,
    reference_index: int,
) -> tuple["np.ndarray", int]:
    r"""Iterate simultaneous updates to the gauge-fixed WHAM solution.

    Parameters
    ----------
    bias_values : np.ndarray
        Finite bias matrix with shape ``(w, n)``.
    sample_counts : np.ndarray
        Finite positive pathway counts with shape ``(w,)``.
    beta : float
        Finite positive inverse temperature.
    tolerance : float
        Finite strictly positive inclusive convergence tolerance.
    max_iterations : int
        Positive integer update cap.
    reference_index : int
        In-range integer pathway index fixed to zero.

    Returns
    -------
    tuple[np.ndarray, int]
        Converged float64 relative offsets with shape ``(w,)`` and the number
        of simultaneous updates performed.

    Raises
    ------
    ValueError
        If shapes or values violate an upstream contract, tolerance is not
        finite and strictly positive, max_iterations is not a positive
        integer, or reference_index is invalid.
    RuntimeError
        If convergence is not reached within max_iterations.

    Notes
    -----
    Initialize every offset to zero and test the maximum absolute change after
    each update with an inclusive comparison.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_solve_wham_offsets(
    bias_values: "np.ndarray",
    sample_counts: "np.ndarray",
    beta: float,
    tolerance: float,
    max_iterations: int,
    reference_index: int,
) -> tuple["np.ndarray", int]:
    bias_values = np.asarray(bias_values, dtype=np.float64)
    if bias_values.ndim != 2 or bias_values.shape[0] == 0 or bias_values.shape[1] == 0:
        raise ValueError("bias_values must have shape (w, n) with w,n > 0")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    if isinstance(max_iterations, (bool, np.bool_)) or not isinstance(
        max_iterations, (int, np.integer)
    ) or int(max_iterations) <= 0:
        raise ValueError("max_iterations must be a positive integer")
    current = np.zeros(bias_values.shape[0], dtype=np.float64)
    for iteration in range(1, int(max_iterations) + 1):
        updated = _oracle_wham_offset_update(
            bias_values,
            sample_counts,
            beta,
            current,
            reference_index,
        )
        if np.max(np.abs(updated - current)) <= tolerance:
            return updated, iteration
        current = updated
    raise RuntimeError("WHAM offsets did not converge")

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return five convergence, reference, boundary, and error cases."""

    return [
        {
            "setup": """import numpy as np
bias=np.array([[0.1,-0.2,0.4,0.0],[0.3,0.0,-0.1,0.2],[0.2,0.1,0.5,-0.3]])
counts=np.array([4.0,6.0,5.0])
""",
            "call": '(lambda r:np.concatenate([np.asarray(r[0],dtype=float).ravel(),np.asarray([r[1]],dtype=float)]))(solve_wham_offsets(bias.copy(),counts.copy(),1.3,1e-12,10000,0))',
            "gold_call": '(lambda r:np.concatenate([np.asarray(r[0],dtype=float).ravel(),np.asarray([r[1]],dtype=float)]))(_oracle_solve_wham_offsets(bias.copy(),counts.copy(),1.3,1e-12,10000,0))',
        },
        {
            "setup": """import numpy as np
bias=np.array([[0.1,-0.2,0.4,0.0]])
counts=np.array([4.0])
""",
            "call": '(lambda r:np.concatenate([np.asarray(r[0],dtype=float).ravel(),np.asarray([r[1]],dtype=float)]))(solve_wham_offsets(bias.copy(),counts.copy(),1.3,1e-12,100,0))',
            "gold_call": '(lambda r:np.concatenate([np.asarray(r[0],dtype=float).ravel(),np.asarray([r[1]],dtype=float)]))(_oracle_solve_wham_offsets(bias.copy(),counts.copy(),1.3,1e-12,100,0))',
        },
        {
            "setup": """import numpy as np
bias=np.array([[0.1,-0.2,0.4],[0.3,0.0,-0.1],[0.2,0.1,0.5]])
counts=np.array([4.0,6.0,5.0])
""",
            "call": '(lambda r:np.concatenate([np.asarray(r[0],dtype=float).ravel(),np.asarray([r[1]],dtype=float)]))(solve_wham_offsets(bias.copy(),counts.copy(),1.3,1e-10,10000,2))',
            "gold_call": '(lambda r:np.concatenate([np.asarray(r[0],dtype=float).ravel(),np.asarray([r[1]],dtype=float)]))(_oracle_solve_wham_offsets(bias.copy(),counts.copy(),1.3,1e-10,10000,2))',
        },
        {
            "setup": """import numpy as np
bias=np.array([[0.1,0.2],[0.3,0.4]]); counts=np.array([2.0,2.0])
def model():
    try: solve_wham_offsets(bias.copy(),counts.copy(),1.0,0.0,100,0); return 0
    except ValueError: return 1
    except Exception: return 2
def oracle():
    try: _oracle_solve_wham_offsets(bias.copy(),counts.copy(),1.0,0.0,100,0); return 0
    except ValueError: return 1
    except Exception: return 2
""",
            "call": 'model()',
            "gold_call": 'oracle()',
        },
        {
            "setup": """import numpy as np
bias=np.array([[0.1,-0.2,0.4],[0.3,0.0,-0.1],[0.2,0.1,0.5]])
counts=np.array([4.0,6.0,5.0])
def model():
    try: solve_wham_offsets(bias.copy(),counts.copy(),1.3,1e-16,1,0); return 0
    except RuntimeError: return 1
    except Exception: return 2
def oracle():
    try: _oracle_solve_wham_offsets(bias.copy(),counts.copy(),1.3,1e-16,1,0); return 0
    except RuntimeError: return 1
    except Exception: return 2
""",
            "call": 'model()',
            "gold_call": 'oracle()',
        },
    ]
