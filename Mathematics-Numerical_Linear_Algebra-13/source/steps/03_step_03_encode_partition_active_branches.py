"""
Encode the active max-range channel throughout every partition cell.



The partition from the previous step removes all nonzero tie points.  At each

cell midpoint, the maximizing channel therefore identifies the active row and

column range law throughout that open cell.  The packed numerical table keeps

the variable-size certificate compatible with strict array comparison.

Returns
-------
finite float np.ndarray of shape (1 + (s-1)*(3+m+n),), packed active-cell table
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def encode_partition_active_branches(
    A: np.ndarray,
    B: np.ndarray,
    partition: np.ndarray,
    tolerance: float = 1e-12,
) -> np.ndarray:
    """Encode active channels for every open cell of a switch partition.

    Raises ``ValueError`` unless ``A`` and ``B`` are finite two-dimensional
    arrays with shapes ``(m, 2)`` and ``(2, n)`` for positive ``m`` and ``n``;
    ``tolerance`` is a non-Boolean scalar that is finite and strictly
    positive; and ``partition`` is a finite one-dimensional packed vector that
    is internally consistent, meaning its header ``s`` is an integer of at
    least 2 matching a total length of ``2 * s``, its ``s`` switches increase
    with consecutive gaps larger than ``tolerance``, and its ``s - 1`` probes
    each lie strictly inside their own cell and equal that cell's midpoint to
    within ``tolerance``.

    Parameters
    ----------
    A : np.ndarray
        Finite left factor of shape ``(m, 2)``.
    B : np.ndarray
        Finite right factor of shape ``(2, n)``.
    partition : np.ndarray
        Packed vector ``[s, switches, probes]`` from the partition step.
    tolerance : float, optional
        Finite positive structural tolerance.

    Returns
    -------
    np.ndarray
        Packed vector beginning with the number of cells.  Each following
        record is ``[lower, upper, probe, a_0, ..., a_{m-1}, b_0, ...,
        b_{n-1}]``, where every active index is exactly 0 or 1.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _validate_branch_inputs(
    A: np.ndarray,
    B: np.ndarray,
    partition: np.ndarray,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    left = np.asarray(A, dtype=float)
    right = np.asarray(B, dtype=float)
    packed = np.asarray(partition, dtype=float)
    if left.ndim != 2 or left.shape[0] < 1 or left.shape[1] != 2:
        raise ValueError("A must have shape (m, 2) with m positive")
    if right.ndim != 2 or right.shape[0] != 2 or right.shape[1] < 1:
        raise ValueError("B must have shape (2, n) with n positive")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("matrix entries must be finite")
    if packed.ndim != 1 or packed.size < 4 or not np.all(np.isfinite(packed)):
        raise ValueError("partition must be a finite one-dimensional vector")
    if isinstance(tolerance, (bool, np.bool_)) or not np.isscalar(tolerance):
        raise ValueError("tolerance must be a finite positive scalar")
    try:
        tol = float(tolerance)
    except (TypeError, ValueError) as exc:
        raise ValueError("tolerance must be a real scalar") from exc
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("tolerance must be a finite positive scalar")
    count = int(round(float(packed[0])))
    if count < 2 or packed[0] != float(count) or packed.size != 2 * count:
        raise ValueError("partition header or packed length is invalid")
    switches = packed[1 : 1 + count]
    probes = packed[1 + count :]
    widths = np.diff(switches)
    if np.any(widths <= tol):
        raise ValueError("partition switches must be separated by tolerance")
    if probes.shape != (count - 1,):
        raise ValueError("partition probe count is invalid")
    if np.any(probes <= switches[:-1]) or np.any(probes >= switches[1:]):
        raise ValueError("every probe must lie strictly inside its cell")
    if not np.allclose(
        probes,
        0.5 * (switches[:-1] + switches[1:]),
        rtol=0.0,
        atol=tol,
    ):
        raise ValueError("partition probes must be cell midpoints")
    return left, right, switches, probes, tol


def _active_channels(
    left: np.ndarray,
    right: np.ndarray,
    probe: float,
) -> tuple[np.ndarray, np.ndarray]:
    scales = np.exp(np.array([probe, -probe], dtype=float))
    active_a = np.argmax(np.abs(left * scales), axis=1).astype(float)
    active_b = np.argmax(np.abs(right / scales[:, None]), axis=0).astype(float)
    return active_a, active_b


def _oracle_encode_partition_active_branches(
    A: np.ndarray,
    B: np.ndarray,
    partition: np.ndarray,
    tolerance: float = 1e-12,
) -> np.ndarray:
    """Reference active-set encoding on every open partition cell."""
    left, right, switches, probes, _tol = _validate_branch_inputs(
        A, B, partition, tolerance
    )
    records = []
    for index, probe in enumerate(probes):
        active_a, active_b = _active_channels(left, right, float(probe))
        records.append(
            np.concatenate(
                (
                    switches[index : index + 2],
                    [probe],
                    active_a,
                    active_b,
                )
            )
        )
    result = np.concatenate(([float(probes.size)], *records))
    if not np.all(np.isfinite(result)):
        raise ValueError("active branch table must be finite")
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return mixed-active, coincident-tie, zero-row, narrow-cell, and invalid cases."""
    return [
        {
            "setup": """import numpy as np
A = np.array([[12.0,0.5],[7.0,-0.4],[0.6,9.0],[0.8,6.0],[1.2,0.7]])
B = np.array([[0.2,0.5,-0.3,0.4],[6.0,-4.0,5.0,3.0]])
partition = np.array([7.0,-1.3862943611198906,-1.0397207708399179,-1.0074515102711323,-0.2694982503663435,1.0074515102711323,1.354025100551105,1.3862943611198906,-1.2130075659799043,-1.023586140555525,-0.6384748803187379,0.3689766299523944,1.1807383054111187,1.3701597308354978])
""",
            "call": "encode_partition_active_branches(A, B, partition)",
            "gold_call": "_oracle_encode_partition_active_branches(A, B, partition)",
            "tol": 1e-12,
        },
        {
            "setup": """import numpy as np
A = np.array([[1.0,1.0],[2.0,2.0],[-3.0,3.0]])
B = np.array([[1.0,-2.0,4.0],[1.0,2.0,-4.0]])
partition = np.array([3.0, -1.0, 0.0, 1.0, -0.5, 0.5])
""",
            "call": "encode_partition_active_branches(A, B, partition)",
            "gold_call": "_oracle_encode_partition_active_branches(A, B, partition)",
            "tol": 1e-12,
        },
        {
            "setup": """import numpy as np
A = np.array([[0.0,0.0],[0.0,2.0],[3.0,0.0]])
B = np.array([[0.0,4.0],[0.0,0.0]])
partition = np.array([2.0, -0.7, 0.7, 0.0])
""",
            "call": "encode_partition_active_branches(A, B, partition)",
            "gold_call": "_oracle_encode_partition_active_branches(A, B, partition)",
            "tol": 1e-12,
        },
        {
            "setup": """import numpy as np
A = np.array([[1.0, np.exp(2e-7)], [2.0, 0.5]])
B = np.array([[1.0, 3.0], [np.exp(-6e-7), 0.2]])
partition = np.array([4.0, -1.0, 1e-7, 3e-7, 1.0, -0.49999995, 2e-7, 0.50000015])
tolerance = 1e-12
""",
            "call": "encode_partition_active_branches(A, B, partition, tolerance)",
            "gold_call": "_oracle_encode_partition_active_branches(A, B, partition, tolerance)",
            "tol": 1e-12,
        },
        {
            "setup": """import numpy as np
A = np.array([[-2.0,5.0],[7.0,-1.0],[0.25,0.5]])
B = np.array([[3.0,-4.0,0.25],[-8.0,1.0,2.0]])
partition = np.array([2.0, -0.4, 0.9, 0.25])
""",
            "call": "encode_partition_active_branches(A, B, partition)",
            "gold_call": "_oracle_encode_partition_active_branches(A, B, partition)",
            "tol": 1e-12,
        },
        {
            "setup": """import numpy as np
A = np.ones((2, 2))
B = np.ones((2, 2))
partition = np.array([3.0, -1.0, 0.0, 1.0, -0.4, 0.5])
def run_model():
    try:
        encode_partition_active_branches(A, B, partition)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_encode_partition_active_branches(A, B, partition)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
            "tol": 0.0,
        },
    ]
