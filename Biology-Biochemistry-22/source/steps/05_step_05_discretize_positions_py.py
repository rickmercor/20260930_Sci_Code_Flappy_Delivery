"""
Convert a continuously relaxed protein-sequence state back into discrete residue-class indices. Select the largest coordinate at each sequence position and reconstruct the corresponding numerical one-hot candidate.

Convert a continuously relaxed protein-sequence state back into discrete residue-class indices. Select the largest coordinate at each sequence position and reconstruct the corresponding numerical one-hot candidate.



The source procedure performs this discretization after every Hamiltonian update. Within the acquisition loop, each updated continuous position is converted into a one-hot proposal before the discretization-aware Metropolis comparison is performed.



Use sitewise `np.argmax` along the residue-class axis. Because NumPy returns the first occurrence of a maximum, an exact tie is resolved deterministically in favour of the smallest zero-based class index. Return both the integer residue-index array and the reconstructed float64 one-hot matrix.

Returns
-------
tuple[np.ndarray, np.ndarray], an integer residue-index array of shape (L,) and a float64 one-hot matrix of shape (L, A)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def discretize_positions(
    q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Discretize each sequence position using sitewise argmax.

    Parameters
    ----------
    q : np.ndarray
        Finite continuous position matrix of shape ``(L, A)``.

    Raises
    ------
    ValueError
        If ``q`` is empty, not two-dimensional, or contains
        non-finite values.

    Returns
    -------
    result : tuple[np.ndarray, np.ndarray]
        Integer residue-index array of shape ``(L,)`` and float64
        one-hot matrix of shape ``(L, A)``.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_discretize_positions(
    q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return sitewise argmax indices and one-hot positions."""

    q_array = np.asarray(
        q,
        dtype=np.float64,
    )

    if (
        q_array.ndim != 2
        or q_array.shape[0] < 1
        or q_array.shape[1] < 1
    ):
        raise ValueError(
            "q must be a non-empty two-dimensional array"
        )

    if not np.all(np.isfinite(q_array)):
        raise ValueError(
            "q must contain finite values"
        )

    residue_indices = np.argmax(
        q_array,
        axis=1,
    ).astype(np.int64)

    one_hot = np.eye(
        q_array.shape[1],
        dtype=np.float64,
    )[residue_indices]

    return residue_indices, one_hot

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return normal, tie-boundary, and invalid-input tests."""

    return [
        {
            "setup": """import numpy as np

q = np.array([
    [0.1192628894, 0.9953579896, 0.3501094789],
    [0.2279118299, 0.9746816529, 0.5493973299],
    [0.2182691570, 0.4970384001, 0.1224047010],
])
""",
            "call": (
                "(lambda r: [r[0].tolist(), r[1].tolist()])("
                "discretize_positions(q))"
            ),
            "gold_call": (
                "(lambda r: [r[0].tolist(), r[1].tolist()])("
                "_oracle_discretize_positions(q))"
            ),
        },
        {
            "setup": """import numpy as np

q = np.array([
    [0.5, 0.5, 0.1],
    [0.0, 0.0, 0.0],
])
""",
            "call": (
                "(lambda r: [r[0].tolist(), r[1].tolist()])("
                "discretize_positions(q))"
            ),
            "gold_call": (
                "(lambda r: [r[0].tolist(), r[1].tolist()])("
                "_oracle_discretize_positions(q))"
            ),
        },
        {
            "setup": """import numpy as np

q = np.array([
    [0.2, np.nan],
    [0.8, 0.2],
])

def _model_error_code():
    try:
        discretize_positions(q)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def _oracle_error_code():
    try:
        _oracle_discretize_positions(q)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "_model_error_code()",
            "gold_call": "_oracle_error_code()",
        },
    ]
