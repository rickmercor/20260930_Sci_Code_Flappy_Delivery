"""
Select a one-based mode from an already ordered finite spectrum.



The scientific convention numbers modes beginning at one.  This boundary

stage enforces that convention and rejects unordered or contaminated spectra

instead of silently selecting from malformed upstream data.

Returns
-------
one finite nonnegative float equal to magnitudes[mode_index - 1]
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def select_one_based_mode(magnitudes: np.ndarray, mode_index: int) -> float:
    """Return the requested one-based entry of a sorted magnitude vector.

    ``magnitudes`` must be nonempty, finite, nonnegative, and nondecreasing.
    ``mode_index`` must be a non-boolean integer in ``[1, len(magnitudes)]``.
    Invalid inputs raise ``ValueError``.

    Parameters
    ----------
    magnitudes : np.ndarray
        Ordered finite mode magnitudes.
    mode_index : int
        One-based mode number.

    Returns
    -------
    float
        The selected mode magnitude.
    """
    return mode_magnitude  # noqa: F821 - model stub

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_select_one_based_mode(magnitudes: np.ndarray, mode_index: int) -> float:
    """Reference validation and one-based selection."""
    values = np.asarray(magnitudes, dtype=float)
    if (
        values.ndim != 1
        or values.size == 0
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or np.any(np.diff(values) < 0.0)
    ):
        raise ValueError("magnitudes must be finite, nonnegative, and sorted")
    if isinstance(mode_index, (bool, np.bool_)) or not isinstance(
        mode_index, (int, np.integer)
    ):
        raise ValueError("mode_index must be a non-boolean integer")
    index = int(mode_index)
    if index < 1 or index > values.size:
        raise ValueError("mode_index lies outside the available spectrum")
    return float(values[index - 1])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return interior, boundary, repeated, unsorted, and invalid-index cases."""
    return [
        {
            "setup": """import numpy as np
magnitudes = np.array([0.125, 2.0, 7.5, 11.0])
mode_index = 3
""",
            "call": "select_one_based_mode(magnitudes, mode_index)",
            "gold_call": "_oracle_select_one_based_mode(magnitudes, mode_index)",
        },
        {
            "setup": """import numpy as np
magnitudes = np.array([1e-14, 1.0, 100.0])
mode_index = 1
""",
            "call": "select_one_based_mode(magnitudes, mode_index)",
            "gold_call": "_oracle_select_one_based_mode(magnitudes, mode_index)",
        },
        {
            "setup": """import numpy as np
magnitudes = np.array([0.0, 3.25, 3.25, 9.0])
mode_index = 4
""",
            "call": "select_one_based_mode(magnitudes, mode_index)",
            "gold_call": "_oracle_select_one_based_mode(magnitudes, mode_index)",
        },
        {
            "setup": """import numpy as np
magnitudes = np.array([1.0, 0.5, 2.0])
mode_index = 2
def run_model():
    try:
        select_one_based_mode(magnitudes, mode_index)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_select_one_based_mode(magnitudes, mode_index)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
magnitudes = np.array([0.5, 1.5])
mode_index = 3
def run_model():
    try:
        select_one_based_mode(magnitudes, mode_index)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_select_one_based_mode(magnitudes, mode_index)
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
