"""
Consolidate the multi-conformer energy predictions into a single molecular free energy metric by evaluating the unweighted central tendency across the resolved structural snapshots.

The final molecular solvation free energy represents the uniform ensemble coordination of the independent structural snapshot states. The consolidation process must treat each individual conformer configuration with identical statistical importance rather than executing thermodynamic exponential weighting or root-mean-square scaling bounds. For instances containing an isolated single configuration, the global output maps directly to that specific state profile without scaling modifications.

Returns
-------
return molecular_delta_g
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def average_conformers(predictions: np.ndarray) -> float:
    """
    Average per-conformer MILC predictions.

    Parameters
    ----------
    predictions : np.ndarray
        Per-conformer MILC values with shape (n_conformers,).

    Returns
    -------
    float
        Arithmetic mean over conformers, in kcal/mol.

    Raises
    ------
    ValueError
        If no conformers are provided or if any prediction is non-finite.
    """
    return molecular_delta_g

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_average_conformers(predictions):
    predictions = np.asarray(predictions, dtype=float).reshape(-1)
    if predictions.size == 0:
        raise ValueError("at least one conformer is required")
    if not np.all(np.isfinite(predictions)):
        raise ValueError("predictions contain non-finite values")
    return float(np.mean(predictions))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return differential test specifications."""
    return [
        {
            "setup": """import numpy as np

# Test case 1: Three-conformer arithmetic mean
predictions = np.array(
    [-36.03842050071194, -19.139228892230275, -2.6413921875983193]
)

def run_model():
    return average_conformers(predictions)

def run_gold():
    return _oracle_average_conformers(predictions)
""",
            "call": "run_model()",
            "gold_call": "run_gold()"
        },
        {
            "setup": """import numpy as np

# Test case 2: Single conformer is returned unchanged
predictions = np.array([-0.55])

def run_model():
    return average_conformers(predictions)

def run_gold():
    return _oracle_average_conformers(predictions)
""",
            "call": "run_model()",
            "gold_call": "run_gold()"
        },
        {
            "setup": """import numpy as np

# Test case 3: Empty input is invalid
predictions = np.array([])

def run_model():
    try:
        average_conformers(predictions)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_average_conformers(predictions)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()"
        },
        {
            "setup": """import numpy as np

# Test case 4: Non-finite prediction is invalid
predictions = np.array([-1.2, np.nan])

def run_model():
    try:
        average_conformers(predictions)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_average_conformers(predictions)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()"
        },
    ]
