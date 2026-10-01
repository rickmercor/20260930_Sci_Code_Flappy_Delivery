"""
Normalize the kinetic observables of each single mutant by the corresponding wild-type kinetic observables to obtain dimensionless single-mutant fold changes. Return the fold changes for kcat, KM, catalytic efficiency, and KD using a fixed wild-type reference.

Mutational effects are compared on a relative scale by dividing each mutant kinetic parameter by its wild-type value. This normalization removes the absolute scale of the kinetic parameter and expresses each mutation as a multiplicative change relative to the wild type. These normalized single-mutant effects are subsequently used to construct the conventional macroscopic null prediction for a double mutant. The paper uses fold changes because the relevant comparison concerns whether the combined double-mutant effect equals the product of the corresponding single-mutant effects.

Returns
-------
np.ndarray, a two-dimensional floating-point array of shape (number_of_single_mutants, 4), containing dimensionless fold changes for [kcat, KM, kcat_over_KM, KD].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_single_fold_changes(
    observables: "np.ndarray",
    anchor_index: int,
) -> "np.ndarray":
    """
    Compute wild-type-normalized single-mutant fold changes.

    Parameters
    ----------
    observables : np.ndarray
        Observable matrix containing the reference state and mutant states.

    anchor_index : int
        Row index of the reference state.

    Returns
    -------
    np.ndarray
        Wild-type-normalized observable matrix.

    Raises
    ------
    ValueError
        If the observable matrix or reference index is invalid.
    """
    return fold_changes

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_compute_single_fold_changes(
    observables: "np.ndarray",
    anchor_index: int,
) -> "np.ndarray":
    x = np.asarray(observables, dtype=float)
    if x.ndim != 2 or x.shape[1] != 4 or x.shape[0] < 1 or not np.all(np.isfinite(x)) or np.any(x <= 0):
        raise ValueError("invalid observables")
    if isinstance(anchor_index, (bool, np.bool_)) or not isinstance(anchor_index, (int, np.integer)) or anchor_index < 0 or anchor_index >= x.shape[0]:
        raise ValueError("invalid anchor")
    out = x / x[int(anchor_index)]
    if not np.all(np.isfinite(out)) or np.any(out <= 0):
        raise ValueError("invalid folds")
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "import numpy as np\nx=np.array([[2.,4.,8.,1.],[4.,2.,16.,2.]],float)\n", "call": "compute_single_fold_changes(x,0)", "gold_call": "_oracle_compute_single_fold_changes(x,0)"},
        {"setup": "import numpy as np\nx=np.ones((3,4))\n", "call": "compute_single_fold_changes(x,1)", "gold_call": "_oracle_compute_single_fold_changes(x,1)"},
        {"setup": "import numpy as np\nx=np.array([[1e-12,2e-9,3e-6,4e-3],[2e-12,4e-9,6e-6,8e-3]],float)\n", "call": "compute_single_fold_changes(x,1)", "gold_call": "_oracle_compute_single_fold_changes(x,1)"},
        {"setup": "import numpy as np\nx=np.ones((1,4))\ndef run_model():\n    try: compute_single_fold_changes(x,2); return 0\n    except ValueError: return 1\n    except Exception: return 2\ndef run_gold():\n    try: _oracle_compute_single_fold_changes(x,2); return 0\n    except ValueError: return 1\n    except Exception: return 2\n", "call": "run_model()", "gold_call": "run_gold()"},
    ]
