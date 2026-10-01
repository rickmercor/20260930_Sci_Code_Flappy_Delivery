"""
Evaluate the simultaneous residual objective across all segments.

The compound method enforces every intermediate condition and the final

terminal condition simultaneously. Empirical endpoint residuals are reduced

over paths for each component, then combined according to the paper's joint

objective convention.

Inputs

------

terminals: Forward-propagated endpoint values by segment and path.

targets: Intermediate compounding and final payoff targets of matching shape.

Returns

-------

loss_summary: Segment mean squared losses followed by the joint objective.

Returns
-------
np.ndarray of shape (m + 1,), segment losses followed by the joint objective
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def compute_joint_objective(terminals: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Compute the segment residual losses and their joint sum.

    Parameters
    ----------
    terminals : np.ndarray
        Forward segment endpoint values of shape ``(m, n_paths)``.
    targets : np.ndarray
        Compounding and terminal targets with the same shape.

    Raises
    ------
    ValueError
        If the arrays are not matching nonempty matrices or contain nonfinite values.

    Returns
    -------
    loss_summary : np.ndarray
        Vector of length ``m + 1`` containing segment MSEs followed by the
        paper-defined joint objective.
    """
    return loss_summary  # noqa: F821

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np  # noqa: E402, F811


def _oracle_compute_joint_objective(terminals: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Reference implementation."""
    terminals = np.asarray(terminals, dtype=float)
    targets = np.asarray(targets, dtype=float)
    if terminals.ndim != 2 or min(terminals.shape) <= 0 or targets.shape != terminals.shape:
        raise ValueError("terminals and targets must be matching nonempty matrices")
    if not np.all(np.isfinite(terminals)) or not np.all(np.isfinite(targets)):
        raise ValueError("terminals and targets must be finite")
    segment_losses = np.mean((terminals - targets) ** 2, axis=1)
    return np.concatenate((segment_losses, np.array([np.sum(segment_losses)])))

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np  # noqa: E402, F811


def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """terminals = np.array([[1.0, 3.0], [2.0, 6.0], [0.0, 4.0]])
targets = np.array([[2.0, 2.0], [1.0, 4.0], [1.0, 1.0]])
""",
            "call": "np.round(compute_joint_objective(terminals, targets), 12).tolist()",
            "gold_call": "np.round(_oracle_compute_joint_objective(terminals, targets), 12).tolist()",
        },
        {
            "setup": "terminals = np.array([[2.0]])\ntargets = np.array([[2.0]])\n",
            "call": "np.round(compute_joint_objective(terminals, targets), 12).tolist()",
            "gold_call": "np.round(_oracle_compute_joint_objective(terminals, targets), 12).tolist()",
        },
        {
            "setup": """terminals = np.ones((2, 2))
targets = np.ones((2, 3))
def run_model():
    try:
        compute_joint_objective(terminals, targets)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_compute_joint_objective(terminals, targets)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
