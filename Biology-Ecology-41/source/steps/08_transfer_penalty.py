"""
Combine the training and test approximation errors into a dimensionless transfer-penalty ratio.

The ratio compares the unchanged sentinel set's test-dynamics error with the error measured on the dynamics used to select the set.

Returns
-------
return float(test_error / training_error)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def transfer_penalty(training_error: float, test_error: float) -> float:
    '''Return the ratio of test-dynamics approximation error to training error.

    Parameters
    ----------
    training_error : float
        Positive approximation error measured on the dynamics used for sentinel
        selection.
    test_error : float
        Non-negative approximation error measured on the second dynamics.

    Returns
    -------
    ratio : float
        Dimensionless ratio test_error / training_error.

    Raises
    ------
    ValueError
        If either input is non-finite, training_error is not positive, or
        test_error is negative.
    '''
    return ratio

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_transfer_penalty(training_error: float, test_error: float) -> float:
    if not np.isfinite(training_error) or not np.isfinite(test_error):
        raise ValueError("errors must be finite")
    if training_error <= 0.0:
        raise ValueError("training_error must be positive")
    if test_error < 0.0:
        raise ValueError("test_error must be non-negative")
    return float(test_error / training_error)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return representative transfer-penalty cases."""
    return [
        {
            "setup": """training_error = 0.2
test_error = 0.3
""",
            "call": "transfer_penalty(training_error, test_error)",
            "gold_call": "_oracle_transfer_penalty(training_error, test_error)",
        },
        {
            "setup": """training_error = 1.0
test_error = 0.0
""",
            "call": "transfer_penalty(training_error, test_error)",
            "gold_call": "_oracle_transfer_penalty(training_error, test_error)",
        },
        {
            "setup": """training_error = 0.0005
test_error = 0.00125
""",
            "call": "transfer_penalty(training_error, test_error)",
            "gold_call": "_oracle_transfer_penalty(training_error, test_error)",
        },
        {
            "setup": """training_error = 0.0
test_error = 0.1
def run_model():
    try:
        transfer_penalty(training_error, test_error)
        return 0
    except ValueError:
        return 1
def run_oracle():
    try:
        _oracle_transfer_penalty(training_error, test_error)
        return 0
    except ValueError:
        return 1
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
