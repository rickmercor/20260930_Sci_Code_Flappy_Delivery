"""
Determine the coefficient assigned to an eligible coalition in the exact pairwise feature-interaction calculation.

Exact pairwise feature attribution combines information across coalition contexts using coefficients determined by the dimensionality of the feature space and the number of features already present in a coalition. Return the coefficient associated with an eligible coalition size.

Returns
-------
Floating-point coefficient assigned to the specified eligible coalition size. Invalid inputs raise ValueError.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_interaction_coalition_weight(
    num_features: int,
    coalition_size: int
) -> float:
    """
    Determine the coefficient assigned to a coalition context.

    Parameters
    ----------
    num_features : int
        Total number of represented features.
    coalition_size : int
        Number of features in the coalition context.

    Returns
    -------
    float
        Coalition coefficient.

    Raises
    ------
    ValueError
        If the feature count or coalition size is outside its valid range.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_interaction_coalition_weight(
    num_features: int,
    coalition_size: int
) -> float:
    import math

    if num_features < 2:
        raise ValueError(
            "num_features must be at least two"
        )

    if coalition_size < 0 or coalition_size > num_features - 2:
        raise ValueError(
            "coalition_size must lie between 0 and num_features - 2"
        )

    numerator = (
        math.factorial(coalition_size)
        * math.factorial(
            num_features - coalition_size - 2
        )
    )

    denominator = (
        2.0
        * math.factorial(num_features - 1)
    )

    return float(numerator / denominator)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """
num_features = 4
coalition_size = 0
""",
            "call": "compute_interaction_coalition_weight(num_features, coalition_size)",
            "gold_call": "_oracle_compute_interaction_coalition_weight(num_features, coalition_size)"
        },

        {
            "setup": """
num_features = 4
coalition_size = 1
""",
            "call": "compute_interaction_coalition_weight(num_features, coalition_size)",
            "gold_call": "_oracle_compute_interaction_coalition_weight(num_features, coalition_size)"
        },

        {
            "setup": """
num_features = 7
coalition_size = 3
""",
            "call": "compute_interaction_coalition_weight(num_features, coalition_size)",
            "gold_call": "_oracle_compute_interaction_coalition_weight(num_features, coalition_size)"
        },

        {
            "setup": """
def _captures_value_error(fn, *args):
    try:
        fn(*args)
    except ValueError:
        return 1.0
    return 0.0

num_features = 4
coalition_size = 3
""",
            "call": "_captures_value_error(compute_interaction_coalition_weight, num_features, coalition_size)",
            "gold_call": "_captures_value_error(_oracle_compute_interaction_coalition_weight, num_features, coalition_size)"
        }
    ]
