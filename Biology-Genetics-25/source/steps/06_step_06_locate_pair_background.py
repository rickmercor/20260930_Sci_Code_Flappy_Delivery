"""
Determine which lexicographically ordered background-distribution row corresponds to a requested unordered feature pair.

The interaction analysis treats feature pairs as unordered and associates each pair with one reference distribution. Background distributions are stored according to the lexicographic ordering of all distinct feature pairs. This step resolves a requested pair to its corresponding row without assuming that the feature indices are supplied in ascending order.

Returns
-------
Zero-based index of the requested unordered pair in the lexicographically ordered set of all distinct feature pairs. Invalid pairs raise ValueError.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def locate_pair_background(
    num_features: int,
    feature_i: int,
    feature_j: int
) -> int:
    """
    Locate the background-distribution row for an unordered feature pair.

    Parameters
    ----------
    num_features : int
        Total number of represented features.
    feature_i : int
        First feature index.
    feature_j : int
        Second feature index.

    Returns
    -------
    int
        Zero-based index of the corresponding pair.

    Raises
    ------
    ValueError
        If the requested feature pair is invalid.
    """
    return 0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_locate_pair_background(
    num_features: int,
    feature_i: int,
    feature_j: int
) -> int:
    if num_features < 2:
        raise ValueError(
            "num_features must be at least two"
        )

    if (
        feature_i < 0
        or feature_j < 0
        or feature_i >= num_features
        or feature_j >= num_features
        or feature_i == feature_j
    ):
        raise ValueError(
            "feature_i and feature_j must be distinct valid feature indices"
        )

    a = min(feature_i, feature_j)
    b = max(feature_i, feature_j)

    pair_index = 0

    for first in range(num_features - 1):
        for second in range(
            first + 1,
            num_features
        ):
            if first == a and second == b:
                return pair_index

            pair_index += 1

    raise ValueError(
        "requested feature pair could not be resolved"
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """
num_features = 4
feature_i = 0
feature_j = 1
""",
            "call": "locate_pair_background(num_features, feature_i, feature_j)",
            "gold_call": "_oracle_locate_pair_background(num_features, feature_i, feature_j)"
        },

        {
            "setup": """
num_features = 4
feature_i = 3
feature_j = 1
""",
            "call": "locate_pair_background(num_features, feature_i, feature_j)",
            "gold_call": "_oracle_locate_pair_background(num_features, feature_i, feature_j)"
        },

        {
            "setup": """
num_features = 6
feature_i = 2
feature_j = 5
""",
            "call": "locate_pair_background(num_features, feature_i, feature_j)",
            "gold_call": "_oracle_locate_pair_background(num_features, feature_i, feature_j)"
        },

        {
            "setup": """
def _captures_value_error(fn, *args):
    try:
        fn(*args)
    except ValueError:
        return 1.0
    return 0.0

num_features = 5
feature_i = 3
feature_j = 3
""",
            "call": "_captures_value_error(locate_pair_background, num_features, feature_i, feature_j)",
            "gold_call": "_captures_value_error(_oracle_locate_pair_background, num_features, feature_i, feature_j)"
        }
    ]
