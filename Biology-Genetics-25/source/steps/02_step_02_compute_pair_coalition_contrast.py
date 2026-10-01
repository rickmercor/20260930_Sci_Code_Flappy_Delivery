"""
Calculate the coalition-level contrast required for a specified pair of genetic features.

Pairwise feature-attribution methods evaluate whether the joint contribution of two features differs from their separate contributions within a given coalition context. For a specified eligible coalition and feature pair, return the corresponding interaction contrast using the coalition-indexed model outputs. Feature indices, coalition masks, and coalition membership must be consistent with the represented feature set.

Returns
-------
Floating-point interaction contrast for the specified eligible coalition and feature pair. Invalid inputs raise ValueError.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_pair_coalition_contrast(
    coalition_values: list[float],
    feature_i: int,
    feature_j: int,
    coalition_mask: int
) -> float:
    """
    Calculate the interaction contrast for one coalition and feature pair.

    Parameters
    ----------
    coalition_values : list[float]
        Complete coalition-value table.
    feature_i : int
        First feature index.
    feature_j : int
        Second feature index.
    coalition_mask : int
        Coalition context encoded as an integer bitmask.

    Returns
    -------
    float
        Coalition-level interaction contrast.

    Raises
    ------
    ValueError
        If the coalition table, feature pair, or coalition mask is invalid.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_pair_coalition_contrast(
    coalition_values: list[float],
    feature_i: int,
    feature_j: int,
    coalition_mask: int
) -> float:
    num_features = _oracle_infer_feature_count(
        coalition_values
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

    block_size = 1 << num_features

    if coalition_mask < 0 or coalition_mask >= block_size:
        raise ValueError(
            "coalition_mask is outside the represented coalition range"
        )

    bit_i = 1 << feature_i
    bit_j = 1 << feature_j

    if coalition_mask & (bit_i | bit_j):
        raise ValueError(
            "coalition_mask must exclude both members of the feature pair"
        )

    s = coalition_mask
    si = s | bit_i
    sj = s | bit_j
    sij = s | bit_i | bit_j

    return float(
        coalition_values[sij]
        - coalition_values[si]
        - coalition_values[sj]
        + coalition_values[s]
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """
coalition_values = [0.0, 0.2, 0.1, 0.7]
feature_i = 0
feature_j = 1
coalition_mask = 0
""",
            "call": "compute_pair_coalition_contrast(coalition_values, feature_i, feature_j, coalition_mask)",
            "gold_call": "_oracle_compute_pair_coalition_contrast(coalition_values, feature_i, feature_j, coalition_mask)"
        },

        {
            "setup": """
coalition_values = [
    0.10, 0.30, 0.00, 1.00,
    0.40, 0.30, 0.55, 1.85,
    0.15, 0.45, -0.10, 0.80,
    0.65, 1.05, 0.55, 2.65
]
feature_i = 0
feature_j = 1
coalition_mask = 4
""",
            "call": "compute_pair_coalition_contrast(coalition_values, feature_i, feature_j, coalition_mask)",
            "gold_call": "_oracle_compute_pair_coalition_contrast(coalition_values, feature_i, feature_j, coalition_mask)"
        },

        {
            "setup": """
coalition_values = [
    float(i * i + 3 * i) / 17.0
    for i in range(32)
]
feature_i = 1
feature_j = 4
coalition_mask = 5
""",
            "call": "compute_pair_coalition_contrast(coalition_values, feature_i, feature_j, coalition_mask)",
            "gold_call": "_oracle_compute_pair_coalition_contrast(coalition_values, feature_i, feature_j, coalition_mask)"
        },

        {
            "setup": """
def _captures_value_error(fn, *args):
    try:
        fn(*args)
    except ValueError:
        return 1.0
    return 0.0

coalition_values = [
    0.10, 0.30, 0.00, 1.00,
    0.40, 0.30, 0.55, 1.85,
    0.15, 0.45, -0.10, 0.80,
    0.65, 1.05, 0.55, 2.65
]
feature_i = 0
feature_j = 1
coalition_mask = 1
""",
            "call": "_captures_value_error(compute_pair_coalition_contrast, coalition_values, feature_i, feature_j, coalition_mask)",
            "gold_call": "_captures_value_error(_oracle_compute_pair_coalition_contrast, coalition_values, feature_i, feature_j, coalition_mask)"
        }
    ]
