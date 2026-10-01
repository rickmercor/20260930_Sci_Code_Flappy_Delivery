"""
Compute the mean exact pairwise interaction value for a specified feature pair across one or more individuals represented by concatenated complete coalition tables.

The source study uses pairwise feature-attribution values to characterize nonlinear relationships among genetic features. This step extends the exact interaction calculation to a batch of individuals while preserving the same interaction convention used for a single complete coalition table. The supplied data must contain one or more complete tables of equal dimensionality.

Returns
-------
Floating-point mean exact interaction value for the requested feature pair across all complete individual blocks. Malformed inputs raise ValueError.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_pair_shap_interaction(
    coalition_values: list[float],
    num_features: int,
    feature_i: int,
    feature_j: int
) -> float:
    """
    Compute the mean pairwise interaction for a specified feature pair.

    Parameters
    ----------
    coalition_values : list[float]
        One or more complete coalition tables concatenated in sequence.
    num_features : int
        Number of features represented by each table.
    feature_i : int
        First feature index.
    feature_j : int
        Second feature index.

    Returns
    -------
    float
        Mean pairwise interaction value.

    Raises
    ------
    ValueError
        If the supplied feature indices or coalition data are invalid.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_pair_shap_interaction(
    coalition_values: list[float],
    num_features: int,
    feature_i: int,
    feature_j: int
) -> float:
    import math

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

    block_size = 1 << num_features

    if len(coalition_values) == 0:
        raise ValueError(
            "at least one complete coalition table is required"
        )

    if len(coalition_values) % block_size != 0:
        raise ValueError(
            "coalition_values length must be a multiple of the complete "
            "coalition-table size"
        )

    if any(
        not math.isfinite(value)
        for value in coalition_values
    ):
        raise ValueError(
            "coalition_values must contain only finite values"
        )

    number_individuals = (
        len(coalition_values) // block_size
    )

    pair_bits = (
        (1 << feature_i)
        | (1 << feature_j)
    )

    total_interaction = 0.0

    for individual_index in range(number_individuals):
        start = individual_index * block_size
        stop = start + block_size

        individual_values = coalition_values[start:stop]
        individual_interaction = 0.0

        for coalition_mask in range(block_size):
            if coalition_mask & pair_bits:
                continue

            coalition_size = bin(
                coalition_mask
            ).count("1")

            contrast = (
                _oracle_compute_pair_coalition_contrast(
                    individual_values,
                    feature_i,
                    feature_j,
                    coalition_mask
                )
            )

            coefficient = (
                _oracle_compute_interaction_coalition_weight(
                    num_features,
                    coalition_size
                )
            )

            individual_interaction += (
                coefficient * contrast
            )

        total_interaction += individual_interaction

    return float(
        total_interaction / number_individuals
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """
coalition_values = [0.0, 0.2, 0.1, 0.7]
num_features = 2
feature_i = 0
feature_j = 1
""",
            "call": "compute_pair_shap_interaction(coalition_values, num_features, feature_i, feature_j)",
            "gold_call": "_oracle_compute_pair_shap_interaction(coalition_values, num_features, feature_i, feature_j)"
        },

        {
            "setup": """
num_features = 4
feature_i = 0
feature_j = 2

coalition_values = []

for pair_strength in [0.40, -0.25, 1.10]:
    for mask in range(1 << num_features):
        x0 = 1.0 if mask & (1 << 0) else 0.0
        x1 = 1.0 if mask & (1 << 1) else 0.0
        x2 = 1.0 if mask & (1 << 2) else 0.0
        x3 = 1.0 if mask & (1 << 3) else 0.0

        value = (
            0.15
            + 0.20 * x0
            - 0.10 * x1
            + 0.35 * x2
            + 0.05 * x3
            + pair_strength * x0 * x2
            + 0.60 * x0 * x2 * x3
            - 0.30 * x0 * x1 * x2
            + 0.25 * x0 * x1 * x2 * x3
        )

        coalition_values.append(value)
""",
            "call": "compute_pair_shap_interaction(coalition_values, num_features, feature_i, feature_j)",
            "gold_call": "_oracle_compute_pair_shap_interaction(coalition_values, num_features, feature_i, feature_j)"
        },

        {
            "setup": """
num_features = 5
feature_i = 1
feature_j = 4

coalition_values = []

for pair_strength in [0.15, 0.55, -0.35, 0.95]:
    for mask in range(1 << num_features):
        x = [
            1.0 if mask & (1 << k) else 0.0
            for k in range(num_features)
        ]

        value = (
            0.10
            + 0.12 * x[0]
            - 0.08 * x[1]
            + 0.18 * x[2]
            + 0.07 * x[3]
            + 0.21 * x[4]
            + pair_strength * x[1] * x[4]
            + 0.45 * x[0] * x[1] * x[4]
            - 0.35 * x[1] * x[2] * x[4]
            + 0.30 * x[1] * x[3] * x[4]
            + 0.50 * x[0] * x[1] * x[2] * x[4]
            - 0.25 * x[1] * x[2] * x[3] * x[4]
            + 0.20 * x[0] * x[1] * x[2] * x[3] * x[4]
        )

        coalition_values.append(value)
""",
            "call": "compute_pair_shap_interaction(coalition_values, num_features, feature_i, feature_j)",
            "gold_call": "_oracle_compute_pair_shap_interaction(coalition_values, num_features, feature_i, feature_j)"
        },

        {
            "setup": """
def _captures_value_error(fn, *args):
    try:
        fn(*args)
    except ValueError:
        return 1.0
    return 0.0

coalition_values = [0.1, 0.2, 0.3, 0.4, 0.5]
num_features = 3
feature_i = 0
feature_j = 2
""",
            "call": "_captures_value_error(compute_pair_shap_interaction, coalition_values, num_features, feature_i, feature_j)",
            "gold_call": "_captures_value_error(_oracle_compute_pair_shap_interaction, coalition_values, num_features, feature_i, feature_j)"
        }
    ]
