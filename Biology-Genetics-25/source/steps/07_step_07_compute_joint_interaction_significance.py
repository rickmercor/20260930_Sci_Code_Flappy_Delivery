"""
Evaluate the complete unordered feature-pair interaction set and return the multiple-testing-adjusted significance assigned to one requested target pair.

The source study evaluates pairwise genetic interactions as a set rather than interpreting each pair independently. Each represented feature pair has its own observed interaction and corresponding background distribution. Apply the interaction-significance convention jointly across all distinct pairs and return the adjusted value assigned specifically to the requested target interaction. Pair ordering, target orientation, and background alignment must be handled consistently.

Returns
-------
Floating-point adjusted significance value assigned to the requested target pair after joint evaluation of every unordered feature pair. Malformed inputs raise ValueError.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_joint_interaction_significance(
    coalition_values: list[float],
    num_features: int,
    feature_i: int,
    feature_j: int,
    background_interactions_by_pair: list[list[float]]
) -> float:
    """
    Evaluate the complete pairwise interaction set and return the adjusted
    significance assigned to one requested target pair.

    Parameters
    ----------
    coalition_values : list[float]
        One or more complete coalition tables.
    num_features : int
        Number of represented features.
    feature_i : int
        First member of the target pair.
    feature_j : int
        Second member of the target pair.
    background_interactions_by_pair : list[list[float]]
        Background distributions for the unordered feature pairs.

    Returns
    -------
    float
        Adjusted significance assigned to the requested pair.

    Raises
    ------
    ValueError
        If the target pair, coalition data, or background distributions
        are invalid.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_joint_interaction_significance(
    coalition_values: list[float],
    num_features: int,
    feature_i: int,
    feature_j: int,
    background_interactions_by_pair: list[list[float]]
) -> float:
    target_pair_index = (
        _oracle_locate_pair_background(
            num_features,
            feature_i,
            feature_j
        )
    )

    expected_pair_count = (
        num_features * (num_features - 1) // 2
    )

    if (
        len(background_interactions_by_pair)
        != expected_pair_count
    ):
        raise ValueError(
            "background_interactions_by_pair must contain exactly one "
            "distribution for every unordered feature pair"
        )

    raw_significance_values = []
    pair_index = 0

    for first in range(num_features - 1):
        for second in range(
            first + 1,
            num_features
        ):
            observed_interaction = (
                _oracle_compute_pair_shap_interaction(
                    coalition_values,
                    num_features,
                    first,
                    second
                )
            )

            raw_significance = (
                _oracle_compute_pair_raw_significance(
                    observed_interaction,
                    background_interactions_by_pair[
                        pair_index
                    ]
                )
            )

            raw_significance_values.append(
                raw_significance
            )

            pair_index += 1

    ranked = sorted(
        [
            (value, original_index)
            for original_index, value
            in enumerate(
                raw_significance_values
            )
        ],
        key=lambda item: (
            item[0],
            item[1]
        )
    )

    number_tests = len(ranked)

    adjusted_ranked = []

    for rank, (value, _) in enumerate(
        ranked,
        start=1
    ):
        adjusted_ranked.append(
            min(
                1.0,
                value * number_tests / rank
            )
        )

    running_minimum = 1.0

    for ranked_index in range(
        number_tests - 1,
        -1,
        -1
    ):
        running_minimum = min(
            running_minimum,
            adjusted_ranked[ranked_index]
        )

        adjusted_ranked[
            ranked_index
        ] = running_minimum

    for ranked_index, (
        _,
        original_index
    ) in enumerate(ranked):
        if original_index == target_pair_index:
            return float(
                adjusted_ranked[ranked_index]
            )

    raise ValueError(
        "target pair was not present in the evaluated pair set"
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """
coalition_values = [
    0.10, 0.30, 0.00, 1.00,
    0.40, 0.30, 0.55, 1.85,
    0.15, 0.45, -0.10, 0.80,
    0.65, 1.05, 0.55, 2.65
]

num_features = 4
feature_i = 0
feature_j = 1

background_interactions_by_pair = [
    [-0.15,0.35,-0.05,0.25,0.00,0.20,-0.10,0.30,0.05,0.15,-0.20,0.40],
    [-0.60,-0.10,-0.50,-0.20,-0.45,-0.25,-0.55,-0.15,-0.40,-0.30,-0.65,-0.05],
    [-0.37,0.13,-0.27,0.03,-0.22,-0.02,-0.32,0.08,-0.17,-0.07,-0.42,0.18],
    [-0.15,0.35,-0.05,0.25,0.00,0.20,-0.10,0.30,0.05,0.15,-0.20,0.40],
    [-0.45,0.05,-0.35,-0.05,-0.30,-0.10,-0.40,0.00,-0.25,-0.15,-0.50,0.10],
    [0.00,0.50,0.10,0.40,0.15,0.35,0.05,0.45,0.20,0.30,-0.05,0.55]
]
""",
            "call": "compute_joint_interaction_significance(coalition_values, num_features, feature_i, feature_j, background_interactions_by_pair)",
            "gold_call": "_oracle_compute_joint_interaction_significance(coalition_values, num_features, feature_i, feature_j, background_interactions_by_pair)"
        },

        {
            "setup": """
coalition_values = [
    0.0, 0.2, 0.1, 0.7,
    0.3, 0.8, 0.5, 1.6
]

num_features = 3
feature_i = 1
feature_j = 0

background_interactions_by_pair = [
    [-0.1,0.0,0.1,0.2,0.3],
    [-0.3,-0.2,-0.1,0.0,0.1],
    [0.0,0.1,0.2,0.3,0.4]
]
""",
            "call": "compute_joint_interaction_significance(coalition_values, num_features, feature_i, feature_j, background_interactions_by_pair)",
            "gold_call": "_oracle_compute_joint_interaction_significance(coalition_values, num_features, feature_i, feature_j, background_interactions_by_pair)"
        },

        {
            "setup": """
num_features = 5
feature_i = 4
feature_j = 1

coalition_values = []

for mask in range(1 << num_features):
    x = [
        1.0 if mask & (1 << k) else 0.0
        for k in range(num_features)
    ]

    coalition_values.append(
        0.1
        + 0.2*x[0]
        - 0.1*x[1]
        + 0.3*x[2]
        + 0.15*x[3]
        + 0.25*x[4]
        + 0.55*x[0]*x[1]
        - 0.35*x[0]*x[3]
        + 0.40*x[1]*x[4]
        + 0.65*x[1]*x[2]*x[4]
        - 0.30*x[0]*x[1]*x[3]*x[4]
    )

background_interactions_by_pair = []

for row in range(
    num_features * (num_features - 1) // 2
):
    center = -0.30 + 0.07 * row

    background_interactions_by_pair.append(
        [
            center - 0.20,
            center + 0.10,
            center - 0.10,
            center + 0.20,
            center - 0.05,
            center + 0.05
        ]
    )
""",
            "call": "compute_joint_interaction_significance(coalition_values, num_features, feature_i, feature_j, background_interactions_by_pair)",
            "gold_call": "_oracle_compute_joint_interaction_significance(coalition_values, num_features, feature_i, feature_j, background_interactions_by_pair)"
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
    0.0, 0.2, 0.1, 0.7,
    0.3, 0.8, 0.5, 1.6
]

num_features = 3
feature_i = 0
feature_j = 1

background_interactions_by_pair = [
    [-0.1,0.0,0.1,0.2],
    [-0.3,-0.2,-0.1,0.0]
]
""",
            "call": "_captures_value_error(compute_joint_interaction_significance, coalition_values, num_features, feature_i, feature_j, background_interactions_by_pair)",
            "gold_call": "_captures_value_error(_oracle_compute_joint_interaction_significance, coalition_values, num_features, feature_i, feature_j, background_interactions_by_pair)"
        }
    ]
