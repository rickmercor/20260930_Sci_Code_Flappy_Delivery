"""
Apply the source study's interaction-analysis convention to the supplied coalition outputs and pair-specific background distributions and return the adjusted significance assigned to the requested target interaction.

The complete task evaluates nonlinear relationships among the represented genetic features and determines the statistical evidence assigned to one specified pair in the context of all pairwise interactions. The coalition table and pair-specific background distributions are supplied directly, so model training is not required. This final step determines the represented feature dimension and delegates the joint interaction analysis to the preceding components.

Returns
-------
Floating-point adjusted significance value assigned to the requested target genetic-feature interaction. Invalid inputs raise ValueError.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_target_interaction_adjusted_significance(
    coalition_values: list[float],
    feature_i: int,
    feature_j: int,
    background_interactions_by_pair: list[list[float]]
) -> float:
    """
    Return the adjusted significance assigned to the requested target
    genetic-feature interaction.

    Parameters
    ----------
    coalition_values : list[float]
        Complete coalition-value table.
    feature_i : int
        First member of the target pair.
    feature_j : int
        Second member of the target pair.
    background_interactions_by_pair : list[list[float]]
        Background distributions for all unordered feature pairs.

    Returns
    -------
    float
        Adjusted significance value assigned to the target interaction.

    Raises
    ------
    ValueError
        If the supplied coalition data, target pair, or background
        distributions are invalid.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_target_interaction_adjusted_significance(
    coalition_values: list[float],
    feature_i: int,
    feature_j: int,
    background_interactions_by_pair: list[list[float]]
) -> float:
    num_features = (
        _oracle_infer_feature_count(
            coalition_values
        )
    )

    return float(
        _oracle_compute_joint_interaction_significance(
            coalition_values,
            num_features,
            feature_i,
            feature_j,
            background_interactions_by_pair
        )
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
            "call": "compute_target_interaction_adjusted_significance(coalition_values, feature_i, feature_j, background_interactions_by_pair)",
            "gold_call": "_oracle_compute_target_interaction_adjusted_significance(coalition_values, feature_i, feature_j, background_interactions_by_pair)"
        },

        {
            "setup": """
coalition_values = [
    0.0, 0.2, 0.1, 0.7,
    0.3, 0.8, 0.5, 1.6
]

feature_i = 1
feature_j = 0

background_interactions_by_pair = [
    [-0.1,0.0,0.1,0.2,0.3],
    [-0.3,-0.2,-0.1,0.0,0.1],
    [0.0,0.1,0.2,0.3,0.4]
]
""",
            "call": "compute_target_interaction_adjusted_significance(coalition_values, feature_i, feature_j, background_interactions_by_pair)",
            "gold_call": "_oracle_compute_target_interaction_adjusted_significance(coalition_values, feature_i, feature_j, background_interactions_by_pair)"
        },

        {
            "setup": """
coalition_values = [
    0.0, 0.2, 0.1, 0.7,
    0.3, 0.8, 0.5, 1.6
]

feature_i = 0
feature_j = 2

background_interactions_by_pair = [
    [-0.1,0.0,0.1,0.2,0.3],
    [-0.3,-0.2,-0.1,0.0,0.1],
    [0.0,0.1,0.2,0.3,0.4]
]
""",
            "call": "compute_target_interaction_adjusted_significance(coalition_values, feature_i, feature_j, background_interactions_by_pair)",
            "gold_call": "_oracle_compute_target_interaction_adjusted_significance(coalition_values, feature_i, feature_j, background_interactions_by_pair)"
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
    0.0, 0.2, 0.1,
    0.7, 0.3
]

feature_i = 0
feature_j = 1

background_interactions_by_pair = [
    [-0.1,0.0,0.1,0.2,0.3]
]
""",
            "call": "_captures_value_error(compute_target_interaction_adjusted_significance, coalition_values, feature_i, feature_j, background_interactions_by_pair)",
            "gold_call": "_captures_value_error(_oracle_compute_target_interaction_adjusted_significance, coalition_values, feature_i, feature_j, background_interactions_by_pair)"
        }
    ]
