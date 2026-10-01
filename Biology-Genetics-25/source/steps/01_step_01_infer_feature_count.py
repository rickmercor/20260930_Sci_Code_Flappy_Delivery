"""
Determine the number of genetic features represented by a complete coalition-value table.

Interaction analyses based on complete feature coalitions require the dimensionality of the represented feature set. The supplied values correspond to model outputs indexed over all coalitions for one individual. Determine the feature dimension represented by the table. An input whose length cannot represent a complete coalition table is invalid.

Returns
-------
Integer number of represented genetic features. Invalid coalition-table lengths raise ValueError.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def infer_feature_count(
    coalition_values: list[float]
) -> int:
    """
    Determine the number of features represented by a complete coalition table.

    Parameters
    ----------
    coalition_values : list[float]
        Model outputs indexed over feature coalitions.

    Returns
    -------
    int
        Number of represented features.

    Raises
    ------
    ValueError
        If the supplied values cannot represent a complete coalition
        table for at least two features.
    """
    return 0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_infer_feature_count(
    coalition_values: list[float]
) -> int:
    n = len(coalition_values)

    if n < 4 or (n & (n - 1)):
        raise ValueError(
            "coalition_values must contain a complete power-of-two "
            "coalition table representing at least two features"
        )

    return n.bit_length() - 1

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """
coalition_values = [0.0, 0.1, 0.2, 0.3]
""",
            "call": "infer_feature_count(coalition_values)",
            "gold_call": "_oracle_infer_feature_count(coalition_values)"
        },

        {
            "setup": """
coalition_values = [
    0.0, 0.1, 0.2, 0.3,
    0.4, 0.5, 0.6, 0.7
]
""",
            "call": "infer_feature_count(coalition_values)",
            "gold_call": "_oracle_infer_feature_count(coalition_values)"
        },

        {
            "setup": """
coalition_values = [float(i) / 10.0 for i in range(16)]
""",
            "call": "infer_feature_count(coalition_values)",
            "gold_call": "_oracle_infer_feature_count(coalition_values)"
        },

        {
            "setup": """
def _captures_value_error(fn, values):
    try:
        fn(values)
    except ValueError:
        return 1.0
    return 0.0

coalition_values = [float(i) for i in range(12)]
""",
            "call": "_captures_value_error(infer_feature_count, coalition_values)",
            "gold_call": "_captures_value_error(_oracle_infer_feature_count, coalition_values)"
        }
    ]
