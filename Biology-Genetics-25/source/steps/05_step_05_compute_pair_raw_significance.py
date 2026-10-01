"""
Determine the raw statistical significance assigned to one observed pairwise interaction relative to its supplied background interaction distribution.

The source study evaluates each pairwise genetic interaction relative to a background distribution before interactions are considered jointly. Given one observed interaction and its corresponding reference values, calculate the raw significance measure required by the analysis. The background distribution must contain sufficient finite variation to support standardization.

Returns
-------
Floating-point raw significance value for the observed interaction. Invalid or non-variable background inputs raise ValueError.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_pair_raw_significance(
    observed_interaction: float,
    background_interactions: list[float]
) -> float:
    """
    Determine the raw significance of one observed pairwise interaction
    relative to its supplied background distribution.

    Parameters
    ----------
    observed_interaction : float
        Observed pairwise interaction value.
    background_interactions : list[float]
        Corresponding background interaction distribution.

    Returns
    -------
    float
        Raw significance value.

    Raises
    ------
    ValueError
        If the observed value or background distribution cannot support
        the requested calculation.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_pair_raw_significance(
    observed_interaction: float,
    background_interactions: list[float]
) -> float:
    import math

    if not math.isfinite(observed_interaction):
        raise ValueError(
            "observed_interaction must be finite"
        )

    if len(background_interactions) < 2:
        raise ValueError(
            "background_interactions must contain at least two values"
        )

    if any(
        not math.isfinite(value)
        for value in background_interactions
    ):
        raise ValueError(
            "background_interactions must contain only finite values"
        )

    n = len(background_interactions)

    mean_background = (
        sum(background_interactions) / n
    )

    squared_deviations = sum(
        (value - mean_background) ** 2
        for value in background_interactions
    )

    sample_variance = (
        squared_deviations / (n - 1)
    )

    if sample_variance <= 0.0:
        raise ValueError(
            "background_interactions must have positive sample variance"
        )

    sample_sd = math.sqrt(sample_variance)

    standardized_value = (
        (observed_interaction - mean_background)
        / sample_sd
    )

    return float(
        math.erfc(
            abs(standardized_value)
            / math.sqrt(2.0)
        )
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """
observed_interaction = 0.5833333333333333

background_interactions = [
    -0.15, 0.35, -0.05, 0.25,
     0.00, 0.20, -0.10, 0.30,
     0.05, 0.15, -0.20, 0.40
]
""",
            "call": "compute_pair_raw_significance(observed_interaction, background_interactions)",
            "gold_call": "_oracle_compute_pair_raw_significance(observed_interaction, background_interactions)"
        },

        {
            "setup": """
observed_interaction = 0.1833333333333333

background_interactions = [
    -0.60, -0.10, -0.50, -0.20,
    -0.45, -0.25, -0.55, -0.15,
    -0.40, -0.30, -0.65, -0.05
]
""",
            "call": "compute_pair_raw_significance(observed_interaction, background_interactions)",
            "gold_call": "_oracle_compute_pair_raw_significance(observed_interaction, background_interactions)"
        },

        {
            "setup": """
observed_interaction = -0.0666666666666667

background_interactions = [
    -0.45, 0.05, -0.35, -0.05,
    -0.30, -0.10, -0.40, 0.00,
    -0.25, -0.15, -0.50, 0.10
]
""",
            "call": "compute_pair_raw_significance(observed_interaction, background_interactions)",
            "gold_call": "_oracle_compute_pair_raw_significance(observed_interaction, background_interactions)"
        },

        {
            "setup": """
def _captures_value_error(fn, *args):
    try:
        fn(*args)
    except ValueError:
        return 1.0
    return 0.0

observed_interaction = 0.5
background_interactions = [0.2, 0.2, 0.2, 0.2]
""",
            "call": "_captures_value_error(compute_pair_raw_significance, observed_interaction, background_interactions)",
            "gold_call": "_captures_value_error(_oracle_compute_pair_raw_significance, observed_interaction, background_interactions)"
        }
    ]
