"""
Return the score statistic of one variant against the residual phenotype.

Once a trait has been corrected for polygenic background, each variant is tested

against the residual by a score statistic rather than by refitting the mixed

model for every variant, which is what makes a genome-wide scan affordable. The

statistic is the squared projection of the residual trait on the variant divided

by the variant's own sum of squares, (g^T y*)^2 / (g^T g), so that variants of

different frequency are placed on a common footing and the value is unchanged if

the variant's column is rescaled.

Returns
-------
float, the score statistic of the variant as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def grammar_gamma_statistic(
    genotype_column: "np.ndarray",
    residual: "np.ndarray",
) -> float:
    """Score one variant against a residual phenotype.

    Args:
        genotype_column: array of shape (n,) holding the centred,
            frequency-scaled genotypes of the variant under test.
        residual: array of shape (n,) holding the residual phenotype.

    Returns:
        float, the score statistic of the variant.

    Raises:
        ValueError: if the two arrays differ in length, or if the variant
            carries no variation, which leaves the statistic undefined.
    """
    return statistic

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_grammar_gamma_statistic(
    genotype_column: "np.ndarray",
    residual: "np.ndarray",
) -> float:
    """Reference implementation."""
    variant = np.asarray(genotype_column, dtype=float).ravel()
    trait = np.asarray(residual, dtype=float).ravel()
    if variant.size != trait.size:
        raise ValueError("genotype column and residual differ in length")
    denominator = float(variant @ variant)
    if denominator <= 0.0:
        raise ValueError("the variant carries no variation")
    projection = float(variant @ trait)
    return projection * projection / denominator

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
column = np.array([0.75446005781, 0.75446005781, -0.50297337187, -0.50297337187, -0.50297337187])
resid = np.array([0.2027, 0.2406, 0.3874, -0.7694, -0.1517])
""",
            "call": "round(grammar_gamma_statistic(column, resid), 12)",
            "gold_call": "round(_oracle_grammar_gamma_statistic(column, resid), 12)",
        },
        {
            "setup": """import numpy as np
column = np.array([1.0, -1.0, 1.0, -1.0, 0.0])
resid = np.array([0.5, 0.5, 0.5, 0.5, 2.0])
""",
            "call": "round(grammar_gamma_statistic(column, resid), 12)",
            "gold_call": "round(_oracle_grammar_gamma_statistic(column, resid), 12)",
        },
        {
            "setup": """import numpy as np

def run_model():
    try:
        grammar_gamma_statistic(np.array([0.0, 0.0, 0.0]), np.array([1.0, 2.0, 3.0]))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_oracle():
    try:
        _oracle_grammar_gamma_statistic(np.array([0.0, 0.0, 0.0]), np.array([1.0, 2.0, 3.0]))
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
