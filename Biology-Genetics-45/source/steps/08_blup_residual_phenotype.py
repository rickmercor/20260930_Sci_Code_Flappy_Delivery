"""
Return the phenotype left over once the polygenic prediction from the retained

chromosomes has been subtracted.

The prediction of a trait from the fitted mixed model is the genetic part of the

leave-one-chromosome-out covariance applied to the solution of the system, and

the residual trait is the standardised phenotype minus that prediction. The

phenotype is standardised here on the same scale the system was solved on, zero

mean and y^T y = n. Subtracting the prediction leaves a trait that carries the

region under test but not the polygenic background shared across the retained

chromosomes, which is what makes a test on the residual comparable across

regions. Because the focal chromosome was held out, its own variants are absent

from the prediction and are therefore not corrected away from the residual.

Returns
-------
np.ndarray of shape (n,), float64 residual phenotype
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def blup_residual_phenotype(
    prediction: "np.ndarray",
    phenotype: "np.ndarray",
) -> "np.ndarray":
    """Subtract the polygenic prediction from the standardised phenotype.

    Args:
        prediction: array of shape (n,) holding the polygenic prediction made
            from the chromosomes that were retained in the fit.
        phenotype: array of shape (n,) of raw phenotype values, standardised
            here to zero mean and y^T y = n before the prediction is removed.

    Returns:
        np.ndarray of shape (n,) holding the residual phenotype.

    Raises:
        ValueError: if the prediction and the phenotype differ in length, or if
            the phenotype has zero variance.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_blup_residual_phenotype(
    prediction: "np.ndarray",
    phenotype: "np.ndarray",
) -> "np.ndarray":
    """Reference implementation."""
    predicted = np.asarray(prediction, dtype=float).ravel()
    trait = np.asarray(phenotype, dtype=float).ravel()
    if predicted.size != trait.size:
        raise ValueError("prediction and phenotype differ in length")
    centred = trait - trait.mean()
    scale = float(centred @ centred)
    if scale <= 0.0:
        raise ValueError("phenotype has zero variance")
    standardised = centred * np.sqrt(trait.size / scale)
    return standardised - predicted

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
prediction = np.array([0.31, -0.42, 0.08, 0.56, -0.53])
trait = np.array([10.4, 12.1, 9.7, 11.5, 13.2])

def rnd(a):
    return [round(float(v), 10) for v in list(a)]
""",
            "call": "rnd(blup_residual_phenotype(prediction, trait))",
            "gold_call": "rnd(_oracle_blup_residual_phenotype(prediction, trait))",
        },
        # A prediction of zero leaves the standardised phenotype untouched, which fixes the
        # scaling the residual is measured on.
        {
            "setup": """import numpy as np
prediction = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
trait = np.array([10.4, 12.1, 9.7, 11.5, 13.2])

def rnd(a):
    return [round(float(v), 10) for v in list(a)]
""",
            "call": "rnd(blup_residual_phenotype(prediction, trait))",
            "gold_call": "rnd(_oracle_blup_residual_phenotype(prediction, trait))",
        },
        {
            "setup": """import numpy as np
trait = np.array([10.4, 12.1, 9.7, 11.5, 13.2])

def run_model():
    try:
        blup_residual_phenotype(np.array([0.1, 0.2]), trait)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_oracle():
    try:
        _oracle_blup_residual_phenotype(np.array([0.1, 0.2]), trait)
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
