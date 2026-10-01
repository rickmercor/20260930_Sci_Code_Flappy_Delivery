"""
Apply MILC as a zero-intercept linear combination of descriptors.

The corrected solvation free energy of each conformer is determined through a linear combination of its resolved molecular descriptors and the prescribed regression parameters. The evaluation protocol assumes a zero-centered baseline projection with no empirical offset constants or secondary optimization parameters applied.

Returns
-------
return milc_predictions
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def apply_milc(
    descriptors: np.ndarray,
    coefficients: np.ndarray,
) -> np.ndarray:
    """
    Evaluate the zero-intercept MILC linear model.

    Parameters
    ----------
    descriptors : np.ndarray
        Descriptor matrix with shape (n_conformers, n_descriptors).
    coefficients : np.ndarray
        Coefficient vector with shape (n_descriptors,).

    Returns
    -------
    np.ndarray
        Per-conformer MILC predictions, shape (n_conformers,).

    Raises
    ------
    ValueError
        If the coefficient vector length does not match the number of
        descriptor columns, or if numerical values are invalid.
    """
    return milc_predictions

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_apply_milc(descriptors, coefficients):
    descriptors = np.asarray(descriptors, dtype=float)
    coefficients = np.asarray(coefficients, dtype=float).reshape(-1)

    if descriptors.ndim != 2:
        raise ValueError("descriptors must be two-dimensional")
    if coefficients.ndim != 1:
        raise ValueError("coefficients must be one-dimensional")
    if descriptors.shape[1] != coefficients.shape[0]:
        raise ValueError(
            "descriptor columns must match the coefficient vector length"
        )
    if descriptors.shape[0] == 0:
        raise ValueError("at least one conformer is required")
    if not np.all(np.isfinite(descriptors)):
        raise ValueError("descriptors contain non-finite values")
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("coefficients contain non-finite values")

    return descriptors @ coefficients

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return differential test specifications."""
    return [
        {
            "setup": """import numpy as np

# Test case 1: Production five-descriptor Sub model
descriptors = np.array([
    [0.572849480159, 0.518357724640, 1.330767547529, 0.981427697918, 0.053114016940],
    [-0.561650950134, -0.584873644966, 0.345598358816, 0.204998679779, 0.026181852835],
])
coefficients = np.array(
    [-0.718451957, 1.62878139, 5.19413792, -3.58716288, -750.513679]
)

def run_model():
    return apply_milc(descriptors, coefficients)

def run_gold():
    return _oracle_apply_milc(descriptors, coefficients)
""",
            "call": "run_model()",
            "gold_call": "run_gold()"
        },
        {
            "setup": """import numpy as np

# Test case 2: Single-conformer HNC-set inner product
descriptors = np.array([[0.7002084, 1.3392138, 0.0531140]])
coefficients = np.array([0.99554824, 1.50999895, -682.54621964])

def run_model():
    return apply_milc(descriptors, coefficients)

def run_gold():
    return _oracle_apply_milc(descriptors, coefficients)
""",
            "call": "run_model()",
            "gold_call": "run_gold()"
        },
        {
            "setup": """import numpy as np

# Test case 3: Linearity: mean(X c) equals (mean X) c
rng = np.random.default_rng(9)
X = rng.normal(size=(4, 5))
c = rng.normal(size=5)

def run_model():
    pred = apply_milc(X, c)
    return np.array([np.mean(pred), float(np.mean(X, axis=0) @ c)])

def run_gold():
    pred = _oracle_apply_milc(X, c)
    return np.array([np.mean(pred), float(np.mean(X, axis=0) @ c)])
""",
            "call": "run_model()",
            "gold_call": "run_gold()"
        },
        {
            "setup": """import numpy as np

# Test case 4: Invalid - coefficient length mismatch
descriptors = np.ones((2, 5))
coefficients = np.ones(3)

def run_model():
    try:
        apply_milc(descriptors, coefficients)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_apply_milc(descriptors, coefficients)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()"
        },
    ]
