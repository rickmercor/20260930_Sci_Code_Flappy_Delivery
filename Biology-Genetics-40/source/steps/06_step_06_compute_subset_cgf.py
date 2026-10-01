"""
Evaluate the conditional cumulant generating function of one subset statistic under the genotype null, together with its first two derivatives.

Conditional on the expression panel, a subset statistic is a fixed linear combination of centred genotypes, and the genotypes of distinct subjects are independent, each taking the values zero, one and two with the Hardy-Weinberg probabilities determined by the allele frequency. The exact null distribution of the statistic is therefore the distribution of a weighted sum of independent three-point variables. It is a lattice distribution with an enormous number of atoms; it is not Gaussian, and for a low allele frequency it is severely skewed, because a subject almost always contributes the same value and only rarely contributes a large one.




The object that makes such a distribution tractable in the tail is its cumulant generating function, the logarithm of the moment generating function. Independence across subjects turns it into a sum of per-subject terms, and each term is the logarithm of a three-term mixture of exponentials in that subject's coefficient. The function is smooth, finite for every real argument because the support is bounded, and convex, and its first derivative is the mean of the statistic under the exponentially tilted measure indexed by that argument. The second derivative is the variance under the same tilted measure, and being a variance, it is strictly positive whenever any coefficient is nonzero, which makes the first derivative strictly increasing. That monotonicity is what allows the argument to be chosen so that the tilted mean lands exactly on a prescribed value, and the second derivative supplies the Newton step that finds it.




Two implementation points decide whether the result is usable. First, the derivatives should be obtained from the same tilted per-subject probabilities that define the function rather than by numerical differencing: normalizing the three tilted weights of a subject gives a probability vector whose mean and variance against the centred genotype values are exactly the per-subject contributions to the first and second derivatives. Second, the exponents are large. A rare variant produces coefficients spread over orders of magnitude, and the tilting argument needed to reach a tail threshold multiplies them further, so the three exponentials of a subject must be combined after subtracting their maximum. Without that shift the evaluation overflows before the tilting argument reaches the value the tail actually requires, and the failure is silent: the overflow appears as an infinite cumulant generating function and a meaningless derivative rather than as an error.

Returns
-------
tuple of three native Python floats: the cumulant generating function at tilt, its first derivative and its second derivative.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def compute_subset_cgf(weight_row: np.ndarray, maf: float, tilt: float) -> tuple:
    """Return the conditional cumulant generating function of one subset statistic.

    The statistic is the sum over subjects of weight_row[i] times the centred
    genotype g_i - 2 * maf, with g_i taking the values 0, 1 and 2 with the
    Hardy-Weinberg probabilities of the allele frequency maf.

    Parameters
    ----------
    weight_row : np.ndarray
        Per-subject coefficients of one subset, of shape (n_subjects,) with
        n_subjects >= 1.
    maf : float
        Minor allele frequency, 0 < maf <= 0.5.
    tilt : float
        Argument of the cumulant generating function.

    Returns
    -------
    values : tuple
        Native Python floats (cgf, first_derivative, second_derivative), the
        value of the cumulant generating function at tilt and its first two
        derivatives with respect to tilt.
    """
    return values  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_compute_subset_cgf(weight_row: np.ndarray, maf: float, tilt: float) -> tuple:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    coefficients = np.asarray(weight_row, dtype=float)
    if coefficients.ndim != 1 or coefficients.size < 1:
        raise ValueError("weight_row must be a non-empty 1D array")
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("weight_row must be finite")
    if not (isinstance(maf, (int, float)) and np.isfinite(maf)
            and 0.0 < float(maf) <= 0.5):
        raise ValueError("maf must be a finite number in the half-open interval (0, 0.5]")
    if not (isinstance(tilt, (int, float)) and np.isfinite(tilt)):
        raise ValueError("tilt must be a finite number")

    frequency = float(maf)
    centred = np.array([0.0, 1.0, 2.0]) - 2.0 * frequency
    log_prior = np.log(np.array([(1.0 - frequency) ** 2,
                                 2.0 * frequency * (1.0 - frequency),
                                 frequency ** 2]))

    # Per-subject log weights of the tilted three-point law, shifted by their
    # maximum so that a large tilt cannot overflow.
    exponents = float(tilt) * np.outer(coefficients, centred) + log_prior[None, :]
    shift = np.max(exponents, axis=1)
    scaled = np.exp(exponents - shift[:, None])
    normaliser = scaled.sum(axis=1)
    cgf = float(np.sum(shift + np.log(normaliser)))

    tilted = scaled / normaliser[:, None]
    tilted_mean = tilted @ centred
    tilted_variance = tilted @ (centred * centred) - tilted_mean * tilted_mean
    first = float(np.sum(coefficients * tilted_mean))
    second = float(np.sum(coefficients * coefficients * tilted_variance))
    return (cgf, first, second)

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: rare variant at a tilt that reaches the tail (normal scenario) ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(21)
weight_row = rng.standard_normal(120) / np.sqrt(120 * 2 * 0.02 * 0.98)
maf, tilt = 0.02, 2.1
def digest(value):
    flat = np.asarray(value, dtype=float).ravel()
    w = np.arange(1.0, flat.size + 1.0) / (flat.size + 1.0)
    return float(np.abs(flat) @ w + flat @ w ** 2)
""",
            "call": "digest(compute_subset_cgf(weight_row, maf, tilt))",
            "gold_call": "digest(_oracle_compute_subset_cgf(weight_row, maf, tilt))",
        },
        # --- Valid: the same coefficients at a large negative tilt ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(21)
weight_row = rng.standard_normal(120) / np.sqrt(120 * 2 * 0.02 * 0.98)
maf, tilt = 0.02, -9.0
def digest(value):
    flat = np.asarray(value, dtype=float).ravel()
    w = np.arange(1.0, flat.size + 1.0) / (flat.size + 1.0)
    return float(np.abs(flat) @ w + flat @ w ** 2)
""",
            "call": "digest(compute_subset_cgf(weight_row, maf, tilt))",
            "gold_call": "digest(_oracle_compute_subset_cgf(weight_row, maf, tilt))",
        },
        # --- Boundary: zero tilt, where the function vanishes and the derivatives are the null moments ---
        {
            "setup": """import numpy as np
weight_row = np.linspace(-0.4, 0.6, 15)
maf, tilt = 0.5, 0.0
def digest(value):
    flat = np.asarray(value, dtype=float).ravel()
    w = np.arange(1.0, flat.size + 1.0) / (flat.size + 1.0)
    return float(np.abs(flat) @ w + flat @ w ** 2)
""",
            "call": "digest(compute_subset_cgf(weight_row, maf, tilt))",
            "gold_call": "digest(_oracle_compute_subset_cgf(weight_row, maf, tilt))",
        },
        # --- Edge: one subject with a large coefficient and an extreme tilt ---
        {
            "setup": """import numpy as np
weight_row = np.array([250.0])
maf, tilt = 0.01, 4.0
def digest(value):
    flat = np.asarray(value, dtype=float).ravel()
    w = np.arange(1.0, flat.size + 1.0) / (flat.size + 1.0)
    return float(np.abs(flat) @ w + flat @ w ** 2)
""",
            "call": "digest(compute_subset_cgf(weight_row, maf, tilt))",
            "gold_call": "digest(_oracle_compute_subset_cgf(weight_row, maf, tilt))",
        },
        # --- Invalid: a non-finite tilt ---
        {
            "setup": """import numpy as np
weight_row = np.ones(4)
def run_model():
    try:
        compute_subset_cgf(weight_row, 0.1, float('inf'))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_subset_cgf(weight_row, 0.1, float('inf'))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a two-dimensional coefficient array ---
        {
            "setup": """import numpy as np
weight_row = np.ones((2, 3))
def run_model():
    try:
        compute_subset_cgf(weight_row, 0.1, 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_subset_cgf(weight_row, 0.1, 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
