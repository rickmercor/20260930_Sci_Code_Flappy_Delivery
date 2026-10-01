"""
Draw one genotype vector from the exponentially tilted Hardy-Weinberg law attached to a subset and a tilt, using a supplied vector of uniforms.

Tilting a joint law by the exponential of a tilt times a statistic looks like a global operation, but when the statistic is a sum of per-subject terms and the subjects are independent under the null, it factorises. The tilted joint law is again a product of independent per-subject laws, and the law of one subject is its Hardy-Weinberg law reweighted by the exponential of the tilt times that subject's own coefficient times the centred genotype value, renormalised over the three genotype values. Sampling therefore stays as cheap as sampling from the null, which is what makes the whole approach practical: no rejection step, no Markov chain, no convergence diagnostic.




The reweighting is what produces the rare event. A subject with a large positive coefficient sees the probability of carrying minor alleles inflated by orders of magnitude relative to the population frequency, while a subject with a negative coefficient sees it deflated, and the joint effect is a genotype configuration that aligns with the subset's coefficient pattern in exactly the way an extreme value of that subset's statistic requires. Under a low allele frequency the null probability of two minor alleles is the square of a small number, so the tilted probability of that value can be raised by many orders of magnitude and still remain small; this is precisely the regime the tail correction has to explore and the regime in which a Gaussian approximation to the statistic is least trustworthy.




Two details keep the procedure exact and reproducible. First, the three reweighted values must be computed in logarithms and shifted by their maximum before exponentiating, because the product of a large tilt with a large coefficient overflows long before the tilted probabilities themselves become extreme. Second, the mapping from uniform variates to genotypes must be a fixed inverse cumulative rule, taking the smallest genotype value whose cumulative tilted probability strictly exceeds the uniform variate, so that the same stream of uniforms always yields the same genotypes. Accepting the uniforms as an argument rather than generating them internally keeps that stream under the control of the caller, which is what allows the estimator to be reproducible replicate by replicate.

Returns
-------
np.ndarray of shape (n_subjects,), float: the sampled genotypes, each equal to 0.0, 1.0 or 2.0.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def sample_tilted_genotypes(weight_row: np.ndarray, maf: float, tilt: float,
                            uniforms: np.ndarray) -> np.ndarray:
    """Map a vector of uniforms to genotypes under the exponentially tilted law.

    Subject i receives genotype k with probability proportional to
    exp(tilt * weight_row[i] * (k - 2 * maf)) times the Hardy-Weinberg
    probability of k. The genotype returned is the smallest value whose
    cumulative tilted probability strictly exceeds uniforms[i].

    Parameters
    ----------
    weight_row : np.ndarray
        Per-subject coefficients of one subset, of shape (n_subjects,).
    maf : float
        Minor allele frequency, 0 < maf <= 0.5.
    tilt : float
        Exponential tilt applied to the subset statistic.
    uniforms : np.ndarray
        Uniform variates of shape (n_subjects,) with entries in [0, 1).

    Returns
    -------
    genotypes : np.ndarray
        Array of shape (n_subjects,) holding the values 0.0, 1.0 or 2.0.
    """
    return genotypes  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_sample_tilted_genotypes(weight_row: np.ndarray, maf: float, tilt: float,
                                    uniforms: np.ndarray) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    coefficients = np.asarray(weight_row, dtype=float)
    draws = np.asarray(uniforms, dtype=float)
    if coefficients.ndim != 1 or coefficients.size < 1:
        raise ValueError("weight_row must be a non-empty 1D array")
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("weight_row must be finite")
    if draws.ndim != 1 or draws.size != coefficients.size:
        raise ValueError("uniforms must be 1D with one entry per subject")
    if not np.all((draws >= 0.0) & (draws < 1.0)):
        raise ValueError("uniforms must lie in the half-open interval [0, 1)")
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

    exponents = float(tilt) * np.outer(coefficients, centred) + log_prior[None, :]
    exponents -= np.max(exponents, axis=1, keepdims=True)
    tilted = np.exp(exponents)
    tilted /= tilted.sum(axis=1, keepdims=True)

    # Inverse cumulative rule: the smallest genotype whose cumulative tilted
    # probability strictly exceeds the uniform variate.
    cumulative = np.cumsum(tilted, axis=1)
    genotypes = (draws[:, None] > cumulative[:, :2]).sum(axis=1)
    return genotypes.astype(float)

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: rare variant strongly tilted towards the positive branch ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(41)
weight_row = rng.standard_normal(120) / np.sqrt(120 * 2 * 0.02 * 0.98)
uniforms = rng.random(120)
maf, tilt = 0.02, 2.2
def digest(value):
    flat = np.asarray(value, dtype=float).ravel()
    w = np.arange(1.0, flat.size + 1.0) / (flat.size + 1.0)
    return float(np.abs(flat) @ w + flat @ w ** 2)
""",
            "call": "digest(sample_tilted_genotypes(weight_row, maf, tilt, uniforms))",
            "gold_call": "digest(_oracle_sample_tilted_genotypes(weight_row, maf, tilt, uniforms))",
        },
        # --- Valid: the same coefficients tilted towards the negative branch ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(41)
weight_row = rng.standard_normal(120) / np.sqrt(120 * 2 * 0.02 * 0.98)
uniforms = rng.random(120)
maf, tilt = 0.02, -8.5
def digest(value):
    flat = np.asarray(value, dtype=float).ravel()
    w = np.arange(1.0, flat.size + 1.0) / (flat.size + 1.0)
    return float(np.abs(flat) @ w + flat @ w ** 2)
""",
            "call": "digest(sample_tilted_genotypes(weight_row, maf, tilt, uniforms))",
            "gold_call": "digest(_oracle_sample_tilted_genotypes(weight_row, maf, tilt, uniforms))",
        },
        # --- Boundary: zero tilt reproduces the untilted Hardy-Weinberg draw ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(43)
weight_row = rng.standard_normal(50)
uniforms = rng.random(50)
maf, tilt = 0.3, 0.0
def digest(value):
    flat = np.asarray(value, dtype=float).ravel()
    w = np.arange(1.0, flat.size + 1.0) / (flat.size + 1.0)
    return float(np.abs(flat) @ w + flat @ w ** 2)
""",
            "call": "digest(sample_tilted_genotypes(weight_row, maf, tilt, uniforms))",
            "gold_call": "digest(_oracle_sample_tilted_genotypes(weight_row, maf, tilt, uniforms))",
        },
        # --- Edge: uniforms at the two ends of their range with an extreme tilt ---
        {
            "setup": """import numpy as np
weight_row = np.array([5.0, -5.0, 0.0])
uniforms = np.array([0.0, 0.9999999999999999, 0.5])
maf, tilt = 0.01, 30.0
def digest(value):
    flat = np.asarray(value, dtype=float).ravel()
    w = np.arange(1.0, flat.size + 1.0) / (flat.size + 1.0)
    return float(np.abs(flat) @ w + flat @ w ** 2)
""",
            "call": "digest(sample_tilted_genotypes(weight_row, maf, tilt, uniforms))",
            "gold_call": "digest(_oracle_sample_tilted_genotypes(weight_row, maf, tilt, uniforms))",
        },
        # --- Invalid: a uniform variate outside its range ---
        {
            "setup": """import numpy as np
weight_row = np.ones(3)
uniforms = np.array([0.5, 1.0, 0.5])
def run_model():
    try:
        sample_tilted_genotypes(weight_row, 0.1, 1.0, uniforms)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_sample_tilted_genotypes(weight_row, 0.1, 1.0, uniforms)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: mismatched lengths ---
        {
            "setup": """import numpy as np
weight_row = np.ones(4)
uniforms = np.full(3, 0.5)
def run_model():
    try:
        sample_tilted_genotypes(weight_row, 0.1, 1.0, uniforms)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_sample_tilted_genotypes(weight_row, 0.1, 1.0, uniforms)
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
