"""
Evaluate the likelihood-ratio contribution of one simulated genotype vector to the estimated tail probability.

An importance sampling estimator is only unbiased if each replicate is divided by the density under which it was actually generated, relative to the null density. Here the generating law is not one tilted law but a uniform mixture of them: no single subset drives the all-subset maximum, and neither sign of the threshold can be ignored in a two-sided scan, so the proposal draws a subset and a sign uniformly and then simulates under the corresponding tilted law. The correct correction is therefore the reciprocal of the mixture density, not of the density of the component that happened to be selected. Using the selected component alone would give an estimator with the wrong mean and, worse, a heavy tailed one, because the replicates it favours are precisely those the other components would have generated more often.




Because every component of the mixture is an exponential tilt of the same null law, the likelihood ratio collapses to something computable without any density evaluation. The ratio of the tilted density to the null density for a given component is the exponential of that component's tilt times that component's subset statistic, divided by the exponential of that component's cumulant generating function at that tilt. Averaging over the mixture and inverting gives a weight equal to the number of components divided by a sum, over all subsets and both signs, of those exponentials evaluated at the simulated genotype vector. Each term needs only the subset statistic of the simulated vector, and all of the subset statistics are one matrix product of the coefficient table with the centred genotype vector.




The indicator matters as much as the ratio. The estimator targets the probability that the all-subset maximum exceeds the threshold, so a replicate whose maximum absolute subset statistic fails to exceed it contributes exactly zero, however large its likelihood ratio would have been. Under a well chosen tilt roughly half of the replicates are discarded this way, which is the signature of a proposal centred on the boundary of the rare event.




Numerically, the sum in the denominator is a sum of exponentials whose exponents span hundreds of units, since the tilts that reach a tail threshold are large and the subset statistics vary across subsets. Forming the terms directly overflows for the dominant subsets and underflows to zero for the rest, so the sum must be accumulated after subtracting the largest exponent and the result reassembled in logarithms. Done that way the weight is finite and accurate over the whole range of replicates; done directly it is either infinite or zero for almost every replicate that matters.

Returns
-------
float: the likelihood-ratio contribution of this replicate, or 0.0 when the all-subset maximum does not exceed the threshold.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def compute_is_weight(weights: np.ndarray, tilts: np.ndarray, cgf_values: np.ndarray,
                      genotypes: np.ndarray, maf: float, threshold: float) -> float:
    """Return the importance sampling contribution of one genotype vector.

    Parameters
    ----------
    weights : np.ndarray
        Table of per-subject coefficients of shape (n_subsets, n_subjects).
    tilts : np.ndarray
        Array of shape (n_subsets, 2) holding the positive-branch and
        negative-branch tilts of every subset.
    cgf_values : np.ndarray
        Array of shape (n_subsets, 2) holding the cumulant generating function
        of every subset evaluated at the matching tilt.
    genotypes : np.ndarray
        Simulated genotypes of shape (n_subjects,), each equal to 0, 1 or 2.
    maf : float
        Minor allele frequency, 0 < maf <= 0.5.
    threshold : float
        Observed value of the all-subset maximum, threshold > 0.

    Returns
    -------
    contribution : float
        Contribution of this replicate as a native Python float; zero when the
        all-subset maximum does not exceed the threshold.
    """
    return contribution  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_compute_is_weight(weights: np.ndarray, tilts: np.ndarray, cgf_values: np.ndarray,
                              genotypes: np.ndarray, maf: float, threshold: float) -> float:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    table = np.asarray(weights, dtype=float)
    tilt_table = np.asarray(tilts, dtype=float)
    cgf_table = np.asarray(cgf_values, dtype=float)
    draw = np.asarray(genotypes, dtype=float)
    if table.ndim != 2 or table.shape[0] < 1 or table.shape[1] < 1:
        raise ValueError("weights must be a non-empty 2D array")
    if tilt_table.shape != (table.shape[0], 2):
        raise ValueError("tilts must have shape (n_subsets, 2)")
    if cgf_table.shape != (table.shape[0], 2):
        raise ValueError("cgf_values must have shape (n_subsets, 2)")
    if draw.ndim != 1 or draw.size != table.shape[1]:
        raise ValueError("genotypes must be 1D with one entry per subject")
    if not np.all(np.isin(draw, (0.0, 1.0, 2.0))):
        raise ValueError("genotypes must take the values 0, 1 or 2")
    if not (np.all(np.isfinite(table)) and np.all(np.isfinite(tilt_table))
            and np.all(np.isfinite(cgf_table))):
        raise ValueError("weights, tilts and cgf_values must be finite")
    if not (isinstance(maf, (int, float)) and np.isfinite(maf)
            and 0.0 < float(maf) <= 0.5):
        raise ValueError("maf must be a finite number in the half-open interval (0, 0.5]")
    if not (isinstance(threshold, (int, float)) and np.isfinite(threshold)
            and float(threshold) > 0.0):
        raise ValueError("threshold must be a finite number > 0")

    statistics = table @ (draw - 2.0 * float(maf))
    if np.max(np.abs(statistics)) <= float(threshold):
        return 0.0

    exponents = np.concatenate([tilt_table[:, 0] * statistics - cgf_table[:, 0],
                                tilt_table[:, 1] * statistics - cgf_table[:, 1]])
    shift = float(np.max(exponents))
    log_mixture = shift + float(np.log(np.sum(np.exp(exponents - shift))))
    return float(2.0 * table.shape[0] * np.exp(-log_mixture))

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: a replicate that exceeds the threshold (normal scenario) ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(51)
weights = rng.standard_normal((7, 30)) / np.sqrt(30 * 2 * 0.05 * 0.95)
tilts = np.column_stack([np.full(7, 1.4), np.full(7, -1.6)])
cgf_values = np.column_stack([np.full(7, 3.1), np.full(7, 3.4)])
genotypes = rng.integers(0, 3, size=30).astype(float)
maf, threshold = 0.05, 0.5
""",
            "call": "compute_is_weight(weights, tilts, cgf_values, genotypes, maf, threshold)",
            "gold_call": "_oracle_compute_is_weight(weights, tilts, cgf_values, genotypes, maf, threshold)",
        },
        # --- Valid: exponents spanning hundreds of units ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(52)
weights = rng.standard_normal((7, 30)) * 4.0
tilts = np.column_stack([np.full(7, 40.0), np.full(7, -40.0)])
cgf_values = np.column_stack([np.linspace(100.0, 400.0, 7), np.linspace(120.0, 380.0, 7)])
genotypes = rng.integers(0, 3, size=30).astype(float)
maf, threshold = 0.02, 1.0
""",
            "call": "compute_is_weight(weights, tilts, cgf_values, genotypes, maf, threshold)",
            "gold_call": "_oracle_compute_is_weight(weights, tilts, cgf_values, genotypes, maf, threshold)",
        },
        # --- Boundary: a replicate that fails the indicator ---
        {
            "setup": """import numpy as np
weights = np.ones((3, 5)) * 0.01
tilts = np.column_stack([np.ones(3), -np.ones(3)])
cgf_values = np.zeros((3, 2))
genotypes = np.zeros(5)
maf, threshold = 0.1, 4.0
""",
            "call": "compute_is_weight(weights, tilts, cgf_values, genotypes, maf, threshold)",
            "gold_call": "_oracle_compute_is_weight(weights, tilts, cgf_values, genotypes, maf, threshold)",
        },
        # --- Edge: one subset and one subject ---
        {
            "setup": """import numpy as np
weights = np.array([[3.0]])
tilts = np.array([[0.7, -0.9]])
cgf_values = np.array([[0.2, 0.3]])
genotypes = np.array([2.0])
maf, threshold = 0.05, 1.0
""",
            "call": "compute_is_weight(weights, tilts, cgf_values, genotypes, maf, threshold)",
            "gold_call": "_oracle_compute_is_weight(weights, tilts, cgf_values, genotypes, maf, threshold)",
        },
        # --- Invalid: a genotype outside the allowed values ---
        {
            "setup": """import numpy as np
weights = np.ones((2, 3))
tilts = np.zeros((2, 2))
cgf_values = np.zeros((2, 2))
genotypes = np.array([0.0, 1.0, 3.0])
def run_model():
    try:
        compute_is_weight(weights, tilts, cgf_values, genotypes, 0.1, 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_is_weight(weights, tilts, cgf_values, genotypes, 0.1, 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: tilt table of the wrong shape ---
        {
            "setup": """import numpy as np
weights = np.ones((2, 3))
tilts = np.zeros((2, 3))
cgf_values = np.zeros((2, 2))
genotypes = np.zeros(3)
def run_model():
    try:
        compute_is_weight(weights, tilts, cgf_values, genotypes, 0.1, 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_is_weight(weights, tilts, cgf_values, genotypes, 0.1, 1.0)
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
