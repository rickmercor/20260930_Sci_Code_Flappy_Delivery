"""
Run the mixture importance sampling loop over replicates, drawing each replicate with sub-problem 08 and weighting it with sub-problem 09, and return the estimated tail probability with its standard error.

The estimator is the sample mean of the per-replicate likelihood-ratio contributions, and it is unbiased for the tail probability by construction, because the mixture proposal has the same support as the null and each contribution is the indicator of the rare event divided by the mixture density ratio. Its standard error is the sample standard deviation of the contributions divided by the square root of the number of replicates, and it is the honest measure of the precision achieved, since the contributions are independent and identically distributed across replicates.




The comparison that justifies the whole construction is with direct simulation from the null. A direct estimator of a probability has variance equal to the probability times one minus it, so reaching a fixed relative precision requires a number of replicates inversely proportional to the probability, which becomes impossible several orders of magnitude before genome-wide significance. The tilted estimator has variance that grows only slowly as the threshold rises, because the proposal follows the rare event, and the ratio of the two variances at equal replicate counts is the efficiency gain. In this problem the gain is what makes an estimate at all possible: the tail probability is far beyond the reach of direct simulation, so there is no alternative estimate to fall back on and the standard error is the only available check.




Each replicate performs three draws in a fixed order from one generator: the index of the driving subset, the sign of the branch, and the vector of uniforms that becomes the genotype vector. Fixing that order is what makes the estimate reproducible from the seed alone, and it is why the uniforms are drawn as one vector rather than one subject at a time. The tilts and the cumulant generating function values are properties of the coefficient table and the threshold, not of any replicate, so they are computed once before the loop; recomputing them per replicate would dominate the cost and change nothing.




The estimate is a sample mean of contributions that are mostly either zero, for replicates that fail the indicator, or of comparable magnitude, for those that succeed. That two-part structure is why the estimator behaves well: the surviving contributions are nearly constant when the tilt is chosen at the boundary of the rare event, so the sample variance is far below the variance of a Bernoulli indicator with the same mean. A sharply skewed set of contributions, with a handful dominating the sum, is the diagnostic that the tilt has been misplaced.

Returns
-------
tuple of two native Python floats: the estimated all-subset tail probability and its standard error.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np 

def run_importance_sampling(weights: np.ndarray, tilts: np.ndarray, cgf_values: np.ndarray,
                            maf: float, threshold: float, n_sims: int, seed: int) -> tuple:
    """Estimate the all-subset tail probability by mixture importance sampling.

    Randomness comes from numpy.random.default_rng(seed). Each replicate draws,
    in this order, the driving subset index with integers(n_subsets), then a
    single random() whose value below one half selects the negative branch, then
    a vector of n_subjects uniforms with random(n_subjects). Genotypes follow
    the exponentially tilted Hardy-Weinberg law of the driving subset at the
    tilt of the selected branch, taken by the inverse cumulative rule.

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
    maf : float
        Minor allele frequency, 0 < maf <= 0.5.
    threshold : float
        Observed value of the all-subset maximum, threshold > 0.
    n_sims : int
        Number of replicates, n_sims >= 2.
    seed : int
        Seed of the random generator.

    Returns
    -------
    estimate : tuple
        Native Python floats (pvalue, standard_error), the estimated tail
        probability and the standard error of that estimate.
    """
    return estimate  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np 

def _oracle_run_importance_sampling(weights: np.ndarray, tilts: np.ndarray, cgf_values: np.ndarray,
                                    maf: float, threshold: float, n_sims: int, seed: int) -> tuple:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import glob
    import importlib.util
    import os
    import sys

    import numpy as np

    table = np.asarray(weights, dtype=float)
    tilt_table = np.asarray(tilts, dtype=float)
    cgf_table = np.asarray(cgf_values, dtype=float)
    if table.ndim != 2 or table.shape[0] < 1 or table.shape[1] < 1:
        raise ValueError("weights must be a non-empty 2D array")
    if tilt_table.shape != (table.shape[0], 2) or cgf_table.shape != (table.shape[0], 2):
        raise ValueError("tilts and cgf_values must have shape (n_subsets, 2)")
    if not (np.all(np.isfinite(table)) and np.all(np.isfinite(tilt_table))
            and np.all(np.isfinite(cgf_table))):
        raise ValueError("weights, tilts and cgf_values must be finite")
    if not (isinstance(maf, (int, float)) and np.isfinite(maf)
            and 0.0 < float(maf) <= 0.5):
        raise ValueError("maf must be a finite number in the half-open interval (0, 0.5]")
    if not (isinstance(threshold, (int, float)) and np.isfinite(threshold)
            and float(threshold) > 0.0):
        raise ValueError("threshold must be a finite number > 0")
    for name, value, floor in (("n_sims", n_sims, 2), ("seed", seed, -(2 ** 63))):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < floor:
            raise ValueError(f"{name} must be an integer >= {floor}")

    # -- Resolve the oracle functions of sub-problems 08-09, which this step
    #    composes rather than reimplements. Preference order: (1) already
    #    present in the executing namespace (shared-namespace harness),
    #    (2) loaded from a sibling sub-problem file matched by name pattern
    #    (standalone execution; file prefixes may vary), (3) the public
    #    function of the same step if the harness injected it.
    def _resolve_step(oracle_name, pattern):
        namespace = globals()
        candidate = namespace.get(oracle_name)
        if callable(candidate):
            return candidate
        search_dirs = []
        if "__file__" in namespace:
            search_dirs.append(os.path.dirname(os.path.abspath(namespace["__file__"])))
        cwd = os.getcwd()
        search_dirs += [cwd, os.path.join(cwd, "sub_problems")]
        if sys.argv and sys.argv[0]:
            search_dirs.append(os.path.dirname(os.path.abspath(sys.argv[0])))
        seen = set()
        search_dirs = [d for d in search_dirs if not (d in seen or seen.add(d))]
        for directory in search_dirs:
            for path in sorted(glob.glob(os.path.join(directory, pattern))):
                spec = importlib.util.spec_from_file_location(
                    os.path.basename(path)[:-3], path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                if hasattr(module, oracle_name):
                    return getattr(module, oracle_name)
        public = namespace.get(oracle_name.replace("_oracle_", "", 1))
        if callable(public):
            return public
        raise RuntimeError(f"cannot resolve required step function {oracle_name}")

    tilted_draw = _resolve_step(
        "_oracle_sample_tilted_genotypes", "*sample_tilted_genotypes*.py")
    replicate_weight = _resolve_step(
        "_oracle_compute_is_weight", "*compute_is_weight*.py")

    n_subsets, n_subjects = table.shape
    rng = np.random.default_rng(int(seed))
    contributions = np.empty(int(n_sims), dtype=float)
    for replicate in range(int(n_sims)):
        driver = int(rng.integers(n_subsets))
        branch = 0 if rng.random() >= 0.5 else 1
        draws = rng.random(n_subjects)
        # -- Sub-problem 08: one genotype vector under the driving subset's tilted law.
        genotypes = tilted_draw(table[driver], maf, float(tilt_table[driver, branch]), draws)
        # -- Sub-problem 09: the likelihood-ratio contribution of that vector.
        contributions[replicate] = replicate_weight(
            table, tilt_table, cgf_table, genotypes, maf, threshold)

    pvalue = float(contributions.mean())
    standard_error = float(np.sqrt(contributions.var(ddof=1) / int(n_sims)))
    return (pvalue, standard_error)

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np 

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: rare variant in the tail (normal scenario) ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(61)
weights = rng.standard_normal((7, 40)) / np.sqrt(40 * 2 * 0.02 * 0.98)
tilts = np.column_stack([np.full(7, 2.0), np.full(7, -6.0)])
cgf_values = np.column_stack([np.full(7, 9.0), np.full(7, 11.0)])
maf, threshold, n_sims, seed = 0.02, 5.0, 400, 3
def digest(value):
    flat = np.asarray(value, dtype=float).ravel()
    w = np.arange(1.0, flat.size + 1.0) / (flat.size + 1.0)
    return float(np.abs(flat) @ w + flat @ w ** 2)
""",
            "call": "digest(run_importance_sampling(weights, tilts, cgf_values, maf, threshold, n_sims, seed))",
            "gold_call": "digest(_oracle_run_importance_sampling(weights, tilts, cgf_values, maf, threshold, n_sims, seed))",
        },
        # --- Valid: common variant with a moderate threshold ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(62)
weights = rng.standard_normal((3, 25)) / np.sqrt(25 * 2 * 0.5 * 0.5)
tilts = np.column_stack([np.full(3, 3.0), np.full(3, -3.0)])
cgf_values = np.column_stack([np.full(3, 4.5), np.full(3, 4.5)])
maf, threshold, n_sims, seed = 0.5, 3.0, 500, 8
def digest(value):
    flat = np.asarray(value, dtype=float).ravel()
    w = np.arange(1.0, flat.size + 1.0) / (flat.size + 1.0)
    return float(np.abs(flat) @ w + flat @ w ** 2)
""",
            "call": "digest(run_importance_sampling(weights, tilts, cgf_values, maf, threshold, n_sims, seed))",
            "gold_call": "digest(_oracle_run_importance_sampling(weights, tilts, cgf_values, maf, threshold, n_sims, seed))",
        },
        # --- Boundary: threshold so high that no replicate survives the indicator ---
        {
            "setup": """import numpy as np
weights = np.ones((2, 6)) * 0.1
tilts = np.zeros((2, 2))
cgf_values = np.zeros((2, 2))
maf, threshold, n_sims, seed = 0.1, 50.0, 50, 1
def digest(value):
    flat = np.asarray(value, dtype=float).ravel()
    w = np.arange(1.0, flat.size + 1.0) / (flat.size + 1.0)
    return float(np.abs(flat) @ w + flat @ w ** 2)
""",
            "call": "digest(run_importance_sampling(weights, tilts, cgf_values, maf, threshold, n_sims, seed))",
            "gold_call": "digest(_oracle_run_importance_sampling(weights, tilts, cgf_values, maf, threshold, n_sims, seed))",
        },
        # --- Edge: the smallest admissible number of replicates ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(63)
weights = rng.standard_normal((1, 10))
tilts = np.array([[1.0, -1.0]])
cgf_values = np.array([[2.0, 2.0]])
maf, threshold, n_sims, seed = 0.2, 0.5, 2, 5
def digest(value):
    flat = np.asarray(value, dtype=float).ravel()
    w = np.arange(1.0, flat.size + 1.0) / (flat.size + 1.0)
    return float(np.abs(flat) @ w + flat @ w ** 2)
""",
            "call": "digest(run_importance_sampling(weights, tilts, cgf_values, maf, threshold, n_sims, seed))",
            "gold_call": "digest(_oracle_run_importance_sampling(weights, tilts, cgf_values, maf, threshold, n_sims, seed))",
        },
        # --- Invalid: fewer than two replicates ---
        {
            "setup": """import numpy as np
weights = np.ones((2, 4))
tilts = np.zeros((2, 2))
cgf_values = np.zeros((2, 2))
def run_model():
    try:
        run_importance_sampling(weights, tilts, cgf_values, 0.1, 1.0, 1, 0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_run_importance_sampling(weights, tilts, cgf_values, 0.1, 1.0, 1, 0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: cumulant table of the wrong shape ---
        {
            "setup": """import numpy as np
weights = np.ones((2, 4))
tilts = np.zeros((2, 2))
cgf_values = np.zeros((3, 2))
def run_model():
    try:
        run_importance_sampling(weights, tilts, cgf_values, 0.1, 1.0, 10, 0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_run_importance_sampling(weights, tilts, cgf_values, 0.1, 1.0, 10, 0)
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
