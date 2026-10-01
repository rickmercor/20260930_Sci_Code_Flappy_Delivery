"""
Chain the sub-problem functions 01-10 end to end on the single-cell eQTL testbed and return the base-ten logarithm of the ratio of the analytic significance to the genotype-conditional estimate.




Orchestrator: yes - this is the final step; it integrates the earlier sub-problems (build_expression_panel, compute_subset_weights, compute_subset_correlation, build_neighbour_table, compute_dlm_pvalue, compute_subset_cgf, solve_tilting_parameters, sample_tilted_genotypes, compute_is_weight, run_importance_sampling) rather than reimplementing them.

This step runs the whole comparison end to end. It (i) builds the standardised zero-inflated expression panel with sub-problem 01, (ii) reduces every subset meta-analysis statistic to a fixed linear form in the centred genotypes with sub-problem 02, (iii) recovers the null correlation structure of the subset family from those coefficients with sub-problem 03 and its one-toggle neighbourhood with sub-problem 04, (iv) evaluates the analytic discrete local maxima significance of the observed maximum with sub-problem 05, (v) locates the exponential tilt that centres each subset and each branch on the threshold with sub-problem 07 and evaluates the cumulant generating function there with sub-problem 06, and (vi) estimates the genotype-conditional significance by the mixture importance sampling loop of sub-problem 10, whose replicates are the tilted genotype draws of sub-problem 08 weighted by the likelihood ratios of sub-problem 09.




The two significances describe the same event for the same data and differ only in what they assume about the randomness. The analytic one propagates a multivariate normal law for the cell-type score vector through the all-subset scan and corrects for the dependence between overlapping subsets. The estimated one conditions on the expression panel and puts the randomness where it actually is, in a genotype vector of independent three-point variables at the observed allele frequency, and makes no distributional approximation at all beyond Hardy-Weinberg equilibrium.




The returned scalar is the base-ten logarithm of their ratio, so its sign says which way the analytic correction errs and its magnitude says by how many orders of magnitude. Zero would mean the normal approximation is adequate at this threshold. A negative value means the analytic significance is smaller than the truth, so the correction is anti-conservative and variants declared significant by it are not, while a positive value means the opposite and signals lost power. A value of large magnitude at a threshold in the range used for genome-wide significance means the analytic correction cannot be trusted for this combination of sample size, allele frequency and expression distribution, and that the conditional calculation is not a refinement but a necessity.

Returns
-------
float: the base-ten logarithm of the analytic significance divided by the genotype-conditional estimate, as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np 

def run_asset_discrepancy_pipeline(n_subjects: int = 120, n_cell_types: int = 7,
                                   zero_fraction: float = 0.6,
                                   factor_correlation: float = 0.4,
                                   panel_seed: int = 20260824, maf: float = 0.02,
                                   threshold: float = 7.0, n_sims: int = 50000,
                                   sampling_seed: int = 2026) -> float:
    """Run the whole comparison on the single-cell eQTL testbed.

    Parameters
    ----------
    n_subjects : int
        Number of subjects, n_subjects >= 2.
    n_cell_types : int
        Number of cell types, 1 <= n_cell_types <= 14.
    zero_fraction : float
        Expected fraction of censored expression entries, 0 < zero_fraction < 1.
    factor_correlation : float
        Fraction of latent expression variance carried by the shared subject
        factor, 0 <= factor_correlation < 1.
    panel_seed : int
        Seed of the generator that builds the expression panel.
    maf : float
        Minor allele frequency, 0 < maf <= 0.5.
    threshold : float
        Observed value of the all-subset maximum, threshold > 0.
    n_sims : int
        Number of importance sampling replicates, n_sims >= 2.
    sampling_seed : int
        Seed of the generator that drives the importance sampling loop.

    Returns
    -------
    discrepancy : float
        Base-ten logarithm of the analytic significance divided by the
        genotype-conditional estimate, as a native Python float.
    """
    return discrepancy  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np 

def _oracle_run_asset_discrepancy_pipeline(n_subjects: int = 120, n_cell_types: int = 7,
                                           zero_fraction: float = 0.6,
                                           factor_correlation: float = 0.4,
                                           panel_seed: int = 20260824, maf: float = 0.02,
                                           threshold: float = 7.0, n_sims: int = 50000,
                                           sampling_seed: int = 2026) -> float:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import glob
    import importlib.util
    import os
    import sys

    import numpy as np

    # -- Resolve the oracle functions of sub-problems 01-10. Preference order:
    #    (1) already present in the executing namespace (shared-namespace
    #    harness), (2) loaded from a sibling sub-problem file matched by name
    #    pattern (standalone execution; file prefixes may vary), (3) the
    #    public function of the same step if the harness injected it.
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

    build_panel = _resolve_step(
        "_oracle_build_expression_panel", "*build_expression_panel*.py")
    subset_weights = _resolve_step(
        "_oracle_compute_subset_weights", "*compute_subset_weights*.py")
    subset_correlation = _resolve_step(
        "_oracle_compute_subset_correlation", "*compute_subset_correlation*.py")
    neighbour_table = _resolve_step(
        "_oracle_build_neighbour_table", "*build_neighbour_table*.py")
    dlm_pvalue = _resolve_step(
        "_oracle_compute_dlm_pvalue", "*compute_dlm_pvalue*.py")
    subset_cgf = _resolve_step(
        "_oracle_compute_subset_cgf", "*compute_subset_cgf*.py")
    tilting_parameters = _resolve_step(
        "_oracle_solve_tilting_parameters", "*solve_tilting_parameters*.py")
    importance_sampling = _resolve_step(
        "_oracle_run_importance_sampling", "*run_importance_sampling*.py")

    # -- Sub-problem 01: the panel every later step conditions on.
    panel = build_panel(n_subjects, n_cell_types, zero_fraction, factor_correlation, panel_seed)

    # -- Sub-problems 02-05: the analytic significance under normality.
    weights = subset_weights(panel, maf)
    correlation = subset_correlation(weights, maf)
    neighbours = neighbour_table(n_cell_types)
    analytic = float(dlm_pvalue(correlation, neighbours, threshold))

    # -- Sub-problems 06-10: the genotype-conditional estimate.
    tilts = tilting_parameters(weights, maf, threshold)
    cgf_values = np.empty_like(np.asarray(tilts, dtype=float))
    for index in range(cgf_values.shape[0]):
        cgf_values[index, 0] = subset_cgf(weights[index], maf, float(tilts[index, 0]))[0]
        cgf_values[index, 1] = subset_cgf(weights[index], maf, float(tilts[index, 1]))[0]
    estimate = importance_sampling(weights, tilts, cgf_values, maf, threshold,
                                   n_sims, sampling_seed)
    conditional = float(estimate[0])

    if not (analytic > 0.0 and conditional > 0.0):
        raise ValueError("both significances must be strictly positive to be compared")
    return float(np.log10(analytic / conditional))

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np 

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Integration: small panel, whole pipeline (normal scenario) ---
        {
            "setup": """import numpy as np
n_subjects, n_cell_types = 40, 4
""",
            "call": "run_asset_discrepancy_pipeline(n_subjects, n_cell_types, n_sims=300, threshold=4.0)",
            "gold_call": "_oracle_run_asset_discrepancy_pipeline(n_subjects, n_cell_types, n_sims=300, threshold=4.0)",
        },
        # --- Integration: the configuration whose result is the task's final answer ---
        {
            "setup": """import numpy as np
""",
            "call": "run_asset_discrepancy_pipeline()",
            "gold_call": "_oracle_run_asset_discrepancy_pipeline()",
        },
        # --- Integration (boundary): a common variant, where the sign of the discrepancy flips ---
        {
            "setup": """import numpy as np
maf = 0.5
""",
            "call": "run_asset_discrepancy_pipeline(60, 4, 0.6, 0.4, 7, maf, 4.0, 400, 11)",
            "gold_call": "_oracle_run_asset_discrepancy_pipeline(60, 4, 0.6, 0.4, 7, maf, 4.0, 400, 11)",
        },
        # --- Integration (edge): a single cell type, where the scan has nothing to correct for ---
        {
            "setup": """import numpy as np
""",
            "call": "run_asset_discrepancy_pipeline(50, 1, 0.3, 0.0, 2, 0.1, 3.0, 300, 4)",
            "gold_call": "_oracle_run_asset_discrepancy_pipeline(50, 1, 0.3, 0.0, 2, 0.1, 3.0, 300, 4)",
        },
        # --- Invalid: allele frequency above one half ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        run_asset_discrepancy_pipeline(30, 3, 0.5, 0.2, 1, 0.9, 3.0, 100, 1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_run_asset_discrepancy_pipeline(30, 3, 0.5, 0.2, 1, 0.9, 3.0, 100, 1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a threshold the subset statistics can never attain ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        run_asset_discrepancy_pipeline(20, 2, 0.4, 0.3, 1, 0.05, 500.0, 100, 1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_run_asset_discrepancy_pipeline(20, 2, 0.4, 0.3, 1, 0.05, 500.0, 100, 1)
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
