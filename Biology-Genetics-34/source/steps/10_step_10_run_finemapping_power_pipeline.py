"""
Chain the sub-problem functions 01-09 end to end and return the number of cases a prospective case-control study needs before its fine-mapped variants carry a target share of the SNP-based heritability.

At any trial sample size the orchestrator rebuilds the prior, turns it into the constants of the posterior inclusion probability, inverts that curve at the PIP threshold, and reduces the resulting cut into the share of SNP-based heritability the detected variants carry. That share rises monotonically with sample size, so bisecting on it locates the equivalent quantitative-trait study that first reaches the target, which the liability-scale conversion then turns into a number of cases.

Returns
-------
float: the number of cases the prospective case-control study must collect, as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def run_finemapping_power_pipeline(n_snps: int, causal_counts: tuple, gamma: tuple,
                                   h2_snp: float, pip_threshold: float,
                                   target_h2_fraction: float, prevalence: float,
                                   case_fraction: float) -> float:
    """Run the full prospective fine-mapping power pipeline for one trait.

    Parameters
    ----------
    n_snps : int
        Total number of SNPs fitted by the model (n_snps > 0).
    causal_counts : tuple
        One non-negative integer per non-null mixture component, giving the
        number of causal variants assigned to that component.
    gamma : tuple
        Prior variance scale factors of the mixture, one per component. The
        first entry is the null component and must be exactly zero.
    h2_snp : float
        SNP-based heritability of the trait on the liability scale
        (0 < h2_snp < 1).
    pip_threshold : float
        Posterior inclusion probability above which a variant counts as
        fine-mapped (0 < pip_threshold < 1).
    target_h2_fraction : float
        Share of the SNP-based heritability the fine-mapped variants are
        required to carry (0 < target_h2_fraction < 1).
    prevalence : float
        Lifetime prevalence of the disease in the population
        (0 < prevalence < 1).
    case_fraction : float
        Fraction of the ascertained sample that are cases
        (0 < case_fraction < 1).

    Returns
    -------
    n_cases : float
        Number of cases the prospective study must collect, as a native
        Python float.

    Raises
    ------
    ValueError
        If ``n_snps`` is not an integer; if ``causal_counts`` is not a
        non-empty sequence; if ``gamma`` does not hold exactly one more
        entry than ``causal_counts``; if any of ``h2_snp``,
        ``pip_threshold``, ``target_h2_fraction``, ``prevalence`` or
        ``case_fraction`` is not a finite real number strictly between 0 and
        1; or if the target share of heritability is out of reach, meaning
        no finite study size attains it. Errors raised by the sub-problem
        functions this step calls propagate unchanged.

    Notes
    -----
    This is the final, orchestrating step: call the public functions of
    sub-problems 01-09 (``mixture_probabilities``, ``mixture_scale_factors``,
    ``pip_curve_constants``, ``pip_threshold_statistic``,
    ``component_detection_power``, ``component_tail_second_moment``,
    ``component_detected_variance``, ``expected_heritability_explained``,
    ``required_case_count``) and feed each returned value into the next,
    rather than reimplementing them. Include every import your implementation
    needs (for example ``import numpy as np``) inside the function body.
    """
    return 0.0  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_run_finemapping_power_pipeline(n_snps: int, causal_counts: tuple, gamma: tuple,
                                           h2_snp: float, pip_threshold: float,
                                           target_h2_fraction: float, prevalence: float,
                                           case_fraction: float) -> float:
    import numpy as np

    # -- Validate the orchestrator inputs.
    if isinstance(n_snps, bool) or not isinstance(n_snps, (int, np.integer)):
        raise ValueError("n_snps must be an integer")
    if not isinstance(causal_counts, (tuple, list)) or len(causal_counts) < 1:
        raise ValueError("causal_counts must be a non-empty sequence")
    if not isinstance(gamma, (tuple, list)) or len(gamma) != len(causal_counts) + 1:
        raise ValueError("gamma must hold a null component plus one per causal count")
    unit_interval = {"h2_snp": h2_snp, "pip_threshold": pip_threshold,
                     "target_h2_fraction": target_h2_fraction,
                     "prevalence": prevalence, "case_fraction": case_fraction}
    for name, val in unit_interval.items():
        if isinstance(val, bool) or not isinstance(val, (int, float, np.integer, np.floating)):
            raise ValueError(f"{name} must be a real number")
        if not np.isfinite(float(val)) or float(val) <= 0.0 or float(val) >= 1.0:
            raise ValueError(f"{name} must lie strictly between 0 and 1")

    # -- Sub-problem 01: the prior of the mixture model, fixed across sizes.
    # The oracle chains the _oracle_ twins of the earlier steps, never their
    # public names, so that the reference answer is independent of the
    # submission and a wrong candidate pipeline fails the differential test.
    probs = _oracle_mixture_probabilities(int(n_snps), tuple(causal_counts))

    def _explained_share(sample_size):
        """Sub-problems 02-08 evaluated at one trial sample size."""
        factors = _oracle_mixture_scale_factors(
            sample_size, float(h2_snp), tuple(gamma))                           # step 02
        constants = _oracle_pip_curve_constants(probs, factors, sample_size)    # step 03
        z_cut = _oracle_pip_threshold_statistic(
            constants, float(pip_threshold))                                    # step 04
        power = _oracle_component_detection_power(factors, z_cut)               # step 05
        moment = _oracle_component_tail_second_moment(factors, z_cut, power)    # step 06
        detected = _oracle_component_detected_variance(
            factors, power, moment, float(h2_snp), sample_size)                 # step 07
        return _oracle_expected_heritability_explained(
            probs, detected, float(h2_snp), int(n_snps))                        # step 08

    # -- The explained share rises monotonically with the sample size, so
    # doubling brackets the target and bisection locates it.
    lo = 1.0e-6
    hi = 1.0
    while _explained_share(hi) < float(target_h2_fraction):
        lo = hi
        hi *= 2.0
        if hi > 1.0e15:
            raise ValueError("the target share of heritability is out of reach")
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if _explained_share(mid) < float(target_h2_fraction):
            lo = mid
        else:
            hi = mid
    n_equivalent = 0.5 * (lo + hi)

    # -- Sub-problem 09: the liability-scale price of ascertainment.
    return _oracle_required_case_count(n_equivalent, float(prevalence), float(case_fraction))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Benchmark architecture of a highly polygenic disorder ---
        {
            "setup": """import numpy as np
n_snps = 12500000
counts = (40000, 3000, 200, 10)
gamma = (0.0, 1e-5, 1e-4, 1e-3, 1e-2)
""",
            "call": ("run_finemapping_power_pipeline(n_snps, counts, gamma, 0.24,"
                     " 0.9, 0.6, 0.01, 0.5)"),
            "gold_call": ("_oracle_run_finemapping_power_pipeline(n_snps, counts, gamma, 0.24,"
                          " 0.9, 0.6, 0.01, 0.5)"),
        },
        # --- A sparser architecture on a smaller panel, where the largest
        # component alone carries most of the heritability ---
        {
            "setup": """import numpy as np
n_snps = 4000000
counts = (9000, 900, 90, 9)
gamma = (0.0, 2e-5, 2e-4, 2e-3, 2e-2)
""",
            "call": ("run_finemapping_power_pipeline(n_snps, counts, gamma, 0.35,"
                     " 0.8, 0.55, 0.03, 0.4)"),
            "gold_call": ("_oracle_run_finemapping_power_pipeline(n_snps, counts, gamma, 0.35,"
                          " 0.8, 0.55, 0.03, 0.4)"),
        },
        # --- Boundary: a two-component prior in which the whole heritability
        # sits in one large-effect class, so the target is reached early ---
        {
            "setup": """import numpy as np
""",
            "call": ("run_finemapping_power_pipeline(1000000, (100,), (0.0, 1e-2), 0.5,"
                     " 0.95, 0.4, 0.005, 0.5)"),
            "gold_call": ("_oracle_run_finemapping_power_pipeline(1000000, (100,), (0.0, 1e-2), 0.5,"
                          " 0.95, 0.4, 0.005, 0.5)"),
        },
        # --- Boundary: a balanced study of a disease as common as its
        # complement, where the ascertainment penalty is at its mildest ---
        {
            "setup": """import numpy as np
""",
            "call": ("run_finemapping_power_pipeline(8000000, (20000, 1500, 100), "
                     "(0.0, 1e-5, 1e-4, 1e-3), 0.18, 0.7, 0.35, 0.5, 0.5)"),
            "gold_call": ("_oracle_run_finemapping_power_pipeline(8000000, (20000, 1500, 100), "
                          "(0.0, 1e-5, 1e-4, 1e-3), 0.18, 0.7, 0.35, 0.5, 0.5)"),
        },
        # --- Edge: a strongly case-poor design on a rare disease, which
        # multiplies the required case count ---
        {
            "setup": """import numpy as np
""",
            "call": ("run_finemapping_power_pipeline(6000000, (15000, 800), "
                     "(0.0, 2e-5, 5e-4), 0.12, 0.5, 0.45, 0.002, 0.1)"),
            "gold_call": ("_oracle_run_finemapping_power_pipeline(6000000, (15000, 800), "
                          "(0.0, 2e-5, 5e-4), 0.12, 0.5, 0.45, 0.002, 0.1)"),
        },
        # --- Invalid: one fewer scale factor than the architecture needs ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        run_finemapping_power_pipeline(12500000, (40000, 3000, 200, 10),
                                       (0.0, 1e-5, 1e-4, 1e-3), 0.24, 0.9, 0.75, 0.01, 0.5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_run_finemapping_power_pipeline(12500000, (40000, 3000, 200, 10),
                                               (0.0, 1e-5, 1e-4, 1e-3), 0.24, 0.9, 0.75, 0.01, 0.5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a target share of one, which no finite study attains ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        run_finemapping_power_pipeline(12500000, (40000, 3000, 200, 10),
                                       (0.0, 1e-5, 1e-4, 1e-3, 1e-2), 0.24, 0.9, 1.0, 0.01, 0.5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_run_finemapping_power_pipeline(12500000, (40000, 3000, 200, 10),
                                               (0.0, 1e-5, 1e-4, 1e-3, 1e-2), 0.24, 0.9, 1.0,
                                               0.01, 0.5)
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
