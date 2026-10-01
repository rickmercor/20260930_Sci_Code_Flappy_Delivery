"""
Orchestrate the fixed-margin conditional haplotype benchmark.

The panel counts determine fixed Gaussian thresholds, working coefficients map

to an oriented unit-variance factor model, and a breakpoint sweep enumerates

partner modes. For each lead state, fixed quadrature builds the conditional

factor mixture, the modes are scored, the leading mode receives its one-allele

neighborhood expansion, and the expanded candidates are rescored and ranked.

The final scalar is the alternate-state top-L mean partner ALT burden minus the

corresponding reference-state burden.

Inputs

------

alt_counts : ALT counts ordered as lead then partners

n_haplotypes : number of phased haplotypes

working_loadings : fitted one-factor working coefficients

psi_min : uniqueness floor

quadrature_order : fixed Gauss-Legendre order

factor_bound : half-width of the factor interval

top_l : number of leading haplotypes retained per lead state

Returns

-------

burden_contrast : alternate-conditioned minus reference-conditioned mean burden

Returns
-------
float, the alternate-minus-reference top-L partner ALT burden as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def run_haploperturb_benchmark(
    alt_counts: np.ndarray,
    n_haplotypes: int,
    working_loadings: np.ndarray,
    psi_min: float,
    quadrature_order: int,
    factor_bound: float,
    top_l: int,
) -> float:
    '''Run the complete conditional haplotype construction and return its burden contrast.

    Parameters
    ----------
    alt_counts : np.ndarray
        ALT counts ordered as lead then partners.
    n_haplotypes : int
        Number of phased haplotypes represented by the counts.
    working_loadings : np.ndarray
        Fitted one-factor working coefficients in the same order.
    psi_min : float
        Minimum admissible residual uniqueness.
    quadrature_order : int
        Number of fixed Gauss-Legendre nodes.
    factor_bound : float
        Positive half-width of the symmetric factor interval.
    top_l : int
        Number of leading configurations retained for each lead state.

    Returns
    -------
    burden_contrast : float
        Alternate-conditioned minus reference-conditioned top-L mean ALT burden.

    Raises
    ------
    ValueError
        If `alt_counts` is not a finite one-dimensional integer vector with at
        least two entries; if `n_haplotypes` is not a positive integer; if any
        count lies outside `[0, n_haplotypes]`; if `working_loadings` is not a
        same-length finite vector with a nonzero lead entry; if `psi_min` is not
        numeric in `(0, 1)` or an implied uniqueness is more than `1e-12` below
        it; if `quadrature_order` is not an integer of at least eight; if
        `factor_bound` is not numeric, positive, and finite; if `top_l` is not
        an integer between one and the expanded candidate count, inclusive; or
        if either lead-conditioned quadrature has no positive finite mass.
    '''
    return burden_contrast  # noqa: F821

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np  # noqa: E402, F811


def _oracle_run_haploperturb_benchmark(
    alt_counts: np.ndarray,
    n_haplotypes: int,
    working_loadings: np.ndarray,
    psi_min: float,
    quadrature_order: int,
    factor_bound: float,
    top_l: int,
) -> float:
    """Reference implementation chaining every earlier benchmark step."""
    _, thresholds = _oracle_compute_fixed_margins(  # noqa: F821
        alt_counts, n_haplotypes
    )
    loadings, uniqueness, _ = _oracle_map_factor_parameters(  # noqa: F821
        working_loadings, thresholds, psi_min
    )
    modes = _oracle_enumerate_factor_modes(  # noqa: F821
        loadings[1:], thresholds[1:]
    )
    burdens = []
    for lead_state in (0, 1):
        _, mixture_weights, partner_probabilities = _oracle_build_conditional_mixture(  # noqa: F821
            loadings,
            uniqueness,
            thresholds,
            lead_state,
            quadrature_order,
            factor_bound,
        )
        mode_probabilities = _oracle_score_conditional_candidates(  # noqa: F821
            modes, mixture_weights, partner_probabilities
        )
        expanded = _oracle_expand_mode_candidates(  # noqa: F821
            modes, mode_probabilities
        )
        expanded_probabilities = _oracle_score_conditional_candidates(  # noqa: F821
            expanded, mixture_weights, partner_probabilities
        )
        mean_burden, _, _, _ = _oracle_summarize_top_burden(  # noqa: F821
            expanded, expanded_probabilities, top_l
        )
        burdens.append(mean_burden)
    return float(burdens[1] - burdens[0])

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np  # noqa: E402, F811


def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """alt_counts = np.array([6, 9, 29, 12, 31, 4, 23, 15])
n_haplotypes = 40
working_loadings = np.array([2.2, 1.7, -1.9, 1.25, -1.55, 0.85, -1.15, 1.45])
psi_min = 0.15
quadrature_order = 64
factor_bound = 8.0
top_l = 4
""",
            "call": "float(run_haploperturb_benchmark(alt_counts, n_haplotypes, working_loadings, psi_min, quadrature_order, factor_bound, top_l))",
            "gold_call": "float(_oracle_run_haploperturb_benchmark(alt_counts, n_haplotypes, working_loadings, psi_min, quadrature_order, factor_bound, top_l))",
        },
        {
            "setup": """alt_counts = np.array([2, 1, 3])
n_haplotypes = 4
working_loadings = np.array([1.0, 0.7, -0.6])
psi_min = 0.3
quadrature_order = 16
factor_bound = 6.0
top_l = 2
""",
            "call": "float(run_haploperturb_benchmark(alt_counts, n_haplotypes, working_loadings, psi_min, quadrature_order, factor_bound, top_l))",
            "gold_call": "float(_oracle_run_haploperturb_benchmark(alt_counts, n_haplotypes, working_loadings, psi_min, quadrature_order, factor_bound, top_l))",
        },
        {
            "setup": """alt_counts = np.array([2, 1, 3])
n_haplotypes = 4
working_loadings = np.array([3.0, 0.7, -0.6])
psi_min = 0.3
quadrature_order = 16
factor_bound = 6.0
top_l = 2
def run_model():
    try:
        run_haploperturb_benchmark(alt_counts, n_haplotypes, working_loadings, psi_min, quadrature_order, factor_bound, top_l)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_run_haploperturb_benchmark(alt_counts, n_haplotypes, working_loadings, psi_min, quadrature_order, factor_bound, top_l)
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
