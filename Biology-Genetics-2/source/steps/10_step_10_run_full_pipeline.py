"""
Run the whole archaic ancestry pipeline on one ancestral recombination graph and

return the resulting estimate of the admixture time in generations.

The analysis chains the pieces in a fixed order. The focal branch spanning the

archaic time cutoff is located in every marginal tree and reduced to one

observation per tree; the upper tail of the genome-wide observation distribution

is flagged by a one-sided extreme Studentized deviate test, and the flagged and

unflagged parts seed the archaic and the modern human gamma emissions of a

two-state hidden Markov chain. Only the archaic emission and the two transition

probabilities are then re-estimated by expectation maximisation, since the

trimmed genome-wide distribution already stands for the modern human state.

Posterior decoding with the fitted parameters gives a probability of archaic

ancestry at every tree, the runs above the posterior threshold that clear both

length filters are the called tracts, and the focal branches of the trees inside

those tracts are reduced to the reported lower bound on the time of gene flow.

Returns
-------
float, the span-weighted admixture time estimate in generations as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def run_full_pipeline(
    coal_times: "np.ndarray",
    focal_mask: "np.ndarray",
    span_bp: "np.ndarray",
    span_cm: "np.ndarray",
    t_archaic: float = 15000.0,
    x_floor: float = 1e-10,
    esd_alpha: float = 0.05,
    esd_max_outlier_fraction: float = 0.2,
    p_init: float = 0.01,
    q_init: float = 0.1,
    pi_archaic: float = 0.05,
    max_iter: int = 200,
    loglik_tol: float = 1e-2,
    post_threshold: float = 0.9,
    min_bp: float = 5e4,
    min_cm: float = 0.05,
) -> float:
    """Return the admixture time estimate for one ancestral recombination graph.

    Parameters
    ----------
    coal_times : np.ndarray
        Coalescence times of shape (m, c), ascending along each row.
    focal_mask : np.ndarray
        Binary focal lineage indicator of shape (m, c).
    span_bp : np.ndarray
        Physical span of each marginal tree in base pairs.
    span_cm : np.ndarray
        Genetic span of each marginal tree in centimorgans.
    t_archaic : float
        Time cutoff in generations defining an archaic event.
    x_floor : float
        Positive floor added to every observation.
    esd_alpha : float
        Significance level of the outlier test.
    esd_max_outlier_fraction : float
        Largest fraction of trees the outlier test may remove.
    p_init : float
        Starting transition probability into the archaic state.
    q_init : float
        Starting transition probability out of the archaic state.
    pi_archaic : float
        Fixed prior probability of the archaic state at the first tree.
    max_iter : int
        Largest number of expectation maximisation iterations.
    loglik_tol : float
        Convergence tolerance on the log-likelihood.
    post_threshold : float
        Posterior probability a tree must exceed to join a candidate tract.
    min_bp : float
        Smallest admissible physical length of a retained tract.
    min_cm : float
        Smallest admissible genetic length of a retained tract.

    Returns
    -------
    t_admix : float
        Span-weighted admixture time estimate in generations.

    Raises
    ------
    ValueError
        If span_bp or span_cm is not one dimensional of length m matching
        coal_times, or if any stage of the pipeline rejects its own inputs, which
        includes an outlier test that flags no tree and a decoding that retains no
        tract.
    """
    return t_admix

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_run_full_pipeline(
    coal_times: "np.ndarray",
    focal_mask: "np.ndarray",
    span_bp: "np.ndarray",
    span_cm: "np.ndarray",
    t_archaic: float = 15000.0,
    x_floor: float = 1e-10,
    esd_alpha: float = 0.05,
    esd_max_outlier_fraction: float = 0.2,
    p_init: float = 0.01,
    q_init: float = 0.1,
    pi_archaic: float = 0.05,
    max_iter: int = 200,
    loglik_tol: float = 1e-2,
    post_threshold: float = 0.9,
    min_bp: float = 5e4,
    min_cm: float = 0.05,
) -> float:
    """Reference implementation."""
    coal_times = np.asarray(coal_times, dtype=float)
    span_bp = np.asarray(span_bp, dtype=float)
    span_cm = np.asarray(span_cm, dtype=float)
    if coal_times.ndim != 2:
        raise ValueError("coal_times must be a two dimensional array of shape (m, c)")
    if span_bp.ndim != 1 or span_bp.size != coal_times.shape[0]:
        raise ValueError("span_bp must be one dimensional of length m")
    if span_cm.ndim != 1 or span_cm.size != coal_times.shape[0]:
        raise ValueError("span_cm must be one dimensional of length m")

    intervals = _oracle_focal_branch_intervals(coal_times, focal_mask, t_archaic=t_archaic)
    _, observations = _oracle_tree_observation_statistic(
        coal_times, intervals, x_floor=x_floor
    )
    flags = _oracle_esd_outlier_flags(
        observations, alpha=esd_alpha, max_outlier_fraction=esd_max_outlier_fraction
    )
    if flags.sum() <= 0.0:
        raise ValueError("the outlier test flagged no tree, so the archaic state has no seed")

    shape_null, rate_null = _oracle_weighted_gamma_mle(observations, 1.0 - flags)
    shape_archaic, rate_archaic = _oracle_weighted_gamma_mle(observations, flags)

    fitted = _oracle_baum_welch_parameters(
        observations,
        shape_null,
        rate_null,
        shape_archaic,
        rate_archaic,
        p=p_init,
        q=q_init,
        pi_archaic=pi_archaic,
        max_iter=max_iter,
        loglik_tol=loglik_tol,
    )
    log_emissions = _oracle_gamma_emission_logpdf(
        observations,
        shape_null,
        rate_null,
        fitted["shape_archaic"],
        fitted["rate_archaic"],
    )
    posteriors, _ = _oracle_forward_backward_posteriors(
        log_emissions, fitted["p"], fitted["q"], pi_archaic=pi_archaic
    )
    segments = _oracle_archaic_segments(
        posteriors[1],
        span_bp,
        span_cm,
        post_threshold=post_threshold,
        min_bp=min_bp,
        min_cm=min_cm,
    )
    return _oracle_admixture_time_estimate(segments, intervals, span_bp)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    fixture = '''import numpy as np

_COAL = np.array([
    [1300, 10930, 11700, 16900, 33800],
    [11320, 11500, 17880, 20960, 21250],
    [2000, 13400, 17340, 21860, 24400],
    [2200, 2970, 10500, 15320, 25700],
    [1300, 3730, 9500, 18600, 23300],
    [4530, 13000, 14510, 15680, 18650],
    [10760, 11130, 11700, 16230, 35200],
    [1500, 6610, 12400, 26630, 31600],
    [3100, 6230, 13400, 14460, 27500],
    [1900, 11450, 12200, 16430, 18200],
    [4700, 9200, 13200, 20850, 21700],
    [5210, 12400, 14510, 14810, 16450],
    [320, 3540, 9600, 10340, 15800],
    [2550, 10800, 13360, 20530, 21050],
    [2183, 4400, 7570, 32420, 48283],
    [1846, 6990, 20090, 32450, 51346],
    [2027, 4840, 14080, 36090, 46827],
    [2314, 17550, 19480, 45670, 49914],
    [870, 2461, 3740, 15900, 25461],
    [1800, 10100, 11540, 16160, 19250],
    [2730, 9550, 11020, 14000, 15300],
    [600, 7540, 11100, 13380, 23000],
    [2300, 12700, 15300, 17750, 21850],
    [700, 5700, 11100, 13150, 19300],
    [900, 1490, 12900, 18160, 27200],
    [1600, 9200, 15560, 19770, 20300],
    [1600, 6730, 9100, 11340, 26800],
    [2210, 2560, 11300, 20060, 35300],
    [7300, 20920, 28530, 30470, 37800],
    [2500, 11630, 12400, 12890, 23300],
    [4460, 5200, 7070, 18080, 28200],
    [1000, 4720, 13200, 15340, 25400],
    [2400, 13000, 15740, 17300, 22050],
    [2650, 10920, 35000, 35050, 47850],
    [9800, 10100, 11190, 16240, 17450],
    [1000, 8240, 12600, 17260, 18500],
    [2000, 11300, 11930, 15680, 17750],
    [1800, 4520, 10100, 19090, 32200],
    [6500, 11700, 13100, 19460, 19750],
    [5260, 6690, 9000, 14780, 21500],
    [5830, 11500, 11830, 16360, 19150],
    [2300, 3940, 11000, 17620, 22400],
    [4850, 13200, 14720, 18080, 18250],
    [8640, 9540, 13200, 24650, 28100],
    [800, 7050, 9300, 14420, 15300],
    [7450, 8690, 11300, 17850, 19900],
    [500, 2940, 11600, 18680, 28900],
    [2910, 11640, 13300, 24550, 26000],
])
_MASK = np.array([
    [1, 0, 1, 0, 1],
    [0, 1, 0, 0, 1],
    [1, 1, 0, 0, 1],
    [1, 0, 1, 0, 1],
    [1, 0, 1, 0, 1],
    [0, 1, 0, 0, 1],
    [0, 0, 1, 0, 1],
    [1, 0, 1, 0, 1],
    [0, 0, 1, 0, 1],
    [1, 0, 1, 0, 1],
    [0, 0, 1, 0, 1],
    [0, 1, 0, 0, 1],
    [0, 0, 1, 0, 1],
    [0, 1, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [0, 1, 0, 0, 1],
    [1, 1, 0, 0, 1],
    [0, 1, 0, 0, 1],
    [1, 0, 1, 0, 1],
    [1, 1, 0, 0, 1],
    [1, 0, 1, 0, 1],
    [1, 0, 1, 0, 1],
    [1, 1, 0, 0, 1],
    [1, 0, 1, 0, 1],
    [0, 0, 1, 0, 1],
    [0, 1, 0, 0, 1],
    [1, 0, 1, 0, 1],
    [0, 1, 0, 0, 1],
    [1, 0, 1, 0, 1],
    [1, 1, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [0, 1, 0, 0, 1],
    [1, 0, 1, 0, 1],
    [1, 1, 0, 0, 1],
    [1, 0, 1, 0, 1],
    [0, 1, 0, 0, 1],
    [0, 0, 1, 0, 1],
    [0, 1, 0, 0, 1],
    [1, 0, 1, 0, 1],
    [0, 1, 0, 0, 1],
    [0, 0, 1, 0, 1],
    [1, 0, 1, 0, 1],
    [0, 0, 1, 0, 1],
    [1, 0, 1, 0, 1],
    [0, 0, 1, 0, 1],
])
_PHYSICAL_SPAN = np.array([
    15800, 12500, 14300, 13000, 10200, 12600, 15600, 15900,
    15800, 13300, 11200, 13000, 13600, 11000, 13000, 12000,
    13700, 13000, 12000, 13300, 11400, 15600, 11100, 10700,
    13200, 11100, 11400, 11700, 13800, 10900, 58000, 11400,
    14400, 62000, 13900, 11600, 13700, 14500, 14400, 9200,
    9600, 11100, 11500, 13900, 9600, 12200, 14600, 15300,
])
_GENETIC_SPAN = np.array([
    0.0160, 0.0141, 0.0157, 0.0144, 0.0117, 0.0133, 0.0135, 0.0126,
    0.0142, 0.0107, 0.0092, 0.0119, 0.0150, 0.0105, 0.0130, 0.0120,
    0.0147, 0.0130, 0.0120, 0.0145, 0.0138, 0.0175, 0.0102, 0.0139,
    0.0144, 0.0131, 0.0131, 0.0142, 0.0166, 0.0109, 0.0610, 0.0119,
    0.0185, 0.0280, 0.0169, 0.0133, 0.0161, 0.0128, 0.0115, 0.0084,
    0.0108, 0.0109, 0.0104, 0.0137, 0.0114, 0.0110, 0.0157, 0.0152,
])
'''
    return [
        # --- Normal scenario: the benchmark configuration ---
        {
            "setup": fixture + """coal_times = _COAL.copy()
focal_mask = _MASK.copy()
span_bp = _PHYSICAL_SPAN.copy()
span_cm = _GENETIC_SPAN.copy()
""",
            "call": "round(run_full_pipeline(coal_times, focal_mask, span_bp, span_cm), 9)",
            "gold_call": "round(_oracle_run_full_pipeline(coal_times, focal_mask, span_bp, span_cm), 9)",
        },
        # --- Boundary case: both length filters switched off ---
        {
            "setup": fixture + """coal_times = _COAL.copy()
focal_mask = _MASK.copy()
span_bp = _PHYSICAL_SPAN.copy()
span_cm = _GENETIC_SPAN.copy()
""",
            "call": "round(run_full_pipeline(coal_times, focal_mask, span_bp, span_cm, min_bp=0.0, min_cm=0.0), 9)",
            "gold_call": "round(_oracle_run_full_pipeline(coal_times, focal_mask, span_bp, span_cm, min_bp=0.0, min_cm=0.0), 9)",
        },
        # --- Edge case: a physical span vector of the wrong length ---
        {
            "setup": fixture + """coal_times = _COAL.copy()
focal_mask = _MASK.copy()
span_bp = _PHYSICAL_SPAN[:-1].copy()
span_cm = _GENETIC_SPAN.copy()
def run_model():
    try:
        run_full_pipeline(coal_times, focal_mask, span_bp, span_cm)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_run_full_pipeline(coal_times, focal_mask, span_bp, span_cm)
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
