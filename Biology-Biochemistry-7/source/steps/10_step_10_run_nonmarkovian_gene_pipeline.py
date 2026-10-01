"""
Chain steps 01-09 and return the signed protein-mean bias.

Orchestrator: yes - packs the parameters (pack_gene_parameters), then evaluates the channel rates (gene_channel_propensities), the integrated completion hazard (completion_hazard_increment), its inversion (invert_completion_waits), the clock advance (advance_remaining_clocks) and the delay-group means (compute_group_completion_means) on the initial state, before running both ensembles (estimate_endpoint_means, which drives simulate_exact_gene_path and simulate_tau_gene_path) and forming the signed ratio; it consumes each step's output rather than reimplementing any of them.

The signed relative difference measures weak endpoint bias introduced by freezing biochemical state and grouping delayed initiations on a coarse grid.

Returns
-------
float: 100 * (approximate mean - exact mean) / exact mean.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def run_nonmarkovian_gene_pipeline(final_time: float = 60.0,
                                    tau: float = 2.5,
                                    n_paths: int = 80,
                                    base_seed: int = 314159) -> float:
    """Return the signed percent bias in approximate mean protein abundance.

    Raises
    ------
    ValueError
        If ``final_time`` or ``tau`` is not positive and finite, if
        ``n_paths`` is not an integer of at least two, if ``base_seed`` is not
        a non-negative integer, or if the exact ensemble mean is not positive.
    """
    return signed_percent_bias  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_run_nonmarkovian_gene_pipeline(final_time: float = 60.0,
                                            tau: float = 2.5,
                                            n_paths: int = 80,
                                            base_seed: int = 314159) -> float:
    parameters = _oracle_pack_gene_parameters(
        10.0, 0.175, 1.0, 0.08, 0.05, 2.5, 10.0, 0.5, 1.5, 5.0)
    # Exercise each paper-specific primitive directly before the ensemble run.
    empty = np.empty(0, dtype=float)
    initial_state = np.zeros(3, dtype=int)
    propensities = _oracle_gene_channel_propensities(initial_state, parameters)
    increments = _oracle_completion_hazard_increment(
        empty, 0.0, initial_state, parameters)
    waits = _oracle_invert_completion_waits(
        empty, empty, initial_state, parameters)
    clocks = _oracle_advance_remaining_clocks(
        np.ones(4), empty, empty, 0.0, initial_state, parameters)
    group_means = _oracle_compute_group_completion_means(
        empty, empty, 0.0, initial_state, parameters)
    exact_probe = _oracle_simulate_exact_gene_path(
        parameters, min(float(final_time), 0.1), int(base_seed))
    tau_probe = _oracle_simulate_tau_gene_path(
        parameters, min(float(final_time), 0.1), min(float(tau), 0.1),
        int(base_seed) + 1)
    certificate = np.concatenate((propensities, increments, waits, clocks,
                                  group_means, exact_probe, tau_probe))
    if not np.all(np.isfinite(certificate)):
        raise ValueError("the direct oracle chain produced a non-finite value")
    estimates = _oracle_estimate_endpoint_means(
        parameters, final_time, tau, n_paths, base_seed)
    exact_mean = float(estimates[0])
    approximate_mean = float(estimates[1])
    if not np.isfinite(exact_mean) or exact_mean <= 0.0:
        raise ValueError("the exact ensemble mean must be positive and finite")
    return float(100.0 * (approximate_mean - exact_mean) / exact_mean)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np",
            "call": "run_nonmarkovian_gene_pipeline(8,2,6,7)",
            "gold_call": "_oracle_run_nonmarkovian_gene_pipeline(8,2,6,7)",
        },
        {
            "setup": "import numpy as np",
            "call": "run_nonmarkovian_gene_pipeline(10,.25,5,0)",
            "gold_call": "_oracle_run_nonmarkovian_gene_pipeline(10,.25,5,0)",
        },
        {
            "setup": "import numpy as np",
            "call": "run_nonmarkovian_gene_pipeline(4,5,4,99)",
            "gold_call": "_oracle_run_nonmarkovian_gene_pipeline(4,5,4,99)",
        },
        {"setup": "import numpy as np\ndef bad():\n    try: run_nonmarkovian_gene_pipeline(8,2,1,7); return 0\n    except ValueError: return 1\n    except Exception: return 2\ndef bad_gold():\n    try: _oracle_run_nonmarkovian_gene_pipeline(8,2,1,7); return 0\n    except ValueError: return 1\n    except Exception: return 2\n", "call": "bad()", "gold_call": "bad_gold()"},
    ]
