"""
Run the whole audit on the prescribed instance for the given sweep budget by calling every earlier step function and using its output, and return the reported diagnostics. The instance is the one prescribed in the problem statement (41-node grid 0.25 + 0.0375 j, centre 1.0, source width 0.15, target width 0.15 sqrt(0.70), one percent mass threshold, capped-tilt cap 0.02, soft-update log cap 1.0, median over the final ten percent of sweeps), and the soft update uses the source's quadratic weight and base step. Entry 5, the active source dispersion, is the unnormalised sum over the active states of the source mass times the squared deviation from the centre, the right-hand side of the conditional-Jensen bound. Entry 6, the first tilt's mass shift, is measured on the coupling after one sweep from the product start: the absolute deviation of row 0's mass from its prescribed source mass before, minus after, one capped tilt of row 0 (axis=1, index 0, cap 0.02) applied to that coupling. Entry 7 is the largest absolute entry of the soft-family gradient at the product start.

The audit assembles the prescribed instance, certifies its feasibility status, runs the priority-split scheme for the given budget, and reports the two constraint-family violations aggregated over the tail of the run.

Returns
-------
ndarray of shape (8,), float64: the marginal-family violation, the martingale-family violation, the feasibility slack, the number of active states, the target dispersion, the active source dispersion (the unnormalised active-restricted sum), the first tilt's mass shift (as defined in the step description), and the largest initial soft-gradient magnitude.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def finite_budget_priority_audit(n_sweeps):
    """Run the whole audit on the prescribed instance for the given sweep budget by calling every earlier step function and using its output, and return the reported diagnostics.

    Returns
    -------
    ndarray of shape (8,), float64: the marginal-family violation, the martingale-family violation, the feasibility slack, the number of active states, the target dispersion, the active source dispersion (the unnormalised active-restricted sum), the first tilt's mass shift (as defined in the step description), and the largest initial soft-gradient magnitude.
    """
    return np.zeros(8, dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_finite_budget_priority_audit(n_sweeps):
    n_sweeps = int(n_sweeps)
    if n_sweeps < 1:
        raise ValueError("n_sweeps must be a positive integer")
    # Evaluation configuration: GIVEN in the prompt.
    x = 0.25 + 0.0375 * np.arange(41, dtype=float)
    centre, sigma_x, mass_frac = 1.0, 0.15, 0.01
    hard_cap, soft_cap = 0.02, 1.0
    # L1: the paper's soft-update magnitude. penalty and base step are ONE lock:
    # their product sets the step actually taken.
    penalty, base_step = 200.0, 0.5

    a = _oracle_discrete_gaussian_weights(x, sigma_x, centre)
    b = _oracle_discrete_gaussian_weights(x, sigma_x * np.sqrt(0.70), centre)
    mask = _oracle_active_state_mask(a, mass_frac)
    gap = _oracle_jensen_feasibility_gap(x, a, b, mask, centre)
    # the two dispersions the certificate compares, reported alongside the slack
    m_target = _oracle_discrete_second_moment(x, b, centre)
    m_source_active = _oracle_discrete_second_moment(x, a * mask, centre) * float((a * mask).sum())

    P = _oracle_product_coupling(a, b)
    # The product coupling already meets both marginals, so a tilt applied to it is
    # the identity. Exercise the tilt on the coupling AFTER one sweep, where the soft
    # update has broken the marginals and the cap actually binds.
    grad0 = _oracle_martingale_soft_gradient(P, x, mask, penalty)
    grad0_max = float(np.abs(grad0).max())
    P1 = _oracle_hybrid_sweep(P, x, a, b, mask, hard_cap, penalty, base_step, soft_cap)
    dev_before = abs(float(P1[0, :].sum()) - float(a[0]))
    P1t = _oracle_capped_marginal_tilt(P1, 1, 0, float(a[0]), hard_cap)
    dev_after = abs(float(P1t[0, :].sum()) - float(a[0]))
    tilt_gain = float(dev_before - dev_after)

    hist = np.empty((n_sweeps, 2), dtype=float)
    for _ in range(n_sweeps):
        P = _oracle_hybrid_sweep(P, x, a, b, mask, hard_cap, penalty,
                                 base_step, soft_cap)
        hist[_, :] = _oracle_residual_pair(P, x, a, b, mask)
    tail = max(1, int(round(0.10 * n_sweeps)))
    r_marg = float(np.median(hist[-tail:, 0]))
    r_cond = float(np.median(hist[-tail:, 1]))
    return np.array([r_marg, r_cond, gap, float(mask.sum()),
                     m_target, m_source_active, tilt_gain, grad0_max], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the step's test specifications."""
    return [
        {
            "setup": "import numpy as np",
            "call": "finite_budget_priority_audit(80)",
            "gold_call": "_oracle_finite_budget_priority_audit(80)",
        },
        {
            "setup": "import numpy as np",
            "call": "finite_budget_priority_audit(1)",
            "gold_call": "_oracle_finite_budget_priority_audit(1)",
        },
        {
            "setup": "import numpy as np",
            "call": "finite_budget_priority_audit(160)",
            "gold_call": "_oracle_finite_budget_priority_audit(160)",
        },
    ]
