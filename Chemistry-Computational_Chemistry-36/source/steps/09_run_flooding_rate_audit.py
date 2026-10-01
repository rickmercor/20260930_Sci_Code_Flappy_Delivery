"""
Run the complete deterministic censored flooding-rate comparison.

The audit joins censoring, source-specific bias restoration, nonlinear
multi-set fitting, a conventional comparator and a stability gate.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def run_flooding_rate_audit(
    seed: int,
    variance_limit: float = 0.003,
    influence_limit: float = 0.2,
) -> float:
    """Return the signed percent difference between two rate estimates.

    Rebuild the supplied seven-set archive for ``seed`` and execute steps
    1--8. Sort sets by full-efficiency log acceleration; scan prefixes from
    three sets; select the largest prefix passing ``variance_limit`` with a
    0.02 interior-gamma margin. Require the maximum absolute leave-one-set-out
    log-rate influence not to exceed ``influence_limit``. Return
    ``100*(k_flooding/k_OPESf - 1)`` without clipping or absolute values.

    Raises
    ------
    ValueError
        If ``seed`` is not an integer or is a Boolean, if ``variance_limit`` is
        nonfinite or negative, if ``influence_limit`` is nonfinite or not
        strictly positive, if an earlier step rejects the generated archive,
        if no eligible prefix exists, if the selected prefix and its direct
        refit disagree, if the leave-one-set-out influence exceeds
        ``influence_limit``, or if the final result is nonfinite.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _flooding_rate_snapshot(seed: int) -> tuple:
    rng = np.random.default_rng(int(seed))
    barriers = np.array([3.0, 4.5, 6.0, 7.5, 9.0, 11.0, 13.5])
    lengths = np.array([
        [18, 18, 17, 18, 16, 18, 18, 15, 18],
        [18, 15, 14, 18, 13, 16, 18, 12, 17],
        [16, 14, 15, 13, 14, 12, 16, 13, 13],
        [14, 13, 12, 14, 11, 13, 14, 12, 12],
        [13, 12, 11, 13, 10, 12, 13, 10, 11],
        [8, 7, 10, 6, 7, 5, 9, 6, 5],
        [7, 6, 8, 5, 6, 4, 7, 5, 4],
    ], dtype=int)
    events = np.array([
        [0, 0, 1, 0, 1, 0, 0, 1, 0],
        [1, 0, 1, 0, 1, 1, 0, 0, 0],
        [1, 1, 0, 1, 1, 0, 0, 1, 0],
        [1, 1, 1, 0, 1, 1, 0, 1, 0],
        [1, 1, 1, 1, 1, 1, 0, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
    ], dtype=int)
    raw_bias = np.full((7, 9, 18), np.nan)
    for set_index in range(7):
        for trajectory in range(9):
            count = int(lengths[set_index, trajectory])
            coordinate = np.linspace(0.0, 1.0, count)
            restored = (
                0.18 + 0.76 * barriers[set_index]
                + 0.18 * np.sin(2.0 * np.pi * (coordinate + 0.07 * trajectory))
                + 0.06 * trajectory
                + rng.normal(scale=0.055, size=count)
            )
            raw_bias[set_index, trajectory, :count] = restored - barriers[set_index]
    return raw_bias, lengths, events, barriers, 0.08, 300.0


def _oracle_run_flooding_rate_audit(
    seed: int,
    variance_limit: float = 0.003,
    influence_limit: float = 0.2,
) -> float:
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, (bool, np.bool_)):
        raise ValueError("seed must be an integer")
    try:
        variance_cutoff = float(variance_limit)
        influence_cutoff = float(influence_limit)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("audit controls must be numeric") from exc
    if (
        not np.isfinite(variance_cutoff) or variance_cutoff < 0.0
        or not np.isfinite(influence_cutoff) or influence_cutoff <= 0.0
    ):
        raise ValueError("invalid variance or influence limit")
    raw, lengths, events, barriers, dt, temperature = _flooding_rate_snapshot(int(seed))
    restored = _oracle_restore_opes_bias(
        raw, lengths, events, barriers, dt, temperature
    )
    observed = _oracle_censored_observed_log_rates(lengths, events, dt)
    full_acceleration = _oracle_ensemble_time_log_acceleration(
        restored, lengths, temperature, 1.0
    )
    order = np.argsort(full_acceleration, kind="stable")
    prefixes = _oracle_scan_eatr_prefixes(
        observed, restored, lengths, temperature, order, 3
    )
    selected_row = _oracle_select_consistent_prefix(
        prefixes, variance_cutoff, 0.02
    )
    selected_count = int(round(float(selected_row[0])))
    selected = order[:selected_count]
    fitted = _oracle_fit_eatr_flooding(
        observed, restored, lengths, temperature, selected
    )
    if not np.allclose(fitted, selected_row[1:], rtol=0.0, atol=5e-10):
        raise ValueError("selected prefix and direct refit disagree")
    opesf_log_rate = _oracle_opesf_cdf_log_rate(
        restored, lengths, events, dt, temperature, selected
    )
    deleted = _oracle_leave_one_set_out_log_rates(
        observed, restored, lengths, temperature, selected
    )
    influence = float(np.max(np.abs(deleted - fitted[1])))
    if not np.isfinite(influence) or influence > influence_cutoff:
        raise ValueError("selected flooding estimate fails the influence check")
    with np.errstate(over="ignore", invalid="ignore"):
        result = 100.0 * np.expm1(float(fitted[1] - opesf_log_rate))
    if not np.isfinite(result):
        raise ValueError("signed rate difference is not finite")
    return float(result)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "", "call": "run_flooding_rate_audit(26091615,.003,.2)", "gold_call": "_oracle_run_flooding_rate_audit(26091615,.003,.2)", "tol": 1e-4},
        {"setup": "", "call": "run_flooding_rate_audit(73,.003,.2)", "gold_call": "_oracle_run_flooding_rate_audit(73,.003,.2)", "tol": 1e-4},
        {"setup": "", "call": "run_flooding_rate_audit(991,.0035,.25)", "gold_call": "_oracle_run_flooding_rate_audit(991,.0035,.25)", "tol": 1e-4},
        {"setup": "import numpy as np\ndef check(fn):\n out=[]\n for args in ((True,.003,.2),(1,-1.,.2),(1,.003,0.)):\n  try: fn(*args)\n  except ValueError: out.append(1)\n  except Exception: out.append(2)\n  else: out.append(0)\n return np.array(out)", "call": "check(run_flooding_rate_audit)", "gold_call": "check(_oracle_run_flooding_rate_audit)", "tol": 0.0},
    ]
