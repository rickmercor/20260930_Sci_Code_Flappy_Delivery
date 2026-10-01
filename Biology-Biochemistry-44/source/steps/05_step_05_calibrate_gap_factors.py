"""
Find the gap-opening and gap-extension factors at which the expected numbers of gap-opening and gap-extension columns in the alignment ensemble of x with y equal two target counts.

Fitting a model whose weights are products of factors by maximum likelihood equates each expected transition count with its observed count, so the gap factors of a pair model can be recovered from the numbers of gap openings and extensions seen in a curated alignment.

Returns
-------
np.ndarray: [gap_open, gap_extend] reproducing both target expected counts.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def calibrate_gap_factors(
    x: "np.ndarray",
    y: "np.ndarray",
    match_factors: "np.ndarray",
    x_gap_factors: "np.ndarray",
    y_gap_factors: "np.ndarray",
    open_count: float,
    extend_count: float,
) -> "np.ndarray":
    """Return the gap factors whose expected transition counts match two targets.

    The pair model is that of ``compute_log_forward_tables``, with the match
    and gap emission factors given and the two transition factors unknown.
    Find positive ``gap_open`` and ``gap_extend`` at which the values
    returned by ``compute_expected_gap_transition_counts`` equal
    ``open_count`` and ``extend_count``. If the equations have more than one
    positive solution, return any positive pair satisfying them. Both
    expected counts at the returned factors must reproduce their targets to
    a relative precision of ``1e-10``.

    Parameters
    ----------
    x, y : np.ndarray
        Non-empty one-dimensional integer arrays of codes 0 to 3.
    match_factors : np.ndarray
        Shape ``(4, 4)``, finite and positive.
    x_gap_factors, y_gap_factors : np.ndarray
        Shape ``(4,)``, finite and positive.
    open_count, extend_count : float
        Finite positive target expected counts.

    Returns
    -------
    np.ndarray
        Float array ``[gap_open, gap_extend]``.

    Raises
    ------
    ValueError
        If a sequence or factor array is invalid as in
        ``compute_log_forward_tables``, if a target count is not a finite
        positive number, or if no positive pair of factors reproduces the
        targets.
    """
    return gap_factors

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_calibrate_gap_factors(
    x: "np.ndarray",
    y: "np.ndarray",
    match_factors: "np.ndarray",
    x_gap_factors: "np.ndarray",
    y_gap_factors: "np.ndarray",
    open_count: float,
    extend_count: float,
) -> "np.ndarray":
    """Reference implementation: damped quasi-Newton iteration in the log factors."""
    import numpy as np

    _validate_pair_model(x, y, match_factors, x_gap_factors, y_gap_factors, 1.0, 1.0)
    targets = np.array([open_count, extend_count], dtype=float)
    if targets.shape != (2,) or not np.all(np.isfinite(targets)) or np.any(targets <= 0.0):
        raise ValueError("target counts must be finite and positive")
    log_targets = np.log(targets)

    def _residual_at(theta):
        # Log counts minus log targets; infinite where the factors leave the float range.
        if np.any(np.abs(theta) > 690.0):
            return np.full(2, np.inf)
        args = (x, y, match_factors, x_gap_factors, y_gap_factors, *np.exp(theta))
        counts = _oracle_compute_expected_gap_transition_counts(
            *args, _oracle_compute_log_forward_tables(*args), _oracle_compute_log_backward_tables(*args))
        with np.errstate(divide="ignore"):
            return np.log(counts) - log_targets

    def _jacobian_at(theta, residual, step=1.0e-6):
        columns = [(_residual_at(theta + step * np.eye(2)[k]) - residual) / step for k in range(2)]
        return np.column_stack(columns)

    theta = np.log([0.05, 0.5])
    residual = _residual_at(theta)
    if not np.all(np.isfinite(residual)):
        raise ValueError("no positive gap factors reproduce the target counts")
    jacobian, fresh = _jacobian_at(theta, residual), True
    for _ in range(100):
        norm = np.max(np.abs(residual))
        # Summed counts carry rounding near 1e-12 for long sequences, so stop well above it.
        if norm <= 1.0e-11:
            break
        accepted = False
        if np.all(np.isfinite(jacobian)) and abs(np.linalg.det(jacobian)) > 1.0e-14:
            step = -np.linalg.solve(jacobian, residual)
            for scale in 0.5 ** np.arange(12):
                trial = theta + scale * step
                trial_residual = _residual_at(trial)
                if np.all(np.isfinite(trial_residual)) and np.max(np.abs(trial_residual)) < norm:
                    accepted = True
                    break
        if accepted:
            # Broyden rank-one update keeps most iterations at one model evaluation.
            delta, change = trial - theta, trial_residual - residual
            jacobian = jacobian + np.outer(change - jacobian @ delta, delta) / (delta @ delta)
            theta, residual, fresh = trial, trial_residual, False
        elif not fresh:
            jacobian, fresh = _jacobian_at(theta, residual), True
        else:
            break
    if np.all(np.isfinite(residual)) and np.max(np.abs(residual)) <= 1.0e-10:
        return np.exp(theta)
    raise ValueError("no positive gap factors reproduce the target counts")

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    model = (
        "import math\n"
        "import numpy as np\n"
        "M = np.array([[0.16, 0.03, 0.05, 0.03], [0.03, 0.16, 0.03, 0.05],\n"
        "              [0.05, 0.03, 0.16, 0.03], [0.03, 0.05, 0.03, 0.16]])\n"
        "EX = np.array([0.12, 0.38, 0.38, 0.12])\n"
        "EY = np.array([0.16, 0.34, 0.34, 0.16])\n"
        "def _lae(*v):\n"
        "    top = max(v)\n"
        "    return top if top == -math.inf else top + math.log(sum(math.exp(t - top) for t in v))\n"
        "def _log_z(x, y, em, ex, ey, go, ge):\n"
        "    n, m = len(x), len(y)\n"
        "    lo, le = math.log(go), math.log(ge)\n"
        "    F = np.full((3, n + 1, m + 1), -math.inf)\n"
        "    F[0, 0, 0] = 0.0\n"
        "    for i in range(n + 1):\n"
        "        for j in range(m + 1):\n"
        "            if i and j:\n"
        "                F[0, i, j] = math.log(em[x[i-1], y[j-1]]) + _lae(*F[:, i-1, j-1])\n"
        "            if i:\n"
        "                F[1, i, j] = math.log(ex[x[i-1]]) + _lae(lo + F[0, i-1, j], le + F[1, i-1, j])\n"
        "            if j:\n"
        "                F[2, i, j] = math.log(ey[y[j-1]]) + _lae(lo + F[0, i, j-1], le + F[2, i, j-1])\n"
        "    return _lae(*F[:, n, m])\n"
        "def _targets(x, y, em, ex, ey, go, ge, h=1.0e-4):\n"
        "    up = _log_z(x, y, em, ex, ey, go * math.exp(h), ge) - _log_z(x, y, em, ex, ey, go * math.exp(-h), ge)\n"
        "    ue = _log_z(x, y, em, ex, ey, go, ge * math.exp(h)) - _log_z(x, y, em, ex, ey, go, ge * math.exp(-h))\n"
        "    return round(up / (2 * h), 6), round(ue / (2 * h), 6)\n"
        "def _pin(factors):\n"
        "    g = np.asarray(factors, dtype=float)\n"
        "    if g.shape != (2,) or np.any(g <= 0):\n"
        "        return -1.0\n"
        "    return float(1.0 + 1.0e-4 * (np.log(g[0]) + 2.0 * np.log(g[1])))\n"
    )
    args = "x.copy(), y.copy(), em.copy(), ex.copy(), ey.copy(), n_open, n_extend"
    instances = [
        ("x = np.array([0, 2, 1, 3, 3, 0, 1, 2, 2]); y = np.array([0, 2, 3, 0, 1, 2, 3])\n"
         "em, ex, ey = M, EX, EY\nn_open, n_extend = _targets(x, y, em, ex, ey, 0.04, 0.4)\n"),
        ("x = np.array([2, 2, 1, 0, 3, 1]); y = np.array([2, 1, 1, 0, 3, 3, 0, 1, 1, 2, 3])\n"
         "em, ex, ey = M, EX, EY\n"
         "n_open, n_extend = _targets(x, y, em, ex, ey, 0.07, 0.3)\n"),
        ("x = np.array([1, 3, 3, 0]); y = np.array([1, 3, 0])\n"
         "em, ex, ey = M, EX, EY\nn_open, n_extend = _targets(x, y, em, ex, ey, 0.5, 0.2)\n"),
        ("x = np.array([3, 0, 0, 1, 2, 2, 1, 0, 3, 3]); y = np.array([3, 0, 2, 2])\n"
         "em, ex, ey = M, np.array([0.05, 0.9, 0.4, 0.02]), np.array([0.7, 0.1, 0.3, 0.8])\n"
         "n_open, n_extend = _targets(x, y, em, ex, ey, 0.2, 0.9)\n"),
    ]
    cases = [
        {  # Normal identifiable calibration.
            "setup": model + instances[0],
            "call": f"_pin(calibrate_gap_factors({args}))",
            "gold_call": f"_pin(_oracle_calibrate_gap_factors({args}))",
        },
        {  # Longer, strongly length-imbalanced pair.
            "setup": model + instances[1],
            "call": f"_pin(calibrate_gap_factors({args}))",
            "gold_call": f"_pin(_oracle_calibrate_gap_factors({args}))",
        },
        {  # Boundary: short pair with high opening factor.
            "setup": model + instances[2],
            "call": f"_pin(calibrate_gap_factors({args}))",
            "gold_call": f"_pin(_oracle_calibrate_gap_factors({args}))",
        },
        {  # Edge: asymmetric residue-dependent gap emissions.
            "setup": model + instances[3],
            "call": f"_pin(calibrate_gap_factors({args}))",
            "gold_call": f"_pin(_oracle_calibrate_gap_factors({args}))",
        },
    ]
    for bad in ("0.0, 1.5", "2.0, float('nan')"):
        cases.append({
            "setup": model + "x = np.array([0, 2, 1]); y = np.array([0, 1])\nem, ex, ey = M, EX, EY\n" + (
                "def _candidate():\n"
                "    try:\n"
                f"        calibrate_gap_factors(x.copy(), y.copy(), em.copy(), ex.copy(), ey.copy(), {bad})\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "def _reference():\n"
                "    try:\n"
                f"        _oracle_calibrate_gap_factors(x.copy(), y.copy(), em.copy(), ex.copy(), ey.copy(), {bad})\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"),
            "call": "_candidate()",
            "gold_call": "_reference()",
        })
    return cases
