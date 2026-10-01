"""
Compose every earlier step to obtain the mean log-scale epistasis over all neighbouring double substitutions of a synthetic reference sequence aligned with its homolog under gap factors fitted to observed transition counts.
Orchestrator: yes - builds the sequence pair (build_homolog_pair), fits the gap factors (calibrate_gap_factors), computes the reference forward and backward tables (compute_log_forward_tables, compute_log_backward_tables), confirms the fitted expected counts (compute_expected_gap_transition_counts), scores every single substitution (evaluate_single_substitution_log_weights) and every neighbouring double substitution (compute_neighbour_double_log_weights), and summarizes the epistasis (summarize_neighbour_epistasis), consuming each output rather than reimplementing any step.

How far neighbouring substitution effects depart from multiplicativity measures how strongly one alignment ensemble couples adjacent residues, a question that arises whenever homology-based scores are used to anticipate the combined effect of several sequence edits.

Returns
-------
float: mean natural-log epistasis over all 9 (length - 1) neighbouring double substitutions.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def estimate_mean_neighbour_epistasis(
    seed: int = 20260914,
    length: int = 2000,
    delete_probability: float = 0.06,
    substitute_probability: float = 0.16,
    match_factors: "tuple" = ((0.16, 0.03, 0.05, 0.03), (0.03, 0.16, 0.03, 0.05),
                              (0.05, 0.03, 0.16, 0.03), (0.03, 0.05, 0.03, 0.16)),
    x_gap_factors: "tuple" = (0.12, 0.38, 0.38, 0.12),
    y_gap_factors: "tuple" = (0.16, 0.34, 0.34, 0.16),
    open_count: float = 115.0,
    extend_count: float = 32.0,
) -> float:
    """Return the mean log-scale epistasis over all neighbouring double substitutions.

    Build ``(x, y)`` with ``build_homolog_pair(seed, length,
    delete_probability, substitute_probability)``. Use the pair model of
    ``compute_log_forward_tables`` with the given emission factors and with
    the gap factors returned by ``calibrate_gap_factors`` for ``open_count``
    and ``extend_count`` on this pair. Let ``Z(s)`` be the summed weight of
    all complete alignments of ``y`` with a sequence ``s``. For every pair of
    neighbouring positions of ``x`` and every choice of new codes at both,
    each different from the base it replaces, the epistasis is ``ln
    Z(double) + ln Z(x) - ln Z(first single) - ln Z(second single)`` as in
    ``summarize_neighbour_epistasis``. Return the mean over all ``9 (length -
    1)`` such double substitutions. The defaults reproduce the problem
    statement.

    Parameters
    ----------
    seed, length, delete_probability, substitute_probability :
        Arguments of ``build_homolog_pair``; ``length`` must be at least 2.
    match_factors : tuple
        Nested ``(4, 4)`` finite positive factors, row = base of ``x``.
    x_gap_factors, y_gap_factors : tuple
        Four finite positive gap emission factors each.
    open_count, extend_count : float
        Target expected numbers of gap-opening and gap-extension columns.

    Returns
    -------
    float
        Mean epistasis as a native Python float.

    Raises
    ------
    ValueError
        If any argument is invalid for the step that consumes it, including
        ``length < 2`` and target counts that no positive gap factors
        reproduce.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_estimate_mean_neighbour_epistasis(
    seed: int = 20260914,
    length: int = 2000,
    delete_probability: float = 0.06,
    substitute_probability: float = 0.16,
    match_factors: "tuple" = ((0.16, 0.03, 0.05, 0.03), (0.03, 0.16, 0.03, 0.05),
                              (0.05, 0.03, 0.16, 0.03), (0.03, 0.05, 0.03, 0.16)),
    x_gap_factors: "tuple" = (0.12, 0.38, 0.38, 0.12),
    y_gap_factors: "tuple" = (0.16, 0.34, 0.34, 0.16),
    open_count: float = 115.0,
    extend_count: float = 32.0,
) -> float:
    """Reference orchestrator using only the reference function chain."""
    import numpy as np

    x, y = _oracle_build_homolog_pair(seed, length, delete_probability, substitute_probability)
    if x.size < 2:
        raise ValueError("length must be at least 2")
    emissions = (np.asarray(match_factors, dtype=float), np.asarray(x_gap_factors, dtype=float),
                 np.asarray(y_gap_factors, dtype=float))
    gap_open, gap_extend = _oracle_calibrate_gap_factors(x, y, *emissions, open_count, extend_count)
    model = (*emissions, float(gap_open), float(gap_extend))
    log_forward = _oracle_compute_log_forward_tables(x, y, *model)
    log_backward = _oracle_compute_log_backward_tables(x, y, *model)
    counts = _oracle_compute_expected_gap_transition_counts(x, y, *model, log_forward, log_backward)
    if np.max(np.abs(counts / np.array([open_count, extend_count], dtype=float) - 1.0)) > 1.0e-8:
        raise ValueError("the fitted gap factors do not reproduce the target counts")
    single = _oracle_evaluate_single_substitution_log_weights(
        x, y, emissions[0], emissions[1], log_forward, log_backward)
    double = _oracle_compute_neighbour_double_log_weights(x, y, *model, log_forward, log_backward)
    summary = _oracle_summarize_neighbour_epistasis(x, single, double)
    return float(summary[0])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    common = (
        "import math\n"
        "import numpy as np\n"
        "M = ((0.16, 0.03, 0.05, 0.03), (0.03, 0.16, 0.03, 0.05), (0.05, 0.03, 0.16, 0.03), (0.03, 0.05, 0.03, 0.16))\n"
        "EX = (0.12, 0.38, 0.38, 0.12)\n"
        "EY = (0.16, 0.34, 0.34, 0.16)\n"
        "def _pair(seed, length, dp, sp):\n"
        "    rng = np.random.default_rng(seed)\n"
        "    x = rng.integers(0, 4, length); u = rng.random(length); v = rng.integers(0, 4, length)\n"
        "    return x, np.where(u < dp + sp, v, x)[u >= dp]\n"
        "def _lae(*v):\n"
        "    top = max(v)\n"
        "    return top if top == -math.inf else top + math.log(sum(math.exp(t - top) for t in v))\n"
        "def _log_z(x, y, em, ex, ey, go, ge):\n"
        "    em, ex, ey = np.asarray(em), np.asarray(ex), np.asarray(ey)\n"
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
        "def _targets(seed, length, dp, sp, em, ex, ey, go, ge, h=1.0e-4):\n"
        "    x, y = _pair(seed, length, dp, sp)\n"
        "    up = _log_z(x, y, em, ex, ey, go * math.exp(h), ge) - _log_z(x, y, em, ex, ey, go * math.exp(-h), ge)\n"
        "    ue = _log_z(x, y, em, ex, ey, go, ge * math.exp(h)) - _log_z(x, y, em, ex, ey, go, ge * math.exp(-h))\n"
        "    return round(up / (2 * h), 5), round(ue / (2 * h), 5)\n"
    )
    other = ("((0.3, 0.02, 0.08, 0.02), (0.02, 0.3, 0.02, 0.08), (0.08, 0.02, 0.3, 0.02), "
             "(0.02, 0.08, 0.02, 0.3)), (0.4, 0.1, 0.1, 0.4), (0.25, 0.25, 0.25, 0.25)")
    value = [
        ("n_open, n_extend = _targets(5, 24, 0.06, 0.16, M, EX, EY, 0.04, 0.4)\n",
         "5, 24, 0.06, 0.16, M, EX, EY, n_open, n_extend"),
        (f"n_open, n_extend = _targets(11, 30, 0.1, 0.3, {other}, 0.08, 0.6)\n",
         f"11, 30, 0.1, 0.3, {other}, n_open, n_extend"),
        ("n_open, n_extend = _targets(2, 3, 0.0, 0.0, M, EX, EY, 0.3, 0.5)\n",
         "2, 3, 0.0, 0.0, M, EX, EY, n_open, n_extend"),
        ("n_open, n_extend = _targets(20260914, 40, 0.06, 0.16, M, EX, EY, 0.05, 0.3)\n",
         "20260914, 40, 0.06, 0.16, M, EX, EY, n_open, n_extend"),
    ]
    cases = [
        {  # Normal end-to-end pipeline.
            "setup": common + value[0][0],
            "call": f"estimate_mean_neighbour_epistasis({value[0][1]})",
            "gold_call": f"_oracle_estimate_mean_neighbour_epistasis({value[0][1]})",
        },
        {  # Alternate emission model and event probabilities.
            "setup": common + value[1][0],
            "call": f"estimate_mean_neighbour_epistasis({value[1][1]})",
            "gold_call": f"_oracle_estimate_mean_neighbour_epistasis({value[1][1]})",
        },
        {  # Boundary: short unchanged homologous pair.
            "setup": common + value[2][0],
            "call": f"estimate_mean_neighbour_epistasis({value[2][1]})",
            "gold_call": f"_oracle_estimate_mean_neighbour_epistasis({value[2][1]})",
        },
        {  # Edge: longer pipeline with a different calibration regime.
            "setup": common + value[3][0],
            "call": f"estimate_mean_neighbour_epistasis({value[3][1]})",
            "gold_call": f"_oracle_estimate_mean_neighbour_epistasis({value[3][1]})",
        },
    ]
    for args in ("3, 1, 0.0, 0.0, M, EX, EY, 1.0, 1.0", "3, 12, 0.0, 0.1, M, EX, EY, -2.0, 1.0"):
        cases.append({
            "setup": common + (
                "def _candidate():\n"
                "    try:\n"
                f"        estimate_mean_neighbour_epistasis({args})\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "def _reference():\n"
                "    try:\n"
                f"        _oracle_estimate_mean_neighbour_epistasis({args})\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"),
            "call": "_candidate()",
            "gold_call": "_reference()",
        })
    return cases
