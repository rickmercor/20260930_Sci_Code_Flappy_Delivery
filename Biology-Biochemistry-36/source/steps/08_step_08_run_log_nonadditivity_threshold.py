"""
Compose class-resolved fixed-sequence, grouped profile-polynomial and nonlinear-series calculations into a mutation-rate design threshold.

The final scalar marks where substitution orders omitted from a chosen Taylor truncation account for a prescribed fraction of the exact log partition-function change.

Returns
-------
float: mutation-rate threshold for the relative log-series remainder.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def run_log_nonadditivity_threshold(
    sequence: "np.ndarray | None" = None,
    mutation_kernel: "np.ndarray | None" = None,
    rule_weights: "np.ndarray | None" = None,
    left_emission: "np.ndarray | None" = None,
    right_emission: "np.ndarray | None" = None,
    pair_emission: "np.ndarray | None" = None,
    stacking_factors: "np.ndarray | None" = None,
    min_loop: "int | None" = None,
    log_order: "int | None" = None,
    target_share: "float | None" = None,
    lower_rate: "float | None" = None,
    upper_rate: "float | None" = None,
    rate_tolerance: float = 1e-12,
) -> float:
    """Return the mutation rate at the requested log-remainder threshold.

    Group valid input positions by their reference nucleotide code, with
    contiguous group labels following the sorted observed codes, and compose
    in order,
    ``build_mutation_profile_polynomials``,
    ``compute_stacked_inside_weights``,
    ``compute_profile_partition_polynomial``,
    ``normalize_order_coefficients``,
    ``compute_log_series_coefficients`` and
    ``locate_log_remainder_threshold``. The grouped polynomial is restricted
    to the common-rate diagonal during normalization, so this internal lift
    leaves the requested one-rate library unchanged. Their validation rules
    apply.

    Any argument left as ``None`` takes its benchmark value: sequence
    ``GGACAUCGAGUCCUAG``; a uniform conditional replacement among the other
    three nucleotides; rule weights ``(0.32, 0.24, 0.18, 0.14, 0.70)``;
    left emissions ``(0.33, 0.21, 0.26, 0.20)``; right emissions
    ``(0.19, 0.30, 0.22, 0.29)``; pair emissions A-U 1.15, U-A 0.85,
    G-C 1.75, C-G 1.40, G-U 0.50 and U-G 0.65 with all others zero;
    stacking factors ``(2.2, 1.5, 1.3, 0.8)``; ``min_loop = 3``;
    ``log_order = 4``; ``target_share = 0.30``; and the unique-root bracket
    ``[0.045, 0.070]``.

    Parameters
    ----------
    sequence, mutation_kernel : np.ndarray or None
        Reference nucleotide codes and the conditional-replacement kernel.
    rule_weights, left_emission, right_emission : np.ndarray or None
        Grammar-rule and single-emission weights.
    pair_emission, stacking_factors : np.ndarray or None
        Ordered pair emissions and the four ordered canonical stack factors.
    min_loop, log_order : int or None
        Minimum paired-loop length and retained logarithmic series order.
    target_share, lower_rate, upper_rate : float or None
        Relative omitted-order target and its unique-crossing bracket.
    rate_tolerance : float
        Finite positive absolute bracket-width tolerance.

    Returns
    -------
    float
        Mutation rate at the unique crossing in the supplied bracket.

    Raises
    ------
    ValueError
        If an explicit or default input violates an upstream contract, if the
        logarithmic-order or solver controls are invalid, or if the requested
        target is not bracketed by ``lower_rate`` and ``upper_rate``.
    """
    return mutation_rate

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_run_log_nonadditivity_threshold(
    sequence: "np.ndarray | None" = None,
    mutation_kernel: "np.ndarray | None" = None,
    rule_weights: "np.ndarray | None" = None,
    left_emission: "np.ndarray | None" = None,
    right_emission: "np.ndarray | None" = None,
    pair_emission: "np.ndarray | None" = None,
    stacking_factors: "np.ndarray | None" = None,
    min_loop: "int | None" = None,
    log_order: "int | None" = None,
    target_share: "float | None" = None,
    lower_rate: "float | None" = None,
    upper_rate: "float | None" = None,
    rate_tolerance: float = 1e-12,
) -> float:
    """Reference orchestrator using every preceding oracle stage."""
    import numpy as np

    if sequence is None:
        sequence = np.array(["ACGU".index(ch) for ch in "GGACAUCGAGUCCUAG"])
    if mutation_kernel is None:
        mutation_kernel = np.ones((4, 4), dtype=float) / 3.0
        np.fill_diagonal(mutation_kernel, 0.0)
    if rule_weights is None:
        rule_weights = np.array([0.32, 0.24, 0.18, 0.14, 0.70])
    if left_emission is None:
        left_emission = np.array([0.33, 0.21, 0.26, 0.20])
    if right_emission is None:
        right_emission = np.array([0.19, 0.30, 0.22, 0.29])
    if pair_emission is None:
        pair_emission = np.zeros((4, 4), dtype=float)
        for a, b, value in (
            (0, 3, 1.15), (3, 0, 0.85), (2, 1, 1.75),
            (1, 2, 1.40), (2, 3, 0.50), (3, 2, 0.65),
        ):
            pair_emission[a, b] = value
    if stacking_factors is None:
        stacking_factors = np.array([2.2, 1.5, 1.3, 0.8])
    if min_loop is None:
        min_loop = 3
    if log_order is None:
        log_order = 4
    if target_share is None:
        target_share = 0.30
    if lower_rate is None:
        lower_rate = 0.045
    if upper_rate is None:
        upper_rate = 0.070

    sequence = _grouped_validated_codes(sequence)
    _, position_groups = np.unique(sequence, return_inverse=True)
    position_groups = position_groups.astype(int)
    profile = _oracle_build_mutation_profile_polynomials(
        sequence, mutation_kernel, position_groups,
    )
    fixed = _oracle_compute_stacked_inside_weights(
        sequence, rule_weights, left_emission, right_emission,
        pair_emission, stacking_factors, min_loop,
    )
    polynomial = _oracle_compute_profile_partition_polynomial(
        profile, position_groups, rule_weights, left_emission, right_emission,
        pair_emission, stacking_factors, min_loop,
    )
    normalized = _oracle_normalize_order_coefficients(fixed, polynomial)
    logarithm = _oracle_compute_log_series_coefficients(normalized, log_order)
    return _oracle_locate_log_remainder_threshold(
        normalized, logarithm, target_share, lower_rate, upper_rate,
        rate_tolerance, 200,
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only whole-chain normal, boundary and validation cases."""
    base = (
        "import numpy as np\n"
        "W = np.array([0.41, 0.27, 0.16, 0.12, 0.80])\n"
        "EL = np.array([0.28, 0.24, 0.19, 0.33])\n"
        "ER = np.array([0.22, 0.31, 0.26, 0.18])\n"
        "EP = np.zeros((4, 4))\n"
        "for a, b, v in [(0, 3, .90), (3, 0, 1.30), (2, 1, 1.60), (1, 2, 1.10), (2, 3, .45), (3, 2, .70)]:\n"
        "    EP[a, b] = v\n"
        "ST = np.array([1.9, 1.4, 1.2, .7])\n"
        "K = np.ones((4, 4)) / 3.0\n"
        "np.fill_diagonal(K, 0.0)\n"
        "def _code(text):\n"
        "    return np.array(['ACGU'.index(ch) for ch in text])\n"
    )
    status = (
        "def _status(fn):\n"
        "    try:\n"
        "        fn()\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
    )
    return [
        {
            "setup": "import numpy as np\n",
            "call": "float(run_log_nonadditivity_threshold())",
            "gold_call": "float(_oracle_run_log_nonadditivity_threshold())",
            "tol": 2e-12,
        },
        {
            "setup": "import numpy as np\n",
            "call": "float(run_log_nonadditivity_threshold(log_order=4, target_share=.2, lower_rate=.04, upper_rate=.055))",
            "gold_call": "float(_oracle_run_log_nonadditivity_threshold(log_order=4, target_share=.2, lower_rate=.04, upper_rate=.055))",
            "tol": 2e-12,
        },
        {
            "setup": base + "S = _code('UGAACGUCAG')\n",
            "call": "float(run_log_nonadditivity_threshold(S.copy(), K.copy(), W.copy(), EL.copy(), ER.copy(), EP.copy(), ST.copy(), 0, 4, .2, .13, .16))",
            "gold_call": "float(_oracle_run_log_nonadditivity_threshold(S.copy(), K.copy(), W.copy(), EL.copy(), ER.copy(), EP.copy(), ST.copy(), 0, 4, .2, .13, .16))",
            "tol": 2e-12,
        },
        {
            "setup": base + "S = _code('CAGUUCAGGU')\n",
            "call": "float(run_log_nonadditivity_threshold(S.copy(), K.copy(), W.copy(), EL.copy(), ER.copy(), EP.copy(), ST.copy(), 2, 3, .3, .03, .05))",
            "gold_call": "float(_oracle_run_log_nonadditivity_threshold(S.copy(), K.copy(), W.copy(), EL.copy(), ER.copy(), EP.copy(), ST.copy(), 2, 3, .3, .03, .05))",
            "tol": 2e-12,
        },
        {
            "setup": base + status + "S = _code('UGAACGUCAG')\n",
            "call": "_status(lambda: run_log_nonadditivity_threshold(S, K, W, EL, ER, EP, ST, 0, 4, .2, .01, .02))",
            "gold_call": "_status(lambda: _oracle_run_log_nonadditivity_threshold(S, K, W, EL, ER, EP, ST, 0, 4, .2, .01, .02))",
        },
    ]
