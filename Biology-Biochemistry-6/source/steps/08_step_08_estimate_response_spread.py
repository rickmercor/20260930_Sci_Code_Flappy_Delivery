"""
Compose every earlier step to compute the growth-optimal states of the pathway across a nutrient-limited growth series, verify the stationary family's local response, and return the mass-weighted spread of the enzymes' zero-growth response factors.

How differently the enzymes of one pathway are induced or repressed as growth slows is a direct readout of how nutrient limitation propagates through a growth-optimal pathway.

Returns
-------
float: the mass-weighted rms spread of the enzymes' zero-growth response factors.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def estimate_response_spread(
    kcat: tuple = (23.0, 9.3, 36.0, 5.9, 16.8, 12.6, 44.0, 7.3, 28.5, 10.7, 19.5, 8.1),
    km: tuple = (2.4e-4, 6.1e-4, 1.3e-4, 9.0e-4, 3.3e-4, 1.9e-3,
                 7.5e-5, 4.4e-4, 2.8e-4, 1.2e-3, 5.2e-4, 8.3e-4),
    kcat_t: float = 22.5,
    km_t: float = 1.2e-3,
    nutrient_hi: float = 0.6,
    max_fold: float = 3.0,
    growth_fractions: tuple = (1.0, 0.85, 0.7, 0.55, 0.4),
    regulated: bool = True,
) -> float:
    """Return the mass-weighted spread of the zero-growth response factors.

    The pathway, the transporter and the balanced-growth model are those of
    ``compute_enzyme_demand`` and ``close_optimal_transporter``. The rich
    reference state is the state that maximises the growth rate over all
    ``n + 1`` amounts at external nutrient level ``nutrient_hi``; call its
    growth rate ``lam_hi`` and its transporter amount ``phi_t_hi``. The
    regulated transporter follows the law of
    ``close_regulated_transporter`` with ``max_fold`` and with the
    ``basal_fraction`` for which that law gives ``phi_t_hi`` at
    ``nutrient_hi``. For each ``f`` in ``growth_fractions``, the state of the
    growth series is the one whose ``n`` enzyme amounts maximise the growth
    rate at the prevailing nutrient level and whose growth rate is
    ``f * lam_hi``, with the transporter regulated when ``regulated`` is
    true and also growth-maximising when it is false. Return the spread
    rich stationary profile and its analytic tangent are first recovered
    through ``propagate_optimal_levels`` as a consistency check on the family
    used by the closures. Return the spread returned by
    ``summarize_growth_response`` for the enzyme amounts of
    these states, their growth rates and ``growth_hi = lam_hi``. Metabolite
    levels are located with ``solve_family_member`` and its default bracket
    and tolerance. The defaults reproduce the problem statement.

    Parameters
    ----------
    kcat, km : tuple
        Positive enzyme constants in pathway order, equal lengths ``n >= 1``.
    kcat_t, km_t : float
        Positive transporter constants.
    nutrient_hi : float
        Positive external nutrient level of the rich reference state.
    max_fold : float
        Largest fold increase of the regulated transporter, at least 1.
    growth_fractions : tuple
        Values of ``lam / lam_hi`` of the series, each in ``(0, 1]``, with at
        least two distinct values.
    regulated : bool
        Whether the transporter is regulated in the growth series.

    Returns
    -------
    float
        The mass-weighted root-mean-square spread of the response factors.

    Raises
    ------
    ValueError
        If any argument is outside its stated domain or any stage rejects
        its input, including when a requested growth rate is not reached.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_estimate_response_spread(
    kcat: tuple = (23.0, 9.3, 36.0, 5.9, 16.8, 12.6, 44.0, 7.3, 28.5, 10.7, 19.5, 8.1),
    km: tuple = (2.4e-4, 6.1e-4, 1.3e-4, 9.0e-4, 3.3e-4, 1.9e-3,
                 7.5e-5, 4.4e-4, 2.8e-4, 1.2e-3, 5.2e-4, 8.3e-4),
    kcat_t: float = 22.5,
    km_t: float = 1.2e-3,
    nutrient_hi: float = 0.6,
    max_fold: float = 3.0,
    growth_fractions: tuple = (1.0, 0.85, 0.7, 0.55, 0.4),
    regulated: bool = True,
) -> float:
    """Reference orchestrator using only the reference function chain."""
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    def _is_flag(value):
        return isinstance(value, (bool, np.bool_))

    if not _is_flag(regulated):
        raise ValueError("regulated must be a boolean")
    for value in (kcat_t, km_t, nutrient_hi):
        if not (_is_number(value) and value > 0.0):
            raise ValueError("kcat_t, km_t and nutrient_hi must be finite positive numbers")
    if not (_is_number(max_fold) and max_fold >= 1.0):
        raise ValueError("max_fold must be a finite number of at least 1")
    try:
        fractions = np.array(growth_fractions, dtype=float)
        rate = np.array(kcat, dtype=float)
        affinity = np.array(km, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("kcat, km and growth_fractions must be numeric sequences") from None
    if (rate.ndim != 1 or rate.size == 0 or rate.shape != affinity.shape
            or not np.all(np.isfinite(rate) & (rate > 0.0))
            or not np.all(np.isfinite(affinity) & (affinity > 0.0))):
        raise ValueError("kcat and km must be equal-length sequences of finite positive numbers")
    if (fractions.ndim != 1 or not np.all(np.isfinite(fractions))
            or not np.all((fractions > 0.0) & (fractions <= 1.0))
            or np.unique(fractions).size < 2):
        raise ValueError("growth_fractions must hold at least two distinct values in (0, 1]")

    def _demand_fn(levels, kc, k):
        return _oracle_compute_enzyme_demand(levels, kc, k)

    def _cost_fn(levels, kc, k):
        return _oracle_compute_metabolite_costs(levels, kc, k)

    def _propagate_fn(last_level, kc, k):
        return _oracle_propagate_optimal_levels(last_level, kc, k)

    def _optimal_fn(levels):
        return _oracle_close_optimal_transporter(levels, rate, affinity, kcat_t, km_t,
                                                 _demand_fn, _cost_fn)

    rich = _oracle_solve_family_member(float(nutrient_hi), 0, rate, affinity,
                                       _propagate_fn, _optimal_fn)
    rich_branch = _oracle_propagate_optimal_levels(
        float(rich[-1]), rate, affinity, "last", True
    )
    if (rich_branch.shape != (2, rate.size)
            or not np.allclose(rich_branch[0], rich, rtol=1.0e-12, atol=0.0)
            or not np.all(np.isfinite(rich_branch[1]))):
        raise ValueError("rich state is inconsistent with the stationary family")
    _, transporter_hi, growth_hi = _optimal_fn(rich)
    basal = transporter_hi / (1.0 + (max_fold - 1.0) / (1.0 + max_fold * nutrient_hi / km_t))

    def _regulated_fn(levels):
        return _oracle_close_regulated_transporter(levels, rate, affinity, kcat_t, km_t,
                                                   basal, max_fold, _demand_fn)

    close_fn = _regulated_fn if regulated else _optimal_fn
    growth, amounts = [], []
    for fraction in fractions:
        levels = _oracle_solve_family_member(float(fraction * growth_hi), 2, rate, affinity,
                                             _propagate_fn, close_fn)
        lam = float(close_fn(levels)[2])
        growth.append(lam)
        amounts.append(lam * _oracle_compute_enzyme_demand(levels, rate, affinity))
    summary = _oracle_summarize_growth_response(np.array(growth), np.array(amounts), growth_hi)
    return float(summary[-1])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only tests comparing the end-to-end chain with the reference."""
    status = (
        "import numpy as np\n"
        "def _status(fn):\n"
        "    try:\n"
        "        fn()\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
    )
    short = (
        "import numpy as np\n"
        "KC = (14.0, 26.0, 8.0, 33.0, 11.0)\n"
        "KM = (3.0e-4, 1.6e-3, 1.2e-4, 9.0e-4, 5.0e-4)\n"
    )
    return [
        {
            "setup": "import numpy as np\n",
            "call": "estimate_response_spread()",
            "gold_call": "_oracle_estimate_response_spread()",
        },
        {
            "setup": "import numpy as np\n",
            "call": "10.0 * estimate_response_spread(regulated=False)",
            "gold_call": "10.0 * _oracle_estimate_response_spread(regulated=False)",
        },
        {
            "setup": short,
            "call": "estimate_response_spread(KC, KM, 40.0, 1.0e-3, 0.5, 2.5, (1.0, 0.8, 0.6))",
            "gold_call": "_oracle_estimate_response_spread(KC, KM, 40.0, 1.0e-3, 0.5, 2.5, (1.0, 0.8, 0.6))",
        },
        {
            "setup": short,
            "call": "estimate_response_spread(KC, KM, 40.0, 1.0e-3, 0.5, 5.0, (0.9, 0.7, 0.5, 0.35))",
            "gold_call": "_oracle_estimate_response_spread(KC, KM, 40.0, 1.0e-3, 0.5, 5.0, (0.9, 0.7, 0.5, 0.35))",
        },
        {
            "setup": status,
            "call": "_status(lambda: estimate_response_spread(growth_fractions=(1.0, 1.2)))",
            "gold_call": "_status(lambda: _oracle_estimate_response_spread(growth_fractions=(1.0, 1.2)))",
        },
        {
            "setup": status,
            "call": "_status(lambda: estimate_response_spread(km=(1.0e-4, 2.0e-4)))",
            "gold_call": "_status(lambda: _oracle_estimate_response_spread(km=(1.0e-4, 2.0e-4)))",
        },
    ]
