"""
Compose every earlier step to obtain the ratio of the steady necking-propagation stress to the necking-bifurcation stress of the ordered stripe.

Necking starts at the load maximum of the homogeneous stress-stretch curve and then propagates at a lower steady stress, so their ratio measures how much load the tissue sheds once the neck forms.

Returns
-------
float: the steady necking-propagation stress divided by the necking-bifurcation stress.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def estimate_propagation_ratio(
    kappa: float = 0.35,
    chi: float = 2.45,
    threshold: float = 0.08,
    tolerance: float = 1e-13,
) -> float:
    """Return the ratio of the propagation stress to the bifurcation stress.

    The reference honeycomb uses the edge-length scale of
    ``solve_stress_free_scale``. Its load-perpendicular edges rearrange at the
    stretch returned by ``locate_rearrangement_stretch`` for ``threshold``;
    the bifurcation stress is ``compute_nominal_stress`` at that stretch; the
    propagation stress is the first entry of ``solve_propagation_state`` for
    that stretch and ``tolerance``. Return the propagation stress divided by
    the bifurcation stress. The defaults reproduce the problem statement.

    Parameters
    ----------
    kappa : float
        Positive perimeter rigidity ratio.
    chi : float
        Positive preferred perimeter.
    threshold : float
        Positive edge length at which load-perpendicular edges rearrange.
    tolerance : float
        Positive absolute accuracy of the propagation stress, at most 1e-6.

    Returns
    -------
    float
        The propagation-to-bifurcation stress ratio.

    Raises
    ------
    ValueError
        If any argument is not a finite positive real number (booleans are
        rejected), if the nominal stress is not at a load maximum at the
        rearrangement stretch (still rising at ``(1 - 1e-6)`` times that
        stretch and lower immediately above it), or if any earlier step
        rejects its input.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_estimate_propagation_ratio(
    kappa: float = 0.35,
    chi: float = 2.45,
    threshold: float = 0.08,
    tolerance: float = 1e-13,
) -> float:
    """Reference orchestrator using only the reference function chain."""
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    arguments = (("kappa", kappa), ("chi", chi), ("threshold", threshold),
                 ("tolerance", tolerance))
    for name, value in arguments:
        if not (_is_number(value) and value > 0.0):
            raise ValueError(f"{name} must be a finite positive number")
    scale = _oracle_solve_stress_free_scale(kappa, chi)
    transition = _oracle_locate_rearrangement_stretch(threshold, scale, kappa, chi)
    bifurcation = _oracle_compute_nominal_stress(transition, transition, scale, kappa, chi)
    # Necking bifurcates at the load maximum of the homogeneous response: the
    # stress must still rise into the rearrangement and drop just past it.
    below = _oracle_compute_nominal_stress(transition * (1.0 - 1e-6), transition, scale, kappa, chi)
    above = _oracle_compute_nominal_stress(float(np.nextafter(transition, np.inf)),
                                           transition, scale, kappa, chi)
    if not (below < bifurcation and above < bifurcation):
        raise ValueError("the rearrangement stretch is not a load maximum")
    state = _oracle_solve_propagation_state(transition, scale, kappa, chi, tolerance)
    return float(state[0]) / float(bifurcation)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only tests over distinct tissue parameters."""
    status = (
        "import numpy as np\n"
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
            "call": "estimate_propagation_ratio()",
            "gold_call": "_oracle_estimate_propagation_ratio()",
        },
        {
            "setup": "import numpy as np\n",
            "call": "estimate_propagation_ratio(0.16, 3.7, 0.1, 1e-13)",
            "gold_call": "_oracle_estimate_propagation_ratio(0.16, 3.7, 0.1, 1e-13)",
        },
        {
            "setup": "import numpy as np\n",
            "call": "estimate_propagation_ratio(0.9, 1.6, 0.05, 1e-13)",
            "gold_call": "_oracle_estimate_propagation_ratio(0.9, 1.6, 0.05, 1e-13)",
        },
        {
            "setup": status,
            "call": "_status(lambda: estimate_propagation_ratio(0.35, 2.45, 0.9, 1e-13))",
            "gold_call": "_status(lambda: _oracle_estimate_propagation_ratio(0.35, 2.45, 0.9, 1e-13))",
        },
        {
            "setup": status,
            "call": "_status(lambda: estimate_propagation_ratio(-0.35, 2.45, 0.08, 1e-13))",
            "gold_call": "_status(lambda: _oracle_estimate_propagation_ratio(-0.35, 2.45, 0.08, 1e-13))",
        },
    ]
