"""
Compose every earlier step to obtain the smallest polarity-regulation constant at which the few-cell analyses allow monolayer inflation.

The monolayer threshold bounds the polarity magnitudes that give single-layered sheets, and the wraparound-inflation boundary rises with polarity magnitude, so the boundary evaluated at the threshold is the lowest regulation constant at which a monolayer can form its lumen by inflation.

Returns
-------
float: the wraparound-inflation boundary evaluated at the monolayer threshold.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
def estimate_inflation_onset(
    rest_distance: float = 2.0,
    cutoff: float = 2.5,
    tau_v: float = 10.0,
    tolerance: float = 1e-12,
) -> float:
    """Return the wraparound-inflation boundary evaluated at the monolayer threshold.

    The kernel range is fixed so that ``U(r) = exp(-r) - exp(-r / beta)`` is
    minimal at ``rest_distance``. The monolayer threshold ``p_c`` is the
    upper end of the polarity-magnitude range, searched in ``(0, 1)``, over
    which three cells with frozen parallel polarities have a stable
    non-collinear equilibrium under the overdamped adhesion velocities with
    interaction range ``cutoff`` and mobility ``tau_v``. The
    wraparound-inflation boundary at magnitude ``p`` is the largest
    regulation constant, searched in ``(1e-6, tau_v)``, for which a
    three-cell arc of two relaxed mirror-symmetric pairs sharing their
    middle cell keeps its end cells out of interaction range, the relaxed
    pairs following the polarity rotation rates with the same parameters.
    Return that boundary at ``p = p_c``; ``tolerance`` is the absolute
    accuracy of both searches. The defaults reproduce the problem
    statement.

    Parameters
    ----------
    rest_distance : float
        Distance at which the kernel is minimal; must exceed 1.
    cutoff : float
        Finite interaction range, larger than ``rest_distance``.
    tau_v : float
        Positive mobility constant.
    tolerance : float
        Positive absolute accuracy of the two searches.

    Returns
    -------
    float
        The regulation constant at which the boundary meets the threshold.

    Raises
    ------
    ValueError
        If any argument is outside the stated domain or any stage rejects
        its input, including when no threshold or boundary lies inside the
        search ranges.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_estimate_inflation_onset(
    rest_distance: float = 2.0,
    cutoff: float = 2.5,
    tau_v: float = 10.0,
    tolerance: float = 1e-12,
) -> float:
    """Reference orchestrator using only the reference function chain."""
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    if not (_is_number(tau_v) and tau_v > 0.0):
        raise ValueError("tau_v must be a finite positive number")
    if not (_is_number(tolerance) and tolerance > 0.0):
        raise ValueError("tolerance must be a finite positive number")
    beta = _oracle_solve_kernel_range(rest_distance)

    def _velocity_fn(positions, angles, magnitudes, kernel_range, reach, mobility):
        return _oracle_compute_adhesion_velocities(
            positions, angles, magnitudes, kernel_range, reach, mobility
        )

    def _rotation_fn(positions, angles, magnitudes, regulation, kernel_range, reach, mobility):
        return _oracle_compute_polarity_rotation_rates(
            positions, angles, magnitudes, regulation, kernel_range, reach, mobility
        )

    def _triad_fn(magnitude):
        return _oracle_solve_stacked_triad(magnitude, beta, cutoff, tau_v, _velocity_fn)

    def _splay_fn(magnitude, regulation):
        return _oracle_compute_pair_splay_angle(
            magnitude, regulation, beta, cutoff, tau_v, _rotation_fn
        )

    threshold = _oracle_locate_monolayer_threshold(_triad_fn, (0.0, 1.0), tolerance)
    onset = _oracle_solve_wraparound_boundary(
        threshold, beta, cutoff, _splay_fn, (1e-6, float(tau_v)), tolerance
    )
    if not (np.isfinite(onset) and onset > 0.0):
        raise ValueError("the boundary must be a finite positive regulation constant")
    return float(onset)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only tests with immutable end-to-end targets."""
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
    return [
        {
            "setup": "import numpy as np\n",
            "call": "estimate_inflation_onset()",
            "gold_call": "_oracle_estimate_inflation_onset()",
        },
        {
            "setup": "import numpy as np\n",
            "call": "estimate_inflation_onset(1.8, 2.2, 6.0)",
            "gold_call": "_oracle_estimate_inflation_onset(1.8, 2.2, 6.0)",
        },
        {
            "setup": "import numpy as np\n",
            "call": "estimate_inflation_onset(2.3, 2.9, 4.0)",
            "gold_call": "_oracle_estimate_inflation_onset(2.3, 2.9, 4.0)",
        },
        {
            "setup": status,
            "call": "_status(lambda: estimate_inflation_onset(0.9, 2.5, 10.0))",
            "gold_call": "_status(lambda: _oracle_estimate_inflation_onset(0.9, 2.5, 10.0))",
        },
        {
            "setup": status,
            "call": "_status(lambda: estimate_inflation_onset(2.0, 1.9, 10.0))",
            "gold_call": "_status(lambda: _oracle_estimate_inflation_onset(2.0, 1.9, 10.0))",
        },
    ]
