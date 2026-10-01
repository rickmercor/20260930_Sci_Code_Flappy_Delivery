"""
Compose every earlier step to obtain the ratio of the stripe's tensile load at necking bifurcation with a free-edge line tension to the bifurcation load of the same stripe without it.

A stretched tissue necks at the maximum of its load, which an ordered stripe reaches when its first junctions exchange neighbours, so comparing bifurcation loads isolates what the free-edge tension adds.

Returns
-------
float: tensile load at necking bifurcation with free-edge line tension divided by that without it.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def estimate_tension_load_ratio(
    n_rows: int = 10,
    kappa: float = 0.16,
    chi: float = 3.5,
    line_tension: float = 0.03,
    threshold: float = 0.1,
    n_columns: int = 2,
    tolerance: float = 1e-12,
) -> float:
    """Return the bifurcation-load ratio of the stripe with and without line tension.

    For the free-edge line tension ``t`` equal to ``line_tension`` and to
    zero, take the stretch ``lam_t`` of ``locate_bifurcation_stretch`` (with
    ``threshold`` and ``tolerance``), relax the affinely stretched
    stress-free stripe of ``build_stress_free_stripe`` at ``lam_t`` with
    ``relax_stripe`` (gradient tolerance ``1e-12``), and sum
    ``compute_row_loads`` over the rows to get the load ``F_t``. Return
    ``F_line_tension / F_0``. The defaults reproduce the problem statement.

    Parameters
    ----------
    n_rows, kappa, chi
        Stripe as in ``build_stress_free_stripe``.
    line_tension : float
        Positive line tension of the free edges.
    threshold : float
        Positive junction length at which a neighbour exchange occurs.
    n_columns : int
        Cells per row within one period, at least 2.
    tolerance : float
        Positive bracket width of the bifurcation stretches, at most ``1e-6``.

    Returns
    -------
    float
        The load ratio ``F_line_tension / F_0``.

    Raises
    ------
    ValueError
        If ``line_tension`` is not a finite positive real number (booleans
        are rejected); if, for either tension, the load at
        ``(1 - 1e-6) * lam_t`` is not below the load at ``lam_t`` or some
        junction at ``lam_t`` is shorter than ``threshold - 1e-6``; or if an
        earlier step rejects its input.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_estimate_tension_load_ratio(
    n_rows: int = 10,
    kappa: float = 0.16,
    chi: float = 3.5,
    line_tension: float = 0.03,
    threshold: float = 0.1,
    n_columns: int = 2,
    tolerance: float = 1e-12,
) -> float:
    """Reference orchestrator using only the reference function chain."""
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    if not (_is_number(line_tension) and line_tension > 0.0):
        raise ValueError("line_tension must be a finite positive number")
    vertices, cells, shifts, period = _oracle_build_stress_free_stripe(n_rows, n_columns, kappa, chi)

    def _state(stretch, tension):
        start = vertices.copy()
        start[:, 0] *= stretch
        relaxed = _oracle_relax_stripe(start, cells, shifts, stretch * period, kappa, chi, tension, 1e-12)
        rows = _oracle_compute_row_loads(relaxed, cells, shifts, stretch * period, kappa, chi, tension, n_rows)
        lengths = _oracle_compute_cell_geometry(relaxed, cells, shifts, stretch * period)[2]
        return float(np.sum(rows)), float(np.min(lengths))

    loads = []
    for tension in (float(line_tension), 0.0):
        stretch = _oracle_locate_bifurcation_stretch(n_rows, n_columns, kappa, chi, tension,
                                                     threshold, tolerance)
        load, shortest = _state(stretch, tension)
        below, _ = _state(stretch * (1.0 - 1e-6), tension)
        # Necking bifurcates at the load maximum: the load must still be rising into
        # the first exchange, and no other junction may have exchanged earlier.
        if not below < load:
            raise ValueError("the load is not rising into the first neighbour exchange")
        if shortest < float(threshold) - 1e-6:
            raise ValueError("another junction reaches the threshold before the first exchange")
        loads.append(load)
    return loads[0] / loads[1]

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only tests over distinct stripes."""
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
            "call": "estimate_tension_load_ratio(5, 0.2, 3.4, 0.04, 0.1, 2, 1e-10)",
            "gold_call": "_oracle_estimate_tension_load_ratio(5, 0.2, 3.4, 0.04, 0.1, 2, 1e-10)",
        },
        {
            "setup": "import numpy as np\n",
            "call": "estimate_tension_load_ratio(3, 0.35, 2.45, 0.05, 0.08, 2, 1e-10)",
            "gold_call": "_oracle_estimate_tension_load_ratio(3, 0.35, 2.45, 0.05, 0.08, 2, 1e-10)",
        },
        {
            "setup": "import numpy as np\n",
            "call": "estimate_tension_load_ratio(2, 0.25, 3.3, 0.02, 0.1, 2, 1e-10)",
            "gold_call": "_oracle_estimate_tension_load_ratio(2, 0.25, 3.3, 0.02, 0.1, 2, 1e-10)",
        },
        {
            "setup": "import numpy as np\n",
            "call": "estimate_tension_load_ratio(4, 0.3, 3.0, 0.1, 0.09, 3, 1e-10)",
            "gold_call": "_oracle_estimate_tension_load_ratio(4, 0.3, 3.0, 0.1, 0.09, 3, 1e-10)",
        },
        {
            "setup": status,
            "call": "_status(lambda: estimate_tension_load_ratio(3, 0.16, 3.5, 0.03, 0.9, 2, 1e-10))",
            "gold_call": "_status(lambda: _oracle_estimate_tension_load_ratio(3, 0.16, 3.5, 0.03, 0.9, 2, 1e-10))",
        },
        {
            "setup": status,
            "call": "_status(lambda: estimate_tension_load_ratio(3, 0.16, 3.5, -0.1, 0.1, 2, 1e-10))",
            "gold_call": "_status(lambda: _oracle_estimate_tension_load_ratio(3, 0.16, 3.5, -0.1, 0.1, 2, 1e-10))",
        },
    ]
