"""
Compute the Boltzmann-averaged reduced proton-transfer rate constant of the trigonometric double well at a given reduced inverse temperature.

The thermal rate generalizes transition-state theory to the quantum regime by weighting every stationary state with its Boltzmann factor, tunnelling states contributing their flux and transmission and states above the barrier passing with certainty.

Returns
-------
float: reduced rate constant k(beta) with converged Boltzmann sums.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_reduced_rate_constant(m: int, p: float, beta: float) -> float:
    """Return the reduced rate constant k(beta) of the trigonometric double well.

    With the levels ``eps_q`` of ``compute_well_energy_levels``::

        k(beta) = sum_q exp(-beta eps_q) g_q / sum_q exp(-beta eps_q)

    where both sums run over all levels ``q = 0, 1, 2, ...``, ``g_q`` is the
    product of ``J_q`` from ``compute_right_moving_flux`` and ``|T_q|**2``
    from ``compute_transmission_probability`` (evaluated with that level's
    ``mu_q**2``) for every level with ``eps_q < 0``, and ``g_q = 1`` for every
    level with ``eps_q >= 0``. The sums must be converged: including further
    levels may change ``k`` by less than ``1e-12`` relative.

    Parameters
    ----------
    m : int
        Order of the potential, a positive integer.
    p : float
        Potential parameter with ``p**2 > m**2 - 1/4``.
    beta : float
        Reduced inverse temperature, the energy unit of the model divided by
        ``k_B T``; finite and positive.

    Returns
    -------
    float
        The reduced rate constant ``k(beta)``.

    Raises
    ------
    ValueError
        If ``beta`` is not a finite positive real number (booleans are
        rejected), if ``m`` or ``p`` is invalid as in
        ``evaluate_well_eigenfunction``, or if the sums do not converge within
        1024 levels.
    """
    return rate

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_compute_reduced_rate_constant(m: int, p: float, beta: float) -> float:
    """Reference implementation: doubling level count until the Boltzmann tail is negligible."""
    import math
    import numpy as np

    if isinstance(beta, bool) or not isinstance(beta, (int, float, np.integer, np.floating)):
        raise ValueError("beta must be a real number")
    if not (math.isfinite(beta) and beta > 0.0):
        raise ValueError("beta must be finite and positive")
    if isinstance(p, bool) or not isinstance(p, (int, float, np.integer, np.floating)):
        raise ValueError("p must be a real number")
    if not (math.isfinite(p) and p * p > m * m - 0.25):
        raise ValueError("p must satisfy p^2 > m^2 - 1/4")
    count = 32
    while True:
        levels = _oracle_compute_well_energy_levels(m, p, count)
        if beta * (levels[-1] - levels[0]) > 45.0:
            break
        count *= 2
        if count > 1024:
            raise ValueError("Boltzmann sums do not converge within 1024 levels")
    boltzmann = np.exp(-beta * (levels - levels[0]))
    weights = np.ones(count)
    for q in np.flatnonzero(levels < 0.0):
        mu2, flux = _oracle_compute_right_moving_flux(m, p, int(q))
        weights[q] = flux * _oracle_compute_transmission_probability(m, p, int(q), float(mu2))
    return float(np.sum(boltzmann * weights) / np.sum(boltzmann))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
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
            "call": "float(compute_reduced_rate_constant(18, 27.089280949903493, 0.07602299053366514))",
            "gold_call": "float(_oracle_compute_reduced_rate_constant(18, 27.089280949903493, 0.07602299053366514))",
        },
        {
            "setup": "import numpy as np\n",
            "call": "float(compute_reduced_rate_constant(18, 27.089280949903493, 0.02))",
            "gold_call": "float(_oracle_compute_reduced_rate_constant(18, 27.089280949903493, 0.02))",
        },
        {
            "setup": "import numpy as np\n",
            "call": "float(compute_reduced_rate_constant(18, 27.089280949903493, 2.5))",
            "gold_call": "float(_oracle_compute_reduced_rate_constant(18, 27.089280949903493, 2.5))",
        },
        {
            "setup": "import numpy as np\n",
            "call": "float(compute_reduced_rate_constant(2, 7.82971, 0.0345))",
            "gold_call": "float(_oracle_compute_reduced_rate_constant(2, 7.82971, 0.0345))",
        },
        {
            "setup": "import numpy as np\n",
            "call": "float(compute_reduced_rate_constant(38, 47.9421, 0.1))",
            "gold_call": "float(_oracle_compute_reduced_rate_constant(38, 47.9421, 0.1))",
        },
        {
            "setup": status,
            "call": "_status(lambda: compute_reduced_rate_constant(18, 27.089280949903493, 0.0))",
            "gold_call": "_status(lambda: _oracle_compute_reduced_rate_constant(18, 27.089280949903493, 0.0))",
        },
        {
            "setup": status,
            "call": "_status(lambda: compute_reduced_rate_constant(18, 27.089280949903493, float('nan')))",
            "gold_call": "_status(lambda: _oracle_compute_reduced_rate_constant(18, 27.089280949903493, float('nan')))",
        },
    ]
