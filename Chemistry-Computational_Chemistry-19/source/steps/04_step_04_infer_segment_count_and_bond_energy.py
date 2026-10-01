"""
Infer the number of Kuhn segments and the bond well depth of a clamped chain from its mean intact and broken dwell times at one end-to-end distance.

A chain clamped at fixed extension switches between intact and broken states, and each mean dwell time reports one thermally activated barrier; the two barriers respond differently to the bond energy and to the chain length.

Returns
-------
np.ndarray: shape (2,) float array holding the inferred segment count and the fitted Lennard-Jones well depth in k_B T.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def infer_segment_count_and_bond_energy(y_bar_hold: float, nu_tau_intact: float, nu_tau_broken: float, n_min: int, n_max: int) -> "np.ndarray":
    """Return the segment count and well depth that reproduce two dwell times.

    At the clamped end-to-end distance ``y_bar_hold`` (Kuhn lengths) the
    stationary points from ``locate_free_energy_extrema`` define the scission
    barrier (transition-state minus intact-state free energy) and the
    re-formation barrier (transition-state minus broken-state free energy),
    both in k_B T from ``compute_link_free_energy``. Scission can occur at any
    one of the ``n_segments`` equivalent segments, while a broken chain re-forms
    only through the bond that broke. Both elementary processes are Arrhenius
    with the same attempt frequency, and times are given in units of its
    inverse. The mean intact dwell time is the inverse of the total scission
    rate and the mean broken dwell time is the inverse of the re-formation rate.

    For each integer ``n_segments`` in ``[n_min, n_max]`` the well depth is the
    value in [20, 150] that reproduces ``nu_tau_intact`` exactly. The returned
    segment count is the one whose fitted well depth brings the natural log of
    the predicted broken dwell time closest to ``ln(nu_tau_broken)``, with ties
    resolved toward the smaller count. The inclusive search range contains at
    most 101 candidates and has ``n_max <= 10000``. Callers supply data for
    which every count in the range admits such a well depth. The well depth
    must be accurate to 1e-8.

    Parameters
    ----------
    y_bar_hold : float
        Clamped end-to-end distance in Kuhn lengths.
    nu_tau_intact : float
        Mean intact dwell time times the attempt frequency, positive.
    nu_tau_broken : float
        Mean broken dwell time times the attempt frequency, positive.
    n_min : int
        Smallest segment count considered, at least 10.
    n_max : int
        Largest segment count considered, at least ``n_min``, at most 10000,
        and no more than 100 above ``n_min``.

    Returns
    -------
    estimate : np.ndarray
        Array ``[n_segments, bond_energy]`` of shape (2,) and float dtype.

    Raises
    ------
    ValueError
        If either dwell time is not positive, if ``n_min`` is not an integer of
        at least 10, or if ``n_max`` is not an integer in
        ``[n_min, min(n_min + 100, 10000)]``.
    """
    return estimate

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_infer_segment_count_and_bond_energy(y_bar_hold: float, nu_tau_intact: float, nu_tau_broken: float, n_min: int, n_max: int) -> "np.ndarray":
    """Reference implementation (bond energy by root finding, segment count by bisection on the mismatch)."""
    import numpy as np
    from scipy.optimize import brentq

    if not (nu_tau_intact > 0.0 and nu_tau_broken > 0.0):
        raise ValueError("dwell times must be positive")
    for value in (n_min, n_max):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError("segment-count bounds must be integers")
    if n_min < 10 or n_max < n_min or n_max > 10000 or n_max - n_min > 100:
        raise ValueError("need 10 <= n_min <= n_max <= 10000 and at most 101 candidates")

    def _barriers(n, depth):
        points = _oracle_locate_free_energy_extrema(y_bar_hold, n, depth)
        if np.isnan(points[0]):
            return -np.inf, np.inf
        energy = _oracle_compute_link_free_energy(points, y_bar_hold, n, depth)
        return energy[1] - energy[0], energy[1] - energy[2]

    cache = {}

    def _fit(n):
        # Returns the fitted depth and the signed log mismatch of the broken dwell time.
        if n not in cache:
            target = np.log(n * nu_tau_intact)
            depth = brentq(lambda d: _barriers(n, d)[0] - target, 20.0, 150.0, xtol=1e-11)
            cache[n] = (depth, _barriers(n, depth)[1] - np.log(nu_tau_broken))
        return cache[n]

    # Compare every admitted integer; the mismatch is not globally monotone.
    best = min(range(int(n_min), int(n_max) + 1), key=lambda n: (abs(_fit(n)[1]), n))
    return np.array([float(best), _fit(best)[0]])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    setup = (
        "import numpy as np\n"
        "def _sig(value, digits):\n"
        "    return float('%.*e' % (digits - 1, float(value)))\n"
        "def _digest(estimate):\n"
        "    return _sig(estimate[1], 8) + 1e-2 * float(estimate[0])\n"
    )
    status = (
        "def run_model():\n"
        "    try:\n"
        "        infer_segment_count_and_bond_energy(29.0, 5.77e11, -1.0, 45, 60)\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "def run_oracle():\n"
        "    try:\n"
        "        _oracle_infer_segment_count_and_bond_energy(29.0, 5.77e11, -1.0, 45, 60)\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
    )
    return [
        {
            "setup": setup,
            "call": "_digest(infer_segment_count_and_bond_energy(29.0, 5.770e11, 5.223e11, 45, 60))",
            "gold_call": "_digest(_oracle_infer_segment_count_and_bond_energy(29.0, 5.770e11, 5.223e11, 45, 60))",
        },
        {
            "setup": setup,
            "call": "_digest(infer_segment_count_and_bond_energy(29.0, 5.770e11, 5.223e11, 53, 58))",
            "gold_call": "_digest(_oracle_infer_segment_count_and_bond_energy(29.0, 5.770e11, 5.223e11, 53, 58))",
        },
        {
            "setup": setup,
            "call": "_digest(infer_segment_count_and_bond_energy(8.0, 2.087e8, 106.9, 12, 21))",
            "gold_call": "_digest(_oracle_infer_segment_count_and_bond_energy(8.0, 2.087e8, 106.9, 12, 21))",
        },
        {
            "setup": setup,
            "call": "_digest(infer_segment_count_and_bond_energy(29.0, 5.770e11, 5.223e11, 51, 51))",
            "gold_call": "_digest(_oracle_infer_segment_count_and_bond_energy(29.0, 5.770e11, 5.223e11, 51, 51))",
        },
        {
            "setup": status,
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
