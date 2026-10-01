"""
Short-time coefficients of the same-junction waiting-time densities.

A detector that resolves one junction of a Markov network in both directions, and that fails to register some of its individual transitions, turns the sequence of registered events into a renewal process. The unregistered part of each transition acts as a second channel between the same two configurations, so a registered pair of identical direction on the same junction must contain exactly one unregistered transition of the opposite direction, while a registered pair of opposite direction on the same junction can follow with no intervening transition at all. Counting one factor of $t$ for each intervening transition gives, for each junction $L$,



$$\psi_{L_-\to L_+}(t) = \eta_{L+} k_{L+} + O(t), \qquad \psi_{L_+\to L_-}(t) = \eta_{L-} k_{L-} + O(t),$$



$$\psi_{L_+\to L_+}(t) = (1-\eta_{L-})\,k_{L-}\,\eta_{L+} k_{L+}\;t + O(t^2), \qquad \psi_{L_-\to L_-}(t) = (1-\eta_{L+})\,k_{L+}\,\eta_{L-} k_{L-}\;t + O(t^2).$$



The two resolved junctions share no configuration, so every density that pairs a transition of one junction with a transition of the other vanishes at $t=0$ and carries no short-time information about either junction.



Extraction convention, to be followed exactly. Use the first six tabulated nodes of the recording, $l = 0,\dots,5$, with the tabulated node times as abscissae and the tabulated values as ordinates. Fit each density by unweighted ordinary least squares with a cubic polynomial in $t$. For the two opposite-direction densities of a junction fit $b_0 + b_1 t + b_2 t^2 + b_3 t^3$ and read the constant term $b_0$. For the two same-direction densities the constant term vanishes identically, so fit $a_1 t + a_2 t^2 + a_3 t^3$ with no constant column and read the linear coefficient $a_1$. Do not weight, do not extend or shorten the window, and do not change the degree.



Write $c_{L+} = \psi_{L_-\to L_+}(0)$, $c_{L-} = \psi_{L_+\to L_-}(0)$, $a_{L++} = \psi_{L_+\to L_+}'(0)$ and $a_{L--} = \psi_{L_-\to L_-}'(0)$.

Returns
-------
A NumPy array of shape `(2, 4)` whose row `L` holds `[c_plus, c_minus, a_pp, a_mm]` for junction `L`. The first two entries of each row are in inverse microseconds and the last two in inverse microseconds squared. Raise `ValueError` for a non-monotonic or non-positive time axis, a shape mismatch, a negative density, or a non-finite entry.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

"""Short-time coefficients of the same-junction waiting-time densities."""

import numpy as np

_N_FIT = 6
_DEGREE = 3


def short_time_coefficients(node_times, recorded_density):
    """Extract the short-time coefficients of both resolved junctions.

    Parameters
    ----------
    node_times : array_like, shape (N,)
        Strictly increasing bin-center times in microseconds, N >= 6.
    recorded_density : array_like, shape (4, 4, N)
        Recorded waiting-time densities in inverse microseconds. The four
        registered transition types are ordered as forward and reverse of
        junction A followed by forward and reverse of junction B, so that
        ``recorded_density[u, v, l]`` is the mean density over bin ``l`` for a
        next registered transition of type ``v`` after one of type ``u``.

    Returns
    -------
    numpy.ndarray, shape (2, 4)
        Row ``L`` holds ``[c_plus, c_minus, a_pp, a_mm]`` for junction ``L``:
        the zero-time values of the two opposite-direction densities of that
        junction in inverse microseconds, followed by the zero-time slopes of
        its two same-direction densities in inverse microseconds squared.
    """
    return np.zeros((2, 4))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _design(times, degree, drop_constant):
    import numpy as np
    powers = np.arange(1, degree + 1) if drop_constant else np.arange(0, degree + 1)
    return np.power.outer(np.asarray(times, dtype=float), powers)


def _fit_leading(times, values, drop_constant):
    import numpy as np
    matrix = _design(times[:6], 3, drop_constant)
    solution, _, _, _ = np.linalg.lstsq(matrix, np.asarray(values, dtype=float)[:6], rcond=None)
    return float(solution[0])


def _oracle_short_time_coefficients(node_times, recorded_density):
    import numpy as np
    times = np.asarray(node_times, dtype=float)
    density = np.asarray(recorded_density, dtype=float)
    if times.ndim != 1 or times.size < 6:
        raise ValueError("node_times must be a one-dimensional array with at least six entries")
    if density.shape != (4, 4, times.size):
        raise ValueError("recorded_density must have shape (4, 4, len(node_times))")
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(density)):
        raise ValueError("node_times and recorded_density must be finite")
    if np.any(np.diff(times) <= 0.0) or times[0] <= 0.0:
        raise ValueError("node_times must be positive and strictly increasing")
    if np.any(density < 0.0):
        raise ValueError("recorded_density must be non-negative")
    out = np.zeros((2, 4))
    for link in range(2):
        plus, minus = 2 * link, 2 * link + 1
        out[link, 0] = _fit_leading(times, density[minus, plus], False)
        out[link, 1] = _fit_leading(times, density[plus, minus], False)
        out[link, 2] = _fit_leading(times, density[plus, plus], True)
        out[link, 3] = _fit_leading(times, density[minus, minus], True)
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Differential cases for the short-time extraction."""
    exact = (
        "import numpy as np\n"
        "t = np.array([0.00125, 0.00375, 0.00625, 0.00875, 0.01125, 0.01375])\n"
        "psi = np.zeros((4, 4, 6))\n"
        "psi[1, 0] = 1.02 - 4.1 * t + 12.0 * t ** 2 - 30.0 * t ** 3\n"
        "psi[0, 1] = 5.37 - 21.0 * t + 60.0 * t ** 2 - 90.0 * t ** 3\n"
        "psi[0, 0] = 0.74 * t - 2.8 * t ** 2 + 6.0 * t ** 3\n"
        "psi[1, 1] = 10.1 * t - 40.0 * t ** 2 + 70.0 * t ** 3\n"
        "psi[3, 2] = 4.84 - 19.0 * t + 55.0 * t ** 2 - 80.0 * t ** 3\n"
        "psi[2, 3] = 1.34 - 5.2 * t + 15.0 * t ** 2 - 25.0 * t ** 3\n"
        "psi[2, 2] = 8.94 * t - 34.0 * t ** 2 + 60.0 * t ** 3\n"
        "psi[3, 3] = 1.83 * t - 7.1 * t ** 2 + 12.0 * t ** 3\n"
        "psi[0, 2] = 0.9 * t\n"
        "psi[2, 0] = 0.4 * t\n"
    )
    longer = (
        "import numpy as np\n"
        "t = 0.0025 * (np.arange(9) + 0.5)\n"
        "psi = np.zeros((4, 4, 9))\n"
        "psi[1, 0] = 1.05 * np.exp(-7.2 * t)\n"
        "psi[0, 1] = 5.40 * np.exp(-9.4 * t)\n"
        "psi[0, 0] = 0.71 * t * np.exp(-5.0 * t)\n"
        "psi[1, 1] = 10.4 * t * np.exp(-4.0 * t)\n"
        "psi[3, 2] = 4.75 * np.exp(-6.1 * t)\n"
        "psi[2, 3] = 1.36 * np.exp(-8.3 * t)\n"
        "psi[2, 2] = 9.10 * t * np.exp(-3.0 * t)\n"
        "psi[3, 3] = 1.79 * t * np.exp(-2.5 * t)\n"
        "psi[1, 2] = 0.6 * t * np.exp(-2.0 * t)\n"
    )
    flat = (
        "import numpy as np\n"
        "t = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06])\n"
        "psi = np.zeros((4, 4, 6))\n"
        "psi[1, 0] = np.full(6, 1.5)\n"
        "psi[0, 1] = np.full(6, 0.8)\n"
        "psi[0, 0] = np.zeros(6)\n"
        "psi[1, 1] = 3.0 * t\n"
        "psi[3, 2] = np.full(6, 2.4)\n"
        "psi[2, 3] = np.full(6, 1.1)\n"
        "psi[2, 2] = 5.0 * t\n"
        "psi[3, 3] = np.zeros(6)\n"
    )
    return [
        {
            "setup": exact,
            "call": "short_time_coefficients(t, psi)",
            "gold_call": "_oracle_short_time_coefficients(t, psi)",
        },
        {
            "setup": longer,
            "call": "short_time_coefficients(t, psi)",
            "gold_call": "_oracle_short_time_coefficients(t, psi)",
        },
        {
            "setup": flat,
            "call": "short_time_coefficients(t, psi)",
            "gold_call": "_oracle_short_time_coefficients(t, psi)",
        },
    ]
