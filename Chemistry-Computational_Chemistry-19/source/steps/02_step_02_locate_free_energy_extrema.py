"""
Locate the intact-state minimum, the transition-state maximum and the broken-state minimum of the free energy profile along the breakable segment's length at a fixed end-to-end distance.

Holding the end-to-end distance fixed lets a dissociated segment settle into a second free energy basin, so the profile can hold a bonded and a broken state separated by a transition state until stretching erases the bonded basin.

Returns
-------
np.ndarray: shape (3,) intact minimum, transition maximum and broken minimum along the segment length, or all NaN without an intact state.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def locate_free_energy_extrema(y_bar: float, n_segments: int, bond_energy: float) -> "np.ndarray":
    """Return the three stationary points that define scission and re-formation.

    The profile is ``compute_link_free_energy(x, y_bar, n_segments, bond_energy)``
    as a function of the breakable-segment length ``x`` (Kuhn lengths). The
    intact state is the local minimum nearest the bond length ``x = 1``, the
    transition state is the local maximum that immediately follows it at larger
    ``x``, and the broken state is the next local minimum after that maximum.
    If the required minimum-maximum-minimum pattern does not exist, every entry
    is NaN. Each reported location must be accurate to 1e-8.

    Parameters
    ----------
    y_bar : float
        Fixed end-to-end distance in Kuhn lengths, nonnegative and below
        ``n_segments + 1``.
    n_segments : int
        Number of Kuhn segments, at least 10.
    bond_energy : float
        Lennard-Jones well depth in k_B T, between 20 and 150.

    Returns
    -------
    locations : np.ndarray
        Array ``[x_intact, x_transition, x_broken]`` of shape (3,), or three
        NaN values when that sequence of stationary points does not exist.

    Raises
    ------
    ValueError
        If ``y_bar`` is negative or not below ``n_segments + 1``, if
        ``n_segments`` is not an integer of at least 10, or if ``bond_energy``
        lies outside [20, 150].
    """
    return locations

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_locate_free_energy_extrema(y_bar: float, n_segments: int, bond_energy: float) -> "np.ndarray":
    """Locate stationary points using Richardson-extrapolated slopes."""
    import numpy as np
    from scipy.optimize import brentq, minimize_scalar

    if (
        isinstance(n_segments, bool)
        or not isinstance(n_segments, (int, np.integer))
        or n_segments < 10
    ):
        raise ValueError("n_segments must be an integer of at least 10")
    if not 0.0 <= y_bar < n_segments + 1:
        raise ValueError("y_bar must lie in [0, n_segments + 1)")
    if not 20.0 <= bond_energy <= 150.0:
        raise ValueError("bond_energy must lie in [20, 150]")

    m = int(n_segments) - 1
    y = float(y_bar)

    def _profile(x):
        return _oracle_compute_link_free_energy(
            np.atleast_1d(np.asarray(x, dtype=float)),
            y,
            n_segments,
            bond_energy,
        )

    def _slope(x):
        def _difference(step):
            values = _profile(np.array([
                x - 2.0 * step,
                x - step,
                x + step,
                x + 2.0 * step,
            ]))
            return float(
                (values[0] - 8.0 * values[1]
                 + 8.0 * values[2] - values[3])
                / (12.0 * step)
            )

        step = 1e-3 * x
        return (
            16.0 * _difference(step / 2.0) - _difference(step)
        ) / 15.0

    def _root(lo, hi):
        while _slope(lo) * _slope(hi) > 0.0:
            lo, hi = (
                lo - 0.25 * (hi - lo),
                hi + 0.25 * (hi - lo),
            )
        return brentq(_slope, lo, hi, xtol=1e-12)

    x_low = max(0.8, y - m + 0.01)
    grid = np.concatenate([
        np.linspace(x_low, 3.0, 161),
        np.geomspace(3.0, y + m - 0.01, 81)[1:],
    ])
    rise = np.diff(_profile(grid)) > 0.0
    bonded = grid[1:-1] <= 3.0
    minima = np.where(~rise[:-1] & rise[1:])[0] + 1
    maxima = np.where(rise[:-1] & ~rise[1:])[0] + 1
    first_min = minima[bonded[minima - 1]] if minima.size else minima

    result = np.full(3, np.nan)
    k = None
    x_intact = None
    x_transition = None

    if first_min.size:
        i = first_min[0]
        x_intact = _root(grid[i - 1], grid[i + 1])
        following_maxima = maxima[maxima > i]
        if following_maxima.size:
            k = following_maxima[0]
            x_transition = _root(grid[k - 1], grid[k + 1])
    else:
        cells = np.where(grid[1:] <= 3.0)[0]
        slopes = np.diff(_profile(grid))[cells] / np.diff(grid)[cells]
        j = cells[int(np.argmax(slopes))]
        lo, hi = grid[max(j - 1, 0)], grid[j + 2]
        best = minimize_scalar(
            lambda x: -_slope(x),
            bounds=(lo, hi),
            method="bounded",
            options={"xatol": 1e-11},
        )
        if -best.fun > 0.0:
            x_intact = brentq(_slope, lo, best.x, xtol=1e-12)
            x_transition = brentq(_slope, best.x, hi, xtol=1e-12)
            k = int(np.searchsorted(grid, x_transition))

    if k is not None:
        later = minima[minima > k]
        if later.size:
            x_broken = _root(
                grid[later[0] - 1],
                grid[later[0] + 1],
            )
            result = np.array([x_intact, x_transition, x_broken])

    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    setup = """import numpy as np
def _locations(value):
    a = np.asarray(value, dtype=float)
    if a.shape != (3,) or np.any(np.isinf(a)):
        raise AssertionError("Expected three locations")
    missing = np.isnan(a)
    if np.any(missing) and not np.all(missing):
        raise AssertionError("Only an all-NaN missing-state result is valid")
    return np.concatenate((missing.astype(float), np.where(missing, 0.0, a)))
"""
    return [
        {
            "setup": setup,
            "tol": 1e-8,
            "call": "_locations(locate_free_energy_extrema(29.0, 51, 35.0))",
            "gold_call": "_locations(_oracle_locate_free_energy_extrema(29.0, 51, 35.0))",
        },
        {
            "setup": setup,
            "tol": 1e-8,
            "call": "_locations(locate_free_energy_extrema(0.5, 51, 35.0))",
            "gold_call": "_locations(_oracle_locate_free_energy_extrema(0.5, 51, 35.0))",
        },
        {
            "setup": setup,
            "tol": 1e-8,
            "call": "_locations(locate_free_energy_extrema(80.0, 101, 50.0))",
            "gold_call": "_locations(_oracle_locate_free_energy_extrema(80.0, 101, 50.0))",
        },
        {
            "setup": setup,
            "tol": 1e-8,
            "call": "_locations(locate_free_energy_extrema(50.577, 51, 35.0))",
            "gold_call": "_locations(_oracle_locate_free_energy_extrema(50.577, 51, 35.0))",
        },
        {
            "setup": setup,
            "tol": 1e-8,
            "call": "_locations(locate_free_energy_extrema(50.6, 51, 35.0))",
            "gold_call": "_locations(_oracle_locate_free_energy_extrema(50.6, 51, 35.0))",
        },
        {
            "setup": setup,
            "tol": 1e-8,
            "call": "_locations(locate_free_energy_extrema(0.0, 51, 35.0))",
            "gold_call": "_locations(_oracle_locate_free_energy_extrema(0.0, 51, 35.0))",
        },
        {
            "setup": setup,
            "tol": 1e-8,
            "call": "_locations(locate_free_energy_extrema(0.5, 10, 150.0))",
            "gold_call": "_locations(_oracle_locate_free_energy_extrema(0.5, 10, 150.0))",
        },
        {
            "setup": setup,
            "tol": 1e-8,
            "call": "_locations(locate_free_energy_extrema(50.57945, 51, 35.0))",
            "gold_call": "_locations(_oracle_locate_free_energy_extrema(50.57945, 51, 35.0))",
        },
        {
            "setup": setup,
            "tol": 1e-8,
            "call": "_locations(locate_free_energy_extrema(50.57946, 51, 35.0))",
            "gold_call": "_locations(_oracle_locate_free_energy_extrema(50.57946, 51, 35.0))",
        },
    ]
