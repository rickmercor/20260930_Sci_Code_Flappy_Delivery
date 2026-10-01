"""
Expected eDNA concentration of the water a sample assays, in copies per ml.

A water sample drawn at a fixed point of a river carries eDNA that was shed

further upstream and has been travelling downstream, decaying, ever since. Which

migrants could have shed it depends on where they were when the shedding

happened, so the record a sample carries is a weighted trace of the run rather

than a snapshot of it. Averaging that trace over the stretch of water a single

sample represents gives the quantity a laboratory assay is calibrated against.

Returns
-------
float, expected concentration of the sampled water in copies per ml, converged in the quadrature to better than 1e-10 relative, as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def mean_sampled_concentration(day: float, site_start: float, window_length: float,
                               reach: "np.ndarray", run: "np.ndarray",
                               shedding_rate: float) -> float:
    '''Expected eDNA concentration of the water a sample assays, in copies per ml.

    The reach is the interval from 0 to its length in km, with x measured upstream
    from the downstream end. River water moves towards x = 0 at the flow speed;
    no eDNA enters at x = reach length and the reach holds none on day 0. Migrants
    pass x = 0 during the season and move upstream at the ground speed, so a
    migrant that passes x = 0 on day t is at x on day t + x/(ground speed). eDNA
    is shed at a rate of shedding_rate times the unit-time migrant count present
    at that place and time, and decays at the decay rate. A sample taken on the
    given day assays the water that occupies the km window from site_start to
    site_start + window_length at that instant; its concentration is the mean of
    the field over that window.

    The expected count profile is the one of step 1; the noise on the field is
    centred and does not enter this expectation.

    Parameters
    ----------
    day : float
        Survey day, a finite value with 0 < day <= season length.
    site_start : float
        Downstream edge of the sampled window in km, finite, at least 0.
    window_length : float
        Length of the sampled window in km, finite and positive, with
        site_start + window_length not exceeding the reach length.
    reach : np.ndarray
        Shape (5,), in the order [reach length in km, flow speed in km per day,
        upstream ground speed of migrants in km per day, eDNA decay rate per day,
        noise intensity]. The first four are positive; the noise intensity is not
        used here and is only required to be finite and not negative.
    run : np.ndarray
        Shape (6,), the run description of step 1.
    shedding_rate : float
        eDNA shed per ml per day per unit of the unit-time migrant count, finite
        and positive.

    Returns
    -------
    concentration : float
        Expected concentration of the sampled water in copies per ml, converged in
        the quadrature to better than 1e-10 relative, as a native Python float.

    Raises
    ------
    ValueError
        If day is not finite and inside the season, if site_start or
        window_length place the window outside the reach, if reach is not a shape
        (5,) array of finite values with positive length, speeds and decay rate
        and a nonnegative noise intensity, if run fails the step 1 contract, or if
        shedding_rate is not finite and positive.
    '''
    return concentration  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _check_reach(reach):
    arr = np.asarray(reach, dtype=float)
    if arr.shape != (5,) or not np.all(np.isfinite(arr)):
        raise ValueError("reach must be a shape (5,) array of finite values")
    length, flow, ground, decay, noise = arr
    if length <= 0.0 or flow <= 0.0 or ground <= 0.0 or decay <= 0.0:
        raise ValueError("reach length, flow speed, ground speed and decay rate must be positive")
    if noise < 0.0:
        raise ValueError("noise intensity must not be negative")
    return length, flow, ground, decay, noise


def _gauss(lo, hi, n):
    nodes, weights = np.polynomial.legendre.leggauss(n)
    mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
    return mid + half * nodes, half * weights


def _panel_nodes(lo, hi, breaks, n):
    edges = [lo] + sorted(b for b in breaks if lo < b < hi) + [hi]
    xs, ws = [], []
    for k in range(len(edges) - 1):
        a, b = edges[k], edges[k + 1]
        if b <= a:
            continue
        x, w = _gauss(a, b, n)
        xs.append(x)
        ws.append(w)
    if not xs:
        return np.zeros(0), np.zeros(0)
    return np.concatenate(xs), np.concatenate(ws)


def _oracle_mean_sampled_concentration(day: float, site_start: float, window_length: float,
                                       reach: "np.ndarray", run: "np.ndarray",
                                       shedding_rate: float) -> float:
    length, flow, ground, decay, noise = _check_reach(reach)
    season = _check_run(run)[0]
    d = float(day)
    if not np.isfinite(d) or d <= 0.0 or d > season:
        raise ValueError("day must be finite and inside the season")
    x0, wd = float(site_start), float(window_length)
    if not np.isfinite(x0) or not np.isfinite(wd) or x0 < 0.0 or wd <= 0.0 or x0 + wd > length:
        raise ValueError("the sampled window must lie inside the reach")
    g = float(shedding_rate)
    if not np.isfinite(g) or g <= 0.0:
        raise ValueError("shedding_rate must be finite and positive")

    slope = 1.0 + flow / ground
    xs, xw = _panel_nodes(x0, x0 + wd, [length - flow * d], 64)
    total = 0.0
    for x, w in zip(xs, xw):
        xi_max = min(d, (length - x) / flow)
        if xi_max <= 0.0:
            continue
        zero_at = (d - x / ground) / slope
        season_at = (d - x / ground - season) / slope
        xi, xiw = _panel_nodes(0.0, xi_max, [zero_at, season_at], 64)
        src = _count_mean(d - xi - (x + flow * xi) / ground, run)
        total += w * float(np.sum(xiw * np.exp(-decay * xi) * src))
    return float(g * total / wd)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    setup = (
        "import numpy as np\n"
        "reach_a = np.array([20.0, 20.0, 2.0, 1.5, 1.2])\n"
        "run_a = np.array([80.0, 1200.0, 6.0, 8.0, 1.8, 30.0])\n"
        "reach_b = np.array([6.0, 45.0, 0.6, 9.0, 0.4])\n"
        "run_b = np.array([45.0, 260.0, 2.5, 2.5, 0.8, 4.0])\n"
        "reach_c = np.array([120.0, 8.0, 5.0, 0.25, 3.0])\n"
        "run_c = np.array([210.0, 9.0e4, 11.0, 4.0, 3.2, 95.0])\n"
    )
    return [
        # --- normal: the shipped reach, the station window, mid-run ---
        {
            "setup": setup,
            "call": "mean_sampled_concentration(26.0, 0.0, 1.0, reach_a, run_a, 0.012)",
            "gold_call": "_oracle_mean_sampled_concentration(26.0, 0.0, 1.0, reach_a, run_a, 0.012)",
        },
        # --- boundary: a window filling a short, fast, fast-decaying reach on the last day of the season ---
        {
            "setup": setup,
            "call": "mean_sampled_concentration(45.0, 0.0, 6.0, reach_b, run_b, 3.5)",
            "gold_call": "_oracle_mean_sampled_concentration(45.0, 0.0, 6.0, reach_b, run_b, 3.5)",
        },
        # --- edge: a long slow reach sampled far upstream, where the travel time truncates the trace ---
        {
            "setup": setup,
            "call": "mean_sampled_concentration(150.0, 88.0, 4.0, reach_c, run_c, 0.0004)",
            "gold_call": "_oracle_mean_sampled_concentration(150.0, 88.0, 4.0, reach_c, run_c, 0.0004)",
        },
    ]
