"""
Laplace exponent per unit concentration carried by the water that will be sampled.

Between shedding and sampling, a patch of eDNA is not simply diluted: biological

decay, deposition and resuspension act on it, and where those processes are

lumped into multiplicative noise whose strength scales with the square root of

the concentration, a patch can vanish outright. The chance that a patch leaves

no trace in the assay therefore depends on how long ago it entered the water,

and that dependence is governed by a single scalar carried along with the water.

Returns
-------
np.ndarray of shape (n,): the Laplace exponent per unit concentration in ml per km, negative, in the order of backward_times
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def washout_exponent(backward_times: "np.ndarray", assayed_volume: float,
                     window_length: float, reach: "np.ndarray") -> "np.ndarray":
    '''Laplace exponent per unit concentration carried by the water that will be sampled.

    Geometry and conventions are those of step 2. The eDNA field obeys

        dY = (flow * dY/dx + shedding - decay * Y) dt + noise * sqrt(Y) W(dt, dx),

    with W space-time white noise on the reach, so that the field over any
    stretch of water is driven by noise independent of that over any other. A
    replicate sample recovers a Poisson number of amplifiable copies whose mean is
    assayed_volume / window_length times the integral of the field over the
    sampled window, and replicates are conditionally independent given the field.

    Fix a backward time s and condition on the whole field s days before the
    survey, with all shedding switched off from then on. The log probability that
    the survey records no copies is then an integral over the reach of a
    coefficient times the field. That coefficient vanishes on water that will have
    left the window by the survey instant and equals one and the same number on
    the water that will be inside it. Return that number at each backward time.
    At s = 0 it is minus assayed_volume divided by window_length.

    Parameters
    ----------
    backward_times : np.ndarray
        Shape (n,), ages in days before the survey, finite and not negative.
    assayed_volume : float
        Effective assayed volume of the whole survey at this window, in ml,
        finite and positive.
    window_length : float
        Length of the sampled window in km, finite and positive.
    reach : np.ndarray
        Shape (5,), the reach description of step 2, whose fifth entry is the
        noise intensity in copies**0.5 km**0.5 ml**-0.5 day**-0.5.

    Returns
    -------
    exponent : np.ndarray
        Shape (n,) of floats, in ml per km, in the order of backward_times. Every
        entry is negative.

    Raises
    ------
    ValueError
        If backward_times is not a non-empty one-dimensional array of finite
        nonnegative values, if assayed_volume or window_length is not finite and
        positive, or if reach fails the step 2 contract.
    '''
    return exponent  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_washout_exponent(backward_times: "np.ndarray", assayed_volume: float,
                             window_length: float, reach: "np.ndarray") -> "np.ndarray":
    length, flow, ground, decay, noise = _check_reach(reach)
    s = np.asarray(backward_times, dtype=float)
    if s.ndim != 1 or s.size == 0 or not np.all(np.isfinite(s)) or np.any(s < 0.0):
        raise ValueError("backward_times must be a non-empty 1-D array of finite nonnegative values")
    vol, wd = float(assayed_volume), float(window_length)
    if not np.isfinite(vol) or vol <= 0.0 or not np.isfinite(wd) or wd <= 0.0:
        raise ValueError("assayed_volume and window_length must be finite and positive")
    lam0 = vol / wd
    if noise == 0.0:
        return -lam0 * np.exp(-decay * s)
    half = noise * noise / (2.0 * decay)
    return 1.0 / (np.exp(decay * s) * (-1.0 / lam0 - half) + half)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    setup = (
        "import numpy as np\n"
        "reach_a = np.array([20.0, 20.0, 2.0, 1.5, 1.2])\n"
        "reach_b = np.array([6.0, 45.0, 0.6, 9.0, 0.4])\n"
        "reach_c = np.array([120.0, 8.0, 5.0, 0.25, 3.0])\n"
        "reach_d = np.array([10.0, 12.0, 1.0, 3.0, 0.0])\n"
        "s_a = np.array([0.0, 0.05, 0.4, 2.0, 9.0])\n"
        "s_b = np.array([0.0, 0.002, 0.03, 0.5])\n"
        "s_c = np.linspace(0.0, 40.0, 9)\n"
    )
    return [
        # --- normal: the shipped reach and the station's three replicates ---
        {
            "setup": setup,
            "call": "washout_exponent(s_a, 3.9, 1.0, reach_a)",
            "gold_call": "_oracle_washout_exponent(s_a, 3.9, 1.0, reach_a)",
        },
        # --- boundary: fast decay with weak noise, where the exponent is nearly a pure decay ---
        {
            "setup": setup,
            "call": "washout_exponent(s_b, 0.6, 6.0, reach_b)",
            "gold_call": "_oracle_washout_exponent(s_b, 0.6, 6.0, reach_b)",
        },
        # --- edge: slow decay with strong noise and a large assayed volume, deep in the nonlinear regime ---
        {
            "setup": setup,
            "call": "washout_exponent(s_c, 260.0, 4.0, reach_c)",
            "gold_call": "_oracle_washout_exponent(s_c, 260.0, 4.0, reach_c)",
        },
        # --- boundary: a reach with no noise at all, where the exponent is a pure decay of the weight ---
        {
            "setup": setup,
            "call": "washout_exponent(s_b, 1.5, 2.0, reach_d)",
            "gold_call": "_oracle_washout_exponent(s_b, 1.5, 2.0, reach_d)",
        },
    ]
