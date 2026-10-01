"""
Bound the emitted photon-number probabilities of a phase-randomised weak coherent pulse whose actual mean photon number is known only to lie within a bounded relative deviation of its nominal setting.

A decoy-state transmitter nominally prepares each pulse at one of a few mean photon numbers, but the intensity modulator never reaches the nominal value exactly, and the residual offset drifts from pulse to pulse in a way that depends on the settings used in nearby rounds. A source characterisation therefore delivers, in place of one emitted photon-number distribution per setting, an admissible interval of actual mean photon numbers around each nominal setting, expressed as a maximum relative deviation. Conditioned on an actual mean photon number, the emitted photon number is Poisson distributed. Because every quantity returned here is extremal over the whole admissible interval, none of them carries any dependence on the settings used in earlier rounds.

Both returned arrays are indexed by photon number from zero up to and including the cut-off.

Returns
-------
lower : np.ndarray — Shape (n_cut + 1,). Entry n is the smallest probability of emitting exactly n photons that is attained over the admissible interval.; upper : np.ndarray — Shape (n_cut + 1,). Entry n is the largest probability of emitting exactly n photons that is attained over the admissible interval.; cut_mass : float — The largest probability of emitting strictly more than `n_cut` photons that is attained over the admissible interval.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_photon_number_bounds(intensity: float, delta_max: float, n_cut: int) -> "tuple[np.ndarray, np.ndarray, float]":
    '''Bound the emitted photon-number probabilities over the admissible intensity interval.

    Parameters
    ----------
    intensity : float
        Nominal mean photon number of the setting. Strictly positive.
    delta_max : float
        Maximum relative deviation of the actual mean photon number from
        `intensity`, in [0, 1). The admissible interval of actual mean photon
        numbers is [intensity * (1 - delta_max), intensity * (1 + delta_max)].
    n_cut : int
        Photon-number cut-off. A non-negative integer.

    Returns
    -------
    lower : np.ndarray
        Shape (n_cut + 1,). Entry n is the smallest probability of emitting
        exactly n photons that is attained over the admissible interval.
    upper : np.ndarray
        Shape (n_cut + 1,). Entry n is the largest probability of emitting
        exactly n photons that is attained over the admissible interval.
    cut_mass : float
        The largest probability of emitting strictly more than `n_cut` photons
        that is attained over the admissible interval.

    Raises
    ------
    ValueError
        If `intensity` is not strictly positive, if `delta_max` is negative or
        is not below one, if `n_cut` is negative, or if
        intensity * (1 + delta_max) exceeds one, beyond which the returned
        bounds are no longer attained at the ends of the admissible interval.
    '''
    return lower, upper, cut_mass

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.special import gammaln


def _oracle_compute_photon_number_bounds(intensity: float, delta_max: float, n_cut: int) -> "tuple[np.ndarray, np.ndarray, float]":
    if isinstance(intensity, bool) or not isinstance(intensity, (int, float, np.integer, np.floating)):
        raise ValueError("intensity must be a real scalar")
    if isinstance(delta_max, bool) or not isinstance(delta_max, (int, float, np.integer, np.floating)):
        raise ValueError("delta_max must be a real scalar")
    if isinstance(n_cut, bool) or not isinstance(n_cut, (int, np.integer)):
        raise ValueError("n_cut must be an integer")
    intensity = float(intensity)
    delta_max = float(delta_max)
    n_cut = int(n_cut)
    if not np.isfinite(intensity) or intensity <= 0.0:
        raise ValueError("intensity must be strictly positive")
    if not np.isfinite(delta_max) or delta_max < 0.0 or delta_max >= 1.0:
        raise ValueError("delta_max must lie in [0, 1)")
    if n_cut < 0:
        raise ValueError("n_cut must be non-negative")

    a_lo = intensity * (1.0 - delta_max)
    a_hi = intensity * (1.0 + delta_max)
    if a_hi > 1.0:
        raise ValueError("intensity * (1 + delta_max) must not exceed one")

    n = np.arange(n_cut + 1, dtype=float)
    log_factorial = gammaln(n + 1.0)

    # e^{-x} x^n / n! increases with x throughout (0, 1) for every n >= 1, so the
    # extremes of each n >= 1 probability sit at the ends of the interval; the
    # vacuum probability e^{-x} decreases with x and its extremes are exchanged.
    lower = np.exp(-a_lo + n * np.log(a_lo) - log_factorial)
    upper = np.exp(-a_hi + n * np.log(a_hi) - log_factorial)
    lower[0] = np.exp(-a_hi)
    upper[0] = np.exp(-a_lo)

    truncated = np.exp(-a_hi + n * np.log(a_hi) - log_factorial)
    cut_mass = float(max(0.0, 1.0 - float(np.sum(truncated))))
    return lower, upper, cut_mass

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    guard = """
def run_model(fn, *args):
    try:
        fn(*args)
        return 0.0
    except ValueError:
        return 1.0
"""
    return [
        # --- normal: benchmark signal setting -------------------------------
        {
            "setup": "intensity = 0.48\ndelta_max = 1e-3\nn_cut = 10\n",
            "call": "compute_photon_number_bounds(intensity, delta_max, n_cut)",
            "gold_call": "_oracle_compute_photon_number_bounds(intensity, delta_max, n_cut)",
            "tol": 1e-13,
        },
        # --- normal: benchmark decoy setting --------------------------------
        {
            "setup": "intensity = 0.10\ndelta_max = 1e-3\nn_cut = 10\n",
            "call": "compute_photon_number_bounds(intensity, delta_max, n_cut)",
            "gold_call": "_oracle_compute_photon_number_bounds(intensity, delta_max, n_cut)",
            "tol": 1e-13,
        },
        # --- boundary: no intensity drift, bounds collapse onto Poisson -----
        {
            "setup": "intensity = 0.48\ndelta_max = 0.0\nn_cut = 10\n",
            "call": "compute_photon_number_bounds(intensity, delta_max, n_cut)",
            "gold_call": "_oracle_compute_photon_number_bounds(intensity, delta_max, n_cut)",
            "tol": 1e-12,
        },
        # --- edge: vacuum-like setting spanning ~40 orders of magnitude -----
        {
            "setup": "intensity = 1e-4\ndelta_max = 1e-3\nn_cut = 10\n",
            "call": "compute_photon_number_bounds(intensity, delta_max, n_cut)",
            "gold_call": "_oracle_compute_photon_number_bounds(intensity, delta_max, n_cut)",
            "tol": 1e-12,
        },
        # --- boundary: single-entry arrays, cut mass carries almost all mass -
        {
            "setup": "intensity = 0.48\ndelta_max = 1e-3\nn_cut = 0\n",
            "call": "compute_photon_number_bounds(intensity, delta_max, n_cut)",
            "gold_call": "_oracle_compute_photon_number_bounds(intensity, delta_max, n_cut)",
        },
        # --- edge: admissible interval reaching to one photon ---------------
        {
            "setup": "intensity = 0.9\ndelta_max = 0.1\nn_cut = 6\n",
            "call": "compute_photon_number_bounds(intensity, delta_max, n_cut)",
            "gold_call": "_oracle_compute_photon_number_bounds(intensity, delta_max, n_cut)",
        },
        # --- edge: very wide admissible interval ----------------------------
        {
            "setup": "intensity = 0.2\ndelta_max = 0.5\nn_cut = 8\n",
            "call": "compute_photon_number_bounds(intensity, delta_max, n_cut)",
            "gold_call": "_oracle_compute_photon_number_bounds(intensity, delta_max, n_cut)",
            "tol": 1e-13,
        },
        # --- edge: cut-off where the truncated mass shares the arrays' scale -
        {
            "setup": "intensity = 0.48\ndelta_max = 1e-3\nn_cut = 4\n",
            "call": "compute_photon_number_bounds(intensity, delta_max, n_cut)",
            "gold_call": "_oracle_compute_photon_number_bounds(intensity, delta_max, n_cut)",
            "tol": 1e-13,
        },
        # --- edge: cut-off low enough that the truncated mass is sizeable ----
        {
            "setup": "intensity = 0.48\ndelta_max = 2e-3\nn_cut = 2\n",
            "call": "compute_photon_number_bounds(intensity, delta_max, n_cut)",
            "gold_call": "_oracle_compute_photon_number_bounds(intensity, delta_max, n_cut)",
        },
        # --- invalid: non-positive intensity --------------------------------
        {
            "setup": guard + "args = (0.0, 1e-3, 10)\n",
            "call": "run_model(compute_photon_number_bounds, *args)",
            "gold_call": "run_model(_oracle_compute_photon_number_bounds, *args)",
        },
        # --- invalid: relative deviation not below one -----------------------
        {
            "setup": guard + "args = (0.48, 1.0, 10)\n",
            "call": "run_model(compute_photon_number_bounds, *args)",
            "gold_call": "run_model(_oracle_compute_photon_number_bounds, *args)",
        },
        # --- invalid: negative cut-off ---------------------------------------
        {
            "setup": guard + "args = (0.48, 1e-3, -1)\n",
            "call": "run_model(compute_photon_number_bounds, *args)",
            "gold_call": "run_model(_oracle_compute_photon_number_bounds, *args)",
        },
        # --- invalid: admissible interval reaching beyond one photon ---------
        {
            "setup": guard + "args = (0.98, 0.05, 10)\n",
            "call": "run_model(compute_photon_number_bounds, *args)",
            "gold_call": "run_model(_oracle_compute_photon_number_bounds, *args)",
        },
    ]
