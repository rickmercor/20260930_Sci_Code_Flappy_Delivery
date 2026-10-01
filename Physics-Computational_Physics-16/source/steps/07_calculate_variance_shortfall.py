"""
Implement calculate_variance_shortfall which compares the ex-core precursor variances predicted by the two benchmark models.

A model can reproduce average inventories while underestimating their
fluctuations. Comparing group-specific variances reveals how precursor
noise and transport affect uncertainty in circulating fuel.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def calculate_variance_shortfall(t_end: float, probabilities: "np.ndarray", generation_time: float, beta: float, source: float, alpha: "np.ndarray", decay: "np.ndarray", rho_schedule: "np.ndarray", tau_core_schedule: "np.ndarray", tau_excore_schedule: "np.ndarray", group: int = 1) -> float:
    """Run the complete pipeline and report the SDE variance shortfall.

    Parameters
    ----------
    t_end : float
        Positive observation time in seconds.
    probabilities : np.ndarray
        Shape (K,), prompt-yield probabilities for yields 0 through K-1.
        The prompt yield is independent of the Poisson total delayed yield,
        which is split among groups by the relative fractions alpha.
    generation_time : float
        Positive prompt neutron generation time in seconds.
    beta : float
        Total delayed fraction, 0 <= beta < 1.
    source : float
        Constant external neutron-source intensity per second, satisfying
        the source model's positivity condition over the interval.
    alpha : np.ndarray
        Shape (6,), nonnegative relative delayed fractions summing to one.
    decay : np.ndarray
        Shape (6,), nonnegative precursor decay constants per second.
    rho_schedule, tau_core_schedule, tau_excore_schedule : np.ndarray
        Shape (3,) each, (x_0, x_1, T) for the reactivity and for the core and
        ex-core residence times in seconds: the quantity follows the linear
        ramp x(t) = x_0 + (x_1 - x_0) * min(t / T, 1) with ramp duration T > 0,
        so it stays at x_1 after t = T. Residence-time values are positive.
    group : int
        Delayed-neutron precursor group, numbered 1 through 6.
        Defaults to 1, the benchmark's ex-core group.
        Initially all populations and covariances are zero.

    Returns
    -------
    shortfall : float
        100 * (V_jump - V_sde) / V_jump for the selected ex-core group.
        Return the unrounded percentage for the continuous model statistics.
        Zero initial conditions and a constant source make both variances
        proportional to source, leaving this percentage invariant.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_calculate_variance_shortfall(t_end: float, probabilities: "np.ndarray", generation_time: float, beta: float, source: float, alpha: "np.ndarray", decay: "np.ndarray", rho_schedule: "np.ndarray", tau_core_schedule: "np.ndarray", tau_excore_schedule: "np.ndarray", group: int = 1) -> float:
    """Implement calculate_variance_shortfall which compares the ex-core precursor variances predicted by the two benchmark models."""
    jump, sde = _oracle_propagate_moments(t_end, probabilities, generation_time, beta, source, alpha, decay, rho_schedule, tau_core_schedule, tau_excore_schedule)
    index = 6 + group
    return float(100.0 * (jump[index, index] - sde[index, index]) / jump[index, index])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return independent candidate/oracle comparisons."""
    return [
        {
            'setup': 'import numpy as np\np = np.array([.027, .158, .339, .305, .133, .038])\nalpha = np.array([.033, .219, .196, .395, .115, .042])\ndecay = np.array([.0124, .0305, .111, .301, 1.14, 3.01])\nrho_s = np.array([-.01, .006, 10.])\ntc_s = np.array([1000., 10., 5.])\nte_s = np.array([1000., 15., 5.])\n',
            'call': 'calculate_variance_shortfall(8., p.copy(), .001, .0065, 8800., alpha.copy(), decay.copy(), rho_s.copy(), tc_s.copy(), te_s.copy(), 1)',
            'gold_call': '_oracle_calculate_variance_shortfall(8., p.copy(), .001, .0065, 8800., alpha.copy(), decay.copy(), rho_s.copy(), tc_s.copy(), te_s.copy(), 1)',
            'tol': 1e-05,
        },
        {
            'setup': 'import numpy as np\np = np.array([.027, .158, .339, .305, .133, .038])\nalpha = np.array([.033, .219, .196, .395, .115, .042])\ndecay = np.array([.0124, .0305, .111, .301, 1.14, 3.01])\nrho_s = np.array([-.01, .006, 10.])\ntc_s = np.array([1000., 10., 5.])\nte_s = np.array([1000., 15., 5.])\n',
            'call': 'calculate_variance_shortfall(5., p.copy(), .001, .0065, 8800., alpha.copy(), decay.copy(), rho_s.copy(), tc_s.copy(), te_s.copy(), 4)',
            'gold_call': '_oracle_calculate_variance_shortfall(5., p.copy(), .001, .0065, 8800., alpha.copy(), decay.copy(), rho_s.copy(), tc_s.copy(), te_s.copy(), 4)',
            'tol': 1e-05,
        },
        {
            'setup': 'import numpy as np\np = np.array([.027, .158, .339, .305, .133, .038])\nalpha = np.array([.033, .219, .196, .395, .115, .042])\ndecay = np.array([.0124, .0305, .111, .301, 1.14, 3.01])\nrho_s = np.array([-.004, .002, 4.])\ntc_s = np.array([50., 50., 1.])\nte_s = np.array([200., 20., 8.])\n',
            'call': 'calculate_variance_shortfall(.5, p.copy(), .002, .0065, 12000., alpha.copy(), decay.copy(), rho_s.copy(), tc_s.copy(), te_s.copy(), 6)',
            'gold_call': '_oracle_calculate_variance_shortfall(.5, p.copy(), .002, .0065, 12000., alpha.copy(), decay.copy(), rho_s.copy(), tc_s.copy(), te_s.copy(), 6)',
            'tol': 1e-05,
        },
    ]
