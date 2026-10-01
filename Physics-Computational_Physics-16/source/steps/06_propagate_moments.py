"""
Implement propagate_moments which evolves the ensemble mean and both population covariance matrices from an empty reactor.

Population means and covariances provide deterministic descriptions of
ensemble statistics. Their evolution spans the reactivity and circulation
ramps while retaining the distinct fluctuation mechanisms of the two models.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def propagate_moments(t_end: float, probabilities: "np.ndarray", generation_time: float, beta: float, source: float, alpha: "np.ndarray", decay: "np.ndarray", rho_schedule: "np.ndarray", tau_core_schedule: "np.ndarray", tau_excore_schedule: "np.ndarray") -> "tuple[np.ndarray, np.ndarray]":
    """Propagate the continuous model moments with zero initial populations and covariance.

    Parameters
    ----------
    t_end : float
        Nonnegative observation time in seconds.
    probabilities : np.ndarray
        Shape (K,), prompt-yield probabilities for 0 through K-1, with positive mean.
    generation_time : float
        Positive prompt neutron generation time in seconds.
    beta : float
        Total delayed fraction, 0 <= beta < 1.
    source : float
        Constant source intensity per second, satisfying the source model's
        positivity condition for the continuous SDE over the interval.
    alpha : np.ndarray
        Shape (6,), nonnegative relative delayed fractions summing to one.
    decay : np.ndarray
        Shape (6,), nonnegative precursor decay constants per second.
    rho_schedule, tau_core_schedule, tau_excore_schedule : np.ndarray
        Shape (3,) each, (x_0, x_1, T) for the reactivity and for the core and
        ex-core residence times in seconds: the quantity follows the linear
        ramp x(t) = x_0 + (x_1 - x_0) * min(t / T, 1) with ramp duration T > 0,
        so it stays at x_1 after t = T. Residence-time values are positive.

    Returns
    -------
    covariance_jump : np.ndarray
        Shape (13, 13), central covariance for the discrete-event process.
    covariance_sde : np.ndarray
        Shape (13, 13), central covariance for the continuous Itô approximation.
        All outputs use neutron, core groups 1-6, ex-core groups 1-6 order.
        Resolve the continuous moments to relative accuracy 1e-7 with
        absolute accuracy 1e-8 for components near zero.

    Raises
    ------
    ValueError
        If t_end is negative.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.integrate import solve_ivp

def _oracle_propagate_moments(t_end: float, probabilities: "np.ndarray", generation_time: float, beta: float, source: float, alpha: "np.ndarray", decay: "np.ndarray", rho_schedule: "np.ndarray", tau_core_schedule: "np.ndarray", tau_excore_schedule: "np.ndarray") -> "tuple[np.ndarray, np.ndarray]":
    """Implement propagate_moments which evolves the ensemble mean and both population covariance matrices from an empty reactor."""
    if t_end < 0:
        raise ValueError("t_end must be nonnegative")
    fission_mean, fission_second = _oracle_compute_fission_moments(probabilities, beta, alpha)
    state = np.zeros(351)
    def _rhs(t, y):
        return _oracle_evaluate_moment_rhs(t, y, fission_mean, fission_second, generation_time, beta, source, alpha, decay, rho_schedule, tau_core_schedule, tau_excore_schedule)
    kinks = [float(s[2]) for s in (rho_schedule, tau_core_schedule, tau_excore_schedule)]
    start = 0.0
    for stop in sorted(set([x for x in kinks if x < t_end] + [t_end])):
        if stop > start:
            result = solve_ivp(_rhs, (start, stop), state, method="DOP853", rtol=2e-11, atol=1e-11)
            state = result.y[:, -1]
            start = stop
    return state[13:182].reshape(13, 13).copy(), state[182:].reshape(13, 13).copy()

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return independent candidate/oracle comparisons."""
    return [
        {
            'setup': 'import numpy as np\np = np.array([.027, .158, .339, .305, .133, .038])\nalpha = np.array([.033, .219, .196, .395, .115, .042])\ndecay = np.array([.0124, .0305, .111, .301, 1.14, 3.01])\nrho_s = np.array([-.01, .006, 10.])\ntc_s = np.array([1000., 10., 5.])\nte_s = np.array([1000., 15., 5.])\n\ndef pack(fn, t, p, generation_time, beta, source, alpha, decay, rho_s, tc_s, te_s):\n    jump, sde = fn(t, p.copy(), generation_time, beta, source, alpha.copy(), decay.copy(), rho_s.copy(), tc_s.copy(), te_s.copy())\n    assert np.shape(jump) == (13,13) and np.shape(sde) == (13,13)\n    return tuple(10.0 * float(x) for x in np.concatenate((np.asarray(jump).ravel(), np.asarray(sde).ravel())))\n',
            'call': 'pack(propagate_moments, .05, p, .001, .0065, 8800., alpha, decay, rho_s, tc_s, te_s)',
            'gold_call': 'pack(_oracle_propagate_moments, .05, p, .001, .0065, 8800., alpha, decay, rho_s, tc_s, te_s)',
            'tol': 1e-07,
        },
        {
            'setup': 'import numpy as np\np = np.array([.027, .158, .339, .305, .133, .038])\nalpha = np.array([.033, .219, .196, .395, .115, .042])\ndecay = np.array([.0124, .0305, .111, .301, 1.14, 3.01])\nrho_s = np.array([-.01, .006, 10.])\ntc_s = np.array([1000., 10., 5.])\nte_s = np.array([1000., 15., 5.])\n\ndef pack(fn, t, p, generation_time, beta, source, alpha, decay, rho_s, tc_s, te_s):\n    jump, sde = fn(t, p.copy(), generation_time, beta, source, alpha.copy(), decay.copy(), rho_s.copy(), tc_s.copy(), te_s.copy())\n    assert np.shape(jump) == (13,13) and np.shape(sde) == (13,13)\n    return tuple(10.0 * float(x) for x in np.concatenate((np.asarray(jump).ravel(), np.asarray(sde).ravel())))\n',
            'call': 'pack(propagate_moments, 0., p, .001, .0065, 8800., alpha, decay, rho_s, tc_s, te_s)',
            'gold_call': 'pack(_oracle_propagate_moments, 0., p, .001, .0065, 8800., alpha, decay, rho_s, tc_s, te_s)',
            'tol': 1e-07,
        },
        {
            'setup': 'import numpy as np\np = np.array([.027, .158, .339, .305, .133, .038])\nalpha = np.array([.033, .219, .196, .395, .115, .042])\ndecay = np.array([.0124, .0305, .111, .301, 1.14, 3.01])\nrho_s = np.array([-.01, .006, 10.])\ntc_s = np.array([1000., 10., 5.])\nte_s = np.array([1000., 15., 5.])\n\ndef pack(fn, t, p, generation_time, beta, source, alpha, decay, rho_s, tc_s, te_s):\n    jump, sde = fn(t, p.copy(), generation_time, beta, source, alpha.copy(), decay.copy(), rho_s.copy(), tc_s.copy(), te_s.copy())\n    assert np.shape(jump) == (13,13) and np.shape(sde) == (13,13)\n    return tuple(10.0 * float(x) for x in np.concatenate((np.asarray(jump).ravel(), np.asarray(sde).ravel())))\n',
            'call': 'pack(propagate_moments, 5.5, p, .001, .0065, 8800., alpha, decay, rho_s, tc_s, te_s)',
            'gold_call': 'pack(_oracle_propagate_moments, 5.5, p, .001, .0065, 8800., alpha, decay, rho_s, tc_s, te_s)',
            'tol': 1e-07,
        },
        {
            'setup': 'import numpy as np\np = np.array([.027, .158, .339, .305, .133, .038])\nalpha = np.array([.033, .219, .196, .395, .115, .042])\ndecay = np.array([.0124, .0305, .111, .301, 1.14, 3.01])\nrho_s = np.array([-.01, .006, 10.])\ntc_s = np.array([1000., 10., 5.])\nte_s = np.array([1000., 15., 5.])\n\ndef check(fn):\n    try:\n        fn(-1., p.copy(), .001, .0065, 8800., alpha.copy(), decay.copy(), rho_s.copy(), tc_s.copy(), te_s.copy())\n    except ValueError:\n        return 1\n    return 0\n',
            'call': 'check(propagate_moments)',
            'gold_call': 'check(_oracle_propagate_moments)',
            'tol': 1e-08,
        },
    ]
