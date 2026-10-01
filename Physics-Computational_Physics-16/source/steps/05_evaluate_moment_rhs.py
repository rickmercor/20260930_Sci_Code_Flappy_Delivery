"""
Implement evaluate_moment_rhs which evaluates the coupled mean and covariance derivatives for both reactor models during a ramp transient.

The ramp transient changes neutron multiplication and precursor circulation
on different time scales. Both models share mean transport, while their
fluctuation sources enter separate covariance evolutions.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def evaluate_moment_rhs(t: float, state: "np.ndarray", fission_mean: "np.ndarray", fission_second: "np.ndarray", generation_time: float, beta: float, source: float, alpha: "np.ndarray", decay: "np.ndarray", rho_schedule: "np.ndarray", tau_core_schedule: "np.ndarray", tau_excore_schedule: "np.ndarray") -> "np.ndarray":
    """Evaluate a shared mean and two covariance derivatives under linear ramp schedules.

    Parameters
    ----------
    t : float
        Nonnegative time in seconds.
    state : np.ndarray
        Shape (351,): mean (13,), jump covariance flattened row-major (169,),
        then SDE covariance flattened row-major (169,). The state coordinate
        order is neutron, core groups 1-6, ex-core groups 1-6.
    fission_mean : np.ndarray
        Shape (7,), mean fission increment in neutron/core coordinates.
    fission_second : np.ndarray
        Shape (7, 7), raw second moment of that increment.
    generation_time : float
        Positive prompt neutron generation time in seconds.
    beta : float
        Total delayed fraction, 0 <= beta < 1.
    source : float
        Nonnegative, constant external neutron-source intensity per second.
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
    derivative : np.ndarray
        Shape (351,), derivatives in the same layout as state. Covariances
        are central second moments. Model parameters are in the source's
        nonnegative SDE regime and give nonnegative event rates.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_evaluate_moment_rhs(t: float, state: "np.ndarray", fission_mean: "np.ndarray", fission_second: "np.ndarray", generation_time: float, beta: float, source: float, alpha: "np.ndarray", decay: "np.ndarray", rho_schedule: "np.ndarray", tau_core_schedule: "np.ndarray", tau_excore_schedule: "np.ndarray") -> "np.ndarray":
    """Implement evaluate_moment_rhs which evaluates the coupled mean and covariance derivatives for both reactor models during a ramp transient."""
    state = np.asarray(state, dtype=float)
    mean = state[:13]
    covariance_jump = state[13:182].reshape(13, 13)
    covariance_sde = state[182:].reshape(13, 13)
    def _ramp(schedule):
        start, end, duration = (float(x) for x in schedule)
        return start + (end - start) * min(t / duration, 1.0)
    rho = _ramp(rho_schedule)
    tau_core = _ramp(tau_core_schedule)
    tau_excore = _ramp(tau_excore_schedule)
    phi = (1.0 - beta) / ((fission_mean[0] + 1.0) * generation_time)
    gamma = (1.0 - rho) / generation_time - phi
    drift = _oracle_build_mean_drift(rho, tau_core, tau_excore, generation_time, beta, alpha, decay)
    jump_noise = _oracle_build_jump_noise(mean, phi, gamma, source, tau_core, tau_excore, decay, fission_second)
    sde_noise = _oracle_build_sde_noise(mean, phi, gamma, fission_second[0, 0])
    dmean = drift @ mean
    dmean[0] += source
    djump = drift @ covariance_jump + covariance_jump @ drift.T + jump_noise
    dsde = drift @ covariance_sde + covariance_sde @ drift.T + sde_noise
    return np.concatenate((dmean, djump.ravel(), dsde.ravel()))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return independent candidate/oracle comparisons."""
    return [
        {
            'setup': 'import numpy as np\np = np.array([.027, .158, .339, .305, .133, .038])\nalpha = np.array([.033, .219, .196, .395, .115, .042])\ndecay = np.array([.0124, .0305, .111, .301, 1.14, 3.01])\nrho_s = np.array([-.01, .006, 10.])\ntc_s = np.array([1000., 10., 5.])\nte_s = np.array([1000., 15., 5.])\n\nmean = np.arange(1., 14.)\nfission_mean = np.r_[1.473, alpha * .0065 * 2.473 / .9935]\nfission_second = np.outer(fission_mean, fission_mean)\nfission_second[0, 0] = p @ (np.arange(6) - 1.) ** 2\nfission_second[np.arange(1, 7), np.arange(1, 7)] += fission_mean[1:]\n\nstate = np.r_[mean, np.eye(13).ravel(), (2*np.eye(13)).ravel()]\n',
            'call': 'evaluate_moment_rhs(2., state.copy(), fission_mean.copy(), fission_second.copy(), .001, .0065, 8800., alpha.copy(), decay.copy(), rho_s.copy(), tc_s.copy(), te_s.copy())',
            'gold_call': '_oracle_evaluate_moment_rhs(2., state.copy(), fission_mean.copy(), fission_second.copy(), .001, .0065, 8800., alpha.copy(), decay.copy(), rho_s.copy(), tc_s.copy(), te_s.copy())',
        },
        {
            'setup': 'import numpy as np\np = np.array([.027, .158, .339, .305, .133, .038])\nalpha = np.array([.033, .219, .196, .395, .115, .042])\ndecay = np.array([.0124, .0305, .111, .301, 1.14, 3.01])\nrho_s = np.array([-.01, .006, 10.])\ntc_s = np.array([1000., 10., 5.])\nte_s = np.array([1000., 15., 5.])\n\nmean = np.arange(1., 14.)\nfission_mean = np.r_[1.473, alpha * .0065 * 2.473 / .9935]\nfission_second = np.outer(fission_mean, fission_mean)\nfission_second[0, 0] = p @ (np.arange(6) - 1.) ** 2\nfission_second[np.arange(1, 7), np.arange(1, 7)] += fission_mean[1:]\n\nstate = np.zeros(351)\n',
            'call': 'evaluate_moment_rhs(0., state.copy(), fission_mean.copy(), fission_second.copy(), .001, .0065, 8800., alpha.copy(), decay.copy(), rho_s.copy(), tc_s.copy(), te_s.copy())',
            'gold_call': '_oracle_evaluate_moment_rhs(0., state.copy(), fission_mean.copy(), fission_second.copy(), .001, .0065, 8800., alpha.copy(), decay.copy(), rho_s.copy(), tc_s.copy(), te_s.copy())',
            'tol': 1e-08,
        },
        {
            'setup': 'import numpy as np\np = np.array([.027, .158, .339, .305, .133, .038])\nalpha = np.array([.033, .219, .196, .395, .115, .042])\ndecay = np.array([.0124, .0305, .111, .301, 1.14, 3.01])\nrho_s = np.array([-.004, .002, 4.])\ntc_s = np.array([50., 50., 1.])\nte_s = np.array([200., 20., 8.])\n\nmean = np.arange(1., 14.)\nfission_mean = np.r_[1.473, alpha * .0065 * 2.473 / .9935]\nfission_second = np.outer(fission_mean, fission_mean)\nfission_second[0, 0] = p @ (np.arange(6) - 1.) ** 2\nfission_second[np.arange(1, 7), np.arange(1, 7)] += fission_mean[1:]\n\nv = np.arange(1., 14.) / 13\nstate = np.r_[mean, np.outer(v,v).ravel(), (np.eye(13)+np.outer(v,v)).ravel()]\n',
            'call': 'evaluate_moment_rhs(12., state.copy(), fission_mean.copy(), fission_second.copy(), .001, .0065, 8800., alpha.copy(), decay.copy(), rho_s.copy(), tc_s.copy(), te_s.copy())',
            'gold_call': '_oracle_evaluate_moment_rhs(12., state.copy(), fission_mean.copy(), fission_second.copy(), .001, .0065, 8800., alpha.copy(), decay.copy(), rho_s.copy(), tc_s.copy(), te_s.copy())',
        },
    ]
