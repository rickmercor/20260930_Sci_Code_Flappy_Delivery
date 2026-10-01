"""
Implement build_mean_drift which constructs the linear mean-population operator for two perfectly mixed fuel regions.

Delayed-neutron precursors circulate with the fuel and can decay in either
region. Only decay in the core contributes a neutron to the modeled neutron
population. Precursor transport couples the two regional inventories.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def build_mean_drift(rho: float, tau_core: float, tau_excore: float, generation_time: float, beta: float, alpha: "np.ndarray", decay: "np.ndarray") -> "np.ndarray":
    """Construct the homogeneous drift operator of the source's six-group model.
    
    Parameters
    ----------
    rho : float
        Reactivity at the current time.
    tau_core, tau_excore : float
        Positive residence times in seconds; positive infinity disables that transfer.
    generation_time : float
        Positive prompt neutron generation time in seconds.
    beta : float
        Total delayed fraction, 0 <= beta < 1.
    alpha : np.ndarray
        Shape (6,), nonnegative relative group fractions summing to one.
    decay : np.ndarray
        Shape (6,), nonnegative group decay constants in inverse seconds.
    
    Returns
    -------
    drift : np.ndarray
        Shape (13, 13), A such that dm/dt = A m + S e_n.
        State order is neutron, core groups 1-6, ex-core groups 1-6;
        e_n selects the neutron coordinate. The external source is separate.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_build_mean_drift(rho: float, tau_core: float, tau_excore: float, generation_time: float, beta: float, alpha: "np.ndarray", decay: "np.ndarray") -> "np.ndarray":
    """Implement build_mean_drift which constructs the linear mean-population operator for two perfectly mixed fuel regions."""
    alpha = np.asarray(alpha, dtype=float)
    decay = np.asarray(decay, dtype=float)
    drift = np.zeros((13, 13))
    drift[0, 0] = (rho - beta) / generation_time
    drift[0, 1:7] = decay
    drift[1:7, 0] = beta * alpha / generation_time
    core = np.arange(1, 7)
    excore = np.arange(7, 13)
    drift[core, core] = -decay - 1.0 / tau_core
    drift[excore, excore] = -decay - 1.0 / tau_excore
    drift[core, excore] = 1.0 / tau_excore
    drift[excore, core] = 1.0 / tau_core
    return drift

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return independent candidate/oracle comparisons."""
    return [
        {
            'setup': 'import numpy as np\np = np.array([.027, .158, .339, .305, .133, .038])\nalpha = np.array([.033, .219, .196, .395, .115, .042])\ndecay = np.array([.0124, .0305, .111, .301, 1.14, 3.01])\n',
            'call': 'build_mean_drift(-.005, 10., 15., .001, .0065, alpha.copy(), decay.copy())',
            'gold_call': '_oracle_build_mean_drift(-.005, 10., 15., .001, .0065, alpha.copy(), decay.copy())',
            'tol': 1e-08,
        },
        {
            'setup': 'import numpy as np\np = np.array([.027, .158, .339, .305, .133, .038])\nalpha = np.array([.033, .219, .196, .395, .115, .042])\ndecay = np.array([.0124, .0305, .111, .301, 1.14, 3.01])\n',
            'call': 'build_mean_drift(0., float("inf"), float("inf"), .001, 0., alpha.copy(), decay.copy())',
            'gold_call': '_oracle_build_mean_drift(0., float("inf"), float("inf"), .001, 0., alpha.copy(), decay.copy())',
            'tol': 1e-08,
        },
        {
            'setup': 'import numpy as np\np = np.array([.027, .158, .339, .305, .133, .038])\nalpha = np.array([.033, .219, .196, .395, .115, .042])\ndecay = np.array([.0124, .0305, .111, .301, 1.14, 3.01])\n\ndecay = np.zeros(6)\n',
            'call': 'build_mean_drift(.003, 2., 4., .002, .01, alpha.copy(), decay.copy())',
            'gold_call': '_oracle_build_mean_drift(.003, 2., 4., .002, .01, alpha.copy(), decay.copy())',
            'tol': 1e-08,
        },
    ]
