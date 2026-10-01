"""
Recover both temperature branches consistent with a finite-temperature splitting-minus-merging calibration observable, then use an independent Sudakov survival measurement to select the physical branch.

At fixed parent momentum, define the channel-balance observable

$$

B(T)=\lambda_{\rm split}(T)-\lambda_{\rm merge}(T).

$$

Because the splitting and merging channels have different thermal dependences, $B(T)$ need not be monotonic. For the supplied calibration intervals, the equation

$$

B(T)=B_{\rm obs}

$$

has exactly two distinct solutions separated by one interior maximum of $B(T)$.

An independent no-interaction measurement provides the Sudakov survival

$$

S(T)=\exp[-(\lambda_{\rm split}(T)+\lambda_{\rm merge}(T))L_{\rm cal}].

$$

Recover both temperature solutions inside the supplied interval, order them from low to high temperature, evaluate the survival for each branch, and select the branch whose survival is closest to $S_{\rm obs}$.

Returns
-------
np.ndarray of shape (3,), containing the lower calibration root, upper calibration root, and selected physical temperature in GeV
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def infer_temperature_branches(
    momentum: float,
    alpha_s: float,
    c_a: float,
    qhat: float,
    z_nodes: "np.ndarray",
    weights: "np.ndarray",
    balance_observed: float,
    survival_observed: float,
    calibration_length: float,
    temperature_bounds: "np.ndarray",
) -> "np.ndarray":
    """Infer the two calibration temperature branches and select the physical one.

    Parameters
    ----------
    momentum : float
        Parent momentum in GeV.
    alpha_s : float
        Strong coupling.
    c_a : float
        Gluon color factor.
    qhat : float
        Transverse-momentum broadening coefficient in GeV^3.
    z_nodes : np.ndarray
        One-dimensional quadrature nodes.
    weights : np.ndarray
        One-dimensional quadrature weights corresponding to z_nodes.
    balance_observed : float
        Observed splitting-minus-merging hazard in GeV.
    survival_observed : float
        Observed no-interaction Sudakov survival probability.
    calibration_length : float
        Calibration propagation length in GeV^(-1).
    temperature_bounds : np.ndarray
        Two-element array [T_min, T_max] in GeV. For the supported
        inputs, the interval contains one interior maximum of the
        balance observable and exactly two distinct calibration roots.

    Returns
    -------
    result : np.ndarray
        Array of shape (3,) containing the lower temperature root,
        the upper temperature root, and the selected physical
        temperature, all in GeV.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import brentq, minimize_scalar


def _oracle_infer_temperature_branches(
    momentum: float,
    alpha_s: float,
    c_a: float,
    qhat: float,
    z_nodes: "np.ndarray",
    weights: "np.ndarray",
    balance_observed: float,
    survival_observed: float,
    calibration_length: float,
    temperature_bounds: "np.ndarray",
) -> "np.ndarray":
    z_nodes = np.asarray(z_nodes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    temperature_bounds = np.asarray(temperature_bounds, dtype=float)

    t_min = float(temperature_bounds[0])
    t_max = float(temperature_bounds[1])

    def _channel_data(temperature):
        hazards = _oracle_integrate_inelastic_hazard(
            momentum,
            float(temperature),
            alpha_s,
            c_a,
            qhat,
            z_nodes,
            weights,
        )
        balance = float(hazards[0] - hazards[1])
        total_hazard = float(hazards[2])
        return balance, total_hazard

    peak_result = minimize_scalar(
        lambda temperature: -_channel_data(temperature)[0],
        bounds=(t_min, t_max),
        method="bounded",
        options={"xatol": 1e-14, "maxiter": 1000},
    )
    t_peak = float(peak_result.x)

    def _residual(temperature):
        return _channel_data(temperature)[0] - balance_observed

    t_low = brentq(
        _residual,
        t_min,
        t_peak,
        xtol=5e-15,
        rtol=1e-14,
        maxiter=500,
    )

    t_high = brentq(
        _residual,
        t_peak,
        t_max,
        xtol=5e-15,
        rtol=1e-14,
        maxiter=500,
    )

    roots = np.array([t_low, t_high], dtype=float)

    survivals = np.empty(2, dtype=float)
    for i, temperature in enumerate(roots):
        total_hazard = _channel_data(float(temperature))[1]
        survivals[i] = np.exp(-total_hazard * calibration_length)

    selected_index = int(
        np.argmin(np.abs(survivals - survival_observed))
    )
    selected_temperature = float(roots[selected_index])

    return np.array(
        [roots[0], roots[1], selected_temperature],
        dtype=float,
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
momentum = 3.0
alpha_s = 0.300
c_a = 3.0
qhat = 0.0150
z_nodes = np.array([
    0.228146046218401,
    0.338459206968295,
    0.500000000000000,
    0.661540793031705,
    0.771853953781599,
], dtype=float)
weights = np.array([
    0.071078065516857,
    0.143588601149810,
    0.170666666666667,
    0.143588601149810,
    0.071078065516857,
], dtype=float)
balance_observed = 0.05369259190766756
survival_observed = 0.9286581077917495
calibration_length = 1.0
temperature_bounds = np.array([0.2, 0.8], dtype=float)""",
            "call": "infer_temperature_branches(momentum, alpha_s, c_a, qhat, z_nodes.copy(), weights.copy(), balance_observed, survival_observed, calibration_length, temperature_bounds.copy())",
            "gold_call": "_oracle_infer_temperature_branches(momentum, alpha_s, c_a, qhat, z_nodes.copy(), weights.copy(), balance_observed, survival_observed, calibration_length, temperature_bounds.copy())",
            "tol": 1e-9,
        },
        {
            "setup": """import numpy as np
momentum = 4.0
alpha_s = 0.300
c_a = 3.0
qhat = 0.0150
z_nodes = np.array([
    0.228146046218401,
    0.338459206968295,
    0.500000000000000,
    0.661540793031705,
    0.771853953781599,
], dtype=float)
weights = np.array([
    0.071078065516857,
    0.143588601149810,
    0.170666666666667,
    0.143588601149810,
    0.071078065516857,
], dtype=float)
balance_observed = 0.04646042843157781
survival_observed = 0.9471547891394752
calibration_length = 1.0
temperature_bounds = np.array([0.3, 0.9], dtype=float)""",
            "call": "infer_temperature_branches(momentum, alpha_s, c_a, qhat, z_nodes.copy(), weights.copy(), balance_observed, survival_observed, calibration_length, temperature_bounds.copy())",
            "gold_call": "_oracle_infer_temperature_branches(momentum, alpha_s, c_a, qhat, z_nodes.copy(), weights.copy(), balance_observed, survival_observed, calibration_length, temperature_bounds.copy())",
            "tol": 1e-9,
        },
        {
            "setup": """import numpy as np
momentum = 4.0
alpha_s = 0.300
c_a = 3.0
qhat = 0.0150
z_nodes = np.array([
    0.228146046218401,
    0.338459206968295,
    0.500000000000000,
    0.661540793031705,
    0.771853953781599,
], dtype=float)
weights = np.array([
    0.071078065516857,
    0.143588601149810,
    0.170666666666667,
    0.143588601149810,
    0.071078065516857,
], dtype=float)
balance_observed = 0.04653681325033222
survival_observed = 0.5500762863533156
calibration_length = 10.0
temperature_bounds = np.array([0.5, 0.7], dtype=float)""",
            "call": "infer_temperature_branches(momentum, alpha_s, c_a, qhat, z_nodes.copy(), weights.copy(), balance_observed, survival_observed, calibration_length, temperature_bounds.copy())",
            "gold_call": "_oracle_infer_temperature_branches(momentum, alpha_s, c_a, qhat, z_nodes.copy(), weights.copy(), balance_observed, survival_observed, calibration_length, temperature_bounds.copy())",
            "tol": 1e-9,
        },
        {
            "setup": """import numpy as np
momentum = 4.0
alpha_s = 0.300
c_a = 3.0
qhat = 0.0150
z_nodes = np.array([
    0.228146046218401,
    0.338459206968295,
    0.500000000000000,
    0.661540793031705,
    0.771853953781599,
], dtype=float)
weights = np.array([
    0.071078065516857,
    0.143588601149810,
    0.170666666666667,
    0.143588601149810,
    0.071078065516857,
], dtype=float)
balance_observed = 0.04653681324882403
survival_observed = 0.5501097085915446
calibration_length = 10.0
temperature_bounds = np.array([0.5, 0.7], dtype=float)""",
            "call": "infer_temperature_branches(momentum, alpha_s, c_a, qhat, z_nodes.copy(), weights.copy(), balance_observed, survival_observed, calibration_length, temperature_bounds.copy())",
            "gold_call": "_oracle_infer_temperature_branches(momentum, alpha_s, c_a, qhat, z_nodes.copy(), weights.copy(), balance_observed, survival_observed, calibration_length, temperature_bounds.copy())",
            "tol": 1e-9,
        },
        {
            "setup": """import numpy as np
momentum = 2.0
alpha_s = 0.300
c_a = 3.0
qhat = 0.0150
z_nodes = np.array([
    0.228146046218401,
    0.338459206968295,
    0.500000000000000,
    0.661540793031705,
    0.771853953781599,
], dtype=float)
weights = np.array([
    0.071078065516857,
    0.143588601149810,
    0.170666666666667,
    0.143588601149810,
    0.071078065516857,
], dtype=float)
balance_observed = 0.06570496800160187
survival_observed = 0.9260921463222397
calibration_length = 1.0
temperature_bounds = np.array([0.1, 0.5], dtype=float)""",
            "call": "infer_temperature_branches(momentum, alpha_s, c_a, qhat, z_nodes.copy(), weights.copy(), balance_observed, survival_observed, calibration_length, temperature_bounds.copy())",
            "gold_call": "_oracle_infer_temperature_branches(momentum, alpha_s, c_a, qhat, z_nodes.copy(), weights.copy(), balance_observed, survival_observed, calibration_length, temperature_bounds.copy())",
            "tol": 1e-9,
        },
    ]
