"""
Infer the physical plasma temperature from the two-branch thermal calibration and use that temperature to evaluate the complete exclusive shower-history probability.

The plasma temperature is determined before evaluating the requested shower history. The channel-balance measurement produces two mathematical temperature branches, while the independent Sudakov survival measurement selects the physical branch.

Using that calibrated temperature, evaluate the parent total hazard, the designated splitting rate, the two daughter total hazards, and finally the exclusive probability that the designated split occurs while no competing interaction occurs before or after it.

Returns
-------
float, the dimensionless exclusive-history probability obtained using the physically selected temperature branch
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def solve_calibrated_thermal_shower_case(
    p0: float,
    length: float,
    alpha_s: float,
    c_a: float,
    qhat: float,
    z_nodes: "np.ndarray",
    weights: "np.ndarray",
    split_index: int,
    balance_observed: float,
    survival_observed: float,
    calibration_length: float,
    temperature_bounds: "np.ndarray",
) -> float:
    """Solve the calibrated finite-temperature exclusive shower case.

    Parameters
    ----------
    p0 : float
        Initial parent momentum in GeV.
    length : float
        Shower propagation length in GeV^(-1).
    alpha_s : float
        Strong coupling.
    c_a : float
        Gluon color factor.
    qhat : float
        Transverse-momentum broadening coefficient in GeV^3.
    z_nodes : np.ndarray
        One-dimensional quadrature nodes.
    weights : np.ndarray
        One-dimensional quadrature weights.
    split_index : int
        Zero-based index of the designated quadrature splitting node.
    balance_observed : float
        Observed splitting-minus-merging hazard in GeV.
    survival_observed : float
        Independent calibration Sudakov survival probability.
    calibration_length : float
        Calibration propagation length in GeV^(-1).
    temperature_bounds : np.ndarray
        Two-element calibration temperature interval in GeV.

    Returns
    -------
    probability : float
        Dimensionless exclusive-history probability after selecting the
        physical calibration branch.
    """
    return probability

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_solve_calibrated_thermal_shower_case(
    p0: float,
    length: float,
    alpha_s: float,
    c_a: float,
    qhat: float,
    z_nodes: "np.ndarray",
    weights: "np.ndarray",
    split_index: int,
    balance_observed: float,
    survival_observed: float,
    calibration_length: float,
    temperature_bounds: "np.ndarray",
) -> float:
    temperature_branches = _oracle_infer_temperature_branches(
        p0,
        alpha_s,
        c_a,
        qhat,
        z_nodes,
        weights,
        balance_observed,
        survival_observed,
        calibration_length,
        temperature_bounds,
    )

    temperature = float(temperature_branches[2])

    parent_hazards = _oracle_integrate_inelastic_hazard(
        p0,
        temperature,
        alpha_s,
        c_a,
        qhat,
        z_nodes,
        weights,
    )
    parent_hazard = float(parent_hazards[2])

    z_star = float(z_nodes[split_index])
    weight_star = float(weights[split_index])

    split_rate = _oracle_compute_designated_split_rate(
        z_star,
        weight_star,
        p0,
        temperature,
        alpha_s,
        c_a,
        qhat,
    )

    daughter_state = _oracle_compute_daughter_shower_state(
        z_star,
        p0,
        temperature,
        alpha_s,
        c_a,
        qhat,
        z_nodes,
        weights,
    )

    daughter_hazard_sum = float(daughter_state[4])

    segment_hazards = np.array(
        [parent_hazard, daughter_hazard_sum],
        dtype=float,
    )
    split_rates = np.array(
        [split_rate],
        dtype=float,
    )

    return _oracle_integrate_exclusive_history_probability(
        segment_hazards,
        split_rates,
        length,
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
p0 = 3.0
length = 2.0
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
split_index = 1
balance_observed = 0.05369259190766756
survival_observed = 0.9286581077917495
calibration_length = 1.0
temperature_bounds = np.array([0.2, 0.8], dtype=float)""",
            "call": "solve_calibrated_thermal_shower_case(p0, length, alpha_s, c_a, qhat, z_nodes.copy(), weights.copy(), split_index, balance_observed, survival_observed, calibration_length, temperature_bounds.copy())",
            "gold_call": "_oracle_solve_calibrated_thermal_shower_case(p0, length, alpha_s, c_a, qhat, z_nodes.copy(), weights.copy(), split_index, balance_observed, survival_observed, calibration_length, temperature_bounds.copy())",
        },
        {
            "setup": """import numpy as np
p0 = 8.0
length = 1.1
alpha_s = 0.250
c_a = 3.0
qhat = 0.0100
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
split_index = 0
balance_observed = 0.02193140201836728
survival_observed = 0.9783052878156464
calibration_length = 1.0
temperature_bounds = np.array([0.1, 2.0], dtype=float)""",
            "call": "solve_calibrated_thermal_shower_case(p0, length, alpha_s, c_a, qhat, z_nodes.copy(), weights.copy(), split_index, balance_observed, survival_observed, calibration_length, temperature_bounds.copy())",
            "gold_call": "_oracle_solve_calibrated_thermal_shower_case(p0, length, alpha_s, c_a, qhat, z_nodes.copy(), weights.copy(), split_index, balance_observed, survival_observed, calibration_length, temperature_bounds.copy())",
        },
        {
            "setup": """import numpy as np
p0 = 4.0
length = 0.75
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
split_index = 3
balance_observed = 0.04653681325033222
survival_observed = 0.5500762863533156
calibration_length = 10.0
temperature_bounds = np.array([0.5, 0.7], dtype=float)""",
            "call": "solve_calibrated_thermal_shower_case(p0, length, alpha_s, c_a, qhat, z_nodes.copy(), weights.copy(), split_index, balance_observed, survival_observed, calibration_length, temperature_bounds.copy())",
            "gold_call": "_oracle_solve_calibrated_thermal_shower_case(p0, length, alpha_s, c_a, qhat, z_nodes.copy(), weights.copy(), split_index, balance_observed, survival_observed, calibration_length, temperature_bounds.copy())",
        },
    ]
