"""
Orchestrator: call prior sub-problems and return the weighted Fx percent increase

**End-to-end SD-SCAN exchange proxy for beta-SiC: compute the material**



 **descriptors from the electronegativities and lattice constant, use them to**



 **condition the grid weights, relax the weighted mean relative percent change**



 **in Fx over the supplied (s, alpha) grid for the SD-SCAN and default-SCAN**



 **parameter pairs, then calibrate a kappa reproducing that metric and return**



 **the metric recomputed at the calibrated kappa. The returned scalar is the task**



 **final answer.**

Returns
-------
float, relaxed descriptor-conditioned weighted mean percent increase in Fx
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def orchestrate_sd_scan_exchange_proxy(
    electronegativity_si: float,
    electronegativity_c: float,
    lattice_a: float,
    s_grid: "np.ndarray",
    alpha_grid: "np.ndarray",
    weight_matrix: "np.ndarray",
    kappa_sd: float,
    c1x_sd: float,
    kappa_default: float,
    c1x_default: float,
) -> float:
    """Run the SD-SCAN exchange enhancement pipeline and return percent increase.

    Parameters
    ----------
    electronegativity_si : float
        Pauling electronegativity of Si.
    electronegativity_c : float
        Pauling electronegativity of C.
    lattice_a : float
        Zincblende lattice constant a in angstrom.
    s_grid : "np.ndarray"
        Dimensionless density-gradient grid.
    alpha_grid : "np.ndarray"
        Bond-strength indicator grid.
    weight_matrix : "np.ndarray"
        Weight matrix over the (s, alpha) grid.
    kappa_sd : float
        SD-SCAN kappa parameter.
    c1x_sd : float
        SD-SCAN c1x parameter.
    kappa_default : float
        Default SCAN kappa parameter.
    c1x_default : float
        Default SCAN c1x parameter.

    Returns
    -------
    percent_increase : float
        Relaxed descriptor-conditioned weighted mean percent increase in Fx.
        Compute the forward metric at the supplied kappa, calibrate a kappa
        reproducing that target using the preceding calibration step, then
        recompute and return the metric at the calibrated kappa. A different
        calibrated kappa is acceptable when the inverse is nonunique.
        The forward target must be bracketed by kappa in [0.01, 0.5].
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_orchestrate_sd_scan_exchange_proxy(
    electronegativity_si: float,
    electronegativity_c: float,
    lattice_a: float,
    s_grid: "np.ndarray",
    alpha_grid: "np.ndarray",
    weight_matrix: "np.ndarray",
    kappa_sd: float,
    c1x_sd: float,
    kappa_default: float,
    c1x_default: float,
) -> float:
    import numpy as np

    descriptors = _oracle_compute_material_descriptors(
        electronegativity_si, electronegativity_c, lattice_a
    )
    percent_increase = _oracle_compute_weighted_fx_percent_increase(
        s_grid,
        alpha_grid,
        weight_matrix,
        descriptors,
        kappa_sd,
        c1x_sd,
        kappa_default,
        c1x_default,
    )
    calibrated_kappa = _oracle_calibrate_kappa_to_target(
        percent_increase,
        s_grid,
        alpha_grid,
        weight_matrix,
        descriptors,
        c1x_sd,
        kappa_default,
        c1x_default,
    )
    return _oracle_compute_weighted_fx_percent_increase(
        s_grid,
        alpha_grid,
        weight_matrix,
        descriptors,
        calibrated_kappa,
        c1x_sd,
        kappa_default,
        c1x_default,
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
electronegativity_si, electronegativity_c, lattice_a = 1.90, 2.55, 4.35
s_grid = np.array([0.75, 1.0, 1.25])
alpha_grid = np.array([0.15, 0.30, 0.45])
weight_matrix = np.array([[1,2,3],[2,3,4],[3,4,5]], dtype=float)
kappa_sd, c1x_sd = 0.15, 0.20
kappa_default, c1x_default = 0.065, 0.667""",
            "call": "orchestrate_sd_scan_exchange_proxy(electronegativity_si, electronegativity_c, lattice_a, s_grid.copy(), alpha_grid.copy(), weight_matrix.copy(), kappa_sd, c1x_sd, kappa_default, c1x_default)",
            "gold_call": "_oracle_orchestrate_sd_scan_exchange_proxy(electronegativity_si, electronegativity_c, lattice_a, s_grid.copy(), alpha_grid.copy(), weight_matrix.copy(), kappa_sd, c1x_sd, kappa_default, c1x_default)",
            "tol": 1e-08,
        },
        {
            "setup": """import numpy as np
electronegativity_si, electronegativity_c, lattice_a = 1.90, 2.55, 4.35
s_grid = np.array([1.0])
alpha_grid = np.array([0.30])
weight_matrix = np.array([[1.0]])
kappa_sd, c1x_sd = 0.15, 0.20
kappa_default, c1x_default = 0.065, 0.667""",
            "call": "orchestrate_sd_scan_exchange_proxy(electronegativity_si, electronegativity_c, lattice_a, s_grid.copy(), alpha_grid.copy(), weight_matrix.copy(), kappa_sd, c1x_sd, kappa_default, c1x_default)",
            "gold_call": "_oracle_orchestrate_sd_scan_exchange_proxy(electronegativity_si, electronegativity_c, lattice_a, s_grid.copy(), alpha_grid.copy(), weight_matrix.copy(), kappa_sd, c1x_sd, kappa_default, c1x_default)",
            "tol": 1e-08,
        },
        {
            "setup": """import numpy as np
electronegativity_si, electronegativity_c, lattice_a = 1.81, 2.18, 5.653
s_grid = np.array([1.0])
alpha_grid = np.array([0.35])
weight_matrix = np.array([[1.0]])
kappa_sd, c1x_sd = 0.027, 0.15
kappa_default, c1x_default = 0.065, 0.667""",
            "call": "orchestrate_sd_scan_exchange_proxy(electronegativity_si, electronegativity_c, lattice_a, s_grid.copy(), alpha_grid.copy(), weight_matrix.copy(), kappa_sd, c1x_sd, kappa_default, c1x_default)",
            "gold_call": "_oracle_orchestrate_sd_scan_exchange_proxy(electronegativity_si, electronegativity_c, lattice_a, s_grid.copy(), alpha_grid.copy(), weight_matrix.copy(), kappa_sd, c1x_sd, kappa_default, c1x_default)",
            "tol": 1e-08,
        },
        {
            "setup": """import numpy as np
electronegativity_si, electronegativity_c, lattice_a = 2.55, 2.55, 3.567
s_grid = np.array([0.5, 1.5, 3.0])
alpha_grid = np.array([0.0, 0.5, 1.0])
weight_matrix = np.array([[5,1,1],[1,5,1],[1,1,5]], dtype=float)
kappa_sd, c1x_sd = 0.5, 0.05
kappa_default, c1x_default = 0.065, 0.667""",
            "call": "orchestrate_sd_scan_exchange_proxy(electronegativity_si, electronegativity_c, lattice_a, s_grid.copy(), alpha_grid.copy(), weight_matrix.copy(), kappa_sd, c1x_sd, kappa_default, c1x_default)",
            "gold_call": "_oracle_orchestrate_sd_scan_exchange_proxy(electronegativity_si, electronegativity_c, lattice_a, s_grid.copy(), alpha_grid.copy(), weight_matrix.copy(), kappa_sd, c1x_sd, kappa_default, c1x_default)",
            "tol": 1e-08,
        },
        {
            "setup": """import numpy as np
electronegativity_si, electronegativity_c, lattice_a = 0.79, 3.98, 6.2
s_grid = np.array([0.25, 2.0])
alpha_grid = np.array([0.05, 0.95])
weight_matrix = np.array([[1.0, 7.0],[3.0, 0.0]])
kappa_sd, c1x_sd = 0.30, 0.10
kappa_default, c1x_default = 0.065, 0.667""",
            "call": "orchestrate_sd_scan_exchange_proxy(electronegativity_si, electronegativity_c, lattice_a, s_grid.copy(), alpha_grid.copy(), weight_matrix.copy(), kappa_sd, c1x_sd, kappa_default, c1x_default)",
            "gold_call": "_oracle_orchestrate_sd_scan_exchange_proxy(electronegativity_si, electronegativity_c, lattice_a, s_grid.copy(), alpha_grid.copy(), weight_matrix.copy(), kappa_sd, c1x_sd, kappa_default, c1x_default)",
            "tol": 1e-08,
        },
    ]
