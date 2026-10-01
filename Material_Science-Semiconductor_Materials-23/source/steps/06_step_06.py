"""
Calibrate the SD-SCAN kappa that reproduces a target weighted Fx metric.

*This step defines an inverse consistency check for the task's weighted exchange*

*proxy: vary kappa with the other inputs fixed until the target metric is*

*reproduced within the admissible bracket. The published SD-SCAN fit instead*

*uses bandgap and lattice data and allows four SCAN parameters to vary.*

Returns
-------
float, calibrated SD-SCAN kappa reproducing the target weighted Fx metric
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def calibrate_kappa_to_target(
    target_percent_increase: float,
    s_grid: "np.ndarray",
    alpha_grid: "np.ndarray",
    weight_matrix: "np.ndarray",
    descriptors: "np.ndarray",
    c1x_sd: float,
    kappa_default: float,
    c1x_default: float,
    kappa_low: float = 0.01,
    kappa_high: float = 0.5,
) -> float:
    """Solve for the SD-SCAN kappa reproducing a target weighted Fx metric.

    Parameters
    ----------
    target_percent_increase : float
        Target weighted mean relative percent increase in Fx.
    s_grid : "np.ndarray"
        Shape (n_s,) dimensionless density gradients.
    alpha_grid : "np.ndarray"
        Shape (n_a,) bond-strength indicators in [0, 1].
    weight_matrix : "np.ndarray"
        Shape (n_s, n_a) nonnegative weights.
    descriptors : "np.ndarray"
        Shape (4,) material descriptors conditioning the weighted metric.
    c1x_sd : float
        SD-SCAN c1x parameter, held fixed during calibration.
    kappa_default : float
        Default SCAN kappa parameter.
    c1x_default : float
        Default SCAN c1x parameter.
    kappa_low : float
        Lower end of the admissible kappa bracket.
    kappa_high : float
        Upper end of the admissible kappa bracket.

    Returns
    -------
    kappa_sd : float
        A kappa reproducing the target, converged to a bracket width of
        1e-12. The inputs must bracket a solution. When the metric is
        constant over the bracket, any kappa in the bracket is valid.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_calibrate_kappa_to_target(
    target_percent_increase: float,
    s_grid: "np.ndarray",
    alpha_grid: "np.ndarray",
    weight_matrix: "np.ndarray",
    descriptors: "np.ndarray",
    c1x_sd: float,
    kappa_default: float,
    c1x_default: float,
    kappa_low: float = 0.01,
    kappa_high: float = 0.5,
) -> float:
    if kappa_low <= 0.0 or kappa_high <= kappa_low:
        raise ValueError("require 0 < kappa_low < kappa_high")

    def _residual(kappa: float) -> float:
        return (
            _oracle_compute_weighted_fx_percent_increase(
                s_grid,
                alpha_grid,
                weight_matrix,
                descriptors,
                kappa,
                c1x_sd,
                kappa_default,
                c1x_default,
            )
            - target_percent_increase
        )

    low, high = float(kappa_low), float(kappa_high)
    r_low, r_high = _residual(low), _residual(high)
    if r_low == 0.0:
        return low
    if r_high == 0.0:
        return high
    if r_low * r_high > 0.0:
        raise ValueError("target is not bracketed by [kappa_low, kappa_high]")

    for _ in range(200):
        mid = 0.5 * (low + high)
        r_mid = _residual(mid)
        if r_mid == 0.0 or (high - low) < 1e-12:
            break
        if r_low * r_mid < 0.0:
            high = mid
            r_high = r_mid
        else:
            low = mid
            r_low = r_mid
    return float(0.5 * (low + high))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
target_percent_increase = 2.4569242254305657
s_grid = np.array([0.75, 1.0, 1.25])
alpha_grid = np.array([0.15, 0.30, 0.45])
weight_matrix = np.array([[1,2,3],[2,3,4],[3,4,5]], dtype=float)
descriptors = _oracle_compute_material_descriptors(1.90, 2.55, 4.35)
c1x_sd, kappa_default, c1x_default = 0.20, 0.065, 0.667""",
            "call": "calibrate_kappa_to_target(target_percent_increase, s_grid.copy(), alpha_grid.copy(), weight_matrix.copy(), descriptors.copy(), c1x_sd, kappa_default, c1x_default)",
            "gold_call": "_oracle_calibrate_kappa_to_target(target_percent_increase, s_grid.copy(), alpha_grid.copy(), weight_matrix.copy(), descriptors.copy(), c1x_sd, kappa_default, c1x_default)",
            "tol": 1e-08,
        },
        {
            "setup": """import numpy as np
target_percent_increase = 2.2
s_grid = np.array([0.75, 1.0, 1.25])
alpha_grid = np.array([0.15, 0.30, 0.45])
weight_matrix = np.array([[1,2,3],[2,3,4],[3,4,5]], dtype=float)
descriptors = _oracle_compute_material_descriptors(1.90, 2.55, 4.35)
c1x_sd, kappa_default, c1x_default = 0.20, 0.065, 0.667""",
            "call": "calibrate_kappa_to_target(target_percent_increase, s_grid.copy(), alpha_grid.copy(), weight_matrix.copy(), descriptors.copy(), c1x_sd, kappa_default, c1x_default)",
            "gold_call": "_oracle_calibrate_kappa_to_target(target_percent_increase, s_grid.copy(), alpha_grid.copy(), weight_matrix.copy(), descriptors.copy(), c1x_sd, kappa_default, c1x_default)",
            "tol": 1e-08,
        },
        {
            "setup": """import numpy as np
target_percent_increase = 2.1
s_grid = np.array([1.0])
alpha_grid = np.array([0.30])
weight_matrix = np.array([[1.0]])
descriptors = _oracle_compute_material_descriptors(1.90, 2.55, 4.35)
c1x_sd, kappa_default, c1x_default = 0.20, 0.065, 0.667""",
            "call": "calibrate_kappa_to_target(target_percent_increase, s_grid.copy(), alpha_grid.copy(), weight_matrix.copy(), descriptors.copy(), c1x_sd, kappa_default, c1x_default)",
            "gold_call": "_oracle_calibrate_kappa_to_target(target_percent_increase, s_grid.copy(), alpha_grid.copy(), weight_matrix.copy(), descriptors.copy(), c1x_sd, kappa_default, c1x_default)",
            "tol": 1e-08,
        },
        {
            "setup": """import numpy as np
target_percent_increase = 1.85
s_grid = np.array([0.75, 1.0, 1.25])
alpha_grid = np.array([0.15, 0.30, 0.45])
weight_matrix = np.array([[1,2,3],[2,3,4],[3,4,5]], dtype=float)
descriptors = _oracle_compute_material_descriptors(1.90, 2.55, 4.35)
c1x_sd, kappa_default, c1x_default = 0.20, 0.065, 0.667""",
            "call": "calibrate_kappa_to_target(target_percent_increase, s_grid.copy(), alpha_grid.copy(), weight_matrix.copy(), descriptors.copy(), c1x_sd, kappa_default, c1x_default)",
            "gold_call": "_oracle_calibrate_kappa_to_target(target_percent_increase, s_grid.copy(), alpha_grid.copy(), weight_matrix.copy(), descriptors.copy(), c1x_sd, kappa_default, c1x_default)",
            "tol": 1e-08,
        },
        {
            "setup": """import numpy as np
target_percent_increase = 2.75
s_grid = np.array([0.75, 1.0, 1.25])
alpha_grid = np.array([0.15, 0.30, 0.45])
weight_matrix = np.array([[1,2,3],[2,3,4],[3,4,5]], dtype=float)
descriptors = _oracle_compute_material_descriptors(1.90, 2.55, 4.35)
c1x_sd, kappa_default, c1x_default = 0.20, 0.065, 0.667""",
            "call": "calibrate_kappa_to_target(target_percent_increase, s_grid.copy(), alpha_grid.copy(), weight_matrix.copy(), descriptors.copy(), c1x_sd, kappa_default, c1x_default)",
            "gold_call": "_oracle_calibrate_kappa_to_target(target_percent_increase, s_grid.copy(), alpha_grid.copy(), weight_matrix.copy(), descriptors.copy(), c1x_sd, kappa_default, c1x_default)",
            "tol": 1e-08,
        },
        {'setup': 'import numpy as np\n'
                  's_grid = np.array([0.75, 1.0, 1.25])\n'
                  'alpha_grid = np.array([0.15, 0.30, 0.45])\n'
                  'weight_matrix = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]], dtype=float)\n'
                  'descriptors = _oracle_compute_material_descriptors(1.90, 2.55, 4.35)\n'
                  'c1x_sd, kappa_default, c1x_default = 0.20, 0.065, 0.667\n'
                  'target_percent_increase = _oracle_compute_weighted_fx_percent_increase(\n'
                  '    s_grid, alpha_grid, weight_matrix, descriptors,\n'
                  '    0.01, c1x_sd, kappa_default, c1x_default)',
         'call': 'calibrate_kappa_to_target(target_percent_increase, s_grid.copy(), alpha_grid.copy(), '
                 'weight_matrix.copy(), descriptors.copy(), c1x_sd, kappa_default, c1x_default)',
         'gold_call': '_oracle_calibrate_kappa_to_target(target_percent_increase, s_grid.copy(), alpha_grid.copy(), '
                      'weight_matrix.copy(), descriptors.copy(), c1x_sd, kappa_default, c1x_default)',
         'tol': 1e-08},
        {'setup': 'import numpy as np\n'
                  's_grid = np.array([0.75, 1.0, 1.25])\n'
                  'alpha_grid = np.array([0.15, 0.30, 0.45])\n'
                  'weight_matrix = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]], dtype=float)\n'
                  'descriptors = _oracle_compute_material_descriptors(1.90, 2.55, 4.35)\n'
                  'c1x_sd, kappa_default, c1x_default = 0.20, 0.065, 0.667\n'
                  'target_percent_increase = _oracle_compute_weighted_fx_percent_increase(\n'
                  '    s_grid, alpha_grid, weight_matrix, descriptors,\n'
                  '    0.5, c1x_sd, kappa_default, c1x_default)',
         'call': 'calibrate_kappa_to_target(target_percent_increase, s_grid.copy(), alpha_grid.copy(), '
                 'weight_matrix.copy(), descriptors.copy(), c1x_sd, kappa_default, c1x_default)',
         'gold_call': '_oracle_calibrate_kappa_to_target(target_percent_increase, s_grid.copy(), alpha_grid.copy(), '
                      'weight_matrix.copy(), descriptors.copy(), c1x_sd, kappa_default, c1x_default)',
         'tol': 1e-08},
    ]
