"""
Compute the task-defined descriptor-conditioned exchange-enhancement metric.

*This numerical proxy combines the published SCAN exchange enhancement with the*

*explicit replay rule below. It is not a DFT bandgap calculation or a model of*

*band-edge-state redistribution. Its damping term -B*(1-alpha)**2 suppresses*

*points farther from alpha=1; the other terms depend on s and running screening.*

Returns
-------
float, three-sweep descriptor-conditioned weighted mean percent increase in Fx
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_weighted_fx_percent_increase(
    s_grid: "np.ndarray",
    alpha_grid: "np.ndarray",
    weight_matrix: "np.ndarray",
    descriptors: "np.ndarray",
    kappa_sd: float,
    c1x_sd: float,
    kappa_default: float,
    c1x_default: float,
) -> float:
    """Descriptor-conditioned weighted mean percent increase in Fx over default SCAN.

    Parameters
    ----------
    s_grid : "np.ndarray"
        Shape (n_s,) dimensionless density gradients.
    alpha_grid : "np.ndarray"
        Shape (n_a,) bond-strength indicators in [0, 1].
    weight_matrix : "np.ndarray"
        Shape (n_s, n_a) nonnegative weights.
    descriptors : "np.ndarray"
        Shape (4,) material descriptors
        [covalency, bond_length, bond_strength, ionic_density].
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
        Descriptor-conditioned weighted mean relative percent increase in Fx
        over the (s, alpha) grid, relative to the default-SCAN reference.

    Notes
    -----
    Rows of weight_matrix index s_grid and columns index alpha_grid. Let
    C = descriptors[0], B = descriptors[2], rho = descriptors[3], and
    q_ij = (Fx_SD(s_i, alpha_j) - Fx_def(s_i, alpha_j)) / Fx_def(s_i, alpha_j).
    Initialize lambda = 0 and perform exactly three sweeps. At each sweep,
    reset numerator and denominator to zero while retaining lambda. Traverse
    alpha_grid in the outer loop and s_grid in the inner loop, preserving
    their supplied order. At each point use the current lambda to compute
    W_ij = weight_matrix[i,j] * exp(-B*(1-alpha_j)**2 - rho*s_i**2
                                  - lambda*s_i**2*(1-alpha_j)**2).
    Add W_ij*q_ij and W_ij to the running numerator and denominator, then
    update lambda = C*numerator/denominator if denominator is positive;
    otherwise retain lambda. Return 100*numerator/denominator from sweep 3.
    Inputs must give a positive final effective-weight sum and nonzero
    default enhancement at every grid point.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_compute_weighted_fx_percent_increase(
    s_grid: "np.ndarray",
    alpha_grid: "np.ndarray",
    weight_matrix: "np.ndarray",
    descriptors: "np.ndarray",
    kappa_sd: float,
    c1x_sd: float,
    kappa_default: float,
    c1x_default: float,
) -> float:
    import numpy as np

    s_grid = np.asarray(s_grid, dtype=float)
    alpha_grid = np.asarray(alpha_grid, dtype=float)
    weight_matrix = np.asarray(weight_matrix, dtype=float)
    descriptors = np.asarray(descriptors, dtype=float)
    if descriptors.shape != (4,):
        raise ValueError("descriptors must have shape (4,)")
    if s_grid.ndim != 1 or alpha_grid.ndim != 1:
        raise ValueError("s_grid and alpha_grid must be one-dimensional")
    if weight_matrix.shape != (s_grid.size, alpha_grid.size):
        raise ValueError("weight_matrix shape must match (len(s_grid), len(alpha_grid))")
    if np.any(s_grid <= 0.0):
        raise ValueError("all s values must be positive")
    if np.any(alpha_grid < 0.0) or np.any(alpha_grid > 1.0):
        raise ValueError("alpha values must lie in [0, 1]")
    if np.any(weight_matrix < 0.0):
        raise ValueError("weights must be non-negative")
    if weight_matrix.sum() == 0.0:
        raise ValueError("weight sum must be positive")

    covalency = float(descriptors[0])
    bond_strength = float(descriptors[2])
    ionic_density = float(descriptors[3])

    ratios = {}
    for i, s in enumerate(s_grid):
        for j, alpha in enumerate(alpha_grid):
            fx_def = _oracle_compute_fx_enhancement(s, alpha, kappa_default, c1x_default)
            if fx_def == 0.0:
                raise ValueError("default Fx must be non-zero")
            fx_sd = _oracle_compute_fx_enhancement(s, alpha, kappa_sd, c1x_sd)
            ratios[(i, j)] = (fx_sd - fx_def) / fx_def

    screening = 0.0
    numerator = 0.0
    denominator = 0.0
    for _ in range(3):
        numerator = 0.0
        denominator = 0.0
        for j, alpha in enumerate(alpha_grid):
            for i, s in enumerate(s_grid):
                w = weight_matrix[i, j] * np.exp(
                    -bond_strength * (1.0 - alpha) ** 2
                    - ionic_density * s**2
                    - screening * s**2 * (1.0 - alpha) ** 2
                )
                numerator += w * ratios[(i, j)]
                denominator += w
                if denominator > 0.0:
                    screening = covalency * numerator / denominator
    if denominator <= 0.0:
        raise ValueError("effective weight sum must be positive")
    return float(100.0 * numerator / denominator)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
s_grid = np.array([1.0])
alpha_grid = np.array([0.30])
weight_matrix = np.array([[1.0]])
descriptors = _oracle_compute_material_descriptors(1.90, 2.55, 4.35)
kappa_sd, c1x_sd = 0.15, 0.20
kappa_default, c1x_default = 0.065, 0.667""",
            "call": "compute_weighted_fx_percent_increase(s_grid.copy(), alpha_grid.copy(), weight_matrix.copy(), descriptors.copy(), kappa_sd, c1x_sd, kappa_default, c1x_default)",
            "gold_call": "_oracle_compute_weighted_fx_percent_increase(s_grid.copy(), alpha_grid.copy(), weight_matrix.copy(), descriptors.copy(), kappa_sd, c1x_sd, kappa_default, c1x_default)",
            "tol": 1e-09,
        },
        {
            "setup": """import numpy as np
s_grid = np.array([0.75, 1.0, 1.25])
alpha_grid = np.array([0.15, 0.30, 0.45])
weight_matrix = np.array([[1,2,3],[2,3,4],[3,4,5]], dtype=float)
descriptors = _oracle_compute_material_descriptors(1.90, 2.55, 4.35)
kappa_sd, c1x_sd = 0.15, 0.20
kappa_default, c1x_default = 0.065, 0.667""",
            "call": "compute_weighted_fx_percent_increase(s_grid.copy(), alpha_grid.copy(), weight_matrix.copy(), descriptors.copy(), kappa_sd, c1x_sd, kappa_default, c1x_default)",
            "gold_call": "_oracle_compute_weighted_fx_percent_increase(s_grid.copy(), alpha_grid.copy(), weight_matrix.copy(), descriptors.copy(), kappa_sd, c1x_sd, kappa_default, c1x_default)",
            "tol": 1e-09,
        },
        {
            "setup": """import numpy as np
s_grid = np.array([1.0, 1.0])
alpha_grid = np.array([0.0, 1.0])
weight_matrix = np.array([[1.0, 0.0],[0.0, 1.0]])
descriptors = np.array([1.0, 1.5457, 0.6469, 0.1758])
kappa_sd, c1x_sd = 0.05, 0.15
kappa_default, c1x_default = 0.065, 0.667""",
            "call": "compute_weighted_fx_percent_increase(s_grid.copy(), alpha_grid.copy(), weight_matrix.copy(), descriptors.copy(), kappa_sd, c1x_sd, kappa_default, c1x_default)",
            "gold_call": "_oracle_compute_weighted_fx_percent_increase(s_grid.copy(), alpha_grid.copy(), weight_matrix.copy(), descriptors.copy(), kappa_sd, c1x_sd, kappa_default, c1x_default)",
            "tol": 1e-09,
        },
        {
            "setup": """import numpy as np
s_grid = np.array([0.5, 1.5, 3.0])
alpha_grid = np.array([0.0, 0.5, 1.0])
weight_matrix = np.array([[5,1,1],[1,5,1],[1,1,5]], dtype=float)
descriptors = _oracle_compute_material_descriptors(2.55, 2.55, 3.567)
kappa_sd, c1x_sd = 0.5, 0.05
kappa_default, c1x_default = 0.065, 0.667""",
            "call": "compute_weighted_fx_percent_increase(s_grid.copy(), alpha_grid.copy(), weight_matrix.copy(), descriptors.copy(), kappa_sd, c1x_sd, kappa_default, c1x_default)",
            "gold_call": "_oracle_compute_weighted_fx_percent_increase(s_grid.copy(), alpha_grid.copy(), weight_matrix.copy(), descriptors.copy(), kappa_sd, c1x_sd, kappa_default, c1x_default)",
            "tol": 1e-09,
        },
        {
            "setup": """import numpy as np
s_grid = np.array([0.25, 2.0])
alpha_grid = np.array([0.05, 0.95])
weight_matrix = np.array([[1.0, 7.0],[3.0, 0.0]])
descriptors = _oracle_compute_material_descriptors(0.79, 3.98, 6.2)
kappa_sd, c1x_sd = 0.30, 0.10
kappa_default, c1x_default = 0.065, 0.667""",
            "call": "compute_weighted_fx_percent_increase(s_grid.copy(), alpha_grid.copy(), weight_matrix.copy(), descriptors.copy(), kappa_sd, c1x_sd, kappa_default, c1x_default)",
            "gold_call": "_oracle_compute_weighted_fx_percent_increase(s_grid.copy(), alpha_grid.copy(), weight_matrix.copy(), descriptors.copy(), kappa_sd, c1x_sd, kappa_default, c1x_default)",
            "tol": 1e-09,
        },
        {'setup': 'import numpy as np\n'
                  's_grid = np.array([0.75, 1.25])\n'
                  'alpha_grid = np.array([0.15, 0.45])\n'
                  'weight_matrix = np.array([[0.0, 1.0], [0.0, 3.0]])\n'
                  'descriptors = _oracle_compute_material_descriptors(1.90, 2.55, 4.35)\n'
                  'kappa_sd, c1x_sd = 0.15, 0.20\n'
                  'kappa_default, c1x_default = 0.065, 0.667',
         'call': 'compute_weighted_fx_percent_increase(s_grid.copy(), alpha_grid.copy(), weight_matrix.copy(), '
                 'descriptors.copy(), kappa_sd, c1x_sd, kappa_default, c1x_default)',
         'gold_call': '_oracle_compute_weighted_fx_percent_increase(s_grid.copy(), alpha_grid.copy(), '
                      'weight_matrix.copy(), descriptors.copy(), kappa_sd, c1x_sd, kappa_default, c1x_default)',
         'tol': 1e-09},
    ]
