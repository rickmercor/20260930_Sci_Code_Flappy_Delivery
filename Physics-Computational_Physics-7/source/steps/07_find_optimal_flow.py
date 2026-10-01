"""
Implement `find_optimal_flow` which finds the volumetric flow rate that maximises the core Cs-135 number density.

The circulation rate controls how long fuel spends in irradiated and external
regions. The resulting isotope concentrations depend on transport together with
nuclear production, decay and chemical removal. Changing the flow can therefore
change the inventory observed in a particular region at a fixed time.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def find_optimal_flow(n_cells: int, m: int,
                      fluxes: "np.ndarray", volumes: "np.ndarray",
                      flow_fractions: "np.ndarray",
                      sigma_gamma: "np.ndarray", sigma_f: "np.ndarray",
                      fission_yields: "np.ndarray",
                      decay_constants: "np.ndarray", branching_ratios: "np.ndarray",
                      removal_rates: "np.ndarray",
                      addition_rates: "np.ndarray",
                      N_0: "np.ndarray",
                      total_time: float, n_steps: int,
                      Q_min: float, Q_max: float) -> float:
    '''Finds the flow rate that maximises Cs-135 in the core.

    Parameters
    ----------
    n_cells : int
        Number of cells.
    m : int
        Number of tracked isotopes.
    fluxes : np.ndarray
        Shape (n_cells,) cell-averaged fluxes [cm⁻² s⁻¹].
    volumes : np.ndarray
        Shape (n_cells,) cell volumes [cm³].
    flow_fractions : np.ndarray
        Shape (n_cells, n_cells) flow fractions.
    sigma_gamma : np.ndarray
        Shape (m,) capture cross-sections [barn].
    sigma_f : np.ndarray
        Shape (m,) fission cross-sections [barn].
    fission_yields : np.ndarray
        Shape (m, m) independent fission yields. fission_yields[j, i] = yield of i from j.
    decay_constants : np.ndarray
        Shape (m,) decay constants [s⁻¹].
    branching_ratios : np.ndarray
        Shape (m, m) decay branching ratios. branching_ratios[j, i] = BR for j → i.
    removal_rates : np.ndarray
        Shape (n_cells, m) per-cell removal rates [s⁻¹].
    addition_rates : np.ndarray
        Shape (n_cells, m) per-cell addition rates [atoms·barn⁻¹·cm⁻¹·s⁻¹].
    N_0 : np.ndarray
        Initial number densities [atoms·barn⁻¹·cm⁻¹]. Shape (m,) for uniform
        composition across all cells, or shape (n_cells*m,) for per-cell state.
    total_time : float
        Non-negative total simulation time [s], as in run_msre_depletion.
    n_steps : int
        Number of equal time steps.
    Q_min : float
        Inclusive lower bound for Q [cm³/s].
    Q_max : float
        Inclusive upper bound for Q [cm³/s].

    Returns
    -------
    Q_opt : float
        Four-significant-figure representation of the flow rate that maximises
        the core Cs-135 density at total_time over [Q_min, Q_max].
        The bounds apply before rounding; the returned value may lie outside them.

    Raises
    ------
    ValueError
        If Q_min exceeds Q_max.
    '''
    return Q_opt

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import minimize_scalar

def _oracle_find_optimal_flow(n_cells: int, m: int, fluxes: "np.ndarray", volumes: "np.ndarray", flow_fractions: "np.ndarray", sigma_gamma: "np.ndarray", sigma_f: "np.ndarray", fission_yields: "np.ndarray", decay_constants: "np.ndarray", branching_ratios: "np.ndarray", removal_rates: "np.ndarray", addition_rates: "np.ndarray", N_0: "np.ndarray", total_time: float, n_steps: int, Q_min: float, Q_max: float) -> float:
    """Reference implementation."""

    if Q_min > Q_max:
        raise ValueError("Q_min must not exceed Q_max")
    def _objective(Q):
        return -_oracle_run_msre_depletion(
            n_cells, m, fluxes, volumes, Q, flow_fractions,
            sigma_gamma, sigma_f, fission_yields,
            decay_constants, branching_ratios, removal_rates,
            addition_rates,
            N_0, total_time, n_steps)

    result = minimize_scalar(_objective, bounds=(Q_min, Q_max), method='bounded')
    candidates = [float(Q_min), float(result.x), float(Q_max)]
    Q_opt = min(candidates, key=lambda Q: (_objective(Q), Q))

    return float(f"{Q_opt:.4g}")

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            # Test 1: Full MSRE problem with pump bowl U-235 source
            "tol": 1e-9,
            "setup": """import copy
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize_scalar
n_cells = 3
m = 4
fluxes = np.array([6.0e14, 0.0, 0.0])
volumes = np.array([1.3e6, 6.2e5, 8.16e4])
flow_fractions = np.array([
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
])
sigma_gamma = np.array([98.71, 80.03, 2.778e6, 8.302])
sigma_f = np.array([585.1, 0.0, 0.0, 0.0])
fission_yields = np.zeros((4, 4))
fission_yields[0, 1] = 0.0628
fission_yields[0, 2] = 0.0016
decay_constants = np.array([3.12e-17, 2.93e-5, 2.11e-5, 9.55e-15])
branching_ratios = np.zeros((4, 4))
branching_ratios[1, 2] = 1.0
branching_ratios[2, 3] = 1.0
removal_rates = np.array([
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 6.72e-4, 0.0],
])
addition_rates = np.zeros((n_cells, m))
addition_rates[2, 0] = 3.5e-10
N_0 = np.array([8.5e-4, 0.0, 0.0, 0.0])
total_time = 100.0 * 3600.0
n_steps = 10
Q_min = 0.0
Q_max = 7.5e5
""",
            "call": 'find_optimal_flow(*copy.deepcopy((n_cells, m, fluxes, volumes, flow_fractions, sigma_gamma, sigma_f, fission_yields, decay_constants, branching_ratios, removal_rates, addition_rates, N_0, total_time, n_steps, Q_min, Q_max)))',
            "gold_call": '_oracle_find_optimal_flow(*copy.deepcopy((n_cells, m, fluxes, volumes, flow_fractions, sigma_gamma, sigma_f, fission_yields, decay_constants, branching_ratios, removal_rates, addition_rates, N_0, total_time, n_steps, Q_min, Q_max)))',
        },
        {
            # Test 2 (boundary case): Reversed flow — core → pump bowl → loop → core
            # With reversed flow, Xe-135 hits the pump bowl off-gas BEFORE spending
            # time in the external loop, so the optimal Q should shift.
            "tol": 1e-9,
            "setup": """import copy
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize_scalar
n_cells = 3
m = 4
fluxes = np.array([6.0e14, 0.0, 0.0])
volumes = np.array([1.3e6, 6.2e5, 8.16e4])
flow_fractions = np.array([
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
])
sigma_gamma = np.array([98.71, 80.03, 2.778e6, 8.302])
sigma_f = np.array([585.1, 0.0, 0.0, 0.0])
fission_yields = np.zeros((4, 4))
fission_yields[0, 1] = 0.0628
fission_yields[0, 2] = 0.0016
decay_constants = np.array([3.12e-17, 2.93e-5, 2.11e-5, 9.55e-15])
branching_ratios = np.zeros((4, 4))
branching_ratios[1, 2] = 1.0
branching_ratios[2, 3] = 1.0
removal_rates = np.array([
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 6.72e-4, 0.0],
])
addition_rates = np.zeros((n_cells, m))
N_0 = np.array([8.5e-4, 0.0, 0.0, 0.0])
total_time = 100.0 * 3600.0
n_steps = 10
Q_min = 0.0
Q_max = 7.5e5
""",
            "call": 'find_optimal_flow(*copy.deepcopy((n_cells, m, fluxes, volumes, flow_fractions, sigma_gamma, sigma_f, fission_yields, decay_constants, branching_ratios, removal_rates, addition_rates, N_0, total_time, n_steps, Q_min, Q_max)))',
            "gold_call": '_oracle_find_optimal_flow(*copy.deepcopy((n_cells, m, fluxes, volumes, flow_fractions, sigma_gamma, sigma_f, fission_yields, decay_constants, branching_ratios, removal_rates, addition_rates, N_0, total_time, n_steps, Q_min, Q_max)))',
        },
        {
            # Test 3 (edge case): No Xe removal — without off-gassing, the peak
            # shifts slightly (to ~17.9) because fission source dilution at high Q
            # still creates a non-monotonic dependence.
            "tol": 1e-9,
            "setup": """import copy
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize_scalar
n_cells = 3
m = 4
fluxes = np.array([6.0e14, 0.0, 0.0])
volumes = np.array([1.3e6, 6.2e5, 8.16e4])
flow_fractions = np.array([
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
])
sigma_gamma = np.array([98.71, 80.03, 2.778e6, 8.302])
sigma_f = np.array([585.1, 0.0, 0.0, 0.0])
fission_yields = np.zeros((4, 4))
fission_yields[0, 1] = 0.0628
fission_yields[0, 2] = 0.0016
decay_constants = np.array([3.12e-17, 2.93e-5, 2.11e-5, 9.55e-15])
branching_ratios = np.zeros((4, 4))
branching_ratios[1, 2] = 1.0
branching_ratios[2, 3] = 1.0
removal_rates = np.array([
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
])
addition_rates = np.zeros((n_cells, m))
N_0 = np.array([8.5e-4, 0.0, 0.0, 0.0])
total_time = 100.0 * 3600.0
n_steps = 10
Q_min = 0.0
Q_max = 7.5e5
""",
            "call": 'find_optimal_flow(*copy.deepcopy((n_cells, m, fluxes, volumes, flow_fractions, sigma_gamma, sigma_f, fission_yields, decay_constants, branching_ratios, removal_rates, addition_rates, N_0, total_time, n_steps, Q_min, Q_max)))',
            "gold_call": '_oracle_find_optimal_flow(*copy.deepcopy((n_cells, m, fluxes, volumes, flow_fractions, sigma_gamma, sigma_f, fission_yields, decay_constants, branching_ratios, removal_rates, addition_rates, N_0, total_time, n_steps, Q_min, Q_max)))',
        },
        {
            "setup": """import copy
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize_scalar
n_cells = 3
m = 4
fluxes = np.array([6.0e14, 0.0, 0.0])
volumes = np.array([1.3e6, 6.2e5, 8.16e4])
flow_fractions = np.array([
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
])
sigma_gamma = np.array([98.71, 80.03, 2.778e6, 8.302])
sigma_f = np.array([585.1, 0.0, 0.0, 0.0])
fission_yields = np.zeros((4, 4))
fission_yields[0, 1] = 0.0628
fission_yields[0, 2] = 0.0016
decay_constants = np.array([3.12e-17, 2.93e-5, 2.11e-5, 9.55e-15])
branching_ratios = np.zeros((4, 4))
branching_ratios[1, 2] = 1.0
branching_ratios[2, 3] = 1.0
removal_rates = np.array([
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 6.72e-4, 0.0],
])
addition_rates = np.zeros((n_cells, m))
addition_rates[2, 0] = 3.5e-10
N_0 = np.array([8.5e-4, 0.0, 0.0, 0.0])
total_time = 100.0 * 3600.0
n_steps = 10
Q_min = 0.0
Q_max = 7.5e5
Q_min = 0.0
Q_max = 0.1
""",
            'call': 'find_optimal_flow(*copy.deepcopy((n_cells, m, fluxes, volumes, flow_fractions, sigma_gamma, sigma_f, fission_yields, decay_constants, branching_ratios, removal_rates, addition_rates, N_0, total_time, n_steps, Q_min, Q_max)))',
            'gold_call': '_oracle_find_optimal_flow(*copy.deepcopy((n_cells, m, fluxes, volumes, flow_fractions, sigma_gamma, sigma_f, fission_yields, decay_constants, branching_ratios, removal_rates, addition_rates, N_0, total_time, n_steps, Q_min, Q_max)))',
        },
        {
            "setup": """import copy
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize_scalar
n_cells = 3
m = 4
fluxes = np.array([6.0e14, 0.0, 0.0])
volumes = np.array([1.3e6, 6.2e5, 8.16e4])
flow_fractions = np.array([
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
])
sigma_gamma = np.array([98.71, 80.03, 2.778e6, 8.302])
sigma_f = np.array([585.1, 0.0, 0.0, 0.0])
fission_yields = np.zeros((4, 4))
fission_yields[0, 1] = 0.0628
fission_yields[0, 2] = 0.0016
decay_constants = np.array([3.12e-17, 2.93e-5, 2.11e-5, 9.55e-15])
branching_ratios = np.zeros((4, 4))
branching_ratios[1, 2] = 1.0
branching_ratios[2, 3] = 1.0
removal_rates = np.array([
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 6.72e-4, 0.0],
])
addition_rates = np.zeros((n_cells, m))
addition_rates[2, 0] = 3.5e-10
N_0 = np.array([8.5e-4, 0.0, 0.0, 0.0])
total_time = 100.0 * 3600.0
n_steps = 10
Q_min = 0.0
Q_max = 7.5e5
Q_min = 1.0
Q_max = 2.0
""",
            'call': 'find_optimal_flow(*copy.deepcopy((n_cells, m, fluxes, volumes, flow_fractions, sigma_gamma, sigma_f, fission_yields, decay_constants, branching_ratios, removal_rates, addition_rates, N_0, total_time, n_steps, Q_min, Q_max)))',
            'gold_call': '_oracle_find_optimal_flow(*copy.deepcopy((n_cells, m, fluxes, volumes, flow_fractions, sigma_gamma, sigma_f, fission_yields, decay_constants, branching_ratios, removal_rates, addition_rates, N_0, total_time, n_steps, Q_min, Q_max)))',
        },
        {
            "setup": """import copy
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize_scalar
n_cells = 3
m = 4
fluxes = np.array([6.0e14, 0.0, 0.0])
volumes = np.array([1.3e6, 6.2e5, 8.16e4])
flow_fractions = np.array([
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
])
sigma_gamma = np.array([98.71, 80.03, 2.778e6, 8.302])
sigma_f = np.array([585.1, 0.0, 0.0, 0.0])
fission_yields = np.zeros((4, 4))
fission_yields[0, 1] = 0.0628
fission_yields[0, 2] = 0.0016
decay_constants = np.array([3.12e-17, 2.93e-5, 2.11e-5, 9.55e-15])
branching_ratios = np.zeros((4, 4))
branching_ratios[1, 2] = 1.0
branching_ratios[2, 3] = 1.0
removal_rates = np.array([
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 6.72e-4, 0.0],
])
addition_rates = np.zeros((n_cells, m))
addition_rates[2, 0] = 3.5e-10
N_0 = np.array([8.5e-4, 0.0, 0.0, 0.0])
total_time = 100.0 * 3600.0
n_steps = 10
Q_min = 0.0
Q_max = 7.5e5
Q_min = 0.0
Q_max = 0.0
""",
            'call': 'find_optimal_flow(*copy.deepcopy((n_cells, m, fluxes, volumes, flow_fractions, sigma_gamma, sigma_f, fission_yields, decay_constants, branching_ratios, removal_rates, addition_rates, N_0, total_time, n_steps, Q_min, Q_max)))',
            'gold_call': '_oracle_find_optimal_flow(*copy.deepcopy((n_cells, m, fluxes, volumes, flow_fractions, sigma_gamma, sigma_f, fission_yields, decay_constants, branching_ratios, removal_rates, addition_rates, N_0, total_time, n_steps, Q_min, Q_max)))',
        },
    ]
