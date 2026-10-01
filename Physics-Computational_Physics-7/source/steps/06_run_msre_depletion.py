"""
Implement `run_msre_depletion` which sets up and runs the full MSRE depletion calculation.

Cs-135 accumulates through the mass-135 fission-product chain as fuel circulates
through the reactor loop. Irradiation, radioactive decay and chemical removal
affect that chain in different regions. The Cs-135 concentration in the core
reflects the combined history of the connected fuel inventory.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def run_msre_depletion(n_cells: int, m: int,
                       fluxes: "np.ndarray", volumes: "np.ndarray",
                       Q: float, flow_fractions: "np.ndarray",
                       sigma_gamma: "np.ndarray", sigma_f: "np.ndarray",
                       fission_yields: "np.ndarray",
                       decay_constants: "np.ndarray", branching_ratios: "np.ndarray",
                       removal_rates: "np.ndarray",
                       addition_rates: "np.ndarray",
                       N_0: "np.ndarray",
                       total_time: float, n_steps: int) -> float:
    '''Sets up and runs the MSRE depletion calculation.

    The supplied physical parameters remain constant over the simulation.

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
    Q : float
        Volumetric flow rate [cm³/s].
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
        Initial number densities [atoms·barn⁻¹·cm⁻¹].
        Shape (m,) for uniform initial composition (tiled across cells),
        or shape (n_cells*m,) for per-cell initial state.
    total_time : float
        Non-negative total simulation time [s]. At zero time, return the initial
        Cs-135 number density in the core cell without advancing the state.
    n_steps : int
        Number of equal time steps.

    Returns
    -------
    Cs135_core : float
        Cs-135 density in the core cell at the end of the simulation
        [atoms·barn⁻¹·cm⁻¹]. The core is cell 0; Cs-135 is isotope index 3.

    Raises
    ------
    ValueError
        If n_steps < 1 or total_time < 0.
    '''
    return Cs135_core

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_run_msre_depletion(n_cells: int, m: int, fluxes: "np.ndarray", volumes: "np.ndarray", Q: float, flow_fractions: "np.ndarray", sigma_gamma: "np.ndarray", sigma_f: "np.ndarray", fission_yields: "np.ndarray", decay_constants: "np.ndarray", branching_ratios: "np.ndarray", removal_rates: "np.ndarray", addition_rates: "np.ndarray", N_0: "np.ndarray", total_time: float, n_steps: int) -> float:
    """Reference implementation."""
    if n_steps < 1 or total_time < 0:
        raise ValueError("need n_steps >= 1 and total_time >= 0")
    cell_matrices = []
    for c in range(n_cells):
        T = _oracle_build_transmutation_matrix(
            fluxes[c], sigma_gamma, sigma_f, fission_yields)
        A = _oracle_build_cell_matrix(
            T, decay_constants, branching_ratios, removal_rates[c])
        cell_matrices.append(A)

    outflow_diags, inflow_blocks = _oracle_build_flow_coupling(
        n_cells, m, volumes, Q, flow_fractions)
    R = _oracle_assemble_system_matrix(cell_matrices, outflow_diags, inflow_blocks)

    N = N_0.copy() if len(N_0) == n_cells * m else np.tile(N_0, n_cells)
    if total_time == 0:
        return float(N[3])
    S = addition_rates.flatten()

    dt = total_time / n_steps
    for _ in range(n_steps):
        N = _oracle_solve_timestep(R, dt, N, S)

    return N[3]

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications.

    Each test case is a dict with:
        setup:      Python code to set up variables
        call:       Expression calling the model's function
        gold_call:  Expression calling the gold function
    """
    return [
        {
            # Test 1 (normal): the full MSRE benchmark configuration, 100 hours in 10 steps,
            # including the pump-bowl U-235 feed.
            "tol": 1e-14,
            "setup": """import copy
import numpy as np
from scipy.linalg import expm
n_cells = 3
m = 4
fluxes = np.array([6.0e14, 0.0, 0.0])
volumes = np.array([1.3e6, 6.2e5, 8.16e4])
Q = 7.5e2
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
""",
            "call": 'run_msre_depletion(*copy.deepcopy((n_cells, m, fluxes, volumes, Q, flow_fractions, sigma_gamma, sigma_f, fission_yields, decay_constants, branching_ratios, removal_rates, addition_rates, N_0, total_time, n_steps)))',
            "gold_call": '_oracle_run_msre_depletion(*copy.deepcopy((n_cells, m, fluxes, volumes, Q, flow_fractions, sigma_gamma, sigma_f, fission_yields, decay_constants, branching_ratios, removal_rates, addition_rates, N_0, total_time, n_steps)))',
        },
        {
            # Test 2 (boundary): zero elapsed time preserves the initial core Cs-135 density.
            # Distinct cell densities distinguish the core value from the loop value
            # or a flow-equilibrated average; a nonzero source must not advance it.
            "setup": """import copy
import numpy as np
n_cells = 2
m = 4
fluxes = np.zeros(n_cells)
volumes = np.array([1000.0, 500.0])
Q = 100.0
flow_fractions = np.array([[0.0, 1.0], [1.0, 0.0]])
sigma_gamma = np.zeros(m)
sigma_f = np.zeros(m)
fission_yields = np.zeros((m, m))
decay_constants = np.array([0.0, 0.0, 0.0, 9.55e-15])
branching_ratios = np.zeros((m, m))
removal_rates = np.zeros((n_cells, m))
addition_rates = np.zeros((n_cells, m))
addition_rates[0, 3] = 1.0e-3
N_0 = np.array([0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.75])
total_time = 0.0
n_steps = 4
""",
            "call": 'run_msre_depletion(*copy.deepcopy((n_cells, m, fluxes, volumes, Q, flow_fractions, sigma_gamma, sigma_f, fission_yields, decay_constants, branching_ratios, removal_rates, addition_rates, N_0, total_time, n_steps)))',
            "gold_call": '_oracle_run_msre_depletion(*copy.deepcopy((n_cells, m, fluxes, volumes, Q, flow_fractions, sigma_gamma, sigma_f, fission_yields, decay_constants, branching_ratios, removal_rates, addition_rates, N_0, total_time, n_steps)))',
        },
        {
            # Test 3 (edge): stationary fuel. A single irradiated cell with no flow, so the
            # mass-135 chain evolves entirely in the core.
            "tol": 1e-14,
            "setup": """import copy
import numpy as np
from scipy.linalg import expm
n_cells = 1
m = 4
fluxes = np.array([6.0e14])
volumes = np.array([1.3e6])
Q = 0.0
flow_fractions = np.array([[0.0]])
sigma_gamma = np.array([98.71, 80.03, 2.778e6, 8.302])
sigma_f = np.array([585.1, 0.0, 0.0, 0.0])
fission_yields = np.zeros((4, 4))
fission_yields[0, 1] = 0.0628
fission_yields[0, 2] = 0.0016
decay_constants = np.array([3.12e-17, 2.93e-5, 2.11e-5, 9.55e-15])
branching_ratios = np.zeros((4, 4))
branching_ratios[1, 2] = 1.0
branching_ratios[2, 3] = 1.0
removal_rates = np.array([[0.0, 0.0, 0.0, 0.0]])
addition_rates = np.zeros((n_cells, m))
N_0 = np.array([8.5e-4, 0.0, 0.0, 0.0])
total_time = 100.0 * 3600.0
n_steps = 10
""",
            "call": 'run_msre_depletion(*copy.deepcopy((n_cells, m, fluxes, volumes, Q, flow_fractions, sigma_gamma, sigma_f, fission_yields, decay_constants, branching_ratios, removal_rates, addition_rates, N_0, total_time, n_steps)))',
            "gold_call": '_oracle_run_msre_depletion(*copy.deepcopy((n_cells, m, fluxes, volumes, Q, flow_fractions, sigma_gamma, sigma_f, fission_yields, decay_constants, branching_ratios, removal_rates, addition_rates, N_0, total_time, n_steps)))',
        },
        {
            # Test 4 (edge): per-cell source placement in a multi-cell loop. Cs-135 is fed
            # only into cell 1 (volume 500) with no nuclear data; after 1 h the
            # 1e-3 * 500 * 3600 = 1800 added atoms are spread over the 1750 cm^3 loop,
            # so the core density approaches 1.03. A source mapped to the wrong cell or
            # applied in every cell gives a different value.
            "setup": """import copy
import numpy as np
n_cells = 3
m = 4
fluxes = np.zeros(3)
volumes = np.array([1000.0, 500.0, 250.0])
Q = 100.0
flow_fractions = np.array([
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
])
sigma_gamma = np.zeros(4)
sigma_f = np.zeros(4)
fission_yields = np.zeros((4, 4))
decay_constants = np.zeros(4)
branching_ratios = np.zeros((4, 4))
removal_rates = np.zeros((3, 4))
addition_rates = np.zeros((3, 4))
addition_rates[1, 3] = 1.0e-3
N_0 = np.zeros(4)
total_time = 3600.0
n_steps = 3
""",
            "call": 'run_msre_depletion(*copy.deepcopy((n_cells, m, fluxes, volumes, Q, flow_fractions, sigma_gamma, sigma_f, fission_yields, decay_constants, branching_ratios, removal_rates, addition_rates, N_0, total_time, n_steps)))',
            "gold_call": '_oracle_run_msre_depletion(*copy.deepcopy((n_cells, m, fluxes, volumes, Q, flow_fractions, sigma_gamma, sigma_f, fission_yields, decay_constants, branching_ratios, removal_rates, addition_rates, N_0, total_time, n_steps)))',
        },
    ]
