"""
Implement `build_flow_coupling` which constructs the flow coupling matrices for the multi-cell system.

Circulating salt carries nuclides between regions with different irradiation and
chemical conditions. A region can gain or lose isotopic inventory through this
exchange even when no local nuclear reaction occurs. Cell volumes and connections
determine how transport affects the concentration history in each region.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def build_flow_coupling(n_cells: int, m: int, volumes: "np.ndarray",
                        Q: float, flow_fractions: "np.ndarray"
                        ) -> tuple[list, list]:
    '''Constructs flow coupling matrices for the multi-cell system.

    Each cell has uniform composition, and flow transports all tracked isotopes
    with the bulk salt. The matrices act on isotope number densities.

    Parameters
    ----------
    n_cells : int
        Number of cells.
    m : int
        Number of tracked isotopes.
    volumes : np.ndarray
        Shape (n_cells,) cell volumes [cm³].
    Q : float
        Volumetric flow rate [cm³/s].
    flow_fractions : np.ndarray
        Shape (n_cells, n_cells) flow fractions. flow_fractions[k, l] = fraction from k to l.
        Row sums should equal 1.0 for cells with outflow.

    Returns
    -------
    outflow_diags : list
        List of n_cells (m, m) negative diagonal matrices for outflow from each cell.
    inflow_blocks : list of lists
        n_cells × n_cells list. inflow_blocks[k][l] is an (m, m) matrix for flow k → l,
        placed at block row l, block column k in the system matrix.

    Raises
    ------
    ValueError
        If n_cells < 1, Q < 0, or any volume is not positive.
    '''
    return outflow_diags, inflow_blocks

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_build_flow_coupling(n_cells: int, m: int, volumes: "np.ndarray", Q: float, flow_fractions: "np.ndarray") -> tuple[list, list]:
    """Reference implementation."""
    if n_cells < 1 or Q < 0 or np.any(np.asarray(volumes) <= 0):
        raise ValueError("need n_cells >= 1, Q >= 0 and positive volumes")
    outflow_diags = []
    inflow_blocks = [[np.zeros((m, m)) for _ in range(n_cells)] for _ in range(n_cells)]

    for k in range(n_cells):
        # Total outflow fraction from cell k
        total_outflow_frac = np.sum(flow_fractions[k, :])
        # Outflow diagonal: -(Q/V_k) * total_outflow_frac * I
        outflow_diags.append(-Q / volumes[k] * total_outflow_frac * np.eye(m))

        for l in range(n_cells):
            if flow_fractions[k, l] > 0.0:
                # Inflow to cell l from cell k: +(Q/V_l) * f_{k->l} * I
                inflow_blocks[k][l] = Q / volumes[l] * flow_fractions[k, l] * np.eye(m)

    return outflow_diags, inflow_blocks

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
            # Test 1: 3-cell MSRE loop (core -> loop -> pump bowl -> core)
            "setup": """import copy
import numpy as np
n_cells = 3
m = 4  # U235, I135, Xe135, Cs135
volumes = np.array([1.3e6, 6.2e5, 8.1e4])  # cm^3
Q = 7.5e4  # cm^3/s
# core(0) -> loop(1) -> pump_bowl(2) -> core(0)
flow_fractions = np.array([
    [0.0, 1.0, 0.0],  # core -> loop
    [0.0, 0.0, 1.0],  # loop -> pump bowl
    [1.0, 0.0, 0.0],  # pump bowl -> core
])
""",
            "call": 'np.concatenate([np.array(x).flatten() for x in build_flow_coupling(*copy.deepcopy((n_cells, m, volumes, Q, flow_fractions)))])',
            "gold_call": 'np.concatenate([np.array(x).flatten() for x in _oracle_build_flow_coupling(*copy.deepcopy((n_cells, m, volumes, Q, flow_fractions)))])',
        },
        {
            # Test 2: 2-cell loop (cell A -> cell B -> cell A)
            "setup": """import copy
import numpy as np
n_cells = 2
m = 2
volumes = np.array([1000.0, 500.0])
Q = 100.0
flow_fractions = np.array([
    [0.0, 1.0],
    [1.0, 0.0],
])
""",
            "call": 'np.concatenate([np.array(x).flatten() for x in build_flow_coupling(*copy.deepcopy((n_cells, m, volumes, Q, flow_fractions)))])',
            "gold_call": 'np.concatenate([np.array(x).flatten() for x in _oracle_build_flow_coupling(*copy.deepcopy((n_cells, m, volumes, Q, flow_fractions)))])',
        },
        {
            # Test 3: Zero flow rate — all coupling matrices should be zero
            "setup": """import copy
import numpy as np
n_cells = 3
m = 4
volumes = np.array([1.3e6, 6.2e5, 8.1e4])
Q = 0.0
flow_fractions = np.array([
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
])
""",
            "call": 'np.concatenate([np.array(x).flatten() for x in build_flow_coupling(*copy.deepcopy((n_cells, m, volumes, Q, flow_fractions)))])',
            "gold_call": 'np.concatenate([np.array(x).flatten() for x in _oracle_build_flow_coupling(*copy.deepcopy((n_cells, m, volumes, Q, flow_fractions)))])',
        },
    ]
