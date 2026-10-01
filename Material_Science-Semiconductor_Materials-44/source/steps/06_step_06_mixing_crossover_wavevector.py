"""
Locate the nonzero recrossing on the continuously tracked Kramers branch.

Track the zone-centre doublet over a 121-point path from zero to k_max using the
projector-overlap rule of the preceding step. Subtract the scalar-mass reference from
the tracked pair-mean energy and find the first strictly positive sign change. Refine
that bracket with Brent's method; inside the narrow bracket, retain the energy-order
pair index selected at its left endpoint. This distinguishes physical band continuation
from independently choosing the lowest two eigenvalues at every momentum.

Returns
-------
float, the nonzero crossover wave vector in 1/nm
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def mixing_crossover_wavevector(width: float, direction: tuple, num_modes: int, params: dict, k_max: float) -> float:
    '''Nonzero recrossing wave vector of the tracked topmost Kramers branch.

    Parameters
    ----------
    width : float
        Well width in nm. Must be finite and strictly positive.
    direction : tuple
        Two finite Cartesian components defining a nonzero in-plane direction.
    num_modes : int
        Number of envelopes retained. Must be an integer >= 1.
    params : dict
        Must hold the finite floats 'gamma1', 'gamma2', 'gamma3', 'eta1', 'eta2',
        'eta3' and 'delta', with 'delta' strictly positive.
    k_max : float
        Strictly positive finite upper search bound in 1/nm.

    Returns
    -------
    float
        First nonzero crossover magnitude strictly between zero and k_max.

    Raises
    ------
    ValueError
        If an argument is invalid or no tracked nonzero crossing is found.
    '''
    return 0.0  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import brentq

def _oracle_mixing_crossover_wavevector(width: float, direction: tuple, num_modes: int, params: dict, k_max: float) -> float:
    top = float(k_max)
    if not np.isfinite(top) or top <= 0.0:
        raise ValueError("k_max must be finite and strictly positive")

    grid = np.linspace(0.0, top, 121)
    tracked = _oracle_tracked_topmost_subband_path(
        width, grid, direction, num_modes, params,
    )
    zone_centre = float(tracked[0, 0])
    stiffness = 38.0998212 * float(params["gamma1"])
    residual = tracked[:, 0] - (zone_centre + stiffness * grid * grid)

    bracket = None
    for index in range(1, grid.size - 1):
        if residual[index] * residual[index + 1] <= 0.0:
            bracket = (index, index + 1)
            break
    if bracket is None:
        raise ValueError("no tracked nonzero crossing below k_max")

    left, right = bracket
    pair_index = int(round(float(tracked[left, 4])))

    def _residual(kk):
        ham = _oracle_hole_hamiltonian(width, float(kk), direction, num_modes, params)
        energies = np.linalg.eigvalsh(ham)
        lo = 2 * pair_index
        exact = 0.5 * float(energies[lo] + energies[lo + 1])
        return exact - (zone_centre + stiffness * float(kk) ** 2)

    return float(brentq(
        _residual, float(grid[left]), float(grid[right]),
        xtol=1e-13, rtol=8.881784197001252e-16,
    ))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    setup = "import numpy as np\nfrom scipy.optimize import brentq\nCU2O = {'gamma1': 1.76, 'gamma2': 0.7532, 'gamma3': -0.3668, 'eta1': -0.020, 'eta2': -0.0037, 'eta3': -0.0337, 'delta': 131.0}"
    return [
        {
            "setup": setup,
            "call": "mixing_crossover_wavevector(6.4, (1.0, 1.0), 16, CU2O, 1.20)",
            "gold_call": "_oracle_mixing_crossover_wavevector(6.4, (1.0, 1.0), 16, CU2O, 1.20)",
        },
        {
            "setup": setup,
            "call": "mixing_crossover_wavevector(10.0, (1.0, 0.0), 16, CU2O, 1.60)",
            "gold_call": "_oracle_mixing_crossover_wavevector(10.0, (1.0, 0.0), 16, CU2O, 1.60)",
        },
        {
            "setup": setup,
            "call": "mixing_crossover_wavevector(4.0, (2.0, 1.0), 12, CU2O, 2.00)",
            "gold_call": "_oracle_mixing_crossover_wavevector(4.0, (2.0, 1.0), 12, CU2O, 2.00)",
        },
    ]
