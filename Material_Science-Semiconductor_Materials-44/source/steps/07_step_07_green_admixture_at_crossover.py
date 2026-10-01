"""
Report the gauge-invariant green content of the tracked Kramers branch at its recrossing.

First locate the nonzero recrossing with the preceding step. Then follow the
zone-centre Kramers subspace to that momentum on a 121-point path, rather than
rediagonalising the endpoint and blindly taking the lowest energy pair. Return the
arithmetic mean of the two green-projector eigenvalues reported for the terminal
tracked subspace. The result is invariant under rotations inside the Kramers pair and
under reversal or rescaling of the in-plane direction.

Returns
-------
float, the dimensionless green-level fraction of the tracked Kramers pair
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def green_admixture_at_crossover(width: float, direction: tuple, num_modes: int, params: dict, k_max: float) -> float:
    '''Gauge-invariant green fraction of the tracked topmost pair at its recrossing.

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
        Mean green-projector eigenvalue in the tracked Kramers subspace, between
        zero and one.

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

def _oracle_green_admixture_at_crossover(width: float, direction: tuple, num_modes: int, params: dict, k_max: float) -> float:
    crossing = _oracle_mixing_crossover_wavevector(
        width, direction, num_modes, params, k_max,
    )
    path = np.linspace(0.0, float(crossing), 121)
    tracked = _oracle_tracked_topmost_subband_path(
        width, path, direction, num_modes, params,
    )
    value = 0.5 * float(tracked[-1, 2] + tracked[-1, 3])
    return float(np.clip(value, 0.0, 1.0))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    setup = "import numpy as np\nCU2O = {'gamma1': 1.76, 'gamma2': 0.7532, 'gamma3': -0.3668, 'eta1': -0.020, 'eta2': -0.0037, 'eta3': -0.0337, 'delta': 131.0}"
    return [
        {
            "setup": setup,
            "call": "green_admixture_at_crossover(6.4, (1.0, 1.0), 16, CU2O, 1.20)",
            "gold_call": "_oracle_green_admixture_at_crossover(6.4, (1.0, 1.0), 16, CU2O, 1.20)",
        },
        {
            "setup": setup,
            "call": "green_admixture_at_crossover(6.4, (-4.0, -4.0), 16, CU2O, 1.20)",
            "gold_call": "_oracle_green_admixture_at_crossover(6.4, (-4.0, -4.0), 16, CU2O, 1.20)",
        },
        {
            "setup": setup,
            "call": "green_admixture_at_crossover(10.0, (1.0, 0.0), 16, CU2O, 1.60)",
            "gold_call": "_oracle_green_admixture_at_crossover(10.0, (1.0, 0.0), 16, CU2O, 1.60)",
        },
    ]
