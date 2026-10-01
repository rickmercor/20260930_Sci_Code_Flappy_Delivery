"""
Evaluate the nearest-neighbour bond amplitude <c+ c> of one spin from a discrete spectral measure given by nodes and weights, using the source's closure kernel built on the bare Bethe density of states, with particle-hole symmetry enforced at the level of the moment update as the source prescribes: the amplitude is one half of the full-line sum over all poles of w_i t rho_0(eps_i) with the scaled Bethe hopping t = D/2, for any particle-hole-symmetric input measure, so a pole at zero energy contributes half its weight and a pole at or beyond the band edge contributes nothing.

The bond amplitude is the correlator through which the reconstructed spectral function feeds back into the moments. At half filling the product of the bare density of states and the spectral function is an even function of energy, which the source uses to make the occupied-side integral well defined for a pole sitting exactly at the Fermi level.

Returns
-------
float, The bond amplitude <c+ c> as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def bond_amplitude_closure(nodes: "np.ndarray", weights: "np.ndarray", half_bandwidth: float) -> float:
    """Evaluate the nearest-neighbour bond amplitude <c+ c> of one spin from a discrete spectral measure given by nodes and weights, using the source's closure kernel built on the bare Bethe density of states, with particle-hole symmetry enforced at the level of the moment update as the source prescribes: the amplitude is one half of the full-line sum over all poles of w_i t rho_0(eps_i) with the scaled Bethe hopping t = D/2, for any particle-hole-symmetric input measure, so a pole at zero energy contributes half its weight and a pole at or beyond the band edge contributes nothing.

    Parameters
    ----------
    nodes : numpy.ndarray
        Finite 1-D array of pole positions.
    weights : numpy.ndarray
        Finite 1-D array of nonnegative pole weights, same shape as nodes.
    half_bandwidth : float
        Positive half-bandwidth D of the semielliptic band.

    Returns
    -------
    amplitude : float
        The bond amplitude <c+ c> as a native Python float.

    Raises
    ------
    ValueError
        If nodes and weights are not 1-D arrays of equal nonzero length, are not finite, a weight is negative, or half_bandwidth is not positive and finite.
    """
    return amplitude

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_bond_amplitude_closure(nodes: "np.ndarray", weights: "np.ndarray", half_bandwidth: float) -> float:
    """<c+c> = int_{-inf}^0 t rho0(w) A(w) dw on the discrete measure, with particle-hole
    symmetry enforced at the moment update.

    rho0 * A is even at half filling, so the occupied integral is half the full one:
    <c+c> = (1/2) sum_i w_i t rho0(eps_i), which gives a Fermi-level pole half its weight
    instead of leaving it to the sign of a machine-zero node.
    """
    nodes = np.asarray(nodes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if nodes.ndim != 1 or nodes.shape != weights.shape or nodes.size < 1:
        raise ValueError("nodes and weights must be 1-D arrays of equal, nonzero length")
    if not (np.all(np.isfinite(nodes)) and np.all(np.isfinite(weights))):
        raise ValueError("nodes and weights must be finite")
    if np.any(weights < 0.0):
        raise ValueError("weights must be nonnegative")
    if not (np.isfinite(half_bandwidth) and half_bandwidth > 0.0):
        raise ValueError("half_bandwidth must be positive and finite")
    t = float(half_bandwidth) / 2.0
    rho = _oracle_semielliptic_dos(nodes, half_bandwidth)
    return 0.5 * float(np.sum(weights * t * rho))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nnodes = np.array([-0.5596881548, 0.0, 0.5596881548])\nweights = np.array([0.4527008105, 0.0945983790, 0.4527008105])\nhalf_bandwidth = 1.0\n",
            "call": "bond_amplitude_closure(nodes, weights, half_bandwidth)",
            "gold_call": "_oracle_bond_amplitude_closure(nodes, weights, half_bandwidth)",
        },
        {
            "setup": "import numpy as np\nnodes = np.array([-0.5, 0.5])\nweights = np.array([0.5, 0.5])\nhalf_bandwidth = 1.0\n",
            "call": "bond_amplitude_closure(nodes, weights, half_bandwidth)",
            "gold_call": "_oracle_bond_amplitude_closure(nodes, weights, half_bandwidth)",
        },
        {
            "setup": "import numpy as np\nnodes = np.array([-1.5, 0.0, 1.5])\nweights = np.array([0.25, 0.5, 0.25])\nhalf_bandwidth = 1.5\n",
            "call": "bond_amplitude_closure(nodes, weights, half_bandwidth)",
            "gold_call": "_oracle_bond_amplitude_closure(nodes, weights, half_bandwidth)",
        },
        {
            "setup": "import numpy as np\nnodes = np.array([-0.5, 0.5])\nweights = np.array([0.5, -0.5])\nhalf_bandwidth = 1.0\ndef run_model():\n    try:\n        bond_amplitude_closure(nodes, weights, half_bandwidth)\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_bond_amplitude_closure(nodes, weights, half_bandwidth)\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
