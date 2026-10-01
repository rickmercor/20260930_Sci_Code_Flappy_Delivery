"""
Reduce weighted phaseless walker energies to a real mixed estimate.

After the phaseless constraint, each walker contributes a nonnegative updated weight and a generally complex local energy. The physical mixed estimator is obtained by normalizing the post-propagation weights and averaging the real parts of the local energies.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def phaseless_mixed_energy(
    local_energies: 'np.ndarray',
    walker_weights: 'np.ndarray',
) -> float:
    """Reduce phaseless walker data to the normalized mixed energy.

    Parameters
    ----------
    local_energies
        Complex local energies, one per walker.
    walker_weights
        Matching nonnegative post-step weights with positive total weight.

    Notes
    -----
    Normalize the weights by their sum and contract them with the real parts
    of the local energies.

    Returns
    -------
    float
        Normalized real mixed energy.

    Raises
    ------
    ValueError
        If the vectors are incompatible or nonfinite, or the weights are
        invalid.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_phaseless_mixed_energy(
    local_energies: 'np.ndarray',
    walker_weights: 'np.ndarray',
) -> float:
    import numpy as np

    energies = np.asarray(local_energies, dtype=complex)
    weights = np.asarray(walker_weights, dtype=float)
    if (
        energies.ndim != 1
        or energies.size == 0
        or weights.shape != energies.shape
    ):
        raise ValueError("energies and weights must be matching nonempty vectors")
    if np.any(~np.isfinite(energies)) or np.any(~np.isfinite(weights)):
        raise ValueError("inputs must be finite")
    if np.any(weights < 0) or not np.any(weights > 0):
        raise ValueError("weights must be nonnegative with positive sum")

    scaled_weights = weights / np.max(weights)
    normalized = scaled_weights / np.sum(scaled_weights)
    return float(np.dot(normalized, np.real(energies)))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\ne=np.array([-1.2+.1j,-.8-.3j,-1.5+.05j]); w=np.array([.2,.5,.3])",
            "call": "phaseless_mixed_energy(e,w)",
            "gold_call": "_oracle_phaseless_mixed_energy(e,w)",
        },
        {
            "setup": "import numpy as np\ne=np.array([2.+9j]); w=np.array([7.])",
            "call": "phaseless_mixed_energy(e,w)",
            "gold_call": "_oracle_phaseless_mixed_energy(e,w)",
        },
        {
            "setup": "import numpy as np\ne=np.array([1.,-2.,4.]); w=np.array([0.,3.,0.])",
            "call": "phaseless_mixed_energy(e,w)",
            "gold_call": "_oracle_phaseless_mixed_energy(e,w)",
        },
        {
            "setup": "import numpy as np\ne=np.array([.1+2j,.2-1j,.3+.5j,.4]); w=np.array([1.,2.,4.,8.])",
            "call": "phaseless_mixed_energy(e,w)",
            "gold_call": "_oracle_phaseless_mixed_energy(e,w)",
        },
        {
            "setup": "import numpy as np\ne=np.array([1.,3.]); w=np.array([1e308,1e308])",
            "call": "phaseless_mixed_energy(e,w)",
            "gold_call": "_oracle_phaseless_mixed_energy(e,w)",
        },
    ]
