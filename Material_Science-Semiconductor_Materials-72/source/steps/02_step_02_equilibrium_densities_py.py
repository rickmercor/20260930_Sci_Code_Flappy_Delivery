"""
Set the equilibrium starting densities of holes and electrons on every node from the local doping and intrinsic density.

Both transport layers are p-type, so before any bias or transient acts, the majority hole population at a node equals the local acceptor doping density and the minority electron population follows from thermal equilibrium. For a non-degenerate semiconductor in equilibrium the product of the two carrier densities equals the square of that material's own intrinsic density, whatever the doping, so each node's electron density is its intrinsic density squared divided by its hole density.

The dynamic range this creates is the numerical point. The two layers' intrinsic densities differ by roughly twelve orders of magnitude, so the equilibrium minority densities on the two sides of the junction differ by about twenty-four. The minority density must be obtained by dividing, never by subtracting comparable quantities, or the small side is lost to rounding.

Densities are in cm^-3.

Returns
-------
np.ndarray of shape (2, N), row 0 the equilibrium hole densities and row 1 the equilibrium electron densities in cm^-3 as float64
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def equilibrium_densities(NA: np.ndarray, ni: np.ndarray) -> np.ndarray:
    '''Equilibrium hole and electron densities of a p-type stack.

    Parameters
    ----------
    NA : np.ndarray
        Array of shape (N,) of acceptor doping densities in cm^-3, all positive.
    ni : np.ndarray
        Array of shape (N,) of intrinsic carrier densities in cm^-3, all non-negative.

    Returns
    -------
    densities : np.ndarray
        Array of shape (2, N); row 0 the hole densities, row 1 the electron densities, in cm^-3.

    Raises
    ------
    ValueError
        If NA and ni are not one-dimensional arrays of the same length, if any NA is not positive,
        or if any ni is negative.
    '''
    return densities

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_equilibrium_densities(NA: np.ndarray, ni: np.ndarray) -> np.ndarray:
    NA = np.array(NA, dtype=float); ni = np.array(ni, dtype=float)
    if NA.ndim != 1 or NA.shape != ni.shape:
        raise ValueError("NA and ni must be one-dimensional arrays of the same length")
    if np.any(NA <= 0.0) or np.any(ni < 0.0):
        raise ValueError("NA must be positive and ni non-negative")
    return np.vstack([NA, ni ** 2 / NA])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
NA = np.concatenate([np.full(21, 2.81e19), np.full(80, 1.00e17)])
ni = np.concatenate([np.full(21, 1.63e6), np.full(80, 1.59e-6)])""",
            "call": "equilibrium_densities(NA, ni)",
            "gold_call": "_oracle_equilibrium_densities(NA, ni)",
        },
        {
            "setup": """import numpy as np
NA = np.array([4.0, 2.0, 0.5]); ni = np.array([2.0, 3.0, 1.0])""",
            "call": "equilibrium_densities(NA, ni)",
            "gold_call": "_oracle_equilibrium_densities(NA, ni)",
        },
        {
            "setup": """import numpy as np
NA = np.array([1.0e18]); ni = np.array([0.0])""",
            "call": "equilibrium_densities(NA, ni)",
            "gold_call": "_oracle_equilibrium_densities(NA, ni)",
        },
        {
            "setup": """import numpy as np
NA = np.array([1.00e17, 1.00e17]); ni = np.array([1.59e-6, 1.63e6])
def _minority(fn):
    return float(fn(NA, ni)[1, 0] * 1.0e29)""",
            "call": "_minority(equilibrium_densities)",
            "gold_call": "_minority(_oracle_equilibrium_densities)",
        },
        {
            "setup": """import numpy as np
def _probe(fn):
    hits = 0
    for NA, ni in ((np.array([1.0, 0.0]), np.array([1.0, 1.0])),
                   (np.array([1.0, 2.0]), np.array([1.0])),
                   (np.array([1.0, 2.0]), np.array([1.0, -1.0]))):
        try:
            fn(NA, ni)
        except ValueError:
            hits += 1
    return float(hits)""",
            "call": "_probe(equilibrium_densities)",
            "gold_call": "_probe(_oracle_equilibrium_densities)",
        },
    ]
