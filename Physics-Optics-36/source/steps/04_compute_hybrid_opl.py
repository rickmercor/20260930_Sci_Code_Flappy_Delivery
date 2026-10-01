"""
Compute round-trip optical pathlengths assigned to hybrid detected packets.

The optical path includes the entire transport history and is expressed in mm of optical distance.

Returns
-------
np.ndarray of shape (K,), the corrected round-trip optical pathlengths in mm, preserving the input record order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_hybrid_opl(records: "np.ndarray", sites: "np.ndarray", refractive_index: float) -> "np.ndarray":
    """Parameters
    ----------
    records : np.ndarray
        Shape (K, 7): exit x, exit y, vx, vy, vz, final segment s,
        total physical path L. Lengths are in mm.
    sites : np.ndarray
        Shape (K, 3), last-scatter x, y, z in mm.
    refractive_index : float
        Positive constant refractive index of the sample.

    Returns
    -------
    opl : np.ndarray
        Shape (K,), corrected round-trip optical pathlengths in mm,
        in the same packet order.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_compute_hybrid_opl(records: "np.ndarray", sites: "np.ndarray", refractive_index: float) -> "np.ndarray":
    records = np.asarray(records, dtype=float)
    return refractive_index * (records[:, 6] - records[:, 5] + sites[:, 2])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return independent normal, boundary and edge cases."""
    return [{'setup': 'import numpy as np\nr=np.array([[.15,.02,.6,0,-.8,.25,.49]])\nx=np.array([[0,.02,.2]])', 'call': 'compute_hybrid_opl(r.copy(), x.copy(), 1.33)', 'gold_call': '_oracle_compute_hybrid_opl(r.copy(), x.copy(), 1.33)', 'tol': 1e-11}, {'setup': 'import numpy as np\nr=np.array([[0,0,0,0,-1,0,0]])\nx=np.zeros((1,3))', 'call': 'compute_hybrid_opl(r.copy(), x.copy(), 1.33)', 'gold_call': '_oracle_compute_hybrid_opl(r.copy(), x.copy(), 1.33)', 'tol': 1e-11}, {'setup': 'import numpy as np\nr=np.array([[.24,.26,.48,.64,-.6,.4,1.],[.01,0,0,0,-1,.1,2.4]])\nx=np.array([[.048,.004,.24],[.01,0,.1]])', 'call': 'compute_hybrid_opl(r.copy(), x.copy(), 1.33)', 'gold_call': '_oracle_compute_hybrid_opl(r.copy(), x.copy(), 1.33)', 'tol': 1e-11}]
