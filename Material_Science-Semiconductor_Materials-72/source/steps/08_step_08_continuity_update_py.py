"""
Advance the nodal hole and electron densities by one explicit time step of the flux-conservative continuity equations.

The continuity equations say that the density at a node changes because current flows across the two faces of its control volume and because carriers recombine inside it. On the non-uniform grid the control volume of interior node l extends halfway to each neighbour, so its width is (z[l+1] - z[l-1]) / 2, and the currents on its two faces are the interval currents on either side of the node. Over one explicit time step dt the change of the density equals dt times the net divergence term minus dt times the local recombination rate.

The divergence term is the difference between the current on the node's right face and on its left face, divided by the elementary charge and by the control-volume width. Holes and electrons carry opposite charge, so they take opposite signs on it: holes change by the negative of this divergence and electrons by the positive. Because each interval current is shared by the two nodes it separates, what leaves one control volume enters its neighbour, which keeps the discretisation charge-conserving.

The two end nodes are contacts: their densities are held at the values supplied and are never updated.

Use q = 1.602176634e-19 C. Densities are in cm^-3, current densities in A cm^-2, recombination rates in cm^-3 s^-1, positions in cm and dt in s.

Returns
-------
np.ndarray of shape (2, N), row 0 the updated hole and row 1 the updated electron densities in cm^-3 as float64
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def continuity_update(p: np.ndarray, n: np.ndarray, Jp: np.ndarray, Jn: np.ndarray, U: np.ndarray,
                      z: np.ndarray, dt: float) -> np.ndarray:
    '''One explicit continuity step with fixed contact densities.

    Parameters
    ----------
    p, n : np.ndarray
        Arrays of shape (N,) of nodal hole and electron densities in cm^-3.
    Jp, Jn : np.ndarray
        Arrays of shape (N-1,) of hole and electron current densities on the intervals in A cm^-2.
    U : np.ndarray
        Array of shape (N,) of nodal recombination rates in cm^-3 s^-1.
    z : np.ndarray
        Array of shape (N,) of strictly increasing node positions in cm.
    dt : float
        Time step in s, positive.

    Returns
    -------
    new : np.ndarray
        Array of shape (2, N); row 0 the updated hole densities, row 1 the updated electron densities,
        with the entries at node 0 and node N - 1 equal to the supplied values.

    Raises
    ------
    ValueError
        If the nodal arrays do not share a one-dimensional shape of length at least 3, if a current
        array does not have length N - 1, if z is not strictly increasing, or if dt is not positive.
    '''
    return new

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_continuity_update(p: np.ndarray, n: np.ndarray, Jp: np.ndarray, Jn: np.ndarray, U: np.ndarray,
                              z: np.ndarray, dt: float) -> np.ndarray:
    q = 1.602176634e-19
    p = np.array(p, dtype=float); n = np.array(n, dtype=float); U = np.array(U, dtype=float)
    Jp = np.array(Jp, dtype=float); Jn = np.array(Jn, dtype=float); z = np.array(z, dtype=float)
    if z.ndim != 1 or z.size < 3 or any(a.shape != z.shape for a in (p, n, U)):
        raise ValueError("nodal arrays must be one-dimensional with a common length of at least 3")
    if Jp.shape != (z.size - 1,) or Jn.shape != (z.size - 1,):
        raise ValueError("current arrays must have length N - 1")
    if np.any(np.diff(z) <= 0.0):
        raise ValueError("z must be strictly increasing")
    if not dt > 0.0:
        raise ValueError("dt must be positive")
    cv = 0.5 * (z[2:] - z[:-2])
    p_new = p.copy()
    n_new = n.copy()
    p_new[1:-1] = p[1:-1] + dt * (-(Jp[1:] - Jp[:-1]) / (q * cv) - U[1:-1])
    n_new[1:-1] = n[1:-1] + dt * ((Jn[1:] - Jn[:-1]) / (q * cv) - U[1:-1])
    return np.vstack([p_new, n_new])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
z = np.concatenate([np.arange(21) * 1.0e-7, 20.0e-7 + np.arange(1, 81) * 0.25e-7])
p = np.concatenate([np.full(21, 2.81e19), np.full(80, 1.00e17)])
n = np.concatenate([np.full(21, 9.4552e-8), np.full(80, 2.5281e-29)])
Jp = np.zeros(100); Jp[20] = 6.4e3; Jp[19] = 3.0e1; Jn = np.zeros(100); U = np.zeros(101)""",
            "call": "continuity_update(p, n, Jp, Jn, U, z, 1.0e-12)",
            "gold_call": "_oracle_continuity_update(p, n, Jp, Jn, U, z, 1.0e-12)",
        },
        {
            "setup": """import numpy as np
z = np.array([0.0, 1.0e-7, 2.0e-7, 2.25e-7, 2.5e-7])
p = np.array([5.0e15, 4.0e15, 3.0e15, 2.0e15, 1.0e15]); n = np.array([1.0e15, 2.0e15, 3.0e15, 4.0e15, 5.0e15])
Jp = np.array([1.0e2, -3.0e2, 2.0e2, 5.0e1]); Jn = np.array([-2.0e2, 1.0e2, 4.0e2, -1.0e2])
U = np.array([0.0, 1.0e26, -5.0e25, 2.0e26, 0.0])""",
            "call": "continuity_update(p, n, Jp, Jn, U, z, 1.0e-12)",
            "gold_call": "_oracle_continuity_update(p, n, Jp, Jn, U, z, 1.0e-12)",
        },
        {
            "setup": """import numpy as np
z = np.array([0.0, 2.0e-7, 2.5e-7]); p = np.array([1.0e16, 2.0e16, 3.0e16]); n = np.array([3.0e16, 2.0e16, 1.0e16])
Jp = np.array([5.0e2, 5.0e2]); Jn = np.array([-4.0e2, 1.0e2]); U = np.array([0.0, 0.0, 0.0])""",
            "call": "continuity_update(p, n, Jp, Jn, U, z, 2.0e-12)",
            "gold_call": "_oracle_continuity_update(p, n, Jp, Jn, U, z, 2.0e-12)",
        },
        {
            "setup": """import numpy as np
z = np.array([0.0, 1.0e-7, 1.5e-7, 2.0e-7]); p = np.array([2.0e16, 1.0e16, 4.0e16, 3.0e16]); n = np.array([1.0e16, 5.0e16, 2.0e16, 6.0e16])
Jp = np.array([-1.0e3, 2.0e3, -5.0e2]); Jn = np.array([3.0e3, -1.0e3, 2.0e3]); U = np.array([7.0e27, -2.0e27, 3.0e27, 9.0e27])""",
            "call": "continuity_update(p, n, Jp, Jn, U, z, 1.0e-12)",
            "gold_call": "_oracle_continuity_update(p, n, Jp, Jn, U, z, 1.0e-12)",
        },
        {
            "setup": """import numpy as np
def _probe(fn):
    hits = 0
    a = np.ones(3); j = np.ones(2)
    for args in ((a, a, j, j, a, np.array([0.0, 1.0, 1.0]), 1.0), (a, a, j, j, a, np.array([0.0, 1.0, 2.0]), 0.0),
                 (a, a, np.ones(3), j, a, np.array([0.0, 1.0, 2.0]), 1.0), (np.ones(2), np.ones(2), np.ones(1), np.ones(1), np.ones(2), np.array([0.0, 1.0]), 1.0)):
        try:
            fn(*args)
        except ValueError:
            hits += 1
    return float(hits)""",
            "call": "_probe(continuity_update)",
            "gold_call": "_probe(_oracle_continuity_update)",
        },
    ]
