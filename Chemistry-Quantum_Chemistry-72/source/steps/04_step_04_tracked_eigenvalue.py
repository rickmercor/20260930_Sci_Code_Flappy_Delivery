"""
Step 04: Eigenvalue that belongs to a given bound state. The eigenvalue of the matching matrix that belongs to a given bound state, and the energies at which it can be followed.

Each bound state is tied to one continuous eigenvalue of the matching matrix, the one that passes through zero at the energy of that state. Following it is not a matter of taking the smallest eigenvalue: the value-ordered spectrum is reshuffled every time some eigenvalue runs through a pole, and near a pole, or in a crowded part of the spectrum, a rule based on the smallest eigenvalue of each sign lands on a different state. Between its own poles the eigenvalue of a state decreases with energy, and at energies where it has been pushed out of the ordered spectrum altogether the state has no eigenvalue to follow at all. Exact symmetry-related degeneracies do not spoil the identification, because degenerate eigenvalues coincide at every energy.

Returns
-------
numpy.ndarray [i, e_i]: position and value of the matching eigenvalue of state m, or [0, 0] where it is unavailable
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def tracked_eigenvalue(params: dict, m: int, E: float, grid: dict) -> "np.ndarray":
    '''Index and value of the matching-matrix eigenvalue that passes through zero at the energy of state m.

    Parameters
    ----------
    params : dict
        Model parameters as in channel_matrix.
    m : int
        Bound-state label, the node count immediately above the state; integer >= 1.
    E : float
        Trial energy in cm^-1, finite.
    grid : dict
        Propagation grid as in matching_spectrum.

    Returns
    -------
    result : np.ndarray
        Float array [i, e] identifying the matching-matrix eigenvalue that passes through zero at the energy of state
        m. The position i is counted from 1 in the ascending matching spectrum at E, and e is its value in inverse
        angstrom. At energies where that state's matching eigenvalue is unavailable, return [0.0, 0.0].

    Raises
    ------
    ValueError
        If m is not an integer >= 1 or E is not finite.
    '''
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _tracked_from_spectrum(spec, m):
    i = m - int(round(spec[0]))
    if 1 <= i <= spec.size - 2:
        return i, float(spec[1 + i])
    return 0, 0.0


def _oracle_tracked_eigenvalue(params: dict, m: int, E: float, grid: dict) -> "np.ndarray":
    import numpy as np
    if isinstance(m, bool) or int(m) != m or m < 1:
        raise ValueError("m must be an integer >= 1")
    if not np.isfinite(E):
        raise ValueError("E must be finite")
    i, e = _tracked_from_spectrum(_oracle_matching_spectrum(params, E, grid), int(m))
    return np.array([float(i), e])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    base = ("import numpy as np\n"
            "P = {'mu': 19.2, 'B': 10.4, 'eps': 180.0, 'Rm': 3.8, 'a': (1.0, 0.25, 0.35), 'b': (1.0, 0.15, 0.30),"
            " 'jmax': 6, 'J': 0, 'parity': 1, 'scale': 1.0}\n"
            "G = {'R_min': 2.5, 'R_match': 4.0, 'R_max': 15.0, 'h': 0.005}\n")
    return [
        # --- Normal: the eigenvalue that carries state 16 of the benchmark, against an independent literal ---
        {"setup": base, "call": "np.asarray(tracked_eigenvalue(dict(P), 16, -30.0, dict(G)), dtype=float)",
         "gold_call": "np.asarray(_oracle_tracked_eigenvalue(dict(P), 16, -30.0, dict(G)), dtype=float)", "tol": 1e-7},
        # --- Normal: benchmark complex, state 17 just above the trial energy ---
        {"setup": base, "call": "np.asarray(tracked_eigenvalue(dict(P), 17, -30.0, dict(G)), dtype=float)",
         "gold_call": "np.asarray(_oracle_tracked_eigenvalue(dict(P), 17, -30.0, dict(G)), dtype=float)", "tol": 1e-7},
        # --- Normal: a state already below the trial energy, negative eigenvalue ---
        {"setup": base, "call": "np.asarray(tracked_eigenvalue(dict(P), 15, -30.0, dict(G)), dtype=float)",
         "gold_call": "np.asarray(_oracle_tracked_eigenvalue(dict(P), 15, -30.0, dict(G)), dtype=float)", "tol": 1e-7},
        # --- Boundary: label too low, every pole of this state already passed ---
        {"setup": base, "call": "np.asarray(tracked_eigenvalue(dict(P), 3, -30.0, dict(G)), dtype=float)",
         "gold_call": "np.asarray(_oracle_tracked_eigenvalue(dict(P), 3, -30.0, dict(G)), dtype=float)", "tol": 1e-12},
        # --- Boundary: label too high for the channel count at this energy ---
        {"setup": base, "call": "np.asarray(tracked_eigenvalue(dict(P, jmax=2), 20, -60.0, dict(G, h=0.01)), dtype=float)",
         "gold_call": "np.asarray(_oracle_tracked_eigenvalue(dict(P, jmax=2), 20, -60.0, dict(G, h=0.01)), dtype=float)",
         "tol": 1e-12},
        # --- Edge: matching distance outside the well, where locally closed channels dominate the spectrum ---
        {"setup": base, "call": "np.asarray(tracked_eigenvalue(dict(P, scale=1.15, J=2, parity=-1), 16, -40.0, dict(G, R_min=2.8, R_match=6.5, R_max=12.0, h=0.01)), dtype=float)",
         "gold_call": "np.asarray(_oracle_tracked_eigenvalue(dict(P, scale=1.15, J=2, parity=-1), 16, -40.0, dict(G, R_min=2.8, R_match=6.5, R_max=12.0, h=0.01)), dtype=float)",
         "tol": 1e-6},
        # --- Normal: heavy J = 4 complex with 25 channels in a dense manifold ---
        {"setup": base + "Q = dict(P, mu=40.0, B=1.1, jmax=6, J=4, parity=1)\n",
         "call": "np.asarray(tracked_eigenvalue(dict(Q), 118, -58.0, dict(G, R_min=2.9, R_max=13.0, h=0.01)), dtype=float)",
         "gold_call": "np.asarray(_oracle_tracked_eigenvalue(dict(Q), 118, -58.0, dict(G, R_min=2.9, R_max=13.0, h=0.01)), dtype=float)",
         "tol": 1e-6},
        # --- Error: non-positive label ---
        {"setup": base + "def _probe(fn):\n    try:\n        fn(dict(P), 0, -30.0, dict(G))\n    except ValueError:\n        return 1\n    return 0\n",
         "call": "_probe(tracked_eigenvalue)", "gold_call": "_probe(_oracle_tracked_eigenvalue)"},
    ]
