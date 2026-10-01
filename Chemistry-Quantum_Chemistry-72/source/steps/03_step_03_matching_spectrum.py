"""
Step 03: Matching-matrix spectrum and node-count split. Matching matrix of a bound-state calculation and the two parts of the multichannel node count.

A bound state must satisfy the boundary conditions at short and at long range simultaneously. One log-derivative matrix is carried outwards from a wall deep in the classically forbidden region at short range and another inwards from a wall far out at long range, and both are evaluated at an intermediate matching distance. A wavefunction that is continuous with a continuous derivative exists only where the difference of the two matrices has a zero eigenvalue, so the eigenvalues of this matching matrix, rather than its determinant, are the natural functions whose zeros mark bound states. The node count of the whole range is the number of bound states below the trial energy, and it splits into a part gathered while the equations are propagated and a part read off at the matching distance. When one eigenvalue passes through a pole the first part rises by one and the second falls by one, so only their sum is a smooth staircase in energy, while each part separately carries information that the sum hides.

Returns
-------
numpy.ndarray of length N + 2: [n_sum, n_match, ascending eigenvalues of the matching matrix in inverse angstrom]
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def matching_spectrum(params: dict, E: float, grid: dict) -> "np.ndarray":
    '''Node-count contributions and ascending eigenvalues of the log-derivative matching matrix at energy E.

    Parameters
    ----------
    params : dict
        Model parameters as in channel_matrix.
    E : float
        Trial energy in cm^-1, finite.
    grid : dict
        Keys 'R_min', 'R_match', 'R_max' (angstrom, 0 < R_min < R_match < R_max) and 'h' (target step, angstrom, > 0).
        The wavefunction vanishes at R_min and at R_max. The outward segment uses
        n_outward = max(1, round((R_match - R_min) / (2 h))) Simpson panels from R_min to R_match, starting from
        Y = 1e30 I, and the inward segment uses n_inward = max(1, round((R_max - R_match) / (2 h))) panels from R_max to
        R_match, starting from Y = -1e30 I, both with propagate_log_derivative (round is Python's round, halves to even).

    Returns
    -------
    result : np.ndarray
        Float array of length N + 2: [n_sum, n_match, e_1, ..., e_N] with N the number of channels of channel_matrix. n_sum is the node count of
        the outward segment plus that of the inward segment. Y_match is the symmetric part of Y_outward - Y_inward, where
        Y_outward and Y_inward are the log-derivative matrices at R_match from the outward and the inward segment; e_1 <= ... <= e_N are
        its eigenvalues (inverse angstrom) and n_match is the number of them below zero.

    Raises
    ------
    ValueError
        If E is not finite, the grid keys are missing, the distances are not ordered as 0 < R_min < R_match < R_max,
        or h is not finite and > 0.
    '''
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _hashable(value):
    if isinstance(value, str) or not hasattr(value, "__len__"):
        return value
    return tuple(float(v) for v in value)


def _spectrum_key(params, E, grid):
    """Hashable form of one matching-matrix evaluation, so that a repeated evaluation costs no propagation."""
    return (tuple(sorted((k, _hashable(v)) for k, v in params.items())), float(E),
            tuple(sorted((k, float(v)) for k, v in grid.items())))


def _grid_segments(grid):
    import numpy as np
    if not isinstance(grid, dict) or any(k not in grid for k in ("R_min", "R_match", "R_max", "h")):
        raise ValueError("grid must contain R_min, R_match, R_max and h")
    r0, rm, r1, h = (float(grid[k]) for k in ("R_min", "R_match", "R_max", "h"))
    if not all(np.isfinite([r0, rm, r1, h])) or not (0.0 < r0 < rm < r1) or h <= 0.0:
        raise ValueError("grid must satisfy 0 < R_min < R_match < R_max and h > 0")
    return r0, rm, r1, max(1, int(round((rm - r0) / (2.0 * h)))), max(1, int(round((r1 - rm) / (2.0 * h))))


def _oracle_matching_spectrum(params: dict, E: float, grid: dict) -> "np.ndarray":
    import numpy as np
    if not np.isfinite(E):
        raise ValueError("E must be finite")
    _check_params(params)
    r0, rm, r1, n_in, n_out = _grid_segments(grid)
    cache = _oracle_matching_spectrum.__dict__.setdefault("cache", {})
    key = _spectrum_key(params, E, grid)
    if key in cache:
        return cache[key].copy()
    Y_out, nodes_out = _oracle_propagate_log_derivative(params, E, r0, rm, n_in, 1e30)
    Y_in, nodes_in = _oracle_propagate_log_derivative(params, E, r1, rm, n_out, -1e30)
    Ym = Y_out - Y_in
    ev = np.linalg.eigvalsh(0.5 * (Ym + Ym.T))
    result = np.concatenate(([float(nodes_out + nodes_in), float(np.sum(ev < 0.0))], ev))
    if len(cache) > 8192:
        cache.clear()
    cache[key] = result
    return result.copy()

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
        # --- Normal: benchmark complex at the observed energy, node-count split and the whole spectrum ---
        {"setup": base, "call": "np.asarray(matching_spectrum(dict(P), -30.0, dict(G)), dtype=float)",
         "gold_call": "np.asarray(_oracle_matching_spectrum(dict(P), -30.0, dict(G)), dtype=float)", "tol": 1e-7},
        # --- Normal: J = 3 odd parity, deeper energy, different matching point near the potential minimum ---
        {"setup": base, "call": "np.asarray(matching_spectrum(dict(P, scale=1.1, J=3, parity=-1), -90.0, dict(G, R_match=3.9)), dtype=float)",
         "gold_call": "np.asarray(_oracle_matching_spectrum(dict(P, scale=1.1, J=3, parity=-1), -90.0, dict(G, R_match=3.9)), dtype=float)",
         "tol": 1e-7},
        # --- Boundary: matching point deep in the classically forbidden region at short range ---
        {"setup": base, "call": "np.asarray(matching_spectrum(dict(P), -60.0, dict(G, R_match=3.0, h=0.005)), dtype=float)",
         "gold_call": "np.asarray(_oracle_matching_spectrum(dict(P), -60.0, dict(G, R_match=3.0, h=0.005)), dtype=float)",
         "tol": 1e-6},
        # --- Boundary: matching point far outside the well, heavy J = 4 complex with 25 channels and a dense spectrum ---
        {"setup": base + "Q = dict(P, mu=40.0, B=1.1, jmax=6, J=4, parity=1)\n",
         "call": "np.asarray(matching_spectrum(dict(Q), -15.0, dict(G, R_match=7.5, R_max=18.0, h=0.005)), dtype=float)",
         "gold_call": "np.asarray(_oracle_matching_spectrum(dict(Q), -15.0, dict(G, R_match=7.5, R_max=18.0, h=0.005)), dtype=float)",
         "tol": 1e-6},
        # --- Edge: single channel, where n_sum and n_match reduce to one-dimensional counts ---
        {"setup": base, "call": "np.asarray(matching_spectrum(dict(P, jmax=0), -45.0, dict(G)), dtype=float)",
         "gold_call": "np.asarray(_oracle_matching_spectrum(dict(P, jmax=0), -45.0, dict(G)), dtype=float)", "tol": 1e-8},
        # --- Edge: target step so large that each segment has a single panel ---
        {"setup": base, "call": "np.asarray(matching_spectrum(dict(P, jmax=2), -90.0, {'R_min': 3.2, 'R_match': 3.8, 'R_max': 4.6, 'h': 5.0}), dtype=float)",
         "gold_call": "np.asarray(_oracle_matching_spectrum(dict(P, jmax=2), -90.0, {'R_min': 3.2, 'R_match': 3.8, 'R_max': 4.6, 'h': 5.0}), dtype=float)",
         "tol": 1e-8},
        # --- Error: matching point outside the propagation range ---
        {"setup": base + "def _probe(fn):\n    try:\n        fn(dict(P), -30.0, dict(G, R_match=16.0))\n    except ValueError:\n        return 1\n    return 0\n",
         "call": "_probe(matching_spectrum)", "gold_call": "_probe(_oracle_matching_spectrum)"},
    ]
