"""
Step 02: Johnson log-derivative propagation with node count. Log-derivative propagation of the coupled-channel equations with a multichannel node count.

Propagating the wavefunction matrix of a set of coupled second-order equations is unstable, because solutions that grow exponentially in locally closed channels swamp the others. The log-derivative matrix, the derivative of the wavefunction matrix multiplied by its inverse, stays bounded away from the points where the wavefunction matrix becomes singular, and it is the quantity that bound-state and scattering programs carry across the radial grid. Those singular points are what the multichannel node count records, and the count gathered along a segment belongs to that segment alone, so counts from propagations running towards larger and towards smaller distances can be combined. Accuracy matters more here than in a scattering calculation, because the count and the eigenvalues that locate a state both come out of the same propagation.

Returns
-------
tuple (Y_end, nodes): symmetric log-derivative matrix at R_end in inverse angstrom and the accumulated multichannel node count (int)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def propagate_log_derivative(params: dict, E: float, R_start: float, R_end: float, n_steps: int,
                             Y_start: float) -> tuple:
    '''Carry the log-derivative matrix of the atom plus rigid rotor coupled equations from R_start to R_end.

    Parameters
    ----------
    params : dict
        Model parameters as in channel_matrix (keys mu, B, eps, Rm, a, b, jmax, J, parity, scale).
    E : float
        Total energy in cm^-1, finite.
    R_start : float
        Starting distance in angstrom, finite and > 0.
    R_end : float
        Final distance in angstrom, finite, > 0 and different from R_start; it may be smaller than R_start.
    n_steps : int
        Number of Simpson panels, integer >= 1. The grid is R_k = R_start + k h, k = 0, ..., 2 n_steps, with
        h = (R_end - R_start) / (2 n_steps), so h is negative for inward propagation.
    Y_start : float
        Finite value y0; the log-derivative matrix at R_start is y0 times the unit matrix.

    Returns
    -------
    result : tuple
        (Y_end, nodes) from Johnson log-derivative propagation on the specified Simpson grid, including its midpoint
        correction. Y_end is the symmetric log-derivative matrix at R_end (np.ndarray of shape (N, N) in the channel
        basis of channel_matrix, inverse angstrom); nodes is the multichannel node count accumulated on this segment,
        returned as a Python int. Y_end is compared only through its eigenvalues.

    Raises
    ------
    ValueError
        If E, R_start, R_end or Y_start is not finite, a distance is <= 0, R_start equals R_end, or n_steps is not a
        positive integer.
    '''
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_propagate_log_derivative(params: dict, E: float, R_start: float, R_end: float, n_steps: int,
                                     Y_start: float) -> tuple:
    import numpy as np
    if not (np.isfinite(E) and np.isfinite(R_start) and np.isfinite(R_end)) or R_start <= 0.0 or R_end <= 0.0 \
            or R_start == R_end:
        raise ValueError("energy and distances must be finite, positive and distinct")
    if isinstance(n_steps, bool) or int(n_steps) != n_steps or n_steps < 1:
        raise ValueError("n_steps must be a positive integer")
    N = _check_params(params)[2][0].shape[0]
    if not np.isfinite(Y_start):
        raise ValueError("Y_start must be finite")
    Y = float(Y_start) * np.eye(N)
    n_steps = int(n_steps)
    h = (R_end - R_start) / (2.0 * n_steps)
    ident = np.eye(N)
    shift = params["mu"] / 16.8576292 * E
    Y = Y - (h / 3.0) * (shift * ident - _oracle_channel_matrix(R_start, params))
    nodes = 0
    for k in range(1, 2 * n_steps + 1):
        Z = ident + h * Y
        nodes += int(np.sum(np.linalg.eigvalsh(0.5 * (Z + Z.T)) < 0.0))
        Y = np.linalg.solve(Z, Y)
        Q = shift * ident - _oracle_channel_matrix(R_start + k * h, params)
        if k == 2 * n_steps:
            Y = Y - (h / 3.0) * Q
        elif k % 2 == 1:
            Y = Y - (4.0 * h / 3.0) * np.linalg.solve(ident + (h * h / 6.0) * Q, Q)
        else:
            Y = Y - (2.0 * h / 3.0) * Q
        Y = 0.5 * (Y + Y.T)
    return Y, nodes

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    base = ("import numpy as np\n"
            "P = {'mu': 19.2, 'B': 10.4, 'eps': 180.0, 'Rm': 3.8, 'a': (1.0, 0.25, 0.35), 'b': (1.0, 0.15, 0.30),"
            " 'jmax': 6, 'J': 0, 'parity': 1, 'scale': 1.0}\n"
            "def _pack(res):\n    Y, n = res\n    Y = np.asarray(Y, dtype=float)\n"
            "    return np.append(np.sort(np.linalg.eigvalsh(0.5 * (Y + Y.T))), float(n))\n")
    return [
        # --- Normal: outward from the repulsive wall to the matching point, benchmark complex below threshold ---
        {"setup": base, "call": "_pack(propagate_log_derivative(dict(P), -30.0, 2.5, 4.0, 150, 1e30))",
         "gold_call": "_pack(_oracle_propagate_log_derivative(dict(P), -30.0, 2.5, 4.0, 150, 1e30))", "tol": 1e-8},
        # --- Normal: inward from the long-range wall at J = 4, many locally closed channels ---
        {"setup": base, "call": "_pack(propagate_log_derivative(dict(P, J=4), -30.0, 15.0, 4.0, 1100, -1e30))",
         "gold_call": "_pack(_oracle_propagate_log_derivative(dict(P, J=4), -30.0, 15.0, 4.0, 1100, -1e30))",
         "tol": 1e-8},
        # --- Boundary: energy above several rotor thresholds, odd parity, oscillatory open channels and many nodes ---
        {"setup": base + "Q = dict(P, B=2.0, mu=35.0, jmax=5, J=2, parity=-1)\n",
         "call": "_pack(propagate_log_derivative(dict(Q), 40.0, 2.8, 12.0, 920, 1e30))",
         "gold_call": "_pack(_oracle_propagate_log_derivative(dict(Q), 40.0, 2.8, 12.0, 920, 1e30))", "tol": 1e-7},
        # --- Boundary: a finite starting value and a coarse grid ---
        {"setup": base, "call": "_pack(propagate_log_derivative(dict(P, scale=1.2, J=1, parity=-1), -55.0, 3.3, 4.9, 20, 0.7))",
         "gold_call": "_pack(_oracle_propagate_log_derivative(dict(P, scale=1.2, J=1, parity=-1), -55.0, 3.3, 4.9, 20, 0.7))",
         "tol": 1e-8},
        # --- Edge: a single channel with one Simpson panel ---
        {"setup": base, "call": "_pack(propagate_log_derivative(dict(P, jmax=0), -100.0, 4.2, 3.6, 1, 2.0))",
         "gold_call": "_pack(_oracle_propagate_log_derivative(dict(P, jmax=0), -100.0, 4.2, 3.6, 1, 2.0))", "tol": 1e-10},
        # --- Normal: J beyond jmax with strong anisotropy, every rotor state with its full set of partial waves ---
        {"setup": base + "Q = dict(P, mu=28.0, B=1.3, a=(1.0, -0.5, 0.8), b=(1.0, 0.35, -0.25), jmax=4, J=7)\n",
         "call": "_pack(propagate_log_derivative(dict(Q), -80.0, 2.9, 6.5, 360, 1e30))",
         "gold_call": "_pack(_oracle_propagate_log_derivative(dict(Q), -80.0, 2.9, 6.5, 360, 1e30))", "tol": 1e-7},
        # --- Error: zero-length propagation ---
        {"setup": base + "def _probe(fn):\n    try:\n        fn(dict(P), -30.0, 4.0, 4.0, 10, 1.0)\n    except ValueError:\n        return 1\n    return 0\n",
         "call": "_probe(propagate_log_derivative)", "gold_call": "_probe(_oracle_propagate_log_derivative)"},
    ]
