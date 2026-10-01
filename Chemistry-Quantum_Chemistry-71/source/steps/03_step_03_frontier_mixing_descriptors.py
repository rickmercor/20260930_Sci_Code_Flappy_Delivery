"""
Step 03: Frontier orbital mixing angle and ionic and covalent weights. Orbital mixing angle of the frontier bond and the ionic and covalent weights it implies.

A two-electron bond described by one bonding and one antibonding orbital can be written as a normalised mixture of the configuration with both electrons in the bonding orbital and the configuration with both electrons in the antibonding orbital. The mixture defines a point on a circle, so the whole family is labelled by one angle, and the same state can be read as a superposition of a purely covalent (diradical) and a purely ionic (zwitterionic) valence-bond structure. The angle is zero for a closed-shell bond, grows as the bond is broken, and reaches its largest value at a perfect open-shell singlet diradical; the two valence-bond weights follow from it. In a correlated many-electron calculation the angle is recovered from the occupation of the highest occupied natural orbital, which turns the qualitative idea of bonding disruption in a forbidden transition state into a number.

Returns
-------
numpy.ndarray [Theta in degrees, W_ionic, W_cov] of the lowest singlet from the highest occupied natural orbital
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def frontier_mixing_descriptors(h: "np.ndarray", U: float) -> "np.ndarray":
    """Frontier orbital mixing angle and ionic/covalent weights of the lowest singlet of a half-filled Hubbard model.

    Parameters
    ----------
    h : np.ndarray
        Real symmetric (n, n) one-electron matrix in eV, as in singlet_ground_state.
    U : float
        On-site repulsion in eV, finite and >= 0.

    Returns
    -------
    d : np.ndarray
        Real array (Theta, W_ionic, W_cov), in that order. Theta is in degrees and is obtained from the highest occupied
        natural-orbital occupation of the lowest singlet using the source paper's frontier-bond construction; W_ionic
        and W_cov are the corresponding source-defined ionic and covalent weights.

    Raises
    ------
    ValueError
        If h or U violates the conditions of singlet_ground_state.
    """
    return d

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_frontier_mixing_descriptors(h: "np.ndarray", U: float) -> "np.ndarray":
    import numpy as np
    res = _oracle_singlet_ground_state(h, U)
    n = len(res) - 1
    n_h = min(max(res[n // 2], 0.0), 2.0)
    theta = np.arccos(np.sqrt(n_h / 2.0))
    return np.array([np.degrees(theta), 0.5 * (1.0 - np.sin(2.0 * theta)), 0.5 * (1.0 + np.sin(2.0 * theta))])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    base = ("import numpy as np\n"
            "def _ring(n, t, tn):\n    h = np.zeros((n, n))\n    for i in range(n - 1):\n        h[i, i + 1] = h[i + 1, i] = t[i % len(t)]\n"
            "    h[0, n - 1] = h[n - 1, 0] = tn\n    return h\n")
    return [
        # --- Normal: Moebius-like hexatriene ring near a frontier level crossing ---
        {"setup": base, "call": "np.asarray(frontier_mixing_descriptors(_ring(6, [-1.9, -2.2], 0.9), 4.0), dtype=float)",
         "gold_call": "np.asarray(_oracle_frontier_mixing_descriptors(_ring(6, [-1.9, -2.2], 0.9), 4.0), dtype=float)", "tol": 1e-8},
        # --- Normal: Hueckel-like ring, small mixing angle ---
        {"setup": base, "call": "np.asarray(frontier_mixing_descriptors(_ring(6, [-1.9, -2.2], -0.9), 4.0), dtype=float)",
         "gold_call": "np.asarray(_oracle_frontier_mixing_descriptors(_ring(6, [-1.9, -2.2], -0.9), 4.0), dtype=float)", "tol": 1e-8},
        # --- Normal: two-site bond at intermediate repulsion ---
        {"setup": base, "call": "np.asarray(frontier_mixing_descriptors(np.array([[0.0, -1.5], [-1.5, 0.0]]), 5.0), dtype=float)",
         "gold_call": "np.asarray(_oracle_frontier_mixing_descriptors(np.array([[0.0, -1.5], [-1.5, 0.0]]), 5.0), dtype=float)", "tol": 1e-9},
        # --- Boundary: weak repulsion, nearly closed shell with a small angle ---
        {"setup": base, "call": "np.asarray(frontier_mixing_descriptors(_ring(6, [-2.8, -2.2], -0.3), 0.3), dtype=float)",
         "gold_call": "np.asarray(_oracle_frontier_mixing_descriptors(_ring(6, [-2.8, -2.2], -0.3), 0.3), dtype=float)", "tol": 1e-8},
        # --- Boundary: strongly correlated two-site bond, close to a pure diradical ---
        {"setup": base, "call": "np.asarray(frontier_mixing_descriptors(np.array([[0.0, -0.2], [-0.2, 0.0]]), 12.0), dtype=float)",
         "gold_call": "np.asarray(_oracle_frontier_mixing_descriptors(np.array([[0.0, -0.2], [-0.2, 0.0]]), 12.0), dtype=float)", "tol": 1e-8},
        # --- Error: negative repulsion ---
        {"setup": base + "def _probe(fn):\n    try:\n        fn(np.array([[0.0, -1.0], [-1.0, 0.0]]), -1.0)\n    except ValueError:\n        return 1\n    return 0\n",
         "call": "_probe(frontier_mixing_descriptors)", "gold_call": "_probe(_oracle_frontier_mixing_descriptors)"},
    ]
