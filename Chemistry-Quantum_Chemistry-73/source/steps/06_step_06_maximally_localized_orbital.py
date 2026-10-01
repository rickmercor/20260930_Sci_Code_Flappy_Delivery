"""
Step 06: Maximally left-localized orbital in a two-state span.

Orbital of largest probability on the left half-line within the span of two orthonormal orbitals.

At large separation the two lowest states of a symmetric one-electron diatomic are nearly degenerate, and the ground
state spreads the electron evenly over both atoms. A localized alternative that dissociates into an integer-charge end
point (the electron on one atom, a bare proton on the other) can be built without leaving the space of those two states:
among all normalized combinations phi = c_1 phi_1 + c_2 phi_2 with c_1^2 + c_2^2 = 1, take the one that puts the most
probability on the half-line x < 0, where the left nucleus sits.

On the uniform grid x_j = -L + j*h_x the probability on the left is h_x sum_j s_j phi_j^2 with weight s_j = 1 for
x_j < 0, s_j = 1/2 for a grid point at x_j = 0 and s_j = 0 for x_j > 0, so that a state symmetric about the origin has
exactly half its probability on each side, and the input orbitals are orthonormal under h_x sum_j. The maximizer is
unique when the two candidate probabilities of the optimization differ; its overall sign is fixed so that
h_x sum_j s_j phi_j is positive.

Returns
-------
numpy.ndarray of shape (M + 1,), normalized combination of the two orbitals with the largest probability on x < 0
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def maximally_localized_orbital(orbital_one: "np.ndarray", orbital_two: "np.ndarray", spacing: float, half_width: float) -> "np.ndarray":
    '''Normalized combination of two orthonormal grid orbitals with the largest probability on x < 0.

    Parameters
    ----------
    orbital_one : np.ndarray
        Shape (M + 1,), real orbital on the grid x_j = -L + j*h_x.
    orbital_two : np.ndarray
        Shape (M + 1,), real orbital on the same grid, orthonormal to orbital_one within 1e-8 under h_x sum_j.
    spacing : float
        Grid spacing h_x > 0 in bohr.
    half_width : float
        Half-width L of the grid in bohr, with 2L / h_x = M.

    Returns
    -------
    result : np.ndarray
        Shape (M + 1,), the combination c_1 orbital_one + c_2 orbital_two with c_1^2 + c_2^2 = 1 that maximizes
        h_x sum_j s_j phi_j^2, with s_j = 1 for x_j < 0, 1/2 at x_j = 0 and 0 for x_j > 0, signed so that
        h_x sum_j s_j phi_j > 0.

    Raises
    ------
    ValueError
        If the orbitals do not have length 2L / h_x + 1, are not orthonormal within 1e-8, or if the maximizing
        combination is not unique because the two extreme left-half probabilities coincide within 1e-12.
    '''
    return result  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_maximally_localized_orbital(orbital_one: "np.ndarray", orbital_two: "np.ndarray", spacing: float, half_width: float) -> "np.ndarray":
    """Reference implementation."""
    import numpy as np
    a = np.asarray(orbital_one, dtype=float).ravel()
    b = np.asarray(orbital_two, dtype=float).ravel()
    m = int(round(2.0 * half_width / spacing))
    if a.size != m + 1 or b.size != m + 1:
        raise ValueError("orbitals must have length 2L / h_x + 1")
    gram = spacing * np.array([[a @ a, a @ b], [a @ b, b @ b]])
    if np.max(np.abs(gram - np.eye(2))) > 1e-8:
        raise ValueError("orbitals must be orthonormal")
    x = -half_width + spacing * np.arange(m + 1)
    weight = np.where(x < -1e-12 * half_width, 1.0, np.where(x > 1e-12 * half_width, 0.0, 0.5))
    p = spacing * np.array([[a @ (weight * a), a @ (weight * b)], [a @ (weight * b), b @ (weight * b)]])
    w, vecs = np.linalg.eigh(p)
    if w[1] - w[0] < 1e-12:
        raise ValueError("the maximally localized combination is not unique")
    c = vecs[:, 1]
    phi = c[0] * a + c[1] * b
    if spacing * float(weight @ phi) < 0.0:
        phi = -phi
    return phi

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: gerade and ungerade states of stretched H2+ built directly with a dense eigensolver ---
        {
            "setup": "import numpy as np\n"
                     "h, L = 0.1, 15.0\n"
                     "x = -L + h * np.arange(301)\n"
                     "v = -1.0 / np.sqrt((x - 2.0) ** 2 + 1.0) - 1.0 / np.sqrt((x + 2.0) ** 2 + 1.0)\n"
                     "H = np.diag(1.0 / h ** 2 + v) - 0.5 / h ** 2 * (np.eye(301, k=1) + np.eye(301, k=-1))\n"
                     "w, U = np.linalg.eigh(H)\n"
                     "g = U[:, 0] / np.sqrt(h)\n"
                     "u = U[:, 1] / np.sqrt(h)\n",
            "call": "maximally_localized_orbital(g.copy(), u.copy(), h, L)",
            "gold_call": "_oracle_maximally_localized_orbital(g.copy(), u.copy(), h, L)",
            "tol": 1e-9,
        },
        # --- Normal: the same pair rotated by an arbitrary angle and with flipped signs spans the same space ---
        {
            "setup": "import numpy as np\n"
                     "h, L = 0.1, 15.0\n"
                     "x = -L + h * np.arange(301)\n"
                     "v = -1.0 / np.sqrt((x - 2.0) ** 2 + 1.0) - 1.0 / np.sqrt((x + 2.0) ** 2 + 1.0)\n"
                     "H = np.diag(1.0 / h ** 2 + v) - 0.5 / h ** 2 * (np.eye(301, k=1) + np.eye(301, k=-1))\n"
                     "w, U = np.linalg.eigh(H)\n"
                     "g = U[:, 0] / np.sqrt(h)\n"
                     "u = U[:, 1] / np.sqrt(h)\n"
                     "t = 0.37\n"
                     "a = -(np.cos(t) * g + np.sin(t) * u)\n"
                     "b = -np.sin(t) * g + np.cos(t) * u\n",
            "call": "maximally_localized_orbital(a.copy(), b.copy(), h, L)",
            "gold_call": "_oracle_maximally_localized_orbital(a.copy(), b.copy(), h, L)",
            "tol": 1e-9,
        },
        # --- Normal: an asymmetric double well, where the optimal weights are not equal ---
        {
            "setup": "import numpy as np\n"
                     "h, L = 0.05, 10.0\n"
                     "x = -L + h * np.arange(401)\n"
                     "v = -1.2 / np.sqrt((x + 1.8) ** 2 + 0.7) - 1.0 / np.sqrt((x - 2.1) ** 2 + 1.0)\n"
                     "H = np.diag(1.0 / h ** 2 + v) - 0.5 / h ** 2 * (np.eye(401, k=1) + np.eye(401, k=-1))\n"
                     "w, U = np.linalg.eigh(H)\n"
                     "p = U[:, 0] / np.sqrt(h)\n"
                     "q = U[:, 1] / np.sqrt(h)\n",
            "call": "maximally_localized_orbital(p.copy(), q.copy(), h, L)",
            "gold_call": "_oracle_maximally_localized_orbital(p.copy(), q.copy(), h, L)",
            "tol": 1e-9,
        },
        # --- Edge: generic orthonormal vectors from a seeded QR factorization ---
        {
            "setup": "import numpy as np\n"
                     "h, L = 0.5, 4.0\n"
                     "rng = np.random.default_rng(11)\n"
                     "Q, _ = np.linalg.qr(rng.normal(size=(17, 2)))\n"
                     "a = Q[:, 0] / np.sqrt(h)\n"
                     "b = Q[:, 1] / np.sqrt(h)\n",
            "call": "maximally_localized_orbital(a.copy(), b.copy(), h, L)",
            "gold_call": "_oracle_maximally_localized_orbital(a.copy(), b.copy(), h, L)",
            "tol": 1e-9,
        },
        # --- Boundary: one input vanishes on the left half-line and is nonzero at the origin, where the weight is 1/2 ---
        {
            "setup": "import numpy as np\n"
                     "h, L = 0.25, 2.0\n"
                     "x = -L + h * np.arange(17)\n"
                     "right = np.where(x >= 0.0, np.exp(-x), 0.0)\n"
                     "right = right / np.sqrt(h * right @ right)\n"
                     "mixed = np.exp(-0.3 * (x + 1.0) ** 2)\n"
                     "mixed = mixed - h * (mixed @ right) * right\n"
                     "mixed = mixed / np.sqrt(h * mixed @ mixed)\n",
            "call": "maximally_localized_orbital(right.copy(), mixed.copy(), h, L)",
            "gold_call": "_oracle_maximally_localized_orbital(right.copy(), mixed.copy(), h, L)",
            "tol": 1e-9,
        },
        # --- Error: orbitals that are not orthonormal must raise ValueError ---
        {
            "setup": "import numpy as np\n"
                     "def _probe(fn):\n"
                     "    try:\n"
                     "        fn(np.ones(9), np.linspace(0.0, 1.0, 9), 0.5, 2.0)\n"
                     "    except ValueError:\n"
                     "        return 1\n"
                     "    except Exception:\n"
                     "        return 2\n"
                     "    return 0\n",
            "call": "_probe(maximally_localized_orbital)",
            "gold_call": "_probe(_oracle_maximally_localized_orbital)",
        },
    ]
