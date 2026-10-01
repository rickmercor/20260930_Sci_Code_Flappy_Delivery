"""
Build the weighted Kirchhoff (Laplacian) matrix of the residue contact graph from the C-alpha coordinates. Every pair of residues whose C-alpha distance does not exceed the contact cutoff r_c is an edge, and each edge carries the source's distance-dependent Boltzmann weight at the effective temperature kT; covalently bonded neighbours are treated exactly like every other contact. Row sums vanish.

A protein backbone becomes a weighted graph once residues are vertices and spatial contacts are edges. The weighted Laplacian of that graph is the single object from which the spanning-tree partition function, the effective resistances and the path probabilities of the framework all follow.

Returns
-------
ndarray of float64, shape (N, N), the weighted Laplacian L = D - W of the contact graph.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def contact_laplacian(coords: "np.ndarray", r_c: float, kT: float) -> "np.ndarray":
    """Build the weighted Kirchhoff (Laplacian) matrix of the residue contact graph from the C-alpha coordinates. Every pair of residues whose C-alpha distance does not exceed the contact cutoff r_c is an edge, and each edge carries the source's distance-dependent Boltzmann weight at the effective temperature kT; covalently bonded neighbours are treated exactly like every other contact. Row sums vanish.

    Parameters
    ----------
    coords : np.ndarray
        C-alpha coordinates in angstrom, shape (N, 3).
    r_c : float
        Contact cutoff in angstrom; pairs with distance <= r_c are edges.
    kT : float
        Effective temperature in angstrom.

    Returns
    -------
    L : np.ndarray
        Weighted Laplacian of shape (N, N).

    Raises
    ------
    ValueError
        If coords is not a finite (N, 3) array with N >= 3, or r_c or kT is not positive.
    """
    return L

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from numpy.linalg import det, slogdet


def _oracle_contact_laplacian(coords: "np.ndarray", r_c: float, kT: float) -> "np.ndarray":
    """Eq. (1): w_ij = exp(-d_ij/kT) for every C-alpha pair with d_ij <= r_c (backbone on the same footing)."""
    X = np.asarray(coords, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != 3 or X.shape[0] < 3 or not np.all(np.isfinite(X)):
        raise ValueError("coords must be a finite (N, 3) array with N >= 3")
    r_c, kT = float(r_c), float(kT)
    if not (np.isfinite(r_c) and r_c > 0.0 and np.isfinite(kT) and kT > 0.0):
        raise ValueError("r_c and kT must be finite positive numbers")
    n = X.shape[0]
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    W = np.where((D <= r_c) & ~np.eye(n, dtype=bool), np.exp(-D / kT), 0.0)
    L = np.diag(W.sum(axis=1)) - W
    return L

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nimport numpy as np\nX = np.array([[2.3, 0.0, 0.0], [-0.4, 2.27, 1.5], [-2.16, -0.79, 3.0], [1.15, -1.99, 4.5], [1.76, 1.48, 6.0], [-1.76, 1.48, 7.5], [-1.15, -1.99, 9.0], [2.16, -0.79, 10.5], [0.4, 2.27, 12.0], [-2.3, 0.0, 13.5], [0.4, -2.27, 15.0], [2.38, 1.63, 17.41], [4.75, 2.3, 18.0], [7.12, 1.63, 17.41], [11.26, 1.48, 15.0], [7.74, 1.48, 13.5], [8.35, -1.99, 12.0], [11.66, -0.78, 10.5], [9.9, 2.27, 9.0], [7.2, -0.0, 7.5], [9.9, -2.26, 6.0], [11.66, 0.79, 4.5], [8.35, 1.99, 3.0], [7.74, -1.48, 1.5], [11.26, -1.48, 0.0], [10.65, 1.99, -1.5]])\nr_c, kT = 7.8, 1.0\n",
            "call": "np.asarray(contact_laplacian(X, r_c, kT))",
            "gold_call": "np.asarray(_oracle_contact_laplacian(X, r_c, kT))",
        },
        {
            "setup": "import numpy as np\nimport numpy as np\nX = np.array([[2.3, 0.0, 0.0], [-0.4, 2.27, 1.5], [-2.16, -0.79, 3.0], [1.15, -1.99, 4.5], [1.76, 1.48, 6.0], [-1.76, 1.48, 7.5], [-1.15, -1.99, 9.0], [2.16, -0.79, 10.5], [0.4, 2.27, 12.0], [-2.3, 0.0, 13.5], [0.4, -2.27, 15.0], [2.38, 1.63, 17.41], [4.75, 2.3, 18.0], [7.12, 1.63, 17.41], [11.26, 1.48, 15.0], [7.74, 1.48, 13.5], [8.35, -1.99, 12.0], [11.66, -0.78, 10.5], [9.9, 2.27, 9.0], [7.2, -0.0, 7.5], [9.9, -2.26, 6.0], [11.66, 0.79, 4.5], [8.35, 1.99, 3.0], [7.74, -1.48, 1.5], [11.26, -1.48, 0.0], [10.65, 1.99, -1.5]])\nr_c, kT = 7.0, 1.2\n",
            "call": "np.asarray(contact_laplacian(X, r_c, kT))",
            "gold_call": "np.asarray(_oracle_contact_laplacian(X, r_c, kT))",
        },
        {
            "setup": "import numpy as np\nimport numpy as np\nX = np.array([[0.0, 0.0, 0.0], [3.8, 0.0, 0.0], [5.5, 3.4, 0.0], [3.8, 6.8, 0.0], [0.0, 6.8, 0.5], [-2.5, 3.4, 1.0], [1.9, 3.4, 4.0]])\nr_c, kT = 5.5, 0.8\n",
            "call": "np.asarray(contact_laplacian(X, r_c, kT))",
            "gold_call": "np.asarray(_oracle_contact_laplacian(X, r_c, kT))",
        },
        {
            "setup": "import numpy as np\nimport numpy as np\nX = np.array([[2.3, 0.0, 0.0], [-0.4, 2.27, 1.5], [-2.16, -0.79, 3.0], [1.15, -1.99, 4.5], [1.76, 1.48, 6.0], [-1.76, 1.48, 7.5], [-1.15, -1.99, 9.0], [2.16, -0.79, 10.5], [0.4, 2.27, 12.0], [-2.3, 0.0, 13.5], [0.4, -2.27, 15.0], [2.38, 1.63, 17.41], [4.75, 2.3, 18.0], [7.12, 1.63, 17.41], [11.26, 1.48, 15.0], [7.74, 1.48, 13.5], [8.35, -1.99, 12.0], [11.66, -0.78, 10.5], [9.9, 2.27, 9.0], [7.2, -0.0, 7.5], [9.9, -2.26, 6.0], [11.66, 0.79, 4.5], [8.35, 1.99, 3.0], [7.74, -1.48, 1.5], [11.26, -1.48, 0.0], [10.65, 1.99, -1.5]])\nr_c, kT = 7.8, 1.0\ndef run_model():\n    try:\n        contact_laplacian(X, -1.0, kT)\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_contact_laplacian(X, -1.0, kT)\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
