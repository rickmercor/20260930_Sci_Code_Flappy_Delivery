"""
Step 03: Lowest eigenstates of soft-Coulomb protons on a finite-difference grid.

Lowest one-electron eigenstates of a one-dimensional chain of soft-Coulomb protons on a uniform grid.

A one-dimensional model of one-electron molecules such as H and H2+ places unit positive charges at fixed positions X_A
on a line and lets the electron feel v(x) = -sum_A 1 / sqrt((x - X_A)^2 + b^2), with the same softening length b as the
electron-electron interaction. The one-electron Hamiltonian is h = -(1/2) d^2/dx^2 + v(x) in atomic units.

The problem is discretized on the uniform grid x_j = -L + j*h_x, j = 0, 1, ..., M, with M = 2L/h_x an integer, and the
orbital is taken to vanish outside the grid. The kinetic operator is the three-point finite difference
(T phi)_j = -(phi_(j+1) - 2 phi_j + phi_(j-1)) / (2 h_x^2) with phi_(-1) = phi_(M+1) = 0, so h becomes a real symmetric
tridiagonal matrix whose eigenvalues and eigenvectors are the discrete energies and orbitals. Orbitals are normalized
with the rectangle rule, h_x sum_j phi_j^2 = 1.

Because eigenvectors are defined only up to sign, a sign convention is fixed: scanning the grid from the left, the first
point at which |phi_j| reaches at least 10^-3 of max_k |phi_k| must carry a positive value. With this convention the
gerade and ungerade states of a symmetric diatomic are both positive on the left atom.

Returns
-------
numpy.ndarray of shape (n_states, M + 2), energies in column 0 and normalized, sign-fixed grid orbitals in columns 1 to M + 1
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def soft_coulomb_states(nuclei: "np.ndarray", softening: float, spacing: float, half_width: float, n_states: int) -> "np.ndarray":
    '''Lowest eigenvalues and normalized eigenvectors of the finite-difference soft-Coulomb one-electron Hamiltonian.

    Parameters
    ----------
    nuclei : np.ndarray
        Shape (P,), P >= 1, positions of unit point charges in bohr.
    softening : float
        Softening length b > 0 of the electron-nucleus attraction, in bohr.
    spacing : float
        Grid spacing h_x > 0 in bohr.
    half_width : float
        Half-width L of the grid in bohr; 2L / h_x must be an integer M within a relative tolerance of 1e-9.
    n_states : int
        Number of lowest states to return, 1 <= n_states <= M + 1.

    Returns
    -------
    result : np.ndarray
        Shape (n_states, M + 2). Row k holds state k in order of increasing energy: column 0 is the energy in hartree and
        columns 1 to M + 1 are the orbital values at x_0, ..., x_M, normalized so that h_x sum_j phi_j^2 = 1 and signed
        so that the first grid point from the left with |phi_j| >= 10^-3 max|phi| is positive.

    Raises
    ------
    ValueError
        If nuclei is empty, the softening, spacing or half-width is not positive, 2L / h_x is not an integer, or
        n_states is outside 1 <= n_states <= M + 1.
    '''
    return result  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.linalg import eigh_tridiagonal


def _oracle_soft_coulomb_states(nuclei: "np.ndarray", softening: float, spacing: float, half_width: float, n_states: int) -> "np.ndarray":
    """Reference implementation."""
    import numpy as np
    from scipy.linalg import eigh_tridiagonal
    centers = np.asarray(nuclei, dtype=float).ravel()
    if centers.size == 0:
        raise ValueError("at least one nucleus is required")
    if not (softening > 0.0 and spacing > 0.0 and half_width > 0.0):
        raise ValueError("softening, spacing and half-width must be positive")
    ratio = 2.0 * half_width / spacing
    m = int(round(ratio))
    if abs(ratio - m) > 1e-9 * max(1.0, ratio) or m < 1:
        raise ValueError("2 * half_width / spacing must be an integer")
    if not 1 <= int(n_states) <= m + 1:
        raise ValueError("n_states must lie between 1 and M + 1")
    x = -half_width + spacing * np.arange(m + 1)
    v = np.zeros_like(x)
    for c in centers:
        v -= 1.0 / np.sqrt((x - c) ** 2 + softening ** 2)
    diag = 1.0 / spacing ** 2 + v
    off = np.full(m, -0.5 / spacing ** 2)
    k = int(n_states)
    w, u = eigh_tridiagonal(diag, off, select="i", select_range=(0, k - 1))
    phi = u / np.sqrt(spacing)
    for i in range(k):
        col = phi[:, i]
        first = int(np.argmax(np.abs(col) >= 1e-3 * np.max(np.abs(col))))
        if col[first] < 0.0:
            phi[:, i] = -col
    out = np.empty((k, m + 2))
    out[:, 0] = w
    out[:, 1:] = phi.T
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: one-dimensional hydrogen atom, three lowest states ---
        {
            "setup": "import numpy as np\n",
            "call": "soft_coulomb_states(np.array([0.0]), 1.0, 0.1, 20.0, 3)",
            "gold_call": "_oracle_soft_coulomb_states(np.array([0.0]), 1.0, 0.1, 20.0, 3)",
            "tol": 1e-8,
        },
        # --- Normal: stretched symmetric H2+ with its gerade and ungerade pair ---
        {
            "setup": "import numpy as np\n",
            "call": "soft_coulomb_states(np.array([-2.6, 2.6]), 1.0, 0.1, 18.0, 2)",
            "gold_call": "_oracle_soft_coulomb_states(np.array([-2.6, 2.6]), 1.0, 0.1, 18.0, 2)",
            "tol": 1e-8,
        },
        # --- Normal: asymmetric two-center arrangement with a shorter softening ---
        {
            "setup": "import numpy as np\n",
            "call": "soft_coulomb_states(np.array([-1.3, 2.2]), 0.7, 0.05, 12.0, 2)",
            "gold_call": "_oracle_soft_coulomb_states(np.array([-1.3, 2.2]), 0.7, 0.05, 12.0, 2)",
            "tol": 1e-8,
        },
        # --- Edge: three centers, where the second state has a node at the central nucleus ---
        {
            "setup": "import numpy as np\n",
            "call": "soft_coulomb_states(np.array([-3.0, 0.0, 3.0]), 1.2, 0.08, 16.0, 3)",
            "gold_call": "_oracle_soft_coulomb_states(np.array([-3.0, 0.0, 3.0]), 1.2, 0.08, 16.0, 3)",
            "tol": 1e-8,
        },
        # --- Boundary: a box so small that the Dirichlet walls raise the energy noticeably ---
        {
            "setup": "import numpy as np\n",
            "call": "soft_coulomb_states(np.array([0.5]), 1.0, 0.125, 2.5, 2)",
            "gold_call": "_oracle_soft_coulomb_states(np.array([0.5]), 1.0, 0.125, 2.5, 2)",
            "tol": 1e-8,
        },
        # --- Error: a half-width that is not an integer multiple of half the spacing must raise ValueError ---
        {
            "setup": "import numpy as np\n"
                     "def _probe(fn):\n"
                     "    try:\n"
                     "        fn(np.array([0.0]), 1.0, 0.3, 10.0, 1)\n"
                     "    except ValueError:\n"
                     "        return 1\n"
                     "    except Exception:\n"
                     "        return 2\n"
                     "    return 0\n",
            "call": "_probe(soft_coulomb_states)",
            "gold_call": "_probe(_oracle_soft_coulomb_states)",
        },
    ]
