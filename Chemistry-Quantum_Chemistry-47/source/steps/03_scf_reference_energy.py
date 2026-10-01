"""
Converged self-consistent-field reference energy of a molecule of the site model.

The Roothaan equations are a generalized symmetric eigenproblem because the site basis is not orthogonal. The closed-shell density is built from the lowest n/2 eigenvectors for n electrons. Starting from an empty density, so that the first Fock matrix equals the core Hamiltonian, the density from each diagonalization is mixed with the previous one until the largest change in any density element drops below a tolerance; the energy is then evaluated for the last diagonalized density.

Returns
-------
float, the converged SCF total energy as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def scf_reference_energy(positions: "np.ndarray", types: "np.ndarray", params: dict, tol: float = 1e-10, max_iter: int = 500, damping: float = 0.5) -> float:
    """Return the converged closed-shell SCF total energy.

    Parameters
    ----------
    positions : np.ndarray
        Shape (n,) atom positions; n must be even.
    types : np.ndarray
        Shape (n,) element types, 0 or 1.
    params : dict
        Model constants used by the matrix builder and the energy evaluation.
    tol : float
        Convergence threshold on the largest absolute change of a density
        element between one diagonalization and the next.
    max_iter : int
        Maximum number of diagonalizations.
    damping : float
        Fraction of the new density mixed into the iterate after a
        non-converged diagonalization; the rest is the previous iterate.

    Returns
    -------
    energy : float
        Total energy of the converged density as a native Python float. Each
        iteration builds the Fock matrix from the current density, solves the
        generalized eigenproblem with the overlap, forms the new density from
        the n/2 lowest eigenvectors and stops when the largest change is below
        tol, returning the energy of that new density; otherwise the damped
        mixture becomes the next iterate. The result does not depend on the
        damping value.

    Raises
    ------
    ValueError
        If the atom count is odd or convergence is not reached within
        max_iter diagonalizations.
    """
    return energy

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.linalg import eigh

def _oracle_scf_reference_energy(positions: "np.ndarray", types: "np.ndarray", params: dict, tol: float = 1e-10, max_iter: int = 500, damping: float = 0.5) -> float:
    pos = np.asarray(positions, dtype=float)
    n = pos.shape[0]
    if n % 2:
        raise ValueError("the model needs an even number of atoms")
    S, H, gamma = _oracle_build_model_matrices(pos, types, params)
    P = np.zeros((n, n))
    for _ in range(int(max_iter)):
        F = _build_fock(H, gamma, P)
        P_new = _occupied_density(F, S, n)
        if np.max(np.abs(P_new - P)) < tol:
            return _oracle_mean_field_energy(H, gamma, P_new, pos, params)
        P = (1.0 - damping) * P + damping * P_new
    raise ValueError("SCF did not converge within max_iter iterations")

def _occupied_density(M: "np.ndarray", S: "np.ndarray", n_electrons: int) -> "np.ndarray":
    if n_electrons % 2:
        raise ValueError("closed-shell density needs an even electron count")
    _, C = eigh(np.asarray(M, dtype=float), np.asarray(S, dtype=float))
    occ = C[:, : n_electrons // 2]
    return 2.0 * occ @ occ.T

def _converged_scf(positions: "np.ndarray", types: "np.ndarray", params: dict, tol: float = 1e-12, max_iter: int = 5000) -> tuple:
    """Overlap, core Hamiltonian, Ohno matrix, density, orbital energies and coefficients at convergence."""
    pos = np.asarray(positions, dtype=float)
    n = pos.shape[0]
    if n % 2:
        raise ValueError("the model needs an even number of atoms")
    S, H, gamma = _oracle_build_model_matrices(pos, types, params)
    P = np.zeros((n, n))
    for _ in range(int(max_iter)):
        F = _build_fock(H, gamma, P)
        w, C = eigh(F, S)
        occ = C[:, : n // 2]
        P_new = 2.0 * occ @ occ.T
        if np.max(np.abs(P_new - P)) < tol:
            return S, H, gamma, P_new, w, C
        P = 0.5 * (P + P_new)
    raise ValueError("SCF did not converge within max_iter iterations")

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Test cases for this step."""
    params = ("params = {'onsite_energy': [-1.0, -1.6], 'onsite_repulsion': [0.9, 1.3], "
              "'kappa': 1.75, 'sigma': 1.1, 'core_repulsion_strength': 6.0, 'core_repulsion_range': 0.45}\n")
    return [
        # homonuclear diatomic near its minimum
        {
            "setup": "import numpy as np\nimport copy\n" + params + "positions = np.array([0.0, 1.2752])\ntypes = np.array([0, 0])\n",
            "call": "scf_reference_energy(*copy.deepcopy((positions, types, params)))",
            "gold_call": "_oracle_scf_reference_energy(positions, types, params)",
        },
        # four-atom chain, two occupied orbitals
        {
            "setup": "import numpy as np\nimport copy\n" + params + "positions = np.array([0.0, 1.0399, 2.7489, 3.7827])\ntypes = np.array([0, 1, 0, 1])\n",
            "call": "scf_reference_energy(*copy.deepcopy((positions, types, params)))",
            "gold_call": "_oracle_scf_reference_energy(positions, types, params)",
        },
        # stretched heteronuclear bond converges slowly; different damping, same energy
        {
            "setup": "import numpy as np\nimport copy\n" + params + "positions = np.array([0.0, 2.6])\ntypes = np.array([0, 1])\n",
            "call": "scf_reference_energy(*copy.deepcopy((positions, types, params, 1e-11, 2000, 0.3)))",
            "gold_call": "_oracle_scf_reference_energy(positions, types, params, 1e-11, 2000, 0.3)",
        },
        # compressed geometry on the repulsive wall
        {
            "setup": "import numpy as np\nimport copy\n" + params + "positions = np.array([0.0, 0.7])\ntypes = np.array([1, 1])\n",
            "call": "scf_reference_energy(*copy.deepcopy((positions, types, params)))",
            "gold_call": "_oracle_scf_reference_energy(positions, types, params)",
        },
        # odd atom count is rejected
        {
            "setup": "import numpy as np\nimport copy\n" + params + """positions = np.array([0.0, 1.1, 2.2])
types = np.array([0, 1, 0])
def run_model():
    try:
        scf_reference_energy(positions, types, params)
        return 0
    except ValueError:
        return 1
def run_oracle():
    try:
        _oracle_scf_reference_energy(positions, types, params)
        return 0
    except ValueError:
        return 1
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
        # too few iterations to converge
        {
            "setup": "import numpy as np\nimport copy\n" + params + """positions = np.array([0.0, 1.0399, 2.7489, 3.7827])
types = np.array([0, 1, 0, 1])
def run_model():
    try:
        scf_reference_energy(positions, types, params, 1e-10, 2, 0.5)
        return 0
    except ValueError:
        return 1
def run_oracle():
    try:
        _oracle_scf_reference_energy(positions, types, params, 1e-10, 2, 0.5)
        return 0
    except ValueError:
        return 1
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
