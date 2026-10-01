"""
Apply a program of matrix primitives to the workspace matrix that approximates the Fock matrix.

A program is an ordered sequence of indices into a fixed library of parameter-free primitives. The workspace matrix is initialized with the core Hamiltonian (the input layer) and each primitive rewrites it in place, using the overlap matrix, the core Hamiltonian and the screened Coulomb matrix of the molecule as fixed inputs. Only diagonalization-free operations are allowed, so that a single diagonalization of the final workspace matrix is all that is needed to obtain orbital coefficients.

Returns
-------
np.ndarray, shape (n, n) float workspace matrix after the last primitive
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def apply_workspace_program(program: "np.ndarray", S: "np.ndarray", H: "np.ndarray", gamma: "np.ndarray") -> "np.ndarray":
    """Return the workspace matrix produced by applying the program to H.

    Parameters
    ----------
    program : np.ndarray
        Shape (N_f,) integer array of primitive indices applied in order.
        The library is: 0, leave M unchanged; 1, add the core Hamiltonian;
        2, add the overlap; 3, replace M by the product M times the inverse
        overlap times M; 4, add the two-electron part of the Fock matrix
        evaluated for the density with one electron on every site (the
        identity matrix); 5, multiply M elementwise by the overlap; 6, halve M.
        An empty program returns H.
    S : np.ndarray
        Shape (n, n) symmetric positive definite overlap matrix.
    H : np.ndarray
        Shape (n, n) core Hamiltonian.
    gamma : np.ndarray
        Shape (n, n) screened Coulomb matrix.

    Returns
    -------
    M : np.ndarray
        Shape (n, n) float workspace matrix after the last primitive.

    Raises
    ------
    ValueError
        If any program entry is not an integer in 0..6, or S, H and gamma are
        not square matrices of one common shape.
    """
    return M

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_apply_workspace_program(program: "np.ndarray", S: "np.ndarray", H: "np.ndarray", gamma: "np.ndarray") -> "np.ndarray":
    S = np.asarray(S, dtype=float)
    H = np.asarray(H, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    n = H.shape[0]
    if S.shape != (n, n) or H.shape != (n, n) or gamma.shape != (n, n):
        raise ValueError("S, H and gamma must share one square shape")
    ops = np.asarray(program).reshape(-1)
    if ops.size and (not np.issubdtype(ops.dtype, np.integer) or ops.min() < 0 or ops.max() > 6):
        raise ValueError("program entries must be integers in 0..6")
    S_inv = np.linalg.inv(S)
    atomic_field = _build_fock(np.zeros((n, n)), gamma, np.eye(n))
    M = H.copy()
    for op in ops:
        if op == 1:
            M = M + H
        elif op == 2:
            M = M + S
        elif op == 3:
            M = M @ S_inv @ M
        elif op == 4:
            M = M + atomic_field
        elif op == 5:
            M = M * S
        elif op == 6:
            M = 0.5 * M
    return M

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Test cases for this step."""
    params = ("params = {'onsite_energy': [-1.0, -1.6], 'onsite_repulsion': [0.9, 1.3], "
              "'kappa': 1.75, 'sigma': 1.1, 'core_repulsion_strength': 6.0, 'core_repulsion_range': 0.45}\n")
    chain = ("positions = np.array([0.0, 1.0399, 2.7489, 3.7827])\n"
             "types = np.array([0, 1, 0, 1])\n"
             "S, H, gamma = build_model_matrices(positions, types, params)\n")
    diatomic = ("positions = np.array([0.0, 1.1659])\n"
                "types = np.array([0, 1])\n"
                "S, H, gamma = build_model_matrices(positions, types, params)\n")
    return [
        # every primitive once, in library order
        {
            "setup": "import numpy as np\nimport copy\n" + params + chain + "program = np.array([1, 2, 3, 4, 5, 6])\n",
            "call": "apply_workspace_program(*copy.deepcopy((program, S, H, gamma)))",
            "gold_call": "_oracle_apply_workspace_program(program, S, H, gamma)",
        },
        # generalized square applied twice on a non-orthogonal basis
        {
            "setup": "import numpy as np\nimport copy\n" + params + diatomic + "program = np.array([3, 3, 4])\n",
            "call": "apply_workspace_program(*copy.deepcopy((program, S, H, gamma)))",
            "gold_call": "_oracle_apply_workspace_program(program, S, H, gamma)",
        },
        # identity-only program returns the core Hamiltonian
        {
            "setup": "import numpy as np\nimport copy\n" + params + chain + "program = np.array([0, 0, 0, 0])\n",
            "call": "apply_workspace_program(*copy.deepcopy((program, S, H, gamma)))",
            "gold_call": "_oracle_apply_workspace_program(program, S, H, gamma)",
        },
        # empty program
        {
            "setup": "import numpy as np\nimport copy\n" + params + diatomic + "program = np.array([], dtype=int)\n",
            "call": "apply_workspace_program(*copy.deepcopy((program, S, H, gamma)))",
            "gold_call": "_oracle_apply_workspace_program(program, S, H, gamma)",
        },
        # primitive index outside the library
        {
            "setup": "import numpy as np\nimport copy\n" + params + chain + """program = np.array([1, 7])
def run_model():
    try:
        apply_workspace_program(program, S, H, gamma)
        return 0
    except ValueError:
        return 1
def run_oracle():
    try:
        _oracle_apply_workspace_program(program, S, H, gamma)
        return 0
    except ValueError:
        return 1
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
