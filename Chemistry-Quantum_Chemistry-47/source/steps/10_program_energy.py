"""
Predict the total energy of a molecule from a workspace program with a single diagonalization.

The output layer of the program solves the generalized eigenproblem of the molecule's workspace matrix with the overlap and occupies the n/2 lowest eigenvectors to form a closed-shell density. That density goes into the exact mean-field energy expression, so the energy is evaluated with the Fock matrix rebuilt from the predicted density rather than with the workspace matrix itself. No self-consistent iteration takes place.

Returns
-------
float, the one-shot total energy of the density built from the n/2 lowest generalized eigenvectors of the final workspace matrix, as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def program_energy(program: "np.ndarray", positions: "np.ndarray", types: "np.ndarray", params: dict) -> float:
    """Return the one-shot total energy predicted by a program.

    Parameters
    ----------
    program : np.ndarray
        Shape (N_f,) integer primitive indices in 0..6, applied in order to
        the core Hamiltonian as in the workspace-program library.
    positions : np.ndarray
        Shape (n,) atom positions; n is even.
    types : np.ndarray
        Shape (n,) element types, 0 or 1.
    params : dict
        Model constants.

    Returns
    -------
    energy : float
        The total mean-field energy of the density built from the n/2 lowest
        generalized eigenvectors of the final workspace matrix, using the
        Fock matrix rebuilt from that density, as a native Python float.

    Raises
    ------
    ValueError
        If the atom count is odd, the program contains an index outside 0..6,
        or positions and types disagree in length.
    """
    return energy

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_program_energy(program: "np.ndarray", positions: "np.ndarray", types: "np.ndarray", params: dict) -> float:
    pos = np.asarray(positions, dtype=float)
    n = pos.shape[0]
    if n % 2:
        raise ValueError("the model needs an even number of atoms")
    S, H, gamma = _oracle_build_model_matrices(pos, types, params)
    M = _oracle_apply_workspace_program(program, S, H, gamma)
    P = _occupied_density(M, S, n)
    return _oracle_mean_field_energy(H, gamma, P, pos, params)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Test cases for this step."""
    params = ("params = {'onsite_energy': [-1.0, -1.6], 'onsite_repulsion': [0.9, 1.3], "
              "'kappa': 1.75, 'sigma': 1.1, 'core_repulsion_strength': 6.0, 'core_repulsion_range': 0.45}\n")
    return [
        # mean-field-corrected program on the four-atom chain
        {
            "setup": "import numpy as np\nimport copy\n" + params + "positions = np.array([0.0, 1.0399, 2.7489, 3.7827])\ntypes = np.array([0, 1, 0, 1])\nprogram = np.array([6, 1, 4, 0])\n",
            "call": "program_energy(*copy.deepcopy((program, positions, types, params)))",
            "gold_call": "_oracle_program_energy(program, positions, types, params)",
        },
        # program with a generalized square on a heteronuclear diatomic
        {
            "setup": "import numpy as np\nimport copy\n" + params + "positions = np.array([0.0, 1.35])\ntypes = np.array([0, 1])\nprogram = np.array([2, 3, 5, 1])\n",
            "call": "program_energy(*copy.deepcopy((program, positions, types, params)))",
            "gold_call": "_oracle_program_energy(program, positions, types, params)",
        },
        # identity program equals the core-guess (first SCF iterate) energy
        {
            "setup": "import numpy as np\nimport copy\n" + params + "positions = np.array([0.0, 1.2752])\ntypes = np.array([0, 0])\nprogram = np.array([0, 0, 0, 0])\n",
            "call": "program_energy(*copy.deepcopy((program, positions, types, params)))",
            "gold_call": "_oracle_program_energy(program, positions, types, params)",
        },
        # program that only rescales and shifts by the overlap
        {
            "setup": "import numpy as np\nimport copy\n" + params + "positions = np.array([0.0, 1.0672])\ntypes = np.array([1, 1])\nprogram = np.array([6, 2, 6, 2])\n",
            "call": "program_energy(*copy.deepcopy((program, positions, types, params)))",
            "gold_call": "_oracle_program_energy(program, positions, types, params)",
        },
        # odd atom count
        {
            "setup": "import numpy as np\nimport copy\n" + params + """positions = np.array([0.0, 1.1, 2.3])
types = np.array([0, 1, 1])
program = np.array([1, 4])
def run_model():
    try:
        program_energy(program, positions, types, params)
        return 0
    except ValueError:
        return 1
def run_oracle():
    try:
        _oracle_program_energy(program, positions, types, params)
        return 0
    except ValueError:
        return 1
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
