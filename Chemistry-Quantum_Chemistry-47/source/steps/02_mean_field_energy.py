"""
Total mean-field energy of the site model for a given closed-shell density matrix.

The two-electron integrals obey zero differential overlap, so an integral survives only when each electron's two orbital indices coincide, and its value is then the Ohno interaction between the two sites. With these integrals the closed-shell Fock matrix of Roothaan theory has a Coulomb part that is diagonal and an exchange part that multiplies the density elementwise. The electronic energy is the standard closed-shell expression built from the density, the core Hamiltonian and the Fock matrix, and the cores repel through the bare Coulomb law plus a short-range exponential wall.

Returns
-------
float, the closed-shell electronic energy plus the core-core repulsion, as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def mean_field_energy(H: "np.ndarray", gamma: "np.ndarray", P: "np.ndarray", positions: "np.ndarray", params: dict) -> float:
    """Return the total mean-field energy of the density matrix P.

    Parameters
    ----------
    H : np.ndarray
        Shape (n, n) core Hamiltonian.
    gamma : np.ndarray
        Shape (n, n) Ohno interaction matrix.
    P : np.ndarray
        Shape (n, n) symmetric closed-shell density matrix.
    positions : np.ndarray
        Shape (n,) atom positions, used only for the core-core repulsion.
    params : dict
        Model constants; uses ``core_repulsion_strength`` (the height of the
        exponential wall) and ``core_repulsion_range`` (its decay length).

    Returns
    -------
    energy : float
        The closed-shell electronic energy of Roothaan theory evaluated with
        the zero-differential-overlap Fock matrix built from P, plus the
        core-core repulsion, which for every pair of atoms is the inverse
        separation plus the wall height times a decaying exponential of the
        separation over the decay length. Returned as a native Python float.

    Raises
    ------
    ValueError
        If H, gamma and P are not square matrices of one common size, or the
        number of positions differs from that size.
    """
    return energy

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_mean_field_energy(H: "np.ndarray", gamma: "np.ndarray", P: "np.ndarray", positions: "np.ndarray", params: dict) -> float:
    H = np.asarray(H, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    P = np.asarray(P, dtype=float)
    pos = np.asarray(positions, dtype=float)
    n = H.shape[0]
    if H.shape != (n, n) or gamma.shape != (n, n) or P.shape != (n, n) or pos.shape != (n,):
        raise ValueError("H, gamma and P must be (n, n) and positions (n,)")
    F = _build_fock(H, gamma, P)
    return float(0.5 * np.sum(P * (H + F)) + _core_repulsion(pos, params))

def _build_fock(H: "np.ndarray", gamma: "np.ndarray", P: "np.ndarray") -> "np.ndarray":
    return H + np.diag(gamma @ np.diag(P)) - 0.5 * P * gamma

def _core_repulsion(positions: "np.ndarray", params: dict) -> float:
    pos = np.asarray(positions, dtype=float)
    strength = float(params["core_repulsion_strength"])
    rho = float(params["core_repulsion_range"])
    iu = np.triu_indices(pos.shape[0], 1)
    R = np.abs(pos[:, None] - pos[None, :])[iu]
    return float(np.sum(1.0 / R + strength * np.exp(-R / rho)))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Test cases for this step."""
    params = ("params = {'onsite_energy': [-1.0, -1.6], 'onsite_repulsion': [0.9, 1.3], "
              "'kappa': 1.75, 'sigma': 1.1, 'core_repulsion_strength': 6.0, 'core_repulsion_range': 0.45}\n")
    diatomic = ("positions = np.array([0.0, 1.2752])\n"
                "types = np.array([0, 0])\n"
                "S, H, gamma = build_model_matrices(positions, types, params)\n")
    chain = ("positions = np.array([0.0, 1.0399, 2.7489, 3.7827])\n"
             "types = np.array([0, 1, 0, 1])\n"
             "S, H, gamma = build_model_matrices(positions, types, params)\n")
    return [
        # bonding orbital of a homonuclear diatomic, normalized with the overlap
        {
            "setup": "import numpy as np\nimport copy\n" + params + diatomic + "c = np.array([1.0, 1.0]) / np.sqrt(2.0 + 2.0 * S[0, 1])\nP = 2.0 * np.outer(c, c)\n",
            "call": "mean_field_energy(*copy.deepcopy((H, gamma, P, positions, params)))",
            "gold_call": "_oracle_mean_field_energy(H, gamma, P, positions, params)",
        },
        # one electron per site on a chain
        {
            "setup": "import numpy as np\nimport copy\n" + params + chain + "P = np.eye(4)\n",
            "call": "mean_field_energy(*copy.deepcopy((H, gamma, P, positions, params)))",
            "gold_call": "_oracle_mean_field_energy(H, gamma, P, positions, params)",
        },
        # no electrons: only the cores contribute
        {
            "setup": "import numpy as np\nimport copy\n" + params + chain + "P = np.zeros((4, 4))\n",
            "call": "mean_field_energy(*copy.deepcopy((H, gamma, P, positions, params)))",
            "gold_call": "_oracle_mean_field_energy(H, gamma, P, positions, params)",
        },
        # dense symmetric density so the exchange part is exercised
        {
            "setup": "import numpy as np\nimport copy\n" + params + chain + "rng = np.random.default_rng(5)\nX = rng.normal(size=(4, 2))\nP = 2.0 * X @ X.T / 3.0\n",
            "call": "mean_field_energy(*copy.deepcopy((H, gamma, P, positions, params)))",
            "gold_call": "_oracle_mean_field_energy(H, gamma, P, positions, params)",
        },
        # density of the wrong size is rejected
        {
            "setup": "import numpy as np\nimport copy\n" + params + diatomic + """P = np.eye(3)
def run_model():
    try:
        mean_field_energy(H, gamma, P, positions, params)
        return 0
    except ValueError:
        return 1
def run_oracle():
    try:
        _oracle_mean_field_energy(H, gamma, P, positions, params)
        return 0
    except ValueError:
        return 1
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
