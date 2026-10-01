"""
Analytic first derivative of the converged SCF energy with respect to every atom position.

Moving one atom changes every quantity of the model at once: the overlap with its neighbours, the core Hamiltonian, the site-site interaction and the core-core repulsion, and the converged orbitals readjust to the new matrices. The derivative of the converged energy with respect to that position is required exactly, so every route by which the position enters the energy has to be accounted for, including the fact that the orbitals stay orthonormal with respect to an overlap that itself depends on the positions.

Returns
-------
np.ndarray, shape (n,) float array, the derivative of the converged SCF total energy with respect to each atom position
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def scf_energy_gradient(positions: "np.ndarray", types: "np.ndarray", params: dict) -> "np.ndarray":
    """Return the exact gradient of the converged SCF total energy.

    Parameters
    ----------
    positions : np.ndarray
        Shape (n,) atom positions; n is even.
    types : np.ndarray
        Shape (n,) element types, 0 or 1.
    params : dict
        Model constants.

    Returns
    -------
    gradient : np.ndarray
        Shape (n,) float array whose entry a is the derivative of the converged
        SCF total energy (the energy the reference-energy routine returns) with
        respect to the position of atom a, holding the other atoms fixed. The
        derivative is the analytic one, evaluated at the SCF solution converged
        until the largest change of a density element between successive
        diagonalizations is below 1e-12, and is checked to 1e-11 in every
        component.

    Raises
    ------
    ValueError
        If the atom count is odd, or if positions and types disagree in length.
    """
    return gradient

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_scf_energy_gradient(positions: "np.ndarray", types: "np.ndarray", params: dict) -> "np.ndarray":
    pos = np.asarray(positions, dtype=float)
    typ = np.asarray(types, dtype=int)
    n = pos.shape[0]
    S, H, gamma, P, w, C = _converged_scf(pos, typ, params)
    kappa = float(params["kappa"])
    sigma = float(params["sigma"])
    strength = float(params["core_repulsion_strength"])
    rho = float(params["core_repulsion_range"])
    occ = C[:, : n // 2]
    weighted = 2.0 * (occ * w[: n // 2]) @ occ.T
    R = np.abs(pos[:, None] - pos[None, :])
    sign = np.sign(pos[:, None] - pos[None, :])
    off = ~np.eye(n, dtype=bool)
    hdiag = np.diag(H)
    p = np.diag(P)
    pair_weight = np.outer(p, p) - 0.5 * P ** 2
    iu = np.triu_indices(n, 1)
    core_force = -1.0 / R[iu] ** 2 - strength / rho * np.exp(-R[iu] / rho)
    gradient = np.zeros(n)
    for a in range(n):
        dR = np.zeros((n, n))
        dR[a, :] = sign[a, :]
        dR[:, a] = -sign[:, a]
        dS = S * (-R / sigma ** 2) * dR * off
        dgamma = -R * gamma ** 3 * dR
        dhdiag = -np.sum(dgamma * off, axis=1)
        dH = np.diag(dhdiag) + kappa * off * (dS * 0.5 * (hdiag[:, None] + hdiag[None, :])
                                              + S * 0.5 * (dhdiag[:, None] + dhdiag[None, :]))
        one_electron = np.sum(P * dH)
        two_electron = 0.5 * np.sum(pair_weight * dgamma)
        pulay = -np.sum(weighted * dS)
        cores = np.sum(core_force * dR[iu])
        gradient[a] = one_electron + two_electron + pulay + cores
    return gradient

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Test cases for this step."""
    params = ("params = {'onsite_energy': [-1.0, -1.6], 'onsite_repulsion': [0.9, 1.3], "
              "'kappa': 1.75, 'sigma': 1.1, 'core_repulsion_strength': 6.0, 'core_repulsion_range': 0.45}\n")
    return [
        # chain away from its minimum, all four components nonzero
        {
            "setup": "import numpy as np\nimport copy\n" + params + "positions = np.array([0.0, 1.1, 2.5, 3.9])\ntypes = np.array([0, 1, 0, 1])\n",
            "call": "scf_energy_gradient(*copy.deepcopy((positions, types, params)))",
            "gold_call": "_oracle_scf_energy_gradient(positions, types, params)",
            "tol": 1e-11,
        },
        # stretched heteronuclear diatomic
        {
            "setup": "import numpy as np\nimport copy\n" + params + "positions = np.array([0.0, 1.6])\ntypes = np.array([0, 1])\n",
            "call": "scf_energy_gradient(*copy.deepcopy((positions, types, params)))",
            "gold_call": "_oracle_scf_energy_gradient(positions, types, params)",
            "tol": 1e-11,
        },
        # compressed homonuclear diatomic on the wall
        {
            "setup": "import numpy as np\nimport copy\n" + params + "positions = np.array([0.0, 0.8])\ntypes = np.array([1, 1])\n",
            "call": "scf_energy_gradient(*copy.deepcopy((positions, types, params)))",
            "gold_call": "_oracle_scf_energy_gradient(positions, types, params)",
            "tol": 1e-11,
        },
        # atoms listed out of spatial order
        {
            "setup": "import numpy as np\nimport copy\n" + params + "positions = np.array([2.4, 0.0, 1.3, 3.5])\ntypes = np.array([1, 0, 1, 0])\n",
            "call": "scf_energy_gradient(*copy.deepcopy((positions, types, params)))",
            "gold_call": "_oracle_scf_energy_gradient(positions, types, params)",
            "tol": 1e-11,
        },
        # odd atom count is rejected
        {
            "setup": "import numpy as np\nimport copy\n" + params + """positions = np.array([0.0, 1.1, 2.2])
types = np.array([0, 1, 0])
def run_model():
    try:
        scf_energy_gradient(positions, types, params)
        return 0
    except ValueError:
        return 1
def run_oracle():
    try:
        _oracle_scf_energy_gradient(positions, types, params)
        return 0
    except ValueError:
        return 1
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
